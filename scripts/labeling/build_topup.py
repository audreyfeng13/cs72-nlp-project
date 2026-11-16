"""Build a top-up set of extra comments to label, because the first 161 contain few sexist comments.

Usage: python -I scripts/labeling/build_topup.py <csv_dir> [--n-random 90] [--n-flagged 90]
Pool  = all comments that pass the SAME filters as build_eval_set.py (English, >=3 words, not emoji-only)
        minus the comments already in eval_set_raw.csv.
Picks = n-random   : equal number per account, uniformly at random (unbiased -> use for prevalence / false-alarm rate)
        n-flagged  : highest mean P(sexist) from the three Stage-2 models (model-selected -> adds positives,
                     but flatters the models; ALWAYS report it separately)
Writes: eval_set_topup_raw.csv      (text, source_account, created_at, selection)
        eval_set_combined_raw.csv   (original 161 as 'systematic' + top-up) -> feed to build_labeling_files.py
The `selection` tag is kept out of the annotators' sheets so it cannot bias their labels.
"""
import argparse, glob, os, re
import numpy as np, pandas as pd, torch
from langdetect import DetectorFactory, detect_langs
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ap = argparse.ArgumentParser()
ap.add_argument('csv_dir'); ap.add_argument('--n-random', type=int, default=90)
ap.add_argument('--n-flagged', type=int, default=90); ap.add_argument('--seed', type=int, default=0)
a = ap.parse_args()
DetectorFactory.seed = 0
MENTION_URL = re.compile(r'@\w[\w.]*|https?://\S+|www\.\S+')
words = lambda t: re.findall(r"[^\W\d_]+(?:['’][^\W\d_]+)*", MENTION_URL.sub(' ', t))  # same as build_eval_set.py


def is_english(t):
    try:
        langs = detect_langs(' '.join(words(t)) or t)
    except Exception:
        return False
    return bool(langs) and langs[0].lang == 'en'


done = set(pd.read_csv('eval_set_raw.csv').text)
parts = []
for f in sorted(glob.glob(os.path.join(a.csv_dir, '*.csv'))):
    d = pd.read_csv(f, usecols=['text', 'created_at']).dropna(subset=['text'])
    d['text'] = d.text.astype(str).str.strip()
    d = d[d.text.map(lambda t: len(words(t)) >= 3)]
    d = d[d.text.map(is_english)]
    d = d[~d.text.isin(done)].drop_duplicates('text')
    d['source_account'] = os.path.splitext(os.path.basename(f))[0]
    parts.append(d)
pool = pd.concat(parts, ignore_index=True)
print(f'pool: {len(pool)} unused comments passing the filters\n{pool.source_account.value_counts().to_string()}')

# score pool with the three Stage-2 models
dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
P = []
for path in ('checkpoints/stage2', 'checkpoints_bert/stage2', 'checkpoints_roberta/stage2'):
    tok = AutoTokenizer.from_pretrained(path); m = AutoModelForSequenceClassification.from_pretrained(path).to(dev).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(pool), 64):
            b = tok(pool.text.iloc[i:i + 64].tolist(), truncation=True, max_length=128, padding=True, return_tensors='pt').to(dev)
            out.append(torch.softmax(m(**b).logits, -1)[:, 1].cpu().numpy())
    P.append(np.concatenate(out)); del m
pool['mean_p'] = np.mean(P, axis=0)

# 1) random, equal per account
k = -(-a.n_random // pool.source_account.nunique())
rnd = pd.concat([g.sample(min(k, len(g)), random_state=a.seed) for _, g in pool.groupby('source_account')])  # (not groupby.apply: pandas 3 drops the key column)
rnd = rnd.sample(min(a.n_random, len(rnd)), random_state=a.seed).assign(selection='random')
# 2) model-flagged: top mean P(sexist) among the rest
rest = pool[~pool.text.isin(rnd.text)].sort_values('mean_p', ascending=False)
flg = rest.head(a.n_flagged).assign(selection='model_flagged')
print(f'\nrandom: {len(rnd)} | flagged: {len(flg)} (mean_p range {flg.mean_p.min():.2f}-{flg.mean_p.max():.2f}; '
      f'{int((flg.mean_p >= .5).sum())} have mean_p>=0.5; pool has {int((pool.mean_p >= .5).sum())} with mean_p>=0.5)')
print('flagged per account:', flg.source_account.value_counts().to_dict())

top = pd.concat([rnd, flg])[['text', 'source_account', 'created_at', 'selection']]
top.to_csv('eval_set_topup_raw.csv', index=False)
orig = pd.read_csv('eval_set_raw.csv').assign(selection='systematic')
pd.concat([orig, top], ignore_index=True).to_csv('eval_set_combined_raw.csv', index=False)
print(f'wrote eval_set_topup_raw.csv ({len(top)}) and eval_set_combined_raw.csv ({len(orig) + len(top)})')
