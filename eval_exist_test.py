"""All six models on the EXIST 2021 English test set (n = 2,208) in one table.

Usage: python -I eval_exist_test.py
Rows: Baseline (TF-IDF + LogReg, trained on Civil Comments toxicity), DistilBERT Stage 1 (Civil Comments only),
DistilBERT / BERT / RoBERTa Stage 2 (Civil Comments -> EXIST), Claude Haiku (zero-shot, answers from the cache
written by llm_zeroshot.py; no API call is made here). Positive class = sexist, threshold 0.5.
Writes exist_test_scores.csv (per-comment probabilities / Haiku answers).
"""
import json, numpy as np, pandas as pd, joblib, torch
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

TEST = 'EXIST2021-2025_datasets/2021 EXIST/test/EXIST2021_test_labeled.tsv'
df = pd.read_csv(TEST, sep='\t', dtype={'id': str})
df = df[df.language == 'en'].copy().reset_index(drop=True)
df['y'] = (df.task1 == 'sexist').astype(int)
print(f'EXIST en test: n={len(df)}, sexist {df.y.mean():.1%}, sources {df.source.value_counts().to_dict()}')

b = joblib.load('models/tfidf_lr.joblib')
df['p_tfidf'] = b['clf'].predict_proba(b['vec'].transform(df.text.astype(str)))[:, 1]

dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
for col, path in (('p_distilbert_stage1', 'checkpoints/stage1'), ('p_distilbert_stage2', 'checkpoints/stage2'),
                  ('p_bert_stage2', 'checkpoints_bert/stage2'), ('p_roberta_stage2', 'checkpoints_roberta/stage2')):
    tok = AutoTokenizer.from_pretrained(path); m = AutoModelForSequenceClassification.from_pretrained(path).to(dev).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(df), 64):
            enc = tok(df.text.iloc[i:i + 64].tolist(), truncation=True, max_length=128, padding=True, return_tensors='pt').to(dev)
            out.append(torch.softmax(m(**enc).logits, -1)[:, 1].cpu().numpy())
    df[col] = np.concatenate(out); del m
    print('scored', col, flush=True)


def parse(reply):  # same rule as llm_zeroshot.py; unparseable counts as "no"
    w = reply.strip().lower().lstrip('"\'*').split()
    w = w[0].strip('.,!:;"\'*') if w else ''
    return {'yes': 1, 'no': 0}.get(w, 0)


cache = {}
for line in open('logs/llm_zeroshot_preds.jsonl'):
    r = json.loads(line)
    if r['reply'].strip():
        cache[r['id']] = r['reply']
df['haiku_pred'] = df.id.map(lambda i: parse(cache[i]) if i in cache else np.nan)
print('Haiku answers found for', int(df.haiku_pred.notna().sum()), 'of', len(df))
df.drop(columns=['text']).to_csv('exist_test_scores.csv', index=False)

ROWS = [('Baseline (TF-IDF + LogReg)', 'p_tfidf'), ('DistilBERT, Stage 1 (Civil Comments only)', 'p_distilbert_stage1'),
        ('DistilBERT', 'p_distilbert_stage2'), ('BERT', 'p_bert_stage2'), ('RoBERTa', 'p_roberta_stage2'),
        ('Claude Haiku (zero-shot)', 'haiku_pred')]
rng = np.random.default_rng(0)


def table(d, title):
    y = d.y.values
    print(f'\n### {title}: n={len(d)}, sexist={y.sum()} ({y.mean():.1%})')
    print('| Model | Macro-F1 (95% CI) | Accuracy | Precision | Recall | AUC |\n|---|---|---|---|---|---|')
    for name, col in ROWS:
        s = d[col].values; pr = s.astype(int) if col == 'haiku_pred' else (s >= .5).astype(int)
        bs = []
        for _ in range(1000):
            i = rng.integers(0, len(y), len(y)); bs.append(f1_score(y[i], pr[i], average='macro'))
        lo, hi = np.percentile(bs, [2.5, 97.5])
        auc = 'n/a' if col == 'haiku_pred' else f'{roc_auc_score(y, s):.3f}'
        print(f'| {name} | {f1_score(y, pr, average="macro"):.3f} ({lo:.3f}-{hi:.3f}) | {accuracy_score(y, pr):.3f} | '
              f'{precision_score(y, pr):.3f} | {recall_score(y, pr):.3f} | {auc} |')


table(df, 'EXIST 2021 English test, all')
for src, g in df.groupby('source'):
    table(g, f'  subset: {src}')
