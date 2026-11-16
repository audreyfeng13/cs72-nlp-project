"""Evaluate Claude Haiku's category predictions against the annotators' categories.

Usage: export ANTHROPIC_API_KEY=...
       python scripts/evaluation/llm_categories.py --input data/labeled/eval_labeled_A.csv [--limit 10] [--workers 4]
Input CSV needs columns id, text and (for scoring) category. Writes <input>_cat.csv (id, cat_reply, cat_pred).
Replies are cached in logs/llm_categories_<input>.jsonl so an interrupted run resumes.
Unparseable replies are counted as wrong (they are listed in the output).
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))  # project root, for category_llm
from concurrent.futures import ThreadPoolExecutor, as_completed

import anthropic
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from tqdm import tqdm

from category_llm import CATEGORY_MODEL, CATEGORY_NAMES, predict_category

ap = argparse.ArgumentParser()
ap.add_argument('--input', required=True)
ap.add_argument('--model', default=CATEGORY_MODEL)
ap.add_argument('--limit', type=int, default=None)
ap.add_argument('--workers', type=int, default=4)
ap.add_argument('--out', default=None)
a = ap.parse_args()
if not os.environ.get('ANTHROPIC_API_KEY'):
    sys.exit('ANTHROPIC_API_KEY is not set.')

stem = os.path.splitext(os.path.basename(a.input))[0]
cache_path, out_path = f'logs/llm_categories_{stem}.jsonl', a.out or os.path.splitext(a.input)[0] + '_cat.csv'
df = pd.read_csv(a.input, dtype={'id': str})
if a.limit:
    df = df.head(a.limit)
client = anthropic.Anthropic(max_retries=6)

done = {}
os.makedirs('logs', exist_ok=True)
if os.path.exists(cache_path):
    for line in open(cache_path):
        r = json.loads(line)
        if r['model'] == a.model and r['reply'].strip():
            done[r['id']] = r['reply']
todo = [(str(i), t) for i, t in zip(df.id, df.text) if str(i) not in done]
print(f'{a.input}: {len(df)} rows | {len(done)} cached, {len(todo)} to query with {a.model}')

with open(cache_path, 'a') as f, ThreadPoolExecutor(a.workers) as ex:
    futs = {ex.submit(predict_category, client, t, a.model): i for i, t in todo}
    for fut in tqdm(as_completed(futs), total=len(futs), desc='Querying'):
        try:
            _, reply = fut.result()
        except Exception as e:
            tqdm.write(f'error: {type(e).__name__}: {e}'); continue
        if reply.strip():
            done[futs[fut]] = reply
            f.write(json.dumps({'id': futs[fut], 'model': a.model, 'reply': reply}) + '\n'); f.flush()

from category_llm import parse_category
df['cat_reply'] = df.id.astype(str).map(done)
missing = int(df.cat_reply.isna().sum())
if missing:
    sys.exit(f'{missing}/{len(df)} rows got no response (see errors above). Rerun to retry; cached rows are kept.')
df['cat_pred'] = df.cat_reply.map(parse_category)
df[['id', 'cat_reply', 'cat_pred']].to_csv(out_path, index=False)
print(f'wrote {out_path} | unparseable replies: {int(df.cat_pred.isna().sum())}')

if 'category' not in df:
    sys.exit('No `category` column: predictions written, nothing to score.')


def report(d, title):
    y, p = d.category.values, d.cat_pred.fillna('UNPARSED').values
    labels = [c for c in CATEGORY_NAMES if (d.category == c).any()]
    pr, rc, f1, sup = precision_recall_fscore_support(y, p, labels=labels, zero_division=0)
    print(f'\n### {title}: n={len(d)} | accuracy {accuracy_score(y, p):.3f} | macro-F1 over {len(labels)} categories present {f1.mean():.3f}')
    print('| Category | Support | Precision | Recall | F1 |\n|---|---|---|---|---|')
    for l, s_, a_, b_, c_ in zip(labels, sup, pr, rc, f1):
        print(f'| {l} | {s_} | {a_:.3f} | {b_:.3f} | {c_:.3f} |')
    missing_cats = [c for c in CATEGORY_NAMES if c not in labels]
    if missing_cats:
        print(f'(no comments labeled {", ".join(missing_cats)}; predicted {sum((d.cat_pred == c).sum() for c in missing_cats)} times)')
    ys, ps = (d.category != 'Safe'), (d.cat_pred.fillna('Safe') != 'Safe')
    tp = int((ys & ps).sum())
    print(f'Not-Safe vs Safe: precision {tp / max(int(ps.sum()), 1):.3f}, recall {tp / max(int(ys.sum()), 1):.3f}')
    cols = CATEGORY_NAMES + (['UNPARSED'] if (p == 'UNPARSED').any() else [])
    cm = pd.DataFrame(confusion_matrix(y, p, labels=cols), index=[f'true {c}' for c in cols], columns=[f'pred {c[:12]}' for c in cols])
    print(cm.loc[(cm.sum(axis=1) > 0)].to_string())


report(df, 'All labeled comments')
if 'selection' in df:
    report(df[df.selection != 'model_flagged'], 'Unbiased subset (systematic + random)')
