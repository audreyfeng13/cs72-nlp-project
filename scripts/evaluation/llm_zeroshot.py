"""Zero-shot LLM baseline: ask Claude Haiku whether each EXIST 2021 English test comment is sexist.

Usage:  export ANTHROPIC_API_KEY=...
        python scripts/evaluation/llm_zeroshot.py [--limit 20] [--model claude-haiku-5-5] [--workers 4]
Own data: python scripts/evaluation/llm_zeroshot.py --input data/labeled/eval_labeled_A.csv   (needs columns id, text; optional 0/1 column `sexist`)
          -> writes <input>_haiku.csv (id, haiku_reply, haiku_pred) for results_table.py --haiku

Predictions are cached in --cache (JSONL), so an interrupted run resumes where it left off.
"""
import argparse, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed

import anthropic
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score
from tqdm import tqdm

ap = argparse.ArgumentParser()
ap.add_argument('--model', default='claude-haiku-5-5')
ap.add_argument('--limit', type=int, default=None, help='only first N rows (for a quick test)')
ap.add_argument('--workers', type=int, default=4)
ap.add_argument('--max-tokens', type=int, default=1024, help='headroom in case the model emits reasoning before the answer')
ap.add_argument('--delay', type=float, default=0.0, help='seconds each worker sleeps after a call')
ap.add_argument('--cache', default='logs/llm_zeroshot_preds.jsonl')
ap.add_argument('--input', default=None, help='your own CSV (id, text[, sexist]) instead of the EXIST test set')
ap.add_argument('--out', default=None, help='with --input: output CSV (default <input>_haiku.csv)')
a = ap.parse_args()
if a.input and a.cache == 'logs/llm_zeroshot_preds.jsonl':
    a.cache = f'logs/llm_zeroshot_{os.path.splitext(os.path.basename(a.input))[0]}.jsonl'

if not os.environ.get('ANTHROPIC_API_KEY'):
    sys.exit('ANTHROPIC_API_KEY is not set.')

if a.input:
    df = pd.read_csv(a.input, dtype={'id': str})
    df['y'] = df['sexist'].astype(int) if 'sexist' in df else None
    if a.limit:
        df = df.head(a.limit)
    print(f'{a.input}: {len(df)} rows' + (f', sexist rate {df.y.mean():.3f}' if 'sexist' in df else ''))
else:
    TEST = 'EXIST2021-2025_datasets/2021 EXIST/test/EXIST2021_test_labeled.tsv'
    df = pd.read_csv(TEST, sep='\t', dtype={'id': str})
    df = df[df.language == 'en'].copy()
    df['y'] = (df.task1 == 'sexist').astype(int)
    if a.limit:
        df = df.head(a.limit)
    print(f'EXIST en test: {len(df)} rows, sexist rate {df.y.mean():.3f}')

PROMPT = "Is the following comment sexist? Answer only 'yes' or 'no'. Comment: {text}"
client = anthropic.Anthropic(max_retries=6)  # SDK backs off on 429/5xx


def parse(reply: str):
    """'yes' -> 1, 'no' -> 0, anything else -> None (unparseable)."""
    w = reply.strip().lower().lstrip('"\'*').split()
    w = w[0].strip('.,!:;"\'*') if w else ''
    return {'yes': 1, 'no': 0}.get(w)


def ask(test_id, text):
    r = client.messages.create(
        model=a.model, max_tokens=a.max_tokens,
        messages=[{'role': 'user', 'content': PROMPT.format(text=text)}])
    reply = ''.join(b.text for b in r.content if b.type == 'text')
    if a.delay:
        time.sleep(a.delay)
    return test_id, reply, r.stop_reason, [b.type for b in r.content]


# Load cache (keyed by test_case id + model)
os.makedirs(os.path.dirname(a.cache), exist_ok=True)
done = {}
if os.path.exists(a.cache):
    for line in open(a.cache):
        r = json.loads(line)
        if r['model'] == a.model and r['reply'].strip():  # empty replies are retried
            done[r['id']] = r['reply']
todo = [(str(i), t) for i, t in zip(df.id, df.text) if str(i) not in done]
print(f'{len(done)} cached, {len(todo)} to query with {a.model}')

stops = {}
with open(a.cache, 'a') as f, ThreadPoolExecutor(a.workers) as ex:
    futs = [ex.submit(ask, i, t) for i, t in todo]
    for fut in tqdm(as_completed(futs), total=len(futs), desc='Querying'):
        try:
            i, reply, stop, types = fut.result()
        except Exception as e:  # keep going; failed rows get retried on the next run
            tqdm.write(f'error: {type(e).__name__}: {e}')
            continue
        stops[f'{stop} {types}'] = stops.get(f'{stop} {types}', 0) + 1
        if not reply.strip():
            continue  # don't cache empty replies
        done[i] = reply
        f.write(json.dumps({'id': i, 'model': a.model, 'reply': reply, 'stop': stop}) + '\n'); f.flush()

print('stop_reason / content block types:', stops)

# Score
df['reply'] = df.id.astype(str).map(done)
missing = df.reply.isna().sum()
df['pred'] = df.reply.map(lambda r: parse(r) if isinstance(r, str) else None)
unparsed = df.pred.isna().sum() - missing
if missing:
    sys.exit(f'\n{missing}/{len(df)} rows got no response (see errors above). '
             'Not scoring a partial run; fix the error and rerun (cached rows are kept).')
print(f'\nNo response (API errors, rerun to retry): {missing} | unparseable replies: {unparsed}')
if unparsed:
    print(df[df.reply.notna() & df.pred.isna()].reply.value_counts().head(5).to_string())

# Unparseable/missing counted as "no" (not sexist) so every row is scored on the same set.
df['haiku_pred'] = df.pred.fillna(0).astype(int)
if a.input:
    out = a.out or os.path.splitext(a.input)[0] + '_haiku.csv'
    df[['id', 'reply', 'haiku_pred']].rename(columns={'reply': 'haiku_reply'}).to_csv(out, index=False)
    print(f'wrote {out}')
if df.y.notna().all():
    y, p = df.y.astype(int).values, df.haiku_pred.values
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, average='binary', zero_division=0)
    print(f'\n[{a.model} zero-shot | {a.input or "EXIST en test"}] n={len(df)}')
    print(f'accuracy={accuracy_score(y, p):.4f} precision={pr:.4f} recall={rc:.4f} '
          f'F1(sexist)={f1:.4f} macroF1={f1_score(y, p, average="macro"):.4f}')
