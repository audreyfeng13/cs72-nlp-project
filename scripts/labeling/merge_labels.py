"""Compare two annotators' sheets, report agreement, and produce the adjudicated eval set.

Step 1 (default):  python scripts/labeling/merge_labels.py [--dir .]
    - validates both sheets, prints raw agreement + Cohen's kappa for toxic / sexist / category
      (pilot rows excluded unless --include-pilot)
    - writes eval_set_adjudication.csv: every row with A's and B's labels; the final_* columns are
      pre-filled where A and B agree and LEFT BLANK where they disagree -> discuss and fill in
      (third person decides unresolved cases)
Step 2:            python scripts/labeling/merge_labels.py --finalize
    - requires no blank final_* cells; writes eval_set_labeled.csv
      (id, text, source_account, created_at, toxic, sexist, category) used for model evaluation.
Agreement is always computed on the annotators' ORIGINAL labels, never the resolved ones.
"""
import argparse, os, sys
import pandas as pd
from sklearn.metrics import cohen_kappa_score

CATS = ['Threats/Intimidation', 'Sexualisation/Objectification', 'Stereotypes/Exclusion', 'Personal Insult', 'Safe']

ap = argparse.ArgumentParser()
ap.add_argument('--dir', default='.')
ap.add_argument('--finalize', action='store_true')
ap.add_argument('--include-pilot', action='store_true')
ap.add_argument('--force', action='store_true', help='overwrite an existing adjudication file')
a = ap.parse_args()
P = lambda f: os.path.join(a.dir, f)
ADJ, FINAL = P('eval_set_adjudication.csv'), P('eval_set_labeled.csv')

if a.finalize:
    adj = pd.read_csv(ADJ)
    blanks = adj[adj[['final_toxic', 'final_sexist', 'final_category']].isna().any(axis=1)]
    if len(blanks):
        sys.exit(f'{len(blanks)} rows still have blank final_* cells (ids: {blanks.id.tolist()[:20]}...). Resolve them first.')
    key = pd.read_csv(P('eval_set_ids.csv'))
    out = key.merge(adj[['id', 'final_toxic', 'final_sexist', 'final_category']], on='id') \
             .rename(columns={'final_toxic': 'toxic', 'final_sexist': 'sexist', 'final_category': 'category'})
    out[['toxic', 'sexist']] = out[['toxic', 'sexist']].astype(int)
    bad = out[~out.category.isin(CATS)]
    if len(bad): sys.exit(f'Invalid final categories in ids {bad.id.tolist()}; allowed: {CATS}')
    out.to_csv(FINAL, index=False)
    print(f'wrote {FINAL}: {len(out)} rows | toxic={out.toxic.sum()} sexist={out.sexist.sum()}')
    print(out.category.value_counts().to_string())
    sys.exit()

if os.path.exists(ADJ) and not a.force:
    sys.exit(f'{ADJ} exists (it may contain your adjudication work). Use --force to regenerate.')

A, B = (pd.read_csv(P(f)).rename(columns={'categories': 'category'}) for f in ('labeling_A.csv', 'labeling_B.csv'))  # sheets may call the column `categories`
assert (A.id.values == B.id.values).all() and (A.text.values == B.text.values).all(), 'sheets are out of sync'

def check(df, who):
    ok = True
    for c, allowed in (('toxic', {0, 1}), ('sexist', {0, 1}), ('category', set(CATS))):
        v = df[c].map(lambda x: int(x) if c != 'category' and pd.notna(x) and str(x).strip() != '' else x)
        v = v.astype(str).str.strip() if c == 'category' else v
        bad = df[~v.isin(allowed)]
        if len(bad):
            ok = False
            print(f'[{who}] {len(bad)} rows with missing/invalid {c} (e.g. ids {bad.id.tolist()[:10]}); allowed {sorted(map(str, allowed))}')
    inc = df[(df.category == 'Safe') & (df.toxic.astype(str).isin(['1', '1.0']) | df.sexist.astype(str).isin(['1', '1.0']))]
    if len(inc): print(f'[{who}] note: {len(inc)} rows have category=Safe but toxic/sexist=1 (ids {inc.id.tolist()[:10]})')
    inc = df[(df.category != 'Safe') & df.toxic.astype(str).isin(['0', '0.0']) & df.sexist.astype(str).isin(['0', '0.0'])]
    if len(inc): print(f'[{who}] note: {len(inc)} rows have a non-Safe category but toxic=0 and sexist=0 (ids {inc.id.tolist()[:10]})')
    return ok

if not (check(A, 'A') & check(B, 'B')):
    sys.exit('Fix the sheets above, then rerun.')
for df in (A, B):
    df[['toxic', 'sexist']] = df[['toxic', 'sexist']].astype(int); df['category'] = df.category.str.strip()

m = A.merge(B[['id', 'toxic', 'sexist', 'category', 'notes']], on='id', suffixes=('_a', '_b'))
ev = m if a.include_pilot else m[m.pilot == 0]
print(f'Agreement on {len(ev)} non-pilot rows ({len(m) - len(ev)} pilot rows excluded)')
for c in ('toxic', 'sexist', 'category'):
    x, y = ev[c + '_a'], ev[c + '_b']
    k = cohen_kappa_score(x, y) if x.nunique() > 1 or y.nunique() > 1 else float('nan')
    extra = ''
    if c != 'category':
        both = int(((x == 1) & (y == 1)).sum()); either = int(((x == 1) | (y == 1)).sum())
        extra = f' | positives A/B: {int(x.sum())}/{int(y.sum())}, both-positive {both}/{either} flagged by either'
    print(f'  {c:9s} raw agreement {(x == y).mean():.3f}  kappa {k:.3f}{extra}')
print('  (kappa is unstable with few positives - report raw agreement and counts alongside it)')

for c in ('toxic', 'sexist', 'category'):
    m['final_' + c] = m[c + '_a'].where(m[c + '_a'] == m[c + '_b'])
m['disagree_on'] = m.apply(lambda r: ','.join(c for c in ('toxic', 'sexist', 'category') if r[c + '_a'] != r[c + '_b']), axis=1)
cols = ['id', 'pilot', 'text', 'source_account', 'toxic_a', 'sexist_a', 'category_a', 'toxic_b', 'sexist_b',
        'category_b', 'disagree_on', 'final_toxic', 'final_sexist', 'final_category', 'notes_a', 'notes_b']
m[cols].sort_values(['disagree_on', 'id'], ascending=[False, True]).to_csv(ADJ, index=False)
n = (m.disagree_on != '').sum()
print(f'\n{n} rows need adjudication -> {ADJ} (disagreements sorted first; fill the blank final_* cells, then run --finalize)')
