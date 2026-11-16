"""Evaluate toxicity and sexism predictions SEPARATELY against human labels.

Usage: python scripts/evaluation/evaluate_labeled.py scored.csv [--toxic-col toxic] [--sexist-col sexist] [--category-col category]
`scored.csv` = a labeled file that has been through score_eval_set.py (has p_<model> columns), e.g.
    python scripts/evaluation/score_eval_set.py eval_set_labeled.csv eval_set_labeled_scored.csv
    python scripts/evaluation/evaluate_labeled.py eval_set_labeled_scored.csv
Every model is scored against every label that exists. '*' marks the intended pairing:
toxic <- stage1 (Civil Comments) models, sexist <- stage2 (EXIST) models.
95% CIs are bootstrap percentile intervals; with few positives they are WIDE - that is the point.
"""
import argparse
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score

ap = argparse.ArgumentParser()
ap.add_argument('scored'); ap.add_argument('--toxic-col', default='toxic')
ap.add_argument('--sexist-col', default='sexist'); ap.add_argument('--category-col', default='category')
ap.add_argument('--boot', type=int, default=1000)
a = ap.parse_args()
df = pd.read_csv(a.scored)
models = [c[2:] for c in df if c.startswith('p_')]
rng = np.random.default_rng(0)


def ci(fn, y, p):
    vals = []
    for _ in range(a.boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        vals.append(fn(y[i], p[i]))
    return (np.percentile(vals, 2.5), np.percentile(vals, 97.5)) if vals else (np.nan, np.nan)


def table(label_col, name, primary_suffix, df=df):
    y = df[label_col].astype(int).values
    print(f'\n=== {name} (label column "{label_col}"): n={len(y)}, positives={y.sum()} ({y.mean():.1%}) ===')
    if y.sum() < 10 or (len(y) - y.sum()) < 10:
        print('  WARNING: fewer than 10 examples in one class - treat all metrics below as anecdotal.')
    if y.min() == y.max():
        print('  Only one class present: no metrics computable.'); return
    rows = []
    for m in models:
        p = df['p_' + m].values; pr = (p >= .5).astype(int)
        lo, hi = ci(lambda yy, pp: f1_score(yy, (pp >= .5).astype(int), zero_division=0), y, p)
        alo, ahi = ci(roc_auc_score, y, p)
        rows.append({'model': ('* ' if m.endswith(primary_suffix) else '  ') + m,
                     'AUC': f'{roc_auc_score(y, p):.3f} [{alo:.2f},{ahi:.2f}]',
                     'acc': f'{accuracy_score(y, pr):.3f}', 'macroF1': f'{f1_score(y, pr, average="macro"):.3f}',
                     'prec': f'{precision_score(y, pr, zero_division=0):.3f}',
                     'rec': f'{recall_score(y, pr, zero_division=0):.3f}',
                     'F1(pos)': f'{f1_score(y, pr, zero_division=0):.3f} [{lo:.2f},{hi:.2f}]'})
    print(pd.DataFrame(rows).to_string(index=False))


has_t, has_s = a.toxic_col in df, a.sexist_col in df
if has_t: table(a.toxic_col, 'TOXICITY', 'stage1')
if has_s: table(a.sexist_col, 'SEXISM', 'stage2')
if 'selection' in df:
    print('\nselection counts:', df.selection.value_counts().to_dict())
    un = df[df.selection != 'model_flagged']
    print('\n##### UNBIASED SUBSET (systematic + random; excludes model_flagged) - use this for prevalence and false-alarm rates #####')
    if has_t: table(a.toxic_col, 'TOXICITY, unbiased subset', 'stage1', un)
    if has_s: table(a.sexist_col, 'SEXISM, unbiased subset', 'stage2', un)
if not (has_t or has_s):
    raise SystemExit('No label columns found.')

if a.category_col in df:
    s2 = [m for m in models if m.endswith('stage2')]
    print(f'\n=== Share of comments flagged sexist (p>=0.5) by category ({a.category_col}) ===')
    g = df.groupby(a.category_col)
    out = pd.DataFrame({'n': g.size()})
    for m in s2:
        out[m] = g.apply(lambda d: (d['p_' + m] >= .5).mean()).round(2)
    print(out.to_string())
    print('  (for sexist categories this is recall per category; for non-sexist ones it is the false-alarm rate)')
