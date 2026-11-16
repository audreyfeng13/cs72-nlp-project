"""Results table (Macro-F1 / Accuracy / Precision / Recall per model) against human labels,
plus a per-category breakdown using the annotators' own categories.

Usage: python -I scripts/evaluation/results_table.py eval_labeled_A_scored.csv [--haiku haiku_preds.csv] [--label-cat categories|category]
  - scored csv   : labels (toxic, sexist, category) + p_<model> columns from score_eval_set.py
  - --haiku csv  : id, haiku_pred  (from llm_zeroshot.py --input ...); optional, adds the Claude Haiku row (sexism only)
Rows: Baseline = TF-IDF + logistic regression (Civil Comments toxicity model); DistilBERT Stage 1 = Civil Comments only;
DistilBERT / BERT / RoBERTa = Stage 2 (Civil Comments -> EXIST). Precision and recall are for the positive class.
The baseline and Stage 1 rows are toxicity models; against the sexism label they act as zero-shot transfer baselines.
Tables are printed for all labeled comments and for the UNBIASED subset (excludes model_flagged selections).
"""
import argparse
import numpy as np, pandas as pd, joblib
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

ap = argparse.ArgumentParser()
ap.add_argument('scored'); ap.add_argument('--haiku'); ap.add_argument('--cat-col', default='category')
ap.add_argument('--boot', type=int, default=1000)
a = ap.parse_args()
df = pd.read_csv(a.scored)
if 'p_tfidf' not in df:
    b = joblib.load('models/tfidf_lr.joblib')
    df['p_tfidf'] = b['clf'].predict_proba(b['vec'].transform(df.text.astype(str)))[:, 1]
if a.haiku:
    h = pd.read_csv(a.haiku)[['id', 'haiku_pred']]
    df = df.merge(h, on='id', how='left')
    print(f"Haiku predictions found for {df.haiku_pred.notna().sum()}/{len(df)} comments")
ROWS = [('Baseline (TF-IDF + LogReg)', 'p_tfidf'), ('DistilBERT, Stage 1 (Civil Comments only)', 'p_distilbert_stage1'),
        ('DistilBERT', 'p_distilbert_stage2'), ('BERT', 'p_bert_stage2'), ('RoBERTa', 'p_roberta_stage2')]
if 'haiku_pred' in df and df.haiku_pred.notna().any():
    ROWS.append(('Claude Haiku (zero-shot, sexism only)', 'haiku_pred'))
rng = np.random.default_rng(0)


def preds(d, col):
    return d[col].astype(float).values if col == 'haiku_pred' else (d[col].values >= .5).astype(int)


def metrics(y, pr):
    return (f1_score(y, pr, average='macro'), accuracy_score(y, pr),
            precision_score(y, pr, zero_division=0), recall_score(y, pr, zero_division=0))


def table(d, label, title):
    y = d[label].astype(int).values
    print(f'\n### {title}: n={len(y)}, positives={y.sum()} ({y.mean():.1%})' +
          ('  (WARNING: very few positives - anecdotal)' if y.sum() < 10 else ''))
    print('| Model | Macro-F1 (95% CI) | Accuracy | Precision | Recall |\n|---|---|---|---|---|')
    for name, col in ROWS:
        if col == 'haiku_pred' and label != 'sexist':
            print(f'| {name} | n/a | n/a | n/a | n/a |'); continue
        dd = d[d[col].notna()]; yy = dd[label].astype(int).values; pr = preds(dd, col).astype(int)
        f1, acc, pre, rec = metrics(yy, pr)
        bs = []
        for _ in range(a.boot):
            i = rng.integers(0, len(yy), len(yy))
            if yy[i].min() != yy[i].max():
                bs.append(f1_score(yy[i], pr[i], average='macro'))
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f'| {name} | {f1:.3f} ({lo:.2f}-{hi:.2f}) | {acc:.3f} | {pre:.3f} | {rec:.3f} |')


un = df[df.selection != 'model_flagged'] if 'selection' in df else df
for label, nm in (('sexist', 'SEXISM'), ('toxic', 'TOXICITY')):
    if label not in df:
        continue
    table(df, label, f'{nm}, all labeled comments')
    if 'selection' in df:
        table(un, label, f'{nm}, unbiased subset (systematic + random)')

cat = a.cat_col
print(f'\n### Per-category breakdown (annotator categories), all labeled comments')
g = df.groupby(cat)
out = pd.DataFrame({'n': g.size(), '% sexist': (g.sexist.mean() * 100).round(0)})
if 'toxic' in df:
    out['% toxic'] = (g.toxic.mean() * 100).round(0)
for name, col in ROWS:
    if col == 'haiku_pred':
        out['flag: Haiku (sexism)'] = g[col].apply(lambda x: x.mean()).round(2)
    else:
        out['flag: ' + name.split(' (')[0].replace('DistilBERT, Stage 1', 'DistilBERT S1')] = g[col].apply(lambda x: (x >= .5).mean()).round(2)
print(out.to_string())
print('\n"flag" = share of that category the model flags at p >= 0.5 (Haiku: answered yes). For the sexist-heavy categories this is a')
print('recall; for Safe it is the false-alarm rate. Baseline and Stage 1 are toxicity models, so read their flags against % toxic.')
