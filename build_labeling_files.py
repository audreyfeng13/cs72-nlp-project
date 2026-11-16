"""Create independent labeling sheets for two annotators from eval_set_raw.csv.

Usage: python build_labeling_files.py [--raw eval_set_raw.csv] [--outdir .] [--pilot 25] [--force]
Writes:  labeling_A.csv, labeling_B.csv   (id, pilot, text, source_account, toxic, sexist, category, notes)
         eval_set_ids.csv                 (id, text, source_account, created_at)  -- key used by merge_labels.py
Rows are shuffled once (seed 0) so both annotators see the same order and are not primed by account.
The first --pilot rows are flagged pilot=1: label them, discuss, refine the guidelines; they are
excluded from the agreement statistics. Existing files are never overwritten without --force.
"""
import argparse, os, sys
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('--raw', default='eval_set_raw.csv')
ap.add_argument('--outdir', default='.')
ap.add_argument('--pilot', type=int, default=25)
ap.add_argument('--force', action='store_true')
a = ap.parse_args()

outs = [os.path.join(a.outdir, f) for f in ('labeling_A.csv', 'labeling_B.csv', 'eval_set_ids.csv')]
existing = [o for o in outs if os.path.exists(o)]
if existing and not a.force:
    sys.exit(f'Refusing to overwrite existing label files (your work!): {existing}. Use --force to replace.')

df = pd.read_csv(a.raw).sample(frac=1, random_state=0).reset_index(drop=True)
df.insert(0, 'id', range(1, len(df) + 1))
key_cols = ['id', 'text', 'source_account', 'created_at'] + (['selection'] if 'selection' in df else [])
df.to_csv(outs[2], index=False, columns=key_cols)  # 'selection' stays out of the annotators' sheets (avoids bias)

sheet = df[['id', 'text', 'source_account']].copy()
sheet.insert(1, 'pilot', (sheet.id <= a.pilot).astype(int))
for c in ('toxic', 'sexist', 'category', 'notes'):
    sheet[c] = ''
for o in outs[:2]:
    sheet.to_csv(o, index=False)
print(f'{len(sheet)} comments -> {outs[0]}, {outs[1]} (first {a.pilot} are pilot rows); key -> {outs[2]}')
