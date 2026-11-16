"""Build eval_set_raw.csv from the collected Instagram comment CSVs.

Usage: python -I scripts/labeling/build_eval_set.py <csv_dir> [out.csv] [--target 17]
Steps: load CSVs -> keep text, created_at -> drop empty / <3 words / emoji-only / non-English
       -> systematic sampling (every Nth row) per account -> save text, source_account, created_at.
"""
import glob, os, re, sys
import pandas as pd
from langdetect import DetectorFactory, detect_langs

DetectorFactory.seed = 0  # make langdetect deterministic
csv_dir = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else 'eval_set_raw.csv'
TARGET = int(sys.argv[sys.argv.index('--target') + 1]) if '--target' in sys.argv else 20  # ~150-180 / 9 accounts

MENTION_URL = re.compile(r'@\w[\w.]*|https?://\S+|www\.\S+')


def body(t):
    """Text with @mentions and URLs removed (what counts as 'actual words')."""
    return MENTION_URL.sub(' ', t)


def words(t):
    return [w for w in re.findall(r"[^\W\d_]+(?:['’][^\W\d_]+)*", body(t))]  # alphabetic tokens


def is_english(t):
    try:
        langs = detect_langs(' '.join(words(t)) or t)
    except Exception:
        return False
    return bool(langs) and langs[0].lang == 'en'


rows, stats = [], []
for f in sorted(glob.glob(os.path.join(csv_dir, '*.csv'))):
    acct = os.path.splitext(os.path.basename(f))[0]
    d = pd.read_csv(f, usecols=['text', 'created_at'])  # drops owner_id, username, profile_url, profile_pic_url (and id)
    n0 = len(d)
    d = d.dropna(subset=['text']).copy()
    d['text'] = d.text.astype(str).str.strip()
    d = d[d.text != '']
    n_empty = n0 - len(d)
    nw = d.text.map(lambda t: len(words(t)))
    n_emoji = int((nw == 0).sum())                     # no letters at all: emoji / punctuation / mention-only
    n_short = int(((nw > 0) & (nw < 3)).sum())
    d = d[nw >= 3]
    en = d.text.map(is_english)
    n_lang = int((~en).sum())
    d = d[en].reset_index(drop=True)
    # systematic sample: every Nth row so that at most TARGET rows are kept (all rows if already <= TARGET)
    N = max(1, -(-len(d) // TARGET))
    s = d.iloc[::N].copy()
    s['source_account'] = acct
    rows.append(s)
    stats.append((acct, n0, n_empty, n_emoji, n_short, n_lang, len(d), N, len(s)))

res = pd.concat(rows, ignore_index=True)[['text', 'source_account', 'created_at']]
res.to_csv(out, index=False)
print(pd.DataFrame(stats, columns=['account', 'raw', 'empty', 'emoji/no-letters', '<3 words', 'non-English',
                                   'kept', 'every_N', 'sampled']).to_string(index=False))
print(f'\nTotal sampled: {len(res)} -> {out}')
