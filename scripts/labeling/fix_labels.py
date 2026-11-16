"""Apply the label corrections from data/labeled/label_review.csv -> data/labeled/eval_labeled_A.csv (+ data/labeled/label_fixes.csv change log).
Original (archive/eval_labeled_A_original.csv) is never modified. Rule used: the draft labeling guideline
(sexualising remarks / desire = Sexualisation + sexist; dismissing women's sport = Stereotypes/Exclusion + sexist;
no gendered content = not sexist). Model/Haiku output was NOT used to pick any label."""
import pandas as pd
SX, ST, PI = 'Sexualisation/Objectification', 'Stereotypes/Exclusion', 'Personal Insult'
FIX = {}  # id -> (changes dict, reason)
for i in (42, 45, 49, 57, 84, 89, 97, 98, 114, 127, 192, 254, 284, 317, 334):
    FIX[i] = ({'sexist': 1}, 'sexualising remark / expressed desire -> sexist (guideline)')
FIX[150] = ({'category': SX}, 'was Safe with sexist=1; romantic proposition -> Sexualisation')
for i in (1, 251, 228):
    FIX[i] = ({'sexist': 0}, 'no gendered content -> not sexist (still a Personal Insult)')
for i in (25, 308):
    FIX[i] = ({'sexist': 1}, 'dismisses women\'s sport -> Stereotypes/Exclusion is sexist (as ids 160, 272, 333)')
FIX[21] = ({'category': ST, 'sexist': 1}, 'same sentiment as id 25 ("no one watching")')
FIX[160] = ({'category': ST}, '"we want the men" = exclusion, not a personal insult')
FIX[272] = ({'category': ST}, 'dismisses women\'s sport = Stereotypes/Exclusion')
d = pd.read_csv('archive/eval_labeled_A_original.csv'); log = []
for i, (ch, why) in FIX.items():
    m = d.id == i; assert m.sum() == 1, i
    for c, v in ch.items():
        old = d.loc[m, c].iloc[0]
        if old != v: log.append({'id': i, 'column': c, 'old': old, 'new': v, 'reason': why, 'text': d.loc[m, 'text'].iloc[0]}); d.loc[m, c] = v
d.to_csv('data/labeled/eval_labeled_A.csv', index=False); pd.DataFrame(log).to_csv('data/labeled/label_fixes.csv', index=False)
print(f'{len(log)} cell changes on {len({r["id"] for r in log})} comments; sexist {pd.read_csv("archive/eval_labeled_A_original.csv").sexist.sum()} -> {d.sexist.sum()}')
