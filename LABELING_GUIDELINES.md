# Labeling guidelines (DRAFT for the team to confirm)

Every comment gets three labels: `toxic` (0/1), `sexist` (0/1) and one `category`.
Label the comment's own text. Use the account only to understand context (e.g. a women's sport).

## Three kinds of negative comment (the distinction the project depends on)

| Kind | `sexist` | `toxic` | Typical category |
|---|---|---|---|
| **Gendered harassment**: hostility, stereotypes, sexualisation or dismissal because of gender | 1 | usually 1, sometimes 0 (patronising or "friendly" sexism) | Sexualisation/Objectification, Stereotypes/Exclusion, Threats/Intimidation |
| **General toxicity**: insults or mockery with no gendered content | 0 | 1 | Personal Insult |
| **Legitimate criticism**: blunt comments about performance, tactics, refereeing or decisions, with no insult | 0 | 0 | Safe |

Examples (synthetic): "She needs to improve her second serve" is legitimate criticism (Safe). "You are a terrible player, retire already" is general toxicity (Personal Insult). "Go back to the kitchen" is gendered harassment (Stereotypes/Exclusion). Criticism that adds a gendered insult is gendered harassment.

## `toxic` (0/1)
1 if the comment is rude, disrespectful, insulting, threatening or likely to make someone want to leave the conversation. Criticism or disagreement without hostility is 0.

## `sexist` (0/1)
1 if the comment expresses prejudice, stereotypes, hostility, sexualisation or discrimination because of gender (here, usually against women), including subtle or friendly forms ("good for a girl") and unwelcome sexual comments or propositions.
0 if the comment criticises or reports sexism ("the sexism in this thread is exhausting") or has no gendered content.

## `category` (exactly one)
- **Threats/Intimidation**: threats, wishing injury on someone, doxxing, or encouraging self-harm.
- **Sexualisation/Objectification**: unsolicited sexual remarks, body-part fixation, or lewd or romantic propositions.
- **Stereotypes/Exclusion**: gender or identity generalisations, gatekeeping, or dismissing women's sports as inferior.
- **Personal Insult**: direct name-calling, mockery, non-sexual appearance policing, or targeted character attacks.
- **Safe**: none of the above.

If several apply, take the first in this order (proposed): Threats/Intimidation, Sexualisation/Objectification, Stereotypes/Exclusion, Personal Insult, Safe.
`Safe` should go with `toxic = 0` and `sexist = 0`. Other categories do not fix the two yes/no labels (for example, a Sexualisation comment can be non-toxic), so label those independently.
The category list and definitions were drafted with Gemini and checked for mutual exclusivity.

## Decide together BEFORE the next pilot (write the answer here)
- Plain appearance compliments ("you look amazing"): proposed **Safe, not sexist**; remarks that sexualise the body or express desire: **Sexualisation/Objectification, sexist**. Confirm or change: ____
- Comments that quote or mock sexism in order to criticise it: proposed **Safe, not sexist**. Confirm or change: ____
- Sexual or romantic remarks that are clearly about men: proposed **not sexist** (the project is about sexism against women). Confirm or change: ____
- Non-English or unreadable comments: **Safe**, with a note. Confirm or change: ____

The 275 labels made so far were created before this written rule. Recheck them against it, in particular any comment with `Safe` and `sexist = 1`.

## Procedure
1. `python build_labeling_files.py` creates `labeling_A.csv` and `labeling_B.csv` (the first 25 rows are a pilot).
2. Both annotators label the pilot independently, compare, and refine this document. Pilot rows are excluded from agreement statistics.
3. Each annotator labels the rest alone, without seeing the other's sheet.
4. `python merge_labels.py` reports agreement (raw agreement and Cohen's kappa for `toxic`, `sexist` and `category`) and writes an adjudication sheet. Resolve every blank `final_*` cell together; a third person decides if you cannot agree.
5. `python merge_labels.py --finalize`, then `score_eval_set.py` and `evaluate_labeled.py`.
