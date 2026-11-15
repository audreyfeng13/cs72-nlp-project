# Social Media Comment Toxicity & Sexism Analyser

A CS72 NLP project that asks: **does toxicity training data (Civil Comments) help detect sexism (EXIST 2021)?**

It compares two models and wraps them in a small Streamlit demo:

| Model | Trained on | Output |
|---|---|---|
| **Baseline**: TF-IDF + Logistic Regression | Civil Comments (toxic = `target ≥ 0.5`) | P(toxic) |
| **Fine-tuned**: DistilBERT, two stages | Stage 1: 50k Civil Comments sample, then Stage 2: EXIST 2021 English (sexist vs. non-sexist) | P(sexist) |

There is also an optional multi-label **category head** (EXIST task 2) and a small **data-augmentation experiment** (hand-written "benevolent sexism" examples).

## Results (summary)

| Model | Eval set | AUC | Macro-F1 |
|---|---|---|---|
| TF-IDF + LogReg (zero-shot) | EXIST en/Twitter test (1,716) | 0.650 | 0.596 |
| DistilBERT, Stage 1 only (zero-shot) | EXIST en test (2,208) | 0.645 | 0.553 |
| DistilBERT, Stage 1 → Stage 2 | EXIST en test (2,208) | 0.808 | 0.738 |
| DistilBERT, EXIST only, lr 1e-5 (control) | EXIST en test (2,208) | 0.802 | 0.730 |
| DistilBERT, EXIST only, lr 2e-5 (control) | EXIST en test (2,208) | 0.820 | 0.739 |

Notes:
- The TF-IDF row uses the Twitter-only English test subset; the DistilBERT rows use all English test rows (Twitter + Gab). They are not directly comparable.
- Each configuration is a single seed. The Stage 1 → Stage 2 vs. EXIST-only difference (≈0.006–0.012 AUC) is within likely noise, so Civil Comments pre-training shows no clear benefit for sexism detection here. It does preserve toxicity ability (Jigsaw validation AUC 0.89 vs. 0.59 for EXIST-only).
- Category head (5 EXIST task-2 categories, threshold 0.5): macro-F1 0.246 on the English test set; recall is low and `misogyny-non-sexual-violence` is never predicted. EXIST task-2 labels are single-label per post, so this is effectively multi-class.

## 1. Get the datasets

The data is **not** included in this repository (see `.gitignore`). Place both folders in the project root.

**Civil Comments / Jigsaw Unintended Bias in Toxicity Classification**
- Download from Kaggle: <https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data> (a free Kaggle account and acceptance of the competition rules are required).
- Only `train.csv` is used. Expected path:
  `jigsaw-unintended-bias-in-toxicity-classification/train.csv`

**EXIST 2021 (sEXism Identification in Social neTworks)**
- Organisers' page: <https://nlp.uned.es/exist2021/> (dataset access/registration is handled by the EXIST organisers).
- Dataset paper: Rodríguez-Sánchez et al., *Overview of EXIST 2021*, Procesamiento del Lenguaje Natural 67 — <https://doi.org/10.26342/2021-67-17> / <https://arxiv.org/abs/2106.04908>
- Expected paths:
  - `EXIST2021-2025_datasets/2021 EXIST/training/EXIST2021_training.tsv`
  - `EXIST2021-2025_datasets/2021 EXIST/test/EXIST2021_test_labeled.tsv`

## 2. Set up the environment

Developed with Python 3.12 on an Apple Silicon Mac (PyTorch MPS GPU acceleration; scripts fall back to CPU).

```bash
python3.12 -m venv hw3-env
source hw3-env/bin/activate
pip install -r requirements.txt
```

`requirements.txt` is a full `pip freeze` of the development environment (macOS/arm64). On another platform you may need to adjust platform-specific packages (e.g. the `torch` build).

## 3. Train the models

Run everything from the project root (paths are relative).

```bash
# Baseline: TF-IDF + Logistic Regression on Civil Comments (~2 min)
# -> models/tfidf_lr.joblib, prints Jigsaw-validation and EXIST (en/Twitter) results
python tfidf_lr.py

# DistilBERT, Stage 1 (50k Civil Comments) -> Stage 2 (EXIST English) (~20 min on an M-series Mac)
# -> checkpoints/stage1, checkpoints/stage2
python distilbert_2stage.py

# Optional: multi-label category head (EXIST task 2), starts from checkpoints/stage2 (~3 min)
# -> checkpoints/stage2_categories
python train_categories.py
```

Useful flags for `distilbert_2stage.py`: `--n-jigsaw`, `--epochs1`, `--epochs2`, `--lr1`, `--lr2`, `--bs`, `--max-len`, `--twitter-only`, `--smoke` (tiny dry run), `--out <dir>`.

**Control: EXIST-only fine-tuning (no Stage 1)**
```bash
python distilbert_2stage.py --skip-stage1 --out checkpoints_baseline
python distilbert_2stage.py --skip-stage1 --lr2 2e-5 --out checkpoints_baseline_2e-5
```

**Augmentation experiment (optional)**
```bash
python augmented/build_augmented.py          # writes augmented/EXIST2021_training_augmented.tsv
python distilbert_2stage.py --skip-stage1 --init-from checkpoints/stage1 \
    --train-file augmented/EXIST2021_training_augmented.tsv --out checkpoints_aug
```
Sentences that also appear in the held-out test list (`augmented/heldout_test_sentences.json`) are excluded from the augmentation file.

## 4. Run the demo

The demo needs `models/tfidf_lr.joblib` and `checkpoints/stage2` (steps above).

```bash
streamlit run app.py
```

- Paste a comment or thread and click **Run Analysis** to see the baseline toxicity score and the fine-tuned sexism score.
- **Past Analyses (N)** lists previous runs (most recent first) from a local SQLite file, `analyses.db`, so history survives restarts. Click an entry to reload its scores; **Clear History** (with a confirmation checkbox) deletes them.
- Limitations: the fine-tuned model reads only the first 128 tokens, and the "Toxicity Category Analyses" table in the demo is still placeholder data (it is not wired to the category head).

## Repository layout

```
app.py                    Streamlit demo
tfidf_lr.py               TF-IDF + LogReg baseline (trains and saves)
distilbert_2stage.py      DistilBERT two-stage fine-tuning (+ baseline / augmentation modes)
train_categories.py       Multi-label category head
augmented/                Augmentation data builder and held-out test sentences
logs/                     Training logs
```
