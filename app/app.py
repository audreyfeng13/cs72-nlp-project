"""Social Media Comment Toxicity Analyser - demo UI.

Three REAL scores: `baseline_score` is a TF-IDF + LogReg model trained on Civil Comments
toxicity (P(toxic)); `stage1_score` is DistilBERT after Stage 1 only (Civil Comments, P(toxic));
`finetuned_score` is DistilBERT after Stage 2 (Civil Comments -> EXIST 2021 English, P(sexist)).
One toxicity score (chosen in the app; TF-IDF by default) and the sexism score are each
thresholded, and the pair is placed in a 2x2 decision table (Neutral / Subtle sexism /
Insult not sexist / Sexist and toxic) that drives the verdict.

The comment's CATEGORY (Threats/Intimidation, Sexualisation/Objectification, Stereotypes/Exclusion,
Personal Insult, Safe: the project's own annotation categories) is predicted by Claude Haiku using
those definitions (category_llm.py); it needs ANTHROPIC_API_KEY, and without it the rest of the app
still works. FLAGGED WORDS are found by deleting each word in turn and ranking words by how much
the relevant score drops (sexism score, or the toxicity score for Personal Insult).

Run (from the project root):  streamlit run app/app.py
"""
import os
import re
import sqlite3
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from category_llm import CATEGORY_MODEL, CATEGORY_NAMES, predict_category

ROOT = Path(__file__).resolve().parent.parent  # project root (this file lives in app/)
BASELINE_PATH = ROOT / "models" / "tfidf_lr.joblib"
STAGE1_DIR = ROOT / "checkpoints" / "stage1"
FINETUNED_DIR = ROOT / "checkpoints" / "stage2"
# Which toxicity score feeds the 2x2 decision table (label shown in the app -> key in the result dict).
TOX_SOURCES = {"TF-IDF baseline": "baseline", "Stage 1 DistilBERT": "stage1"}
DB_PATH = Path(os.environ.get("ANALYSES_DB", ROOT / "analyses.db"))
HISTORY_LIMIT = 200      # most recent rows listed in the Past Analyses panel
MAX_LEN = 128            # same truncation the DistilBERT model was trained with

TOXIC_THRESHOLD = 0.50   # toxicity score at/above this -> Toxic = 1 (adjustable in the app)
SEXIST_THRESHOLD = 0.50  # sexism score at/above this -> Sexist = 1 (adjustable in the app)
NEAR_MARGIN = 0.20       # a score within this margin below its threshold is called "close"
TOP_WORDS = 3            # flagged words shown per comment
MAX_WORDS = 80           # words examined for flagging (the models only see ~128 tokens anyway)
MIN_DROP = 0.02          # a word must lower the score by at least this much to be flagged

st.set_page_config(page_title="Social Media Comment Toxicity Analyser", layout="wide")


# ---------------------------------------------------------------- models
@st.cache_resource(show_spinner="Loading models...")
def _load_baseline():
    bundle = joblib.load(BASELINE_PATH)
    return bundle["vec"], bundle["clf"]


@st.cache_resource(show_spinner="Loading models...")
def _load_distilbert(path: str):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).to(device).eval()
    return tok, model, device


def _distil_probs(path, texts: list) -> list:
    """P(class 1) from a DistilBERT checkpoint for each text. Input is truncated to MAX_LEN tokens, as in training."""
    tok, model, device = _load_distilbert(str(path))
    out = []
    for i in range(0, len(texts), 32):
        enc = tok(texts[i:i + 32], truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            out += torch.softmax(model(**enc).logits, dim=-1)[:, 1].cpu().tolist()
    return out


def _distil_prob(path, text: str) -> float:
    return _distil_probs(path, [text])[0]


def _tfidf_probs(texts: list) -> list:
    vec, clf = _load_baseline()
    return clf.predict_proba(vec.transform(texts))[:, 1].tolist()


def baseline_score(text: str) -> float:
    """P(toxic) from the TF-IDF + Logistic Regression model (trained on Civil Comments)."""
    return _tfidf_probs([text])[0]


def stage1_score(text: str) -> float:
    """P(toxic) from DistilBERT after Stage 1 only (trained on Civil Comments, no sexism data)."""
    return _distil_prob(STAGE1_DIR, text)


def finetuned_score(text: str) -> float:
    """P(sexist) from the stage-2 DistilBERT model (Civil Comments -> EXIST English)."""
    return _distil_prob(FINETUNED_DIR, text)


# ---------------------------------------------------------------- category (Claude Haiku) and flagged words
def category_of(text: str):
    """Predict the comment's category with Claude Haiku. Returns (category or None, message if None)."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None, "Category prediction needs ANTHROPIC_API_KEY (Claude Haiku). Set it and restart the app."
    try:
        import anthropic
    except ImportError:
        return None, "Category prediction needs the `anthropic` package."
    try:
        category, _ = predict_category(anthropic.Anthropic(max_retries=3), text, CATEGORY_MODEL)
    except Exception as e:  # network, auth, rate limit...
        return None, f"Category prediction failed ({type(e).__name__}); the scores above are unaffected."
    return (category, "") if category else (None, "Claude's reply did not name one of the categories.")


def _scorer(kind: str):
    """Batch scoring function for 'sexism', 'baseline' (TF-IDF toxicity) or 'stage1' (DistilBERT toxicity)."""
    if kind == "baseline":
        return _tfidf_probs
    return lambda texts: _distil_probs(STAGE1_DIR if kind == "stage1" else FINETUNED_DIR, texts)


@st.cache_data(show_spinner=False)
def flagged_words(text: str, kind: str) -> list:
    """Words whose removal lowers the `kind` score the most (word-deletion / occlusion attribution)."""
    words = text.split()[:MAX_WORDS]
    if len(words) < 2:
        return []
    score = _scorer(kind)
    base = score([" ".join(words)])[0]
    variants = [" ".join(words[:i] + words[i + 1:]) for i in range(len(words))]
    ranked = sorted(zip((base - p for p in score(variants)), words), key=lambda t: t[0], reverse=True)
    out = []
    for drop, w in ranked:
        w = w.strip(".,!?:;\"'()[]")
        if drop >= MIN_DROP and w and w.lower() not in [o.lower() for o in out]:
            out.append(w)
        if len(out) == TOP_WORDS:
            break
    return out


def analyze(text: str) -> dict:
    """Real baseline / Stage 1 / fine-tuned scores plus the Haiku category."""
    category, note = category_of(text)
    return {"baseline": round(baseline_score(text), 2), "stage1": round(stage1_score(text), 2),
            "fine_tuned": round(finetuned_score(text), 2), "category": category, "category_note": note}


# Decision table, keyed by (toxic, sexist): (quadrant name, streamlit level, verdict, what happens).
# Edit the verdict wording / actions here to match your moderation policy.
QUADRANTS = {
    (0, 0): ("Neutral", "success", "NO ACTION NEEDED",
             "Neither score reaches its threshold."),
    (0, 1): ("Subtle sexism", "warning", "FLAGGED FOR REVIEW (LOW PRIORITY)",
             "Sexist content without toxic language, such as a patronising or stereotyping remark. "
             "Queued for a moderator at low priority."),
    (1, 0): ("Insult (not sexist)", "warning", "TOXIC - APPLY GENERAL CONDUCT POLICY",
             "Toxic or insulting but not sexist. Handle under the general conduct rules."),
    (1, 1): ("Sexist and toxic", "error", "FLAGGED FOR MODERATOR REVIEW",
             "Both sexist and toxic. Queued for a moderator at high priority."),
}


def classify(toxicity: float, sexism: float, tox_thr: float, sex_thr: float, tox_name: str = "") -> dict:
    """Threshold both scores and look the pair up in the decision table."""
    t, s = int(toxicity >= tox_thr), int(sexism >= sex_thr)
    name, level, verdict, action = QUADRANTS[(t, s)]
    near = [n for n, flag, score, thr in (("toxicity", t, toxicity, tox_thr), ("sexism", s, sexism, sex_thr))
            if not flag and score >= thr - NEAR_MARGIN]
    feedback = (f"Toxicity{f' ({tox_name})' if tox_name else ''} {toxicity:.2f} ({'≥' if t else '<'} {tox_thr:.2f}) and sexism {sexism:.2f} "
                f"({'≥' if s else '<'} {sex_thr:.2f}) place this comment in **{name}**. {action}")
    if near:
        feedback += f" The {' and '.join(near)} score is close to its threshold, so this is a borderline call."
    return dict(toxic=t, sexist=s, name=name, level=level, verdict=verdict, feedback=feedback)


def quadrant_grid(active: tuple) -> str:
    """The 2x2 decision table as HTML, with the cell this comment falls in highlighted."""
    rgb = {"success": "46,160,67", "warning": "224,155,0", "error": "229,57,53"}

    def cell(t: int, s: int) -> str:
        name, level = QUADRANTS[(t, s)][:2]
        style = "padding:8px 10px;border:1px solid rgba(128,128,128,.45);"
        if (t, s) == active:
            style += f"background:rgba({rgb[level]},.28);font-weight:600;outline:2px solid rgb({rgb[level]});"
        return f"<td style='{style}'>{name}</td>"

    th = "style='padding:6px 10px;border:1px solid rgba(128,128,128,.45);opacity:.8'"
    return ("<table style='width:100%;border-collapse:collapse;font-size:.9rem;text-align:center;margin:4px 0 12px'>"
            f"<tr><th {th}></th><th {th}>Sexist = 0</th><th {th}>Sexist = 1</th></tr>"
            f"<tr><th {th}>Toxic = 0</th>{cell(0, 0)}{cell(0, 1)}</tr>"
            f"<tr><th {th}>Toxic = 1</th>{cell(1, 0)}{cell(1, 1)}</tr></table>")


# ---------------------------------------------------------------- history (SQLite)
def _db() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.execute("""CREATE TABLE IF NOT EXISTS analyses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        comment TEXT NOT NULL,
        baseline REAL NOT NULL,
        fine_tuned REAL NOT NULL)""")
    # Older databases predate the Stage 1 score: add the column (existing rows are backfilled below).
    if "stage1" not in {r[1] for r in con.execute("PRAGMA table_info(analyses)")}:
        con.execute("ALTER TABLE analyses ADD COLUMN stage1 REAL")
    if "category" not in {r[1] for r in con.execute("PRAGMA table_info(analyses)")}:
        con.execute("ALTER TABLE analyses ADD COLUMN category TEXT")
    return con


def save_analysis(text: str, baseline: float, fine_tuned: float, stage1: float, category) -> None:
    with _db() as con:  # commits on success
        con.execute("INSERT INTO analyses (timestamp, comment, baseline, fine_tuned, stage1, category) VALUES (?,?,?,?,?,?)",
                    (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), text, baseline, fine_tuned, stage1, category))


@st.cache_resource(show_spinner="Scoring saved analyses with the Stage 1 model...")
def backfill_stage1() -> int:
    """Once per server run: give saved analyses that predate the Stage 1 column a Stage 1 score."""
    with _db() as con:
        rows = con.execute("SELECT id, comment FROM analyses WHERE stage1 IS NULL").fetchall()
        for rid, text in rows:
            con.execute("UPDATE analyses SET stage1=? WHERE id=?", (round(stage1_score(text), 2), rid))
    return len(rows)


def history_count() -> int:
    with _db() as con:
        return con.execute("SELECT COUNT(*) FROM analyses").fetchone()[0]


def clear_history_db() -> None:
    with _db() as con:
        con.execute("DELETE FROM analyses")


def load_history(limit: int = HISTORY_LIMIT) -> list[dict]:
    """Most recent first."""
    with _db() as con:
        rows = con.execute("SELECT id, timestamp, comment, baseline, fine_tuned, stage1, category FROM analyses "
                           "ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(id=r[0], time=r[1], text=r[2], baseline=r[3], fine_tuned=r[4], stage1=r[5], category=r[6]) for r in rows]


# ---------------------------------------------------------------- state / callbacks
st.session_state.setdefault("comment", "")
st.session_state.setdefault("result", None)
st.session_state.setdefault("show_history", False)
st.session_state.setdefault("confirm_clear", False)
st.session_state.setdefault("tox_src", "TF-IDF baseline")
st.session_state.setdefault("tox_thr", TOXIC_THRESHOLD)
st.session_state.setdefault("sex_thr", SEXIST_THRESHOLD)


def run_analysis():
    text = st.session_state.comment.strip()
    if not text:
        st.session_state.result = None
        st.session_state.empty_warning = True
        return
    st.session_state.empty_warning = False
    result = analyze(text)
    result["text"] = text
    result["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.result = result
    save_analysis(text, result["baseline"], result["fine_tuned"], result["stage1"], result["category"])


def reload_analysis(h: dict):
    """Show a stored analysis in the results panel (and its comment in the input box)."""
    st.session_state.comment = h["text"]
    st.session_state.empty_warning = False
    stage1 = h["stage1"] if h["stage1"] is not None else round(stage1_score(h["text"]), 2)
    result = {"baseline": h["baseline"], "stage1": stage1, "fine_tuned": h["fine_tuned"], "category": h["category"],
              "category_note": "" if h["category"] else "No category was recorded for this saved analysis.",
              "text": h["text"], "time": h["time"]}
    st.session_state.result = result


def clear_history():
    """Delete every saved analysis, but only if the confirmation box is ticked."""
    if st.session_state.confirm_clear:
        clear_history_db()
    st.session_state.confirm_clear = False


def start_new():
    st.session_state.comment = ""
    st.session_state.result = None
    st.session_state.empty_warning = False


def toggle_history():
    st.session_state.show_history = not st.session_state.show_history


# ---------------------------------------------------------------- header
backfill_stage1()
n_history = history_count()
head_l, head_r = st.columns([5, 1], vertical_alignment="center")
head_l.title("Social Media Comment Toxicity Analyser")
head_r.button(
    "Past Analyses" + (f" ({n_history})" if n_history else ""),
    on_click=toggle_history,
    width="stretch",
)
st.divider()

if st.session_state.show_history:
    with st.container(border=True):
        st.subheader("Past Analyses")
        history = load_history()
        c_chk, c_btn = st.columns([4, 1], vertical_alignment="center")
        c_chk.checkbox("Confirm: permanently delete all saved analyses", key="confirm_clear")
        c_btn.button("Clear History", on_click=clear_history, width="stretch",
                     disabled=not (st.session_state.confirm_clear and history))
        if not history:
            st.caption("No analyses yet.")
        else:
            st.caption("Most recent first. Click an entry to reload its scores into the results panel.")
        for h in history:
            preview = h["text"][:80].replace("\n", " ") + ("…" if len(h["text"]) > 80 else "")
            h_tox = h[TOX_SOURCES[st.session_state.tox_src]]
            quad = classify(h_tox, h["fine_tuned"], st.session_state.tox_thr, st.session_state.sex_thr)["name"]
            st.button(
                f"{h['time']}  |  toxicity {h['baseline']:.2f} TF-IDF / {h['stage1']:.2f} Stage 1  |  "
                f"sexism {h['fine_tuned']:.2f}  |  {quad}  |  {h['category'] or 'no category'}  |  {preview}",
                key=f"hist_{h['id']}", on_click=reload_analysis, args=(h,), width="stretch",
            )

# ---------------------------------------------------------------- main layout
left, right = st.columns(2, gap="large")

with left:
    st.text_area(
        "Input Comment or Thread To Analyze",
        key="comment",
        height=380,
        placeholder="Paste a comment or an entire thread here...",
    )
    b1, b2 = st.columns(2)
    b1.button("Start New Analysis", on_click=start_new, width="stretch")
    b2.button("Run Analysis", on_click=run_analysis, type="primary", width="stretch")
    if st.session_state.get("empty_warning"):
        st.warning("Enter a comment or thread first.")

with right:
    res = st.session_state.result
    if res is None:
        st.info("Results will appear here after you run an analysis.")
        st.caption("Scores come from our models; the category comes from Claude Haiku and flagged words from word deletion.")
    else:
        tox_thr, sex_thr = st.session_state.tox_thr, st.session_state.sex_thr
        src = st.session_state.tox_src
        c = classify(res[TOX_SOURCES[src]], res["fine_tuned"], tox_thr, sex_thr, tox_name=src)

        def used(name: str) -> str:
            return " Drives the decision table." if name == src else ""

        st.subheader("Model Scores")
        m1, m2, m3 = st.columns(3)
        m1.metric("Toxicity (TF-IDF baseline)", f"{res['baseline']:.2f}",
                  help="P(toxic) from TF-IDF + logistic regression trained on Civil Comments.")
        m1.caption(f"{'Toxic' if res['baseline'] >= tox_thr else 'Not toxic'} at {tox_thr:.2f}.{used('TF-IDF baseline')}")
        m2.metric("Toxicity (Stage 1 DistilBERT)", f"{res['stage1']:.2f}",
                  help="P(toxic) from DistilBERT trained on Civil Comments only (Stage 1, before any sexism training).")
        m2.caption(f"{'Toxic' if res['stage1'] >= tox_thr else 'Not toxic'} at {tox_thr:.2f}.{used('Stage 1 DistilBERT')}")
        m3.metric("Sexism (Stage 2 DistilBERT)", f"{res['fine_tuned']:.2f}",
                  help="P(sexist) from DistilBERT trained on Civil Comments, then EXIST 2021 English (Stage 2).")
        m3.caption(f"{'Sexist' if c['sexist'] else 'Not sexist'} at {sex_thr:.2f}.")

        st.subheader("Category Analysis")
        if res["category"] is None:
            st.info(res.get("category_note") or "No category available.")
        else:
            kind = TOX_SOURCES[src] if res["category"] == "Personal Insult" else "sexism"
            words = [] if res["category"] == "Safe" else flagged_words(res["text"], kind)
            st.dataframe(pd.DataFrame([{
                "Category": n,
                "Predicted": "◀ predicted" if n == res["category"] else "",
                "Flagged words": ", ".join(words) if n == res["category"] else "",
            } for n in CATEGORY_NAMES]), hide_index=True, width="stretch")
            st.caption("One category per comment, chosen by Claude Haiku from the project's definitions. Flagged words are "
                       f"the words whose removal lowers the {'toxicity' if kind != 'sexism' else 'sexism'} score the most. "
                       "The category and the scores come from different models and can disagree; the verdict below uses the scores.")

        st.subheader("Summary of Analysis")
        st.markdown(quadrant_grid((c["toxic"], c["sexist"])), unsafe_allow_html=True)
        st.markdown(c["feedback"])
        getattr(st, c["level"])(f"VERDICT: [ {c['verdict']} ]")

    with st.expander("Decision table settings"):
        st.radio("Toxicity score that drives the decision table", list(TOX_SOURCES), key="tox_src", horizontal=True,
                 help="The decision table uses one toxicity score. Both are always shown above.")
        st.slider("Toxicity threshold", 0.05, 0.95, step=0.05, key="tox_thr",
                  help="Toxicity score at or above this counts as Toxic = 1.")
        st.slider("Sexism threshold", 0.05, 0.95, step=0.05, key="sex_thr",
                  help="Sexism score at or above this counts as Sexist = 1.")
        st.caption("Both start at 0.50. The toxicity model is weak on short social media comments, so try a "
                   "lower toxicity threshold if it misses obvious insults.")
