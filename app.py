"""Social Media Comment Toxicity Analyser - demo UI.

Model scores are REAL: `baseline_score` is a TF-IDF + LogReg model trained on Civil
Comments toxicity; `finetuned_score` is DistilBERT (Civil Comments -> EXIST 2021 English)
and returns P(sexist). The per-category table is still a PLACEHOLDER.

Run:  streamlit run app.py
"""
import hashlib
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

ROOT = Path(__file__).parent
BASELINE_PATH = ROOT / "models" / "tfidf_lr.joblib"
FINETUNED_DIR = ROOT / "checkpoints" / "stage2"
DB_PATH = Path(os.environ.get("ANALYSES_DB", ROOT / "analyses.db"))
HISTORY_LIMIT = 200      # most recent rows listed in the Past Analyses panel
MAX_LEN = 128            # same truncation the DistilBERT model was trained with

FLAG_THRESHOLD = 0.50    # fine-tuned score at/above this -> moderator review
WARN_THRESHOLD = 0.30    # between WARN and FLAG -> borderline
HIGH_PROB = 0.50         # category probability labelled "High"
MED_PROB = 0.25          # category probability labelled "Medium"

CATEGORIES = [
    "Sexism",
    "Insult",
    "Identity attack",
    "Obscene",
    "Threat",
]

st.set_page_config(page_title="Social Media Comment Toxicity Analyser", layout="wide")


# ---------------------------------------------------------------- models
@st.cache_resource(show_spinner="Loading models...")
def _load_baseline():
    bundle = joblib.load(BASELINE_PATH)
    return bundle["vec"], bundle["clf"]


@st.cache_resource(show_spinner="Loading models...")
def _load_finetuned():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(FINETUNED_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(FINETUNED_DIR).to(device).eval()
    return tok, model, device


def baseline_score(text: str) -> float:
    """P(toxic) from the TF-IDF + Logistic Regression model (trained on Civil Comments)."""
    vec, clf = _load_baseline()
    return float(clf.predict_proba(vec.transform([text]))[0, 1])


def finetuned_score(text: str) -> float:
    """P(sexist) from the stage-2 DistilBERT model (Civil Comments -> EXIST English).
    Input is truncated to the first MAX_LEN tokens, as in training."""
    tok, model, device = _load_finetuned()
    enc = tok(text, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(device)
    with torch.no_grad():
        return float(torch.softmax(model(**enc).logits, dim=-1)[0, 1])


# ---------------------------------------------------------------- placeholder categories
def _fake_score(text: str, salt: str) -> float:
    """Deterministic pseudo-random float in [0, 1) so results are stable per input."""
    digest = hashlib.sha256(f"{salt}:{text}".encode()).digest()
    return int.from_bytes(digest[:4], "big") / 2**32


def analyze(text: str) -> dict:
    """Real baseline / fine-tuned scores; category table remains placeholder."""
    return build_result(text, round(baseline_score(text), 2), round(finetuned_score(text), 2))


def build_result(text: str, baseline: float, fine_tuned: float) -> dict:
    """Assemble the results-panel data from scores. The placeholder category table is
    deterministic in `text`, so a stored analysis reloads without re-running the models."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
    rows = []
    for cat in CATEGORIES:
        p = round(_fake_score(text, cat), 2)
        flagged = []
        if p >= MED_PROB and sentences:
            # Placeholder "flagged tokens": pick a sentence deterministically.
            idx = int(_fake_score(text, cat + "idx") * len(sentences))
            flagged.append(f"“{sentences[idx]}”")
        level = "High" if p >= HIGH_PROB else "Medium" if p >= MED_PROB else "Low"
        rows.append({
            "Category": cat,
            "Flagged Tokens": ", ".join(flagged),
            "Probability": f"{level} ({p:.2f})",
            "_p": p,
        })
    rows.sort(key=lambda r: r["_p"], reverse=True)
    return {"baseline": baseline, "fine_tuned": fine_tuned, "categories": rows}


def verdict_for(score: float) -> tuple[str, str, str]:
    """Return (label, streamlit-level, feedback) for a toxicity score."""
    if score >= FLAG_THRESHOLD:
        return ("FLAGGED FOR MODERATOR REVIEW", "error",
                f"Score {score:.2f} exceeds the {FLAG_THRESHOLD:.2f} threshold. "
                "This comment is highly toxic and has been queued for a moderator.")
    if score >= WARN_THRESHOLD:
        return ("BORDERLINE - MONITOR", "warning",
                f"Score {score:.2f} is elevated but below the {FLAG_THRESHOLD:.2f} threshold. "
                "No action taken; the thread will be monitored.")
    return ("NO ACTION NEEDED", "success",
            f"Score {score:.2f} is low. This comment appears non-toxic.")


# ---------------------------------------------------------------- history (SQLite)
def _db() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.execute("""CREATE TABLE IF NOT EXISTS analyses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        comment TEXT NOT NULL,
        baseline REAL NOT NULL,
        fine_tuned REAL NOT NULL)""")
    return con


def save_analysis(text: str, baseline: float, fine_tuned: float) -> None:
    with _db() as con:  # commits on success
        con.execute("INSERT INTO analyses (timestamp, comment, baseline, fine_tuned) VALUES (?,?,?,?)",
                    (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), text, baseline, fine_tuned))


def history_count() -> int:
    with _db() as con:
        return con.execute("SELECT COUNT(*) FROM analyses").fetchone()[0]


def clear_history_db() -> None:
    with _db() as con:
        con.execute("DELETE FROM analyses")


def load_history(limit: int = HISTORY_LIMIT) -> list[dict]:
    """Most recent first."""
    with _db() as con:
        rows = con.execute("SELECT id, timestamp, comment, baseline, fine_tuned FROM analyses "
                           "ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(id=r[0], time=r[1], text=r[2], baseline=r[3], fine_tuned=r[4]) for r in rows]


# ---------------------------------------------------------------- state / callbacks
st.session_state.setdefault("comment", "")
st.session_state.setdefault("result", None)
st.session_state.setdefault("show_history", False)
st.session_state.setdefault("confirm_clear", False)


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
    save_analysis(text, result["baseline"], result["fine_tuned"])


def reload_analysis(h: dict):
    """Show a stored analysis in the results panel (and its comment in the input box)."""
    st.session_state.comment = h["text"]
    st.session_state.empty_warning = False
    result = build_result(h["text"], h["baseline"], h["fine_tuned"])
    result["text"], result["time"] = h["text"], h["time"]
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
            st.button(
                f"{h['time']}  |  baseline {h['baseline']:.2f} / fine-tuned {h['fine_tuned']:.2f}  |  {preview}",
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
        st.caption("Model scores are real; the category table is still placeholder.")
    else:
        st.subheader("Model Scores")
        m1, m2 = st.columns(2)
        m1.metric("Baseline Toxicity Score", f"{res['baseline']:.2f}")
        m2.metric("Fine-Tuned Sexism Score", f"{res['fine_tuned']:.2f}",
                  delta=f"{res['fine_tuned'] - res['baseline']:+.2f} vs baseline",
                  delta_color="off")

        st.subheader("Toxicity Category Analyses")
        df = pd.DataFrame(res["categories"]).drop(columns="_p")
        st.dataframe(df, hide_index=True, width="stretch")

        st.subheader("Summary of Analysis")
        label, level, feedback = verdict_for(res["fine_tuned"])
        st.write(feedback)
        getattr(st, level)(f"VERDICT: [ {label} ]")
        st.caption("Baseline = P(toxic), TF-IDF + LogReg. Fine-tuned = P(sexist), DistilBERT. "
                   "Category table values are placeholders.")
