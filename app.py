"""Streamlit demo: paste an email or upload a .eml file and get a verdict with reasons.

Run:  streamlit run app.py
"""
from __future__ import annotations

from pathlib import Path

import joblib
import streamlit as st

from src.features import red_flags
from src.model import explain, predict_proba

MODEL_PATH = Path("models/baseline.joblib")
SAMPLES = {p.stem.replace("_", " ").title(): p for p in sorted(Path("samples").glob("*.eml"))}
SEV_ICON = {"high": "🔴", "medium": "🟠", "low": "🟡"}

st.set_page_config(page_title="Phishing Email Detector", page_icon="🛡️", layout="centered")
st.title("Phishing Email Detector")
st.caption("A machine learning score plus the specific red flags an analyst would check.")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


model = load_model()
if model is None:
    st.error("No trained model found. Run `python train_baseline.py` first.")
    st.stop()

tab_paste, tab_upload = st.tabs(["Paste email", "Upload .eml"])
raw = ""
with tab_paste:
    choice = st.selectbox("Or load a sample", ["(none)"] + list(SAMPLES))
    default = SAMPLES[choice].read_text() if choice in SAMPLES else ""
    raw = st.text_area("Email text (with headers if you have them)", value=default, height=260)
with tab_upload:
    up = st.file_uploader("Upload a .eml file", type=["eml", "txt"])
    if up is not None:
        raw = up.read().decode("utf-8", errors="replace")

if st.button("Analyse", type="primary") and raw.strip():
    email, urls, flags = red_flags(raw)
    text_for_model = f"{email.subject}\n{email.text}"
    prob = float(predict_proba(model, [text_for_model])[0])
    high = sum(f.severity == "high" for f in flags)

    if prob >= 0.8 or (prob >= 0.5 and high):
        st.error(f"### Likely phishing  ({prob:.0%} model score)")
    elif prob >= 0.5 or high:
        st.warning(f"### Suspicious, review before acting  ({prob:.0%} model score)")
    else:
        st.success(f"### Looks legitimate  ({prob:.0%} model score)")
    st.progress(prob)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Red flags")
        if flags:
            for f in flags:
                st.markdown(f"{SEV_ICON[f.severity]} **{f.category.title()}**: {f.message}")
        else:
            st.write("No rule-based red flags found.")
    with col2:
        st.subheader("Words that raised the score")
        reasons = explain(model, text_for_model)
        if reasons:
            for term, w in reasons:
                st.markdown(f"`{term}`  +{w}")
        else:
            st.write("Nothing in the wording pushed the score up.")

    with st.expander("Details"):
        if email.has_headers:
            st.write({"From": email.sender, "Reply-To": email.reply_to or "-",
                      "Return-Path": email.return_path or "-", "Subject": email.subject,
                      "Authentication-Results": email.auth_results or "-"})
        else:
            st.info("No headers found, so sender checks (SPF, DKIM, DMARC) were skipped. "
                    "Paste the full source of the email to include them.")
        st.write(f"Links found: {len(urls)}")
        for u in urls:
            st.code(u, language=None)

st.divider()
st.caption("Portfolio project. Don't paste emails containing personal or confidential data.")
