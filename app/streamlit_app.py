import streamlit as st
import pandas as pd
from joblib import load

@st.cache_resource
def load_model():
    return load("models/toxicity_clf.joblib")

st.title("Toxic Content Scoring — Simple Reviewer")
st.write("Paste a comment to get a toxicity probability. Upload CSV for batch scoring.")

model = None
try:
    model = load_model()
except Exception as e:
    st.warning("Model not found. Please run `python src/train.py` first.")
    st.stop()

text = st.text_area("Enter text", height=120, placeholder="Type something...")
if st.button("Score single text"):
    if text.strip():
        import numpy as np
        p = model.predict_proba([text])[:,1][0]
        st.metric("toxicity_prob", f"{p:.3f}")
    else:
        st.info("Please enter some text.")

st.markdown("---")
uploaded = st.file_uploader("Batch score a CSV (must include a 'text' column)", type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded)
    if "text" not in df.columns:
        st.error("CSV must contain a 'text' column.")
    else:
        df["toxicity_prob"] = model.predict_proba(df["text"].astype(str))[:,1]
        st.dataframe(df.head(50))
        st.download_button("Download scored CSV", df.to_csv(index=False).encode("utf-8"), file_name="scored.csv", mime="text/csv")
