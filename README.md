# Trust & Safety MVP — Toxic Content Scoring (Open Source)

**What it is:** A tiny, end‑to‑end project that loads an open dataset, trains a text toxicity classifier, and serves a simple UI for manual review.

**Why it maps to T&S roles:** Shows you can build an ingestion→modeling→evaluation→review loop with sensible metrics and clean code, using fully open tools.

## Stack
- Python 3.10+
- [datasets](https://huggingface.co/docs/datasets) to load **civil_comments** (open Jigsaw dataset)
- scikit‑learn (TF‑IDF + LogisticRegression)
- Streamlit (minimal reviewer UI)
- pandas / numpy / joblib

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1) Ingest & split the open dataset
python src/data_pipeline.py

# 2) Train the model and see metrics
python src/train.py

# 3) Try interactive scoring UI
streamlit run app/streamlit_app.py
```

## Files
- `src/data_pipeline.py` — loads **civil_comments**, cleans, stratified split → `data/train.csv` & `data/test.csv`
- `src/train.py` — trains TF‑IDF+LR pipeline, evaluates, saves `models/toxicity_clf.joblib` and `models/metrics.json`
- `src/score.py` — CLI scoring for ad‑hoc strings or CSVs
- `app/streamlit_app.py` — tiny UI for reviewers (paste text or upload CSV)
- `tests/test_sanity.py` — minimal unit test


## Notes
- The **civil_comments** dataset has a continuous `toxicity` score in [0,1]. We binarize at 0.5 by default (configurable).
- For a stronger model, swap LR for LinearSVC or fine‑tune a small transformer. The scaffolding stays the same.
- This repo intentionally stays light (no Airflow/dbt). You can layer them later.
