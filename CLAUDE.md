# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Flask API (`app.py`) serving fake-news predictions from a scikit-learn pipeline, a retraining script (`train_model.py`), and a static dark-themed chat frontend (`frontend/`). See `README.md` for full setup/API/architecture docs — this file only covers things future edits need to know that aren't obvious from reading the code.

## Commands

```bash
pip install -r requirements-dev.txt   # runtime + pytest
python app.py                         # serve the API (localhost:5000)
python train_model.py                 # retrain model.pkl from news_dataset.csv
pytest                                 # run tests/
cd frontend && python -m http.server 8000   # serve the frontend (localhost:8000)
```

No lint config or build step in this repository.

## Architecture

- `news_dataset.csv` — training corpus (`label`, `text`), read only by `train_model.py`.
- `train_model.py` — builds a `FeatureUnion` of word (1,2)-gram + char (3,5)-gram TF-IDF, feeds a soft-voting `LogisticRegression`+`SVC` ensemble, wraps it all in one `sklearn.Pipeline`, and saves it to `model.pkl`. Prints 5-fold CV macro-F1 and a held-out classification report on every run.
- `model.pkl` — the trained pipeline (vectorizers + classifier bundled together), loaded once via `joblib.load` at `app.py` import time. `app.py` never touches `news_dataset.csv` or refits anything at runtime.
- `app.py` — `POST /predict` (validates `news_text`, returns `{"prediction", "confidence"}`), `GET /health`. Config via `HOST`/`PORT`/`FLASK_DEBUG`/`MODEL_PATH`/`ALLOWED_ORIGINS` env vars.
- `frontend/` — static HTML/CSS/JS, no build step, no dependencies. Calls the API via `fetch`; `API_BASE` in `script.js` assumes the API is on the same host at port 5000. Chat history persists to `localStorage` (`truthlens.history.v1`).
- `tests/` — `test_app.py` (Flask test client, input validation) and `test_train_model.py` (guards the dedup/null-handling in `train_model.load_data()`).

## Model training notes

`news_dataset.csv` has heavy exact-duplicate rows (~40% of rows before dedup) and label imbalance among unique articles (unique REAL articles are outnumbered ~5:1 by unique FAKE articles — the raw 50/50 label split was an artifact of duplicated REAL rows). `train_model.py` handles both: it drops duplicate `text` rows before any train/test split (to avoid leaking identical articles across the split) and trains with `class_weight="balanced"`. Current 5-fold CV macro-F1 is ~0.995 — if a change to the pipeline moves this number, regenerate it and update the figure in `README.md` too.

The `FAKE`-labeled articles in this dataset are BOOM fact-check writeups *about* misinformation (formal journalism discussing/debunking a claim), not the raw viral/forwarded misinformation text itself. The model reflects that: it distinguishes real news reporting from fact-check reporting, and is not trained to score short informal text (tweets, WhatsApp-forward-style messages) — that's a dataset coverage limit, not a training bug. Evaluate any change to the training pipeline against `classification_report` on a held-out split (already wired up in `train_model.py`), not just training-set accuracy, since training-set/full-dataset accuracy on this data has previously masked leakage.

## Frontend notes

`frontend/style.css`: `white-space: pre-wrap` is scoped to `.msg--user .bubble` and `.bubble--error` only, not the base `.bubble`. The bot result bubble (`#tpl-bot`) is built from structured `<div>`s with insignificant whitespace between them in the `<template>` markup — applying `pre-wrap` to it renders that literal template whitespace as visible gaps. If adding new bot-bubble templates, keep `pre-wrap` off the base `.bubble` class.
