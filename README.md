# Zepto Data & AI Platform

This repository is a student capstone composed of three connected modules. The
data pipeline turns catalogue pages into a clean SQLite database, the analytics
module explores and models the Titanic dataset, and the support assistant serves
grounded answers from a small Zepto policy corpus.

## Repository layout

- `data_pipeline/` — scrape, clean, enrich, store and query book catalogue data.
- `analytics/` — a two-stage Titanic EDA and modelling workflow.
- `support_assistant/` — an offline-first LangGraph/FastAPI policy assistant.

Each module has its own `requirements.txt` and README. Install dependencies in a
virtual environment for the module you want to run. The scripts are deliberately
separate: it keeps each stage inspectable and makes it easy to rerun only the
part that changed.

## Quick start

```bash
cd data_pipeline && pip install -r requirements.txt && python run_pipeline.py
cd ../analytics && pip install -r requirements.txt && python 01_eda.py && python 02_modeling.py
cd ../support_assistant && pip install -r requirements.txt && uvicorn main:app --reload
```

`analytics/01_eda.py` fetches the Titanic dataset once with `sns.load_dataset`
and immediately saves `analytics/titanic.csv`; later stages only read that file.
If the first download is unavailable, place the canonical Seaborn Titanic CSV at
`analytics/titanic.csv` and rerun. No result files in this repository are claimed
to have been generated in this environment.

## Design decisions

The data pipeline uses a fixed assignment conversion of **1 GBP = 105.50 INR**;
it intentionally never calls a currency service. SQLite is used because it is
portable and lets analysts query the same cleaned data without a server. The
analytics workflow splits before modelling preprocessing to prevent leakage. The
support assistant is mock-first: `MOCK_LLM` is on by default, so grading and local
development require no key or LLM network call.

## Reproducibility and Git workflow

Run the commands above from a clean checkout. Commit work on a feature branch,
make at least two meaningful commits, and merge it into `main`; Git history is a
submission requirement and cannot be truthfully manufactured by source files.
