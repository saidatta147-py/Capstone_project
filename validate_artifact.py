"""Checks that the regenerated joblib file is a complete raw-input pipeline."""
from pathlib import Path
import joblib
import pandas as pd

root = Path(__file__).parents[1]
artifact = root / "output" / "best_pipeline.joblib"
data = root / "output" / "titanic_cleaned.csv"
if not artifact.exists() or not data.exists():
    raise SystemExit("Run 01_eda.py and 02_modeling.py before this validation.")
pipeline = joblib.load(artifact)
raw = pd.read_csv(data)[["pclass", "age", "sibsp", "parch", "fare", "sex", "embarked"]].head(3)
assert len(pipeline.predict(raw)) == 3
print("Saved complete pipeline accepts raw rows.")
