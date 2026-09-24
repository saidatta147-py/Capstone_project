"""Stage 1: load Titanic once, profile it, and write reusable EDA artefacts."""
from __future__ import annotations
from pathlib import Path
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent; OUT = ROOT / "output"; CSV = ROOT / "titanic.csv"

def load_once() -> pd.DataFrame:
    if CSV.exists():
        return pd.read_csv(CSV)
    df = sns.load_dataset("titanic")  # the module's only network/cache loader
    df.to_csv(CSV, index=False)
    return df

def iqr_outliers(series: pd.Series) -> int:
    q1, q3 = series.quantile([.25, .75]); iqr = q3 - q1
    return int(((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum())

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True); df = load_once()
    missing = (df.isna().mean() * 100).loc[lambda s: s.gt(0)].sort_values(ascending=False)
    print(df.info()); print(df.describe(include="all")); print("shape:", df.shape); print("missing %:\n", missing)
    # Exact thresholds are applied after the percentages above have been measured.
    working = df.drop(columns=["deck"]).copy()  # deck is far above 30%, so do not invent cabins
    low = missing[missing < 5].index.intersection(working.columns)
    working = working.dropna(subset=list(low))
    for col in missing[(missing >= 5) & (missing <= 30)].index.intersection(working.columns):
        working[col] = working[col].fillna(working[col].median() if pd.api.types.is_numeric_dtype(working[col]) else working[col].mode()[0])
    working.to_csv(OUT / "titanic_cleaned.csv", index=False)
    fare_mode = working.fare.mode().iloc[0]
    rates_sex = working.groupby("sex").survived.mean(); rates_class = working.groupby("pclass").survived.mean()
    rates_both = working.groupby(["sex", "pclass"]).survived.mean()
    # Boolean masks explicitly demonstrate the requested filtering form.
    women_first = working[(working.sex == "female") & (working.pclass == 1)].survived.mean()
    six = ["survived", "pclass", "age", "sibsp", "parch", "fare"]; corr = working[six].corr()
    pairs = sorted(((abs(corr.loc[a,b]), a, b, corr.loc[a,b]) for a,b in itertools.combinations(six,2)), reverse=True)[:2]
    report = ["# EDA run report", f"\nShape: {df.shape}", "\n## Missing values (%)", missing.to_markdown(),
              "\nStrategy: <5% rows dropped; 5–30% median/mode imputed; deck (>30%) dropped.",
              f"\nAge IQR outliers: {iqr_outliers(working.age)}; Fare IQR outliers: {iqr_outliers(working.fare)}.",
              f"\nFare mean={working.fare.mean():.2f}, median={working.fare.median():.2f}, mode={fare_mode:.2f}. Mean above median/mode indicates right skew.",
              "\n## Survival rates by sex\n" + rates_sex.to_markdown(), "\n## By class\n" + rates_class.to_markdown(),
              "\n## By sex and class\n" + rates_both.to_markdown(), f"\nFemale first-class masked rate: {women_first:.3f}",
              "\n## Strongest correlations\n" + "\n".join(f"{a}/{b}: {v:.3f}" for _,a,b,v in pairs)]
    for col in ["age", "fare"]:
        fig, ax = plt.subplots(1,2, figsize=(9,3)); sns.histplot(working[col], kde=True, ax=ax[0]); sns.boxplot(x=working[col], ax=ax[1]); fig.tight_layout(); fig.savefig(OUT/f"{col}_distribution.png"); plt.close(fig)
    charts = [("survival_by_sex_class", sns.catplot(data=working, x="pclass", y="survived", hue="sex", kind="bar"), "Survival differs sharply by sex and class; class is a useful proxy for access and circumstances."),
              ("age_by_survival", sns.boxplot(data=working, x="survived", y="age"), "Age distributions overlap, so age alone is unlikely to separate outcomes."),
              ("fare_by_survival_class", sns.boxplot(data=working, x="pclass", y="fare", hue="survived"), "Fare is right-skewed and varies by class, so class-aware interpretation is important.")]
    for name, plot, text in charts:
        fig = plot.fig if hasattr(plot, "fig") else plot.get_figure(); fig.savefig(OUT/f"{name}.png", bbox_inches="tight"); plt.close(fig); report.append(f"\n### {name}\n{text}")
    plt.figure(figsize=(7,5)); sns.heatmap(corr, annot=True, cmap="coolwarm", center=0); plt.tight_layout(); plt.savefig(OUT/"correlation_heatmap.png"); plt.close(); report.append("\n### correlation_heatmap\nThe heatmap uses exactly survived, pclass, age, sibsp, parch, and fare; redundant adult_male and alone are excluded.")
    z = (working[["age","fare"]] - working[["age","fare"]].mean()) / working[["age","fare"]].std(ddof=0)
    report.append("\n## Z-score check\n" + pd.DataFrame({"mean":z.mean(), "std":z.std(ddof=0)}).to_markdown())
    (OUT/"report.md").write_text("\n".join(report), encoding="utf-8")
