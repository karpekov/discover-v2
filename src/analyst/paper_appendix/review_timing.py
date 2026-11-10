"""Appendix section 5: time each rater spent per home and step, from the rated_at stamps in the review CSVs.

    conda activate discover-v2-env
    python src/analyst/paper_appendix/review_timing.py     # → results/paper_appendix/review_timing.csv
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
REVIEW = ROOT / "results/analyst/runs/analyst_questions/review/review_v2"
OUT = ROOT / "results/paper_appendix/review_timing.csv"
MAX_GAP_MIN = 10  # gaps longer than this count as breaks, not rating time
TZ = "America/New_York"


def active_minutes(ts: pd.Series) -> float:
    d = ts.sort_values().diff().dt.total_seconds().div(60).dropna()
    return float(d[d <= MAX_GAP_MIN].sum())


def summarize(g: pd.DataFrame) -> pd.Series:
    ts = g["rated_at"].sort_values()
    gaps = ts.diff().dt.total_seconds().dropna()
    return pd.Series({
        "n_rated": len(g), "first_local": ts.iloc[0].strftime("%Y-%m-%d %H:%M"),
        "last_local": ts.iloc[-1].strftime("%Y-%m-%d %H:%M"), "active_min": round(active_minutes(ts), 1),
        "median_s_per_item": round(float(gaps.median()), 1) if len(gaps) else None,
    })


def main():
    df = pd.concat([pd.read_csv(p) for p in sorted(REVIEW.glob("review_review_v2_*.csv"))], ignore_index=True)
    df = df[df["rating"].isin(["yes", "no", "unsure"])].copy()
    df["rated_at"] = pd.to_datetime(df["rated_at"], utc=True).dt.tz_convert(TZ)
    df["rater"] = df["reviewer"].str.replace(r"_oct\d+$", "", regex=True)
    by_step = df.groupby(["rater", "home", "step"]).apply(summarize).reset_index()
    by_home = df.groupby(["rater", "home"]).apply(summarize).reset_index().assign(step="all")
    out = pd.concat([by_home, by_step], ignore_index=True).sort_values(["rater", "home", "step"], key=lambda s: s.astype(str))
    out.to_csv(OUT, index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
