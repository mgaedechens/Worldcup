"""Assemble the match-level training table for the outcome classifier.

One row per match. Every feature is computed using ONLY information available before
kickoff (pre-match), which is what keeps the model leakage-free.

Features
--------
- ``elo_diff``    : home_elo_pre - away_elo_pre   (the single strongest predictor)
- ``is_neutral``  : 1 if played on neutral ground  (home advantage on/off)
- ``form_diff``   : home recent form - away recent form (points-per-game, last N)
- ``rest_diff``   : home rest days - away rest days
Target
------
- ``target`` in {``H``, ``D``, ``A``}  (home win / draw / away win)

Design note (quant link): recent ``form`` is a *short-window* momentum signal, the
complement to Elo's *long-memory* level — much like pairing a fast and slow moving
average. Elo says "how good are you in general"; form says "how hot are you right now".

Run:
    python -m src.features.build_features
"""

from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.elo import compute_elo

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

FORM_WINDOW = 10          # matches used for the recent-form signal
TRAIN_START_YEAR = 2002   # decided in the EDA (dense, modern data)
_POINTS = {"win": 3.0, "draw": 1.0, "loss": 0.0}


def _result_char(home_score: int, away_score: int) -> str:
    if home_score > away_score:
        return "H"
    if home_score == away_score:
        return "D"
    return "A"


def build_features(
    matches: pd.DataFrame,
    *,
    form_window: int = FORM_WINDOW,
    train_start_year: int = TRAIN_START_YEAR,
) -> pd.DataFrame:
    """Build the leakage-safe feature table, filtered to ``train_start_year`` onward.

    Elo is warmed on the FULL history (passed in via ``matches``); only the final
    table is filtered to the modern window.
    """
    # Sort once; Elo's internal stable sort keeps the same order, so we can align by
    # position safely.
    df = matches.sort_values("date", kind="stable").reset_index(drop=True)

    elo = compute_elo(df)
    df["home_elo_pre"] = elo.match_features["home_elo_pre"].to_numpy()
    df["away_elo_pre"] = elo.match_features["away_elo_pre"].to_numpy()
    df["elo_diff"] = elo.match_features["elo_diff"].to_numpy()

    # Single chronological pass for form (rolling points-per-game) and rest days.
    recent_points: dict[str, deque] = defaultdict(lambda: deque(maxlen=form_window))
    last_played: dict[str, pd.Timestamp] = {}

    home_form, away_form, home_rest, away_rest = [], [], [], []

    for row in df.itertuples(index=False):
        home, away, date = row.home_team, row.away_team, row.date

        # Pre-match form = average points over the team's last N matches (NaN if none).
        hf = np.mean(recent_points[home]) if recent_points[home] else np.nan
        af = np.mean(recent_points[away]) if recent_points[away] else np.nan
        home_form.append(hf)
        away_form.append(af)

        # Pre-match rest = days since the team last played (NaN if first appearance).
        home_rest.append((date - last_played[home]).days if home in last_played else np.nan)
        away_rest.append((date - last_played[away]).days if away in last_played else np.nan)

        # --- update state AFTER recording pre-match features (no leakage) ---
        res = _result_char(row.home_score, row.away_score)
        if res == "H":
            recent_points[home].append(_POINTS["win"])
            recent_points[away].append(_POINTS["loss"])
        elif res == "D":
            recent_points[home].append(_POINTS["draw"])
            recent_points[away].append(_POINTS["draw"])
        else:
            recent_points[home].append(_POINTS["loss"])
            recent_points[away].append(_POINTS["win"])
        last_played[home] = date
        last_played[away] = date

    df["home_form"] = home_form
    df["away_form"] = away_form
    df["form_diff"] = df["home_form"] - df["away_form"]
    df["rest_diff"] = np.array(home_rest, dtype="float") - np.array(away_rest, dtype="float")
    df["is_neutral"] = df["neutral"].astype(int)
    df["target"] = [
        _result_char(h, a) for h, a in zip(df["home_score"], df["away_score"])
    ]

    # Keep the modern window for training.
    df["year"] = df["date"].dt.year
    out = df[df["year"] >= train_start_year].copy()

    # Fill the few remaining NaNs (teams with little prior history) with neutral values:
    # average PPG for form, median rest for rest. Documented simplification.
    out["form_diff"] = out["form_diff"].fillna(0.0)
    out["rest_diff"] = out["rest_diff"].fillna(out["rest_diff"].median())

    cols = [
        "date", "home_team", "away_team",
        "elo_diff", "form_diff", "rest_diff", "is_neutral",
        "home_elo_pre", "away_elo_pre", "tournament", "target",
    ]
    return out[cols].reset_index(drop=True)


def run() -> None:
    matches = pd.read_csv(PROCESSED_DIR / "matches_clean.csv", parse_dates=["date"])
    features = build_features(matches)
    out_path = PROCESSED_DIR / "features.csv"
    features.to_csv(out_path, index=False)

    print(f"[ok] features -> {out_path.name}  ({len(features):,} rows)")
    print("\nClass balance (target):")
    print((features["target"].value_counts(normalize=True) * 100).round(1).to_string())
    print("\nSanity: mean elo_diff by outcome (expect H>0, A<0):")
    print(features.groupby("target")["elo_diff"].mean().round(1).to_string())


if __name__ == "__main__":
    run()
