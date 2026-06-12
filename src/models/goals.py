"""Poisson goals model: simulate realistic scorelines driven by Elo.

Why a goals model (in addition to the W/D/L classifier)?
A tournament needs *scorelines*, not just win/draw/loss: group standings are decided by
points AND goal difference, and the new 48-team format ranks the 8 best third-placed
teams largely on goal difference. A goals model gives a fully coherent simulation.

Method (independent Poisson — the football-analytics standard):
    goals_team ~ Poisson(lambda),   lambda = exp(b0 + b1 * elo_adv + b2 * is_home)
where ``elo_adv`` is the team's pre-match Elo minus the opponent's, and ``is_home`` flags
a real home side (non-neutral venue). We fit ONE symmetric model on a team-perspective
("long") view of every match, so each match contributes two rows.

Cross-check: the W/D/L probabilities *implied* by this Poisson model should agree with the
separately-built logistic classifier — a nice internal consistency test.

Limitation (documented): independent Poisson slightly under-predicts draws because it
ignores the mild negative correlation between the two scores; the Dixon-Coles correction
fixes this and is a natural v2 improvement.

Run:
    python -m src.models.goals
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features.elo import compute_elo

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

TRAIN_START_YEAR = 2002
_MAX_GOALS = 15  # truncation for implied-probability sums (P(>15 goals) ~ 0)


def build_long_table(matches: pd.DataFrame, *, start_year: int = TRAIN_START_YEAR) -> pd.DataFrame:
    """Two rows per match (one per team) with [scored, elo_adv, is_home]."""
    df = matches.sort_values("date", kind="stable").reset_index(drop=True)
    elo = compute_elo(df)
    df["home_elo_pre"] = elo.match_features["home_elo_pre"].to_numpy()
    df["away_elo_pre"] = elo.match_features["away_elo_pre"].to_numpy()
    df = df[df["date"].dt.year >= start_year]

    non_neutral = (~df["neutral"].astype(bool)).astype(int)
    home_rows = pd.DataFrame({
        "scored": df["home_score"].to_numpy(),
        "elo_adv": (df["home_elo_pre"] - df["away_elo_pre"]).to_numpy(),
        "is_home": non_neutral.to_numpy(),
    })
    away_rows = pd.DataFrame({
        "scored": df["away_score"].to_numpy(),
        "elo_adv": (df["away_elo_pre"] - df["home_elo_pre"]).to_numpy(),
        "is_home": np.zeros(len(df), dtype=int),  # away side never gets home advantage
    })
    return pd.concat([home_rows, away_rows], ignore_index=True)


def fit_goals_model(long: pd.DataFrame) -> Pipeline:
    """Fit the Poisson GLM (log link) for expected goals.

    Features are standardized first: ``elo_adv`` spans hundreds of points, which makes the
    exponential link overflow during optimization if left unscaled (the optimizer then
    collapses to an intercept-only model). Scaling keeps the problem well-conditioned.
    """
    X = long[["elo_adv", "is_home"]].to_numpy()
    y = long["scored"].to_numpy()
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("poisson", PoissonRegressor(alpha=1e-4, max_iter=1000)),
    ])
    model.fit(X, y)
    return model


def expected_goals(model: Pipeline, elo_adv: float, is_home: int = 0) -> float:
    """Predicted mean goals (lambda) for a team with the given Elo advantage."""
    return float(model.predict([[elo_adv, is_home]])[0])


def implied_outcome_probs(lam_home: float, lam_away: float, max_goals: int = _MAX_GOALS):
    """P(away win), P(draw), P(home win) implied by two independent Poissons.

    Order [A, D, H] to match the classifier's class order.
    """
    g = np.arange(max_goals + 1)
    ph = poisson.pmf(g, lam_home)
    pa = poisson.pmf(g, lam_away)
    joint = np.outer(ph, pa)  # joint[i, j] = P(home=i, away=j)
    p_home = np.tril(joint, -1).sum()  # i > j
    p_draw = np.trace(joint)           # i == j
    p_away = np.triu(joint, 1).sum()   # i < j
    return np.array([p_away, p_draw, p_home])


def run() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    matches = pd.read_csv(PROCESSED_DIR / "matches_clean.csv", parse_dates=["date"])
    long = build_long_table(matches)
    model = fit_goals_model(long)

    baseline = expected_goals(model, 0, 0)
    home_even = expected_goals(model, 0, 1)
    print("Poisson goals model (features standardized internally):")
    print(f"  baseline goals (even teams, neutral) = {baseline:.2f}")
    print(f"  home side, even teams                = {home_even:.2f}  "
          f"({home_even / baseline:.2f}x)")

    print("\nExample expected goals (neutral venue):")
    for adv in (0, 100, 200, 400):
        lf = expected_goals(model, adv)
        lu = expected_goals(model, -adv)
        probs = implied_outcome_probs(lf, lu)
        print(f"  +{adv:>3} Elo favorite: {lf:.2f} - {lu:.2f} goals | "
              f"P(win/draw/lose) = {probs[2]:.0%}/{probs[1]:.0%}/{probs[0]:.0%}")

    joblib.dump(model, MODELS_DIR / "goals_model.joblib")
    print(f"\n[ok] saved -> {MODELS_DIR / 'goals_model.joblib'}")


if __name__ == "__main__":
    run()
