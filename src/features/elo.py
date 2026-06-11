"""Elo ratings for international football teams.

Elo turns a history of match results into a single strength number per team. We use
it as a *feature* for the downstream classifier (not as the final model).

--------------------------------------------------------------------------------
Quant-finance connection (EWMA)
--------------------------------------------------------------------------------
The Elo update rule

    R_new = R_old + K * (S - E)

is an **exponentially weighted** estimator. Each new result nudges the rating by a
fraction K of the "surprise" (actual score S minus expected score E), so older
results decay geometrically in influence -- exactly like an EWMA (RiskMetrics)
estimator of returns/volatility. K is the analogue of the smoothing parameter:
a larger K means faster reaction and shorter memory.

--------------------------------------------------------------------------------
Football-specific adaptations (the "World Football Elo" recipe)
--------------------------------------------------------------------------------
1. Home advantage: add a fixed Elo bonus to the home side *in the expectation only*,
   and only when the match is not on neutral ground (justified by the EDA).
2. Margin of victory: scale K by a goal-difference factor (a 5-0 means more than a 1-0).
3. Match importance: scale K by the tournament weight (a World Cup means more than a
   friendly).

Note on penalty shootouts: a knockout level after 90/120 minutes is recorded as a
draw here. For rating purposes we treat it as a draw (S = 0.5). This is a documented
simplification.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import pandas as pd

BASE_RATING = 1500.0
HOME_ADVANTAGE = 100.0  # Elo points added to the home side on non-neutral venues.


def _tournament_weight(name: str) -> float:
    """Map a tournament name to a base K (match-importance weight).

    Values follow the spirit of the World Football Elo index: the more important the
    match, the more a result should move the ratings.
    """
    n = name.lower()
    if "friendly" in n:
        return 20.0
    if "qualification" in n or "qualifier" in n:
        return 40.0
    if "world cup" in n:  # FIFA World Cup final tournament (qualifiers handled above)
        return 60.0
    # Continental finals (Euro, Copa América, AFCON, Asian Cup, Gold Cup, Nations League...)
    continental = ("euro", "copa am", "african cup", "asian cup", "gold cup",
                   "nations league", "confederations")
    if any(token in n for token in continental):
        return 50.0
    return 30.0  # other minor tournaments


def _mov_multiplier(goal_diff: int) -> float:
    """Margin-of-victory multiplier. Bigger wins move ratings more, with diminishing
    returns (a common World Football Elo form)."""
    gd = abs(goal_diff)
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11 + gd) / 8.0


def expected_score(rating_home: float, rating_away: float, home_adv: float) -> float:
    """Probability-style expected score for the home team (0..1).

    A 400-point edge ~= 10x more likely to win. ``home_adv`` is folded into the home
    rating before comparing.
    """
    diff = (rating_home + home_adv) - rating_away
    return 1.0 / (1.0 + 10 ** (-diff / 400.0))


@dataclass
class EloResult:
    """Container for everything the Elo pass produces."""

    match_features: pd.DataFrame  # per-match PRE-match ratings (no leakage) + elo_diff
    history: pd.DataFrame         # post-match rating timeline: [date, team, rating]
    final_ratings: dict[str, float] = field(default_factory=dict)


def compute_elo(
    matches: pd.DataFrame,
    *,
    base_rating: float = BASE_RATING,
    home_advantage: float = HOME_ADVANTAGE,
) -> EloResult:
    """Run Elo chronologically over ``matches`` (must have the cleaned schema).

    Returns, for every match, the **pre-match** ratings of both teams -- this is the
    leakage-safe quantity the classifier may use, because it reflects only information
    available *before* kickoff.
    """
    required = {"date", "home_team", "away_team", "home_score", "away_score",
                "tournament", "neutral"}
    missing = required - set(matches.columns)
    if missing:
        raise ValueError(f"matches is missing columns: {sorted(missing)}")

    # Chronological order is essential; stabilize ties by original position.
    df = matches.sort_values("date", kind="stable").reset_index(drop=True)

    ratings: dict[str, float] = defaultdict(lambda: base_rating)
    home_pre: list[float] = []
    away_pre: list[float] = []
    history: list[tuple] = []

    for row in df.itertuples(index=False):
        home, away = row.home_team, row.away_team
        r_home, r_away = ratings[home], ratings[away]
        home_pre.append(r_home)
        away_pre.append(r_away)

        # Home advantage applies only off neutral ground.
        adv = 0.0 if bool(row.neutral) else home_advantage
        exp_home = expected_score(r_home, r_away, adv)

        # Actual score from the home perspective.
        if row.home_score > row.away_score:
            s_home = 1.0
        elif row.home_score == row.away_score:
            s_home = 0.5
        else:
            s_home = 0.0

        k = _tournament_weight(row.tournament) * _mov_multiplier(
            row.home_score - row.away_score
        )
        delta = k * (s_home - exp_home)

        # Zero-sum update: points transfer from one side to the other.
        ratings[home] = r_home + delta
        ratings[away] = r_away - delta

        history.append((row.date, home, ratings[home]))
        history.append((row.date, away, ratings[away]))

    match_features = df[["date", "home_team", "away_team"]].copy()
    match_features["home_elo_pre"] = home_pre
    match_features["away_elo_pre"] = away_pre
    match_features["elo_diff"] = match_features["home_elo_pre"] - match_features["away_elo_pre"]

    history_df = pd.DataFrame(history, columns=["date", "team", "rating"])
    return EloResult(match_features, history_df, dict(ratings))


if __name__ == "__main__":
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    data = pd.read_csv(root / "data" / "processed" / "matches_clean.csv",
                       parse_dates=["date"])
    result = compute_elo(data)
    top = pd.Series(result.final_ratings).sort_values(ascending=False).head(20)
    print("Top 20 current Elo ratings:")
    print(top.round(0).to_string())
