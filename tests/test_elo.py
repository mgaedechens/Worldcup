"""Invariants for the Elo engine."""

import pandas as pd

from src.features.elo import compute_elo, expected_score


def test_expected_score_even_is_half():
    assert abs(expected_score(1500, 1500, 0) - 0.5) < 1e-9


def test_expected_score_complementary():
    # Without home advantage, E_home(a,b) + E_home(b,a) == 1.
    assert abs(expected_score(1700, 1500, 0) + expected_score(1500, 1700, 0) - 1) < 1e-9


def test_home_advantage_increases_expectation():
    assert expected_score(1500, 1500, 100) > 0.5


def _toy_matches() -> pd.DataFrame:
    # A wins twice and draws once -> A should finish clearly above the baseline.
    return pd.DataFrame({
        "date": pd.to_datetime(["2000-01-01", "2000-02-01", "2000-03-01"]),
        "home_team": ["A", "A", "A"],
        "away_team": ["B", "B", "B"],
        "home_score": [2, 1, 1],
        "away_score": [0, 0, 1],
        "tournament": ["Friendly"] * 3,
        "neutral": [False, False, True],
    })


def test_first_match_uses_base_ratings():
    res = compute_elo(_toy_matches())
    first = res.match_features.iloc[0]
    assert first["home_elo_pre"] == 1500 and first["away_elo_pre"] == 1500


def test_elo_is_zero_sum():
    # Every update transfers points, so total rating == n_teams * base.
    res = compute_elo(_toy_matches())
    assert abs(sum(res.final_ratings.values()) - 2 * 1500) < 1e-6


def test_winner_gains_rating():
    res = compute_elo(_toy_matches())
    # A won 2 and drew 1, so it should end above the 1500 baseline and B below.
    assert res.final_ratings["A"] > 1500 > res.final_ratings["B"]
