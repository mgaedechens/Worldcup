"""Invariants for the Poisson goals model and the match engine."""

import numpy as np

from src.models.goals import implied_outcome_probs
from src.simulation.engine import GoalsParams, knockout_winner, simulate_scoreline


def test_lambda_positive_and_monotonic():
    p = GoalsParams(b0=0.1, b_elo=0.001, b_home=0.3)
    assert p.lam(0) > 0
    # A bigger Elo advantage means more expected goals.
    assert p.lam(300) > p.lam(0) > p.lam(-300)


def test_implied_probs_are_a_distribution():
    probs = implied_outcome_probs(1.5, 1.0)
    assert abs(probs.sum() - 1.0) < 1e-6
    assert (probs >= 0).all()


def test_stronger_team_has_higher_win_prob():
    # implied_outcome_probs returns [away_win, draw, home_win].
    probs = implied_outcome_probs(2.0, 0.8)  # home much stronger
    assert probs[2] > probs[0]


def test_simulate_scoreline_nonnegative_ints():
    p = GoalsParams(0.1, 0.001, 0.0)
    rng = np.random.default_rng(0)
    gh, ga = simulate_scoreline(1800, 1500, p, rng)
    assert isinstance(gh, int) and isinstance(ga, int) and gh >= 0 and ga >= 0


def test_knockout_returns_one_of_the_two():
    p = GoalsParams(0.1, 0.001, 0.0)
    rng = np.random.default_rng(0)
    ratings = {"A": 1800, "B": 1500}
    assert knockout_winner("A", "B", ratings, p, rng) in ("A", "B")
