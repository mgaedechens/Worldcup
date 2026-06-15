"""Invariants for the composite strength rating (Elo + market + squad value)."""

import numpy as np

from src.features.strength import (
    StrengthWeights, compose_strength, market_implied_probs,
)


def test_market_implied_probs_sum_to_title_mass():
    odds = {"A": 5.0, "B": 10.0, "C": 20.0}
    probs = market_implied_probs(odds, title_mass=0.9)
    assert abs(sum(probs.values()) - 0.9) < 1e-9
    # Shorter odds must map to a higher probability.
    assert probs["A"] > probs["B"] > probs["C"]


def test_composite_blends_toward_market_and_squad():
    # Three teams, equal Elo. Market and squad both favour B, disfavour C.
    elo = {"A": 1800.0, "B": 1800.0, "C": 1800.0}
    market = {"A": 1800.0, "B": 1900.0, "C": 1700.0}
    squad = {"A": 1800.0, "B": 1900.0, "C": 1700.0}
    out = compose_strength(elo, market=market, squad=squad)
    assert out["B"] > out["A"] > out["C"]
    # A's signals are all equal, so its composite stays put.
    assert abs(out["A"] - 1800.0) < 1e-9


def test_composite_falls_back_to_available_signals():
    # A team with no market quote is rated from Elo + squad only (weights renormalise).
    elo = {"A": 1800.0}
    squad = {"A": 2000.0}
    w = StrengthWeights(elo=0.5, market=0.3, squad=0.2)
    out = compose_strength(elo, market={}, squad=squad, weights=w)
    expected = (0.5 * 1800.0 + 0.2 * 2000.0) / (0.5 + 0.2)
    assert abs(out["A"] - expected) < 1e-9


def test_composite_is_pure_elo_when_no_aux_signals():
    elo = {"A": 1750.0, "B": 1600.0}
    out = compose_strength(elo, market={}, squad={})
    assert out == elo


def test_composite_stays_on_elo_scale():
    # Composite values should sit within the spread of the inputs, never blow up.
    rng = np.random.default_rng(0)
    teams = [f"t{i}" for i in range(20)]
    elo = {t: float(1500 + rng.normal(0, 150)) for t in teams}
    market = {t: float(1500 + rng.normal(0, 150)) for t in teams[:8]}
    squad = {t: float(1500 + rng.normal(0, 150)) for t in teams}
    out = compose_strength(elo, market=market, squad=squad)
    lo = min(min(elo.values()), min(market.values()), min(squad.values())) - 1
    hi = max(max(elo.values()), max(market.values()), max(squad.values())) + 1
    assert all(lo <= v <= hi for v in out.values())
