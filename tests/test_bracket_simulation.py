"""Invariants for the bracket structure and a full synthetic tournament."""

import numpy as np

from src.simulation.bracket import (
    GROUP_ANCHORS, THIRD_SLOT_ALLOWED, assign_thirds_to_slots, label_groups_official,
)
from src.simulation.engine import GoalsParams
from src.simulation.tournament import Context, simulate_tournament


def test_label_groups_official_covers_a_to_l():
    reconstructed = {
        f"g{i}": [anchor, f"x{i}", f"y{i}", f"z{i}"]
        for i, anchor in enumerate(GROUP_ANCHORS)
    }
    official = label_groups_official(reconstructed)
    assert set(official) == set("ABCDEFGHIJKL")
    # The group containing 'Mexico' must be labeled 'A'.
    assert "Mexico" in official["A"]


def test_assign_thirds_respects_allowed_sets():
    thirds = [(g, f"team_{g}") for g in "ABCDEFGH"]
    slot_team = assign_thirds_to_slots(thirds)
    assert len(slot_team) == 8
    group_of = {team: g for g, team in thirds}
    # Every assigned third must come from a group allowed in that slot.
    for slot, team in slot_team.items():
        assert group_of[team] in THIRD_SLOT_ALLOWED[slot]


def _synthetic_context() -> Context:
    groups, ratings = {}, {}
    for letter, anchor in zip("ABCDEFGHIJKL", GROUP_ANCHORS):
        teams = [anchor, f"{letter}2", f"{letter}3", f"{letter}4"]
        groups[letter] = teams
        for k, t in enumerate(teams):
            ratings[t] = 1500 + 20 * k  # mild spread so ranking is non-degenerate
    group_fixtures = {
        letter: [(teams[i], teams[j]) for i in range(4) for j in range(i + 1, 4)]
        for letter, teams in groups.items()
    }
    return Context(ratings, groups, group_fixtures, GoalsParams(0.1, 0.001, 0.0))


def test_tournament_has_exactly_one_champion_and_32_qualifiers():
    ctx = _synthetic_context()
    stage = simulate_tournament(ctx, np.random.default_rng(0))
    assert len(stage) == 32                      # exactly 32 teams reach the Round of 32
    assert sum(1 for s in stage.values() if s == 5) == 1  # one champion
    assert max(stage.values()) == 5
