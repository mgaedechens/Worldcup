"""Simulate ONE full World Cup: group stage -> knockouts -> champion.

Returns, for every participating team, the furthest stage it reached, so the Monte Carlo
runner can aggregate stage-reach and title probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.elo import compute_elo
from src.simulation.bracket import (
    FINAL_MATCH, QUARTERFINALS, ROUND_OF_16, ROUND_OF_32, SEMIFINALS,
    assign_thirds_to_slots, label_groups_official,
)
from src.simulation.engine import GoalsParams, knockout_winner, load_goals_params, simulate_scoreline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"


HOST_NATIONS = frozenset({"United States", "Canada", "Mexico"})


@dataclass
class Context:
    """Everything needed to simulate, loaded once and reused across Monte Carlo runs."""

    ratings: dict[str, float]                  # current Elo for the 48 teams
    groups: dict[str, list[str]]               # official letter -> 4 teams
    group_fixtures: dict[str, list[tuple[str, str]]]  # letter -> 6 (home, away) pairs
    params: GoalsParams
    hosts: frozenset[str] = HOST_NATIONS       # teams eligible for a host boost
    host_bonus: float = 0.0                    # Elo points added to hosts (0 = neutral, v1 default)

    def effective_ratings(self) -> dict[str, float]:
        """Ratings with the optional host boost folded in (v2 'home World Cup' effect)."""
        if self.host_bonus == 0.0:
            return self.ratings
        return {t: r + (self.host_bonus if t in self.hosts else 0.0)
                for t, r in self.ratings.items()}


def build_context() -> Context:
    """Load current Elo, official groups, and the group fixture list."""
    matches = pd.read_csv(PROCESSED_DIR / "matches_clean.csv", parse_dates=["date"])
    ratings = compute_elo(matches).final_ratings

    groups_raw = pd.read_csv(EXTERNAL_DIR / "wc2026_groups.csv")
    reconstructed = {g: sub["team"].tolist() for g, sub in groups_raw.groupby("group")}
    groups = label_groups_official(reconstructed)

    team_to_group = {t: g for g, teams in groups.items() for t in teams}
    fixtures = pd.read_csv(EXTERNAL_DIR / "wc2026_fixtures.csv")
    group_fixtures: dict[str, list[tuple[str, str]]] = {g: [] for g in groups}
    for home, away in zip(fixtures["home_team"], fixtures["away_team"]):
        g = team_to_group.get(home)
        if g is not None and team_to_group.get(away) == g:
            group_fixtures[g].append((home, away))

    return Context(ratings, groups, group_fixtures, load_goals_params())


def _rank_group(teams, stats, rng):
    """Order teams by FIFA criteria: points, goal difference, goals for, then random.

    (Overall GD/GF precede head-to-head in World Cup rules; deeper ties are resolved by
    drawing of lots, approximated here by a random key.)"""
    return sorted(
        teams,
        key=lambda t: (stats[t]["pts"], stats[t]["gd"], stats[t]["gf"], rng.random()),
        reverse=True,
    )


def simulate_group_stage(ctx: Context, ratings: dict[str, float], rng: np.random.Generator):
    """Play all 72 group matches. Returns winners, runners-up, and ranked thirds."""
    winners, runners = {}, {}
    thirds: list[tuple[str, str, dict]] = []  # (group, team, stats)

    for g, teams in ctx.groups.items():
        stats = {t: {"pts": 0, "gf": 0, "ga": 0, "gd": 0} for t in teams}
        for home, away in ctx.group_fixtures[g]:
            gh, ga = simulate_scoreline(ratings[home], ratings[away], ctx.params, rng)
            stats[home]["gf"] += gh
            stats[home]["ga"] += ga
            stats[away]["gf"] += ga
            stats[away]["ga"] += gh
            if gh > ga:
                stats[home]["pts"] += 3
            elif ga > gh:
                stats[away]["pts"] += 3
            else:
                stats[home]["pts"] += 1
                stats[away]["pts"] += 1
        for t in teams:
            stats[t]["gd"] = stats[t]["gf"] - stats[t]["ga"]

        ranked = _rank_group(teams, stats, rng)
        winners[g], runners[g] = ranked[0], ranked[1]
        thirds.append((g, ranked[2], stats[ranked[2]]))

    return winners, runners, thirds


def _best_eight_thirds(thirds, rng):
    """Pick the 8 best third-placed teams (points, GD, GF, random)."""
    ranked = sorted(
        thirds,
        key=lambda x: (x[2]["pts"], x[2]["gd"], x[2]["gf"], rng.random()),
        reverse=True,
    )
    return [(g, team) for g, team, _ in ranked[:8]]


def simulate_tournament(ctx: Context, rng: np.random.Generator) -> dict[str, int]:
    """Run a full tournament. Returns team -> knockout wins (0..5; 5 = champion).

    Stage mapping: 0 reached R32, 1 R16, 2 QF, 3 SF, 4 Final, 5 Champion.
    """
    ratings = ctx.effective_ratings()
    winners, runners, thirds = simulate_group_stage(ctx, ratings, rng)
    qualifying_thirds = _best_eight_thirds(thirds, rng)
    third_by_slot = assign_thirds_to_slots(qualifying_thirds)

    # Every qualifier has reached at least the Round of 32.
    stage = {t: 0 for t in [*winners.values(), *runners.values(),
                            *[team for _, team in qualifying_thirds]]}

    def resolve(src):
        kind, key = src
        if kind == "W":
            return winners[key]
        if kind == "R":
            return runners[key]
        return third_by_slot[key]  # ("3", slot match-id)

    # Round of 32
    results: dict[int, str] = {}
    for mid, sa, sb in ROUND_OF_32:
        w = knockout_winner(resolve(sa), resolve(sb), ratings, ctx.params, rng)
        results[mid] = w
        stage[w] = 1

    # Later rounds: each is (match_id -> two feeder match ids), winner bumps its stage.
    for round_map, reached in (
        (ROUND_OF_16, 2), (QUARTERFINALS, 3), (SEMIFINALS, 4),
    ):
        for mid, (fa, fb) in round_map.items():
            w = knockout_winner(results[fa], results[fb], ratings, ctx.params, rng)
            results[mid] = w
            stage[w] = reached

    champion = knockout_winner(results[FINAL_MATCH[0]], results[FINAL_MATCH[1]],
                               ratings, ctx.params, rng)
    stage[champion] = 5
    return stage
