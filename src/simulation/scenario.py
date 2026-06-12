"""Simulate ONE detailed tournament for visualization.

Unlike the Monte Carlo runner (which only needs the stage each team reached, aggregated over
thousands of runs), this records the full story of a single simulated tournament: every group
table and every knockout scoreline. It is a *sample scenario* — re-rolling the seed yields a
different plausible tournament — which is exactly how to honestly show "exact results" from a
probabilistic model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.simulation.bracket import (
    FINAL_MATCH, QUARTERFINALS, ROUND_OF_16, ROUND_OF_32, SEMIFINALS,
    assign_thirds_to_slots,
)
from src.simulation.engine import knockout_result, simulate_scoreline
from src.simulation.tournament import Context

_ROUND_NAMES = {
    "R32": "Round of 32", "R16": "Round of 16", "QF": "Quarterfinal",
    "SF": "Semifinal", "F": "Final",
}


@dataclass
class KOMatch:
    round_key: str
    team_a: str
    team_b: str
    goals_a: int
    goals_b: int
    winner: str
    pens: bool


@dataclass
class GroupMatch:
    home: str
    away: str
    goals_h: int
    goals_a: int


@dataclass
class Scenario:
    group_tables: dict[str, list[dict]]      # letter -> ordered standings rows
    group_matches: dict[str, list[GroupMatch]]  # letter -> the 6 played fixtures
    knockouts: list[KOMatch]
    champion: str
    stats: dict = field(default_factory=dict)


def _standings_row(team: str) -> dict:
    return {"team": team, "pld": 0, "w": 0, "d": 0, "l": 0, "gf": 0, "ga": 0, "gd": 0, "pts": 0}


def simulate_scenario(ctx: Context, rng: np.random.Generator) -> Scenario:
    ratings = ctx.effective_ratings()
    params = ctx.params

    group_tables: dict[str, list[dict]] = {}
    group_matches: dict[str, list[GroupMatch]] = {}
    winners, runners = {}, {}
    thirds: list[tuple[str, str, dict]] = []
    total_goals = 0

    # --- Group stage -------------------------------------------------------- #
    for g, teams in ctx.groups.items():
        rows = {t: _standings_row(t) for t in teams}
        group_matches[g] = []
        for home, away in ctx.group_fixtures[g]:
            gh, ga = simulate_scoreline(ratings[home], ratings[away], params, rng)
            group_matches[g].append(GroupMatch(home, away, gh, ga))
            total_goals += gh + ga
            for t, gf, gag in ((home, gh, ga), (away, ga, gh)):
                rows[t]["pld"] += 1
                rows[t]["gf"] += gf
                rows[t]["ga"] += gag
            if gh > ga:
                rows[home]["w"] += 1
                rows[home]["pts"] += 3
                rows[away]["l"] += 1
            elif ga > gh:
                rows[away]["w"] += 1
                rows[away]["pts"] += 3
                rows[home]["l"] += 1
            else:
                rows[home]["d"] += 1
                rows[away]["d"] += 1
                rows[home]["pts"] += 1
                rows[away]["pts"] += 1
        for r in rows.values():
            r["gd"] = r["gf"] - r["ga"]
        ordered = sorted(rows.values(),
                         key=lambda r: (r["pts"], r["gd"], r["gf"], rng.random()), reverse=True)
        group_tables[g] = ordered
        winners[g], runners[g] = ordered[0]["team"], ordered[1]["team"]
        thirds.append((g, ordered[2]["team"], ordered[2]))

    # Best 8 thirds + slot assignment.
    best_thirds = sorted(thirds, key=lambda x: (x[2]["pts"], x[2]["gd"], x[2]["gf"], rng.random()),
                         reverse=True)[:8]
    third_by_slot = assign_thirds_to_slots([(g, t) for g, t, _ in best_thirds])

    # --- Knockouts ---------------------------------------------------------- #
    def resolve(src):
        kind, key = src
        return winners[key] if kind == "W" else runners[key] if kind == "R" else third_by_slot[key]

    knockouts: list[KOMatch] = []
    results: dict[int, str] = {}
    shootouts = 0

    def play(round_key, a, b):
        nonlocal total_goals, shootouts
        w, ga, gb, pens = knockout_result(a, b, ratings, params, rng)
        total_goals += ga + gb
        shootouts += int(pens)
        knockouts.append(KOMatch(round_key, a, b, ga, gb, w, pens))
        return w

    for mid, sa, sb in ROUND_OF_32:
        results[mid] = play("R32", resolve(sa), resolve(sb))
    for round_map, key in ((ROUND_OF_16, "R16"), (QUARTERFINALS, "QF"), (SEMIFINALS, "SF")):
        for mid, (fa, fb) in round_map.items():
            results[mid] = play(key, results[fa], results[fb])
    champion = play("F", results[FINAL_MATCH[0]], results[FINAL_MATCH[1]])

    n_matches = 72 + len(knockouts)
    biggest = max(knockouts, key=lambda m: abs(m.goals_a - m.goals_b))
    stats = {
        "total_goals": total_goals,
        "n_matches": n_matches,
        "avg_goals": round(total_goals / n_matches, 2),
        "shootouts": shootouts,
        "biggest_win": biggest,
    }
    return Scenario(group_tables, group_matches, knockouts, champion, stats)
