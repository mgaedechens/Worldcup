"""Monte Carlo runner: play the tournament N times and aggregate probabilities.

Why Monte Carlo? Football is high-variance: the favorite does not always win. Rather than
predict one outcome, we replay the whole tournament thousands of times and read off how
often each team reaches each stage. This is the same simulation technique used in finance
(portfolio/retirement Monte Carlo).

Run:
    python -m src.simulation.montecarlo            # default 10,000 simulations
    python -m src.simulation.montecarlo --n 2000
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src.simulation.bracket import STAGES
from src.simulation.scenario import simulate_scenario
from src.simulation.tournament import Context, build_context, simulate_tournament

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"
META_PATH = REPORTS_DIR / "simulation_results.meta.json"


def run_montecarlo(
    n: int = 10_000, seed: int = 42, ctx: Context | None = None,
    host_bonus: float | None = None, strength_noise: float | None = None,
) -> pd.DataFrame:
    """Simulate ``n`` tournaments and return per-team stage-reach probabilities.

    ``host_bonus`` optionally overrides the flat Elo boost given to host nations and
    ``strength_noise`` the per-tournament rating uncertainty; both are sensitivity knobs.
    """
    ctx = ctx or build_context()
    if host_bonus is not None:
        ctx.host_bonus = host_bonus
    if strength_noise is not None:
        ctx.strength_noise = strength_noise
    rng = np.random.default_rng(seed)

    all_teams = [t for teams in ctx.groups.values() for t in teams]
    team_to_group = {t: g for g, teams in ctx.groups.items() for t in teams}
    # counts[team][k] = times the team reached AT LEAST stage level k (0..5).
    counts = {t: np.zeros(len(STAGES), dtype=np.int64) for t in all_teams}

    for _ in range(n):
        stage = simulate_tournament(ctx, rng)
        for team, reached in stage.items():
            counts[team][: reached + 1] += 1

    rows = []
    for team in all_teams:
        probs = counts[team] / n
        rows.append({
            "team": team,
            "group": team_to_group[team],
            "elo": round(ctx.elo[team]),
            "strength": round(ctx.ratings[team]),
            **{stage: probs[i] for i, stage in enumerate(STAGES)},
        })
    df = pd.DataFrame(rows).sort_values("Champion", ascending=False).reset_index(drop=True)
    return df


def find_reference_seed(ctx: Context, champion: str, max_seed: int = 2000) -> int:
    """Smallest RNG seed whose detailed scenario crowns ``champion``.

    The dashboard's Bracket tab shows ONE fixed example tournament; tying its seed to the
    Monte Carlo's modal champion keeps that tab consistent with the Title race by
    construction, instead of by a hardcoded number that can silently go stale.
    """
    for seed in range(max_seed):
        if simulate_scenario(ctx, np.random.default_rng(seed)).champion == champion:
            return seed
    return 0


def save_results(df: pd.DataFrame, ctx: Context, n: int, seed: int) -> dict:
    """Persist the results CSV plus a metadata sidecar binding it to the engine settings.

    The sidecar is the single-source-of-truth guard: the dashboard compares it against the
    live engine parameters and warns if the cached probabilities were produced by a
    different engine (the exact failure mode found in the 2026-06-12 audit).
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(REPORTS_DIR / "simulation_results.csv", index=False)

    favorite = df.iloc[0]
    meta = {
        "n_simulations": n,
        "seed": seed,
        "strength_noise": ctx.strength_noise,
        "host_bonus": ctx.host_bonus,
        "favorite": favorite["team"],
        "favorite_prob": round(float(favorite["Champion"]), 4),
        "reference_seed": find_reference_seed(ctx, favorite["team"]),
        "generated": date.today().isoformat(),
    }
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def run() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10_000, help="number of simulations")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Simulating the 2026 World Cup {args.n:,} times...")
    ctx = build_context()
    df = run_montecarlo(args.n, args.seed, ctx=ctx)
    meta = save_results(df, ctx, args.n, args.seed)
    out = REPORTS_DIR / "simulation_results.csv"
    print(f"[ok] sidecar -> {META_PATH.name}  (reference bracket seed: {meta['reference_seed']})")

    pd.set_option("display.float_format", lambda v: f"{v:.1%}")
    print("\nTitle probabilities — top 20:\n")
    print(df.head(20)[["team", "group", "elo", "Champion", "Final", "Semifinal"]]
          .to_string(index=False))
    print(f"\n[ok] full table -> {out}")


if __name__ == "__main__":
    run()
