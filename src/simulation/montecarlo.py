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
from pathlib import Path

import numpy as np
import pandas as pd

from src.simulation.bracket import STAGES
from src.simulation.tournament import Context, build_context, simulate_tournament

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"


def run_montecarlo(
    n: int = 10_000, seed: int = 42, ctx: Context | None = None,
    host_bonus: float | None = None,
) -> pd.DataFrame:
    """Simulate ``n`` tournaments and return per-team stage-reach probabilities.

    ``host_bonus`` optionally overrides the Elo boost given to host nations (default 0 keeps
    the neutral v1 model); used by the robustness analysis.
    """
    ctx = ctx or build_context()
    if host_bonus is not None:
        ctx.host_bonus = host_bonus
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
            "elo": round(ctx.ratings[team]),
            **{stage: probs[i] for i, stage in enumerate(STAGES)},
        })
    df = pd.DataFrame(rows).sort_values("Champion", ascending=False).reset_index(drop=True)
    return df


def run() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10_000, help="number of simulations")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Simulating the 2026 World Cup {args.n:,} times...")
    df = run_montecarlo(args.n, args.seed)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "simulation_results.csv"
    df.to_csv(out, index=False)

    pd.set_option("display.float_format", lambda v: f"{v:.1%}")
    print("\nTitle probabilities — top 20:\n")
    print(df.head(20)[["team", "group", "elo", "Champion", "Final", "Semifinal"]]
          .to_string(index=False))
    print(f"\n[ok] full table -> {out}")


if __name__ == "__main__":
    run()
