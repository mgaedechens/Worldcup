"""Calibrate the per-tournament rating-uncertainty sigma (see ADR-005).

Why this exists: treating Elo ratings as exact truths makes the Monte Carlo overconfident,
because a rating error repeats across all of a team's matches in the same tournament. We fix
that by drawing each team's strength once per simulated tournament from N(rating, sigma).

How sigma is chosen: we sweep candidate sigmas and compare the favorite's title probability
against the external benchmark quants use, the betting market, where the 2026 favorite trades
around 15-20% implied probability, consistent with the historical record of pre-tournament
favorites (the bookmakers' favorite has rarely won the World Cup in the modern era).

Run:
    python scripts/calibrate_sigma.py        # writes reports/sigma_calibration.csv
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.simulation.montecarlo import run_montecarlo  # noqa: E402
from src.simulation.tournament import build_context  # noqa: E402

REPORTS = Path(__file__).resolve().parents[1] / "reports"
SIGMAS = (0, 75, 100, 125, 150)
N = 4000
SEED = 11


def main() -> None:
    ctx = build_context()
    rows = []
    for sigma in SIGMAS:
        t0 = time.time()
        df = run_montecarlo(N, seed=SEED, ctx=ctx, strength_noise=sigma)
        d = df.set_index("team")["Champion"]
        top4 = df.head(4)
        rows.append({
            "sigma": sigma,
            "favorite": top4.iloc[0]["team"],
            "favorite_prob": round(top4.iloc[0]["Champion"], 4),
            "second": top4.iloc[1]["team"],
            "second_prob": round(top4.iloc[1]["Champion"], 4),
            "top4_prob_sum": round(top4["Champion"].sum(), 4),
            "mexico_prob": round(float(d.get("Mexico", 0.0)), 4),
            "n_sims": N,
            "seconds": round(time.time() - t0, 1),
        })
        print(f"sigma={sigma:>3}: favorite {rows[-1]['favorite']} "
              f"{rows[-1]['favorite_prob']:.1%} | top4 {rows[-1]['top4_prob_sum']:.1%} "
              f"| Mexico {rows[-1]['mexico_prob']:.1%}")

    out = pd.DataFrame(rows)
    REPORTS.mkdir(parents=True, exist_ok=True)
    out.to_csv(REPORTS / "sigma_calibration.csv", index=False)
    print(f"\n[ok] evidence -> {REPORTS / 'sigma_calibration.csv'}")
    print("Benchmark: market-implied favorite probability ~15-20%.")


if __name__ == "__main__":
    main()
