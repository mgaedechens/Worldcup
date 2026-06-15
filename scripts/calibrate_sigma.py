"""Calibrate the per-tournament rating-uncertainty sigma against the betting market (ADR-005/006).

Why this exists: treating ratings as exact truths makes the Monte Carlo overconfident, because a
rating error repeats across all of a team's matches in the same tournament. We fix that by drawing
each team's strength once per simulated tournament from N(rating, sigma).

How sigma is chosen (upgraded): now that team strengths blend in the de-vigged betting market, we
no longer eyeball "favorite in 15-20%". Instead we sweep sigma and pick the value whose *full
simulated title distribution* best matches the market's de-vigged title distribution over the
teams the books price, scored by sum-of-squared error (and reported alongside mean absolute error).
This ties the one free knob directly to the most accurate public forecast available.

Run:
    python scripts/calibrate_sigma.py        # writes reports/sigma_calibration.csv
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.strength import MARKET_ODDS_CSV, market_implied_probs  # noqa: E402
from src.simulation.montecarlo import run_montecarlo  # noqa: E402
from src.simulation.tournament import build_context  # noqa: E402

REPORTS = Path(__file__).resolve().parents[1] / "reports"
SIGMAS = (0, 75, 100, 125, 150, 175, 200)
N = 5000
SEED = 11


def _market_title_probs() -> dict[str, float]:
    odds_df = pd.read_csv(MARKET_ODDS_CSV, comment="#")
    return market_implied_probs(dict(zip(odds_df["team"], odds_df["decimal_odds"])))


def main() -> None:
    ctx = build_context()
    market = _market_title_probs()
    priced = list(market)
    m = np.array([market[t] for t in priced])

    rows = []
    for sigma in SIGMAS:
        t0 = time.time()
        df = run_montecarlo(N, seed=SEED, ctx=ctx, strength_noise=sigma)
        champ = df.set_index("team")["Champion"]
        s = np.array([float(champ.get(t, 0.0)) for t in priced])
        sse = float(np.sum((s - m) ** 2))
        mae = float(np.mean(np.abs(s - m)))
        top = df.head(1).iloc[0]
        rows.append({
            "sigma": sigma,
            "favorite": top["team"],
            "favorite_prob": round(float(top["Champion"]), 4),
            "market_favorite_prob": round(float(max(market.values())), 4),
            "vs_market_sse": round(sse, 5),
            "vs_market_mae": round(mae, 5),
            "n_sims": N,
            "seconds": round(time.time() - t0, 1),
        })
        print(f"sigma={sigma:>3}: favorite {rows[-1]['favorite']} "
              f"{rows[-1]['favorite_prob']:.1%} | vs-market SSE {sse:.4f} | MAE {mae:.4f}")

    out = pd.DataFrame(rows)
    best = out.loc[out["vs_market_sse"].idxmin()]
    REPORTS.mkdir(parents=True, exist_ok=True)
    out.to_csv(REPORTS / "sigma_calibration.csv", index=False)
    print(f"\n[ok] evidence -> {REPORTS / 'sigma_calibration.csv'}")
    print(f"Best fit to the market: sigma={int(best['sigma'])} "
          f"(SSE {best['vs_market_sse']:.4f}, favorite {best['favorite_prob']:.1%} "
          f"vs market {best['market_favorite_prob']:.1%}).")


if __name__ == "__main__":
    main()
