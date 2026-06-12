"""Run the full pipeline end to end, from raw download to title probabilities.

This is the single reproducible entry point: a fresh clone becomes a finished forecast.

    python scripts/run_pipeline.py            # full run, 10,000 simulations
    python scripts/run_pipeline.py --n 2000   # fewer simulations (faster)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow running as a plain script (python scripts/run_pipeline.py) by putting the project
# root on the import path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import clean, download
from src.features import build_features
from src.models import goals, train
from src.simulation import montecarlo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10_000, help="number of simulations")
    args = parser.parse_args()

    steps = [
        ("Download raw data", lambda: download.download_all()),
        ("Clean & extract WC2026 fixture", clean.run),
        ("Build features (Elo, form, ...)", build_features.run),
        ("Train & select classifier", train.run),
        ("Fit Poisson goals model", goals.run),
    ]
    for i, (label, fn) in enumerate(steps, 1):
        print(f"\n=== [{i}/{len(steps) + 1}] {label} ===")
        t = time.time()
        fn()
        print(f"    done in {time.time() - t:.1f}s")

    print(f"\n=== [{len(steps) + 1}/{len(steps) + 1}] Monte Carlo simulation ===")
    df = montecarlo.run_montecarlo(args.n)
    out = montecarlo.REPORTS_DIR / "simulation_results.csv"
    df.to_csv(out, index=False)
    print("\nTop 10 title contenders:\n")
    print(df.head(10)[["team", "group", "elo", "Champion"]].to_string(index=False))
    print(f"\nFull results -> {out}")
    print("Done. The 2026 World Cup has been simulated.")


if __name__ == "__main__":
    main()
