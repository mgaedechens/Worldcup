# ADR-005 — Per-tournament rating uncertainty + data-fitted host advantage

**Date:** 2026-06-12 · **Status:** Accepted

## Context
The 2026-06-12 audit found the Monte Carlo's title odds were overconfident (favorite at
27.7% vs a market/historical benchmark of ~15-20%), and identified the cause: Elo ratings
were treated as exact truths. A rating error is **correlated within a tournament** (it
follows the team through all of its matches), so point-estimate ratings compound into
over-concentrated title probabilities even when match-level calibration is good.

Separately, the fixture data marks the hosts' own-stadium group games as non-neutral
(9 matches: 3 each for USA/Canada/Mexico), but the simulation ignored this.

## Options considered
1. Keep point-estimate ratings (status quo) — simple, but demonstrably overconfident at
   tournament level.
2. Flatten probabilities post-hoc — a fudge with no mechanism; rejected.
3. **Draw each team's strength once per simulated tournament from N(rating, σ)** — models
   exactly the correlated-error mechanism; standard practice in professional forecasting
   models. σ chosen by sweeping candidates against an external benchmark.
4. Host advantage: (a) ignore; (b) flat Elo bonus (arbitrary); (c) **use the goals model's
   own home-advantage coefficient (fitted on 150 years of data) for the 9 group games the
   fixture list marks as real home matches**; knockouts stay neutral (venues for specific
   teams are not guaranteed).

## Decision
Adopt **option 3** with **σ = 125** and **option 4(c)**.

Calibration evidence (`scripts/calibrate_sigma.py` → `reports/sigma_calibration.csv`,
4,000 sims per row, benchmark: market favorite ~15-20%):

| σ | Favorite (Spain) | Top-4 share | Mexico |
|---|---|---|---|
| 0 | 26.3% | 63.8% | 1.9% |
| 75 | 23.8% | 57.6% | 1.5% |
| 100 | 20.9% | 54.1% | 1.8% |
| **125** | **19.3%** | **50.1%** | **2.4%** |
| 150 | 17.3% | 46.0% | 2.4% |

σ=150 would match the market mid-point exactly, but the goal is to correct the
overconfidence mechanism, not to copy the market (which also prices squad/injury
information we deliberately do not use). σ=125 enters the supported band while preserving
more of the model's own signal; the full sweep is published in the dashboard's Validation
tab for transparency. Mexico at ~2.4% with the host boost matches bookmaker quotes (~2.5-3%).

## Consequences
- ✅ Headline odds are realistic: Spain 19.5%, Argentina 14.4%, France 9.0% (10k run).
- ✅ A **metadata sidecar** (`simulation_results.meta.json`) now pins n, seed, σ, host
  settings and the reference bracket seed to the published CSV; the dashboard warns if the
  cached numbers disagree with the live engine. This guards against the exact audit finding
  (artifact/engine divergence) recurring.
- ✅ The Bracket tab's fixed scenario seed is found **programmatically** (smallest seed whose
  champion equals the modal favorite) instead of being hardcoded.
- ⚠️ Scenario tabs (Bracket/Simulator) share the same noise process, so a single displayed
  tournament can feature strength draws far from the ratings — that variance is the point,
  and the UI copy explains it.
- ⚠️ σ is a judgment within an evidence band, not a uniquely determined constant; the sweep
  documents the sensitivity honestly.
