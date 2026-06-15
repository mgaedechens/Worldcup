# ADR-006 — Composite team strength: blend Elo with the betting market and squad value

**Date:** 2026-06-15 · **Status:** Accepted

## Context
Until now every simulated match was driven by a single number: each team's **Elo**, learned
purely from past results. Elo is a strong, leakage-free backbone, but on its own it lags two
things that matter for a *forward-looking* forecast:

- **Squad quality today.** A young core on the rise (or an ageing side coasting on reputation)
  is mispriced by results that are months or years old.
- **The market's information.** Bookmakers' de-vigged outright odds are the single most
  accurate public forecast of football outcomes; they aggregate injuries, form, squad news and
  sharp money that a results-only model cannot see.

The user's feedback was direct: the bracket looked implausible and the model should "use
statistics, use the current players, be far more complete." The aggregate title odds were
already reasonable, but the engine was thin and Elo-only, and Argentina (old squad, strong
Elo) and a few others were visibly over-rated relative to the market.

## Options considered
1. **Keep Elo-only** — simple, but ignores freely available, more accurate signals.
2. **Post-hoc blend of title probabilities** toward the market — fixes the headline numbers
   but not the bracket: the knockout *paths* still run on Elo-only matchups. Rejected.
3. **Composite strength on the Elo scale** — fuse Elo, market-implied rating and squad-value
   rating into one number that drives every match, so the whole bracket becomes
   market-consistent. Chosen.
4. Player-level micro data (xG, individual ratings, lineups) — highest ceiling, but no clean
   free source for the 2026 field and not reproducible. Deferred.

## Decision
Adopt **option 3**. New module `src/features/strength.py`:

- Each auxiliary signal is reduced to a z-score over the teams it covers, then mapped onto the
  Elo scale using Elo's own mean/std over that same set, so it expresses a peer-relative
  adjustment in Elo units.
  - **Squad value** (log of Transfermarkt aggregate, all 48 teams) — `data/external/squad_value_2026.csv`.
  - **Market** (log of de-vigged title probability; only the ~13 teams the books price) —
    `data/external/market_odds_2026.csv`. The quoted teams are normalised to hold ~90% of the
    title mass (the unpriced long tail holds the rest), removing both the bookmaker overround
    and the missing-teams mass.
- The composite is a weighted average over whichever signals a team has, renormalised when one
  is missing: **Elo 0.45, market 0.35, squad 0.20**. Most minnows have no market quote and fall
  back to Elo + squad automatically. The market gets the most weight because it is the most
  accurate; Elo is the dynamic backbone; squad value is a smaller current-players nudge.

Data are **curated snapshots** committed to the repo (source + date in each file header),
retrieved 2026-06-15. They are not auto-scraped, so they are reproducible and auditable; they
can be refreshed by editing the CSVs.

## σ recalibration (supersedes the ADR-005 sweep)
`scripts/calibrate_sigma.py` now scores the **full simulated title distribution against the
de-vigged market** (sum of squared error over priced teams), instead of eyeballing "favorite
in 15–20%". Because the ratings now already fold in the market, less artificial noise is
needed than before. SSE is flat across σ = 75–125 (within Monte Carlo noise); **σ = 125 is
kept** because it also reproduces the market favourite almost exactly (16.0% vs 15.8%) while
preserving realistic knockout variance.

| σ | Favourite (Spain) | vs-market SSE | MAE |
|---|---|---|---|
| 0 | 22.8% | 0.0085 | 0.0193 |
| 75 | 17.9% | 0.0030 | 0.0126 |
| 100 | 17.6% | 0.0038 | 0.0140 |
| **125** | **16.0%** | **0.0035** | **0.0127** |
| 150 | 15.1% | 0.0052 | 0.0148 |
| 200 | 11.6% | 0.0087 | 0.0190 |

## Consequences
- ✅ Headline odds now track the market closely: Spain 15.8%, France 11.3%, Argentina 10.9%,
  England 9.2%, Portugal 6.5%, Brazil 6.4% (10k, seed 42). Argentina is cooled from its
  Elo-only 14.4%; France/England warmed.
- ✅ The whole bracket — group scorelines and knockout paths — is now driven by market- and
  squad-aware ratings, which is what makes individual simulated tournaments look credible.
- ✅ The match predictor and the bracket read the **same** composite ratings (single source of
  truth preserved); the Validation tab publishes the per-team Elo → market → squad → composite
  breakdown.
- ✅ Honest limitation narrowed: the model now reads squad value (a players signal) but still
  does not know day-of injuries, suspensions or the manager's lineup.
- ⚠️ Market + squad partly overlap (the market already prices squad quality); the blend is an
  ensemble, not a causal decomposition, and the weights are a documented judgment, not fitted.
- ⚠️ Curated data snapshots go stale; they carry a retrieval date and are trivially updatable.
