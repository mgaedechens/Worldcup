# ADR-002 — Training window (2002+) with full-history Elo warm-up

**Date:** 2026-06-11 · **Status:** Accepted

## Context
The dataset spans 1872–2026. Early football is sparse and barely resembles the modern game,
but Elo ratings need long memory so teams are not cold-started at a flat 1500.

## Options considered
1. Train on the **full history** — maximizes rows but mixes eras and injects noise.
2. Train on a **modern window only** and compute Elo only on that window — clean era, but
   ratings start cold and are wrong for years.
3. **Warm Elo on the full history, train the classifier on the modern window** (hybrid).

## Decision
Use **option 3**: Elo is computed over *all* matches; the supervised training table is
filtered to **2002 onward** (EDA showed data becomes dense and stable in the modern era).

## Consequences
- ✅ Ratings are warm and meaningful from the first modern match; the model learns only
  from relevant, dense data.
- ✅ A reusable pattern (long-memory feature + short training window) common in time series.
- ⚠️ The 2002 cutoff is a judgement call; sensitivity to it is a candidate robustness check.
