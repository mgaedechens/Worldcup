# ADR-001 — Data source & acquisition

**Date:** 2026-06-11 · **Status:** Accepted

## Context
We need a long, reliable history of international match results to rate teams and train a
model, plus the official 2026 World Cup fixture to simulate.

## Options considered
1. **Kaggle "International football results" dataset** — popular, but needs an account/API key.
2. **`martj42/international_results` GitHub repo** — same canonical data, raw CSV over HTTPS.
3. Scrape FIFA / federation sites — fragile, rate-limited, legally murky.

## Decision
Use **option 2**. Download `results.csv` (+ `shootouts.csv`) directly from the GitHub raw
URLs via `src/data/download.py`.

## Consequences
- ✅ Fully reproducible: anyone who clones the repo regenerates `data/raw/` with no secrets.
- ✅ The dataset already contains the 72 scheduled WC2026 group matches → we extracted the
  48 teams and 12 groups for free (`src/data/clean.py`).
- ⚠️ Single-source dependency on one maintainer's repo; mitigated because the download is
  pinned to a known schema and the cleaned output is validated.
- ⚠️ Team names change across history (West Germany, etc.) → handled by a documented
  normalization map; still a simplification.
