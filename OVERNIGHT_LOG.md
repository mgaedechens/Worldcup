# 🌙 Overnight autonomous work log

This file is a human-readable journal of the autonomous overnight session, so Matías can
quickly see in the morning what happened. Detailed engineering state lives in
`PROJECT_STATUS.md`; design rationale lives in `docs/decisions/`.

**Session start:** 2026-06-11 evening
**Agent:** Claude Code (Opus 4.8), running autonomously
**Goal:** finish a complete, polished v1 — simulation, results, tests, docs — committing at
every milestone so the morning state is always coherent and reversible.

**Decision discipline:** every significant choice is stress-tested via the 4-lens panel in
`docs/decisions/README.md` and recorded as an ADR.

---

## Timeline (newest first)

### ✅ v1 COMPLETE — full pipeline finished
The project is now an end-to-end, reproducible, tested forecast. A fresh clone runs
`python scripts/run_pipeline.py` and produces the title probabilities.

What got built tonight (in order):
1. **Decision-team protocol + ADRs 001–004** (`docs/decisions/`).
2. **Official FIFA bracket research** — fetched all 12 official groups + the full Round-of-32
   → Final structure (matches 73–104). Verified our reconstructed groups match the official
   draw; mapped via unique anchor teams.
3. **`src/simulation/engine.py`** — fast closed-form Poisson scoreline + knockout resolution
   (regulation draw → shootout weighted by Elo). Fixed a numpy-2 bug (`np.math.factorial`).
4. **`src/simulation/bracket.py`** — official structure + best-thirds assignment via
   constraint-respecting matching (scipy `linear_sum_assignment`).
5. **`src/simulation/tournament.py`** — group stage with FIFA tie-breaks + knockouts.
6. **`src/simulation/montecarlo.py`** — N-run aggregation → stage/title probabilities.
7. **10,000-run simulation:** 🇪🇸 Spain 27.7%, 🇦🇷 Argentina 19.7%, 🇫🇷 France 10.3%,
   England 7.3%, Brazil 5.0%. (`reports/simulation_results.csv`)
8. **`notebooks/04_simulation.ipynb`** — headline chart, survival curves, and a cross-check
   proving the Poisson and logistic models agree out-of-sample (log loss 0.8765 vs 0.8768).
9. **Test suite** — 14 pytest invariants, all passing.
10. **`scripts/run_pipeline.py`** end-to-end (validated), **MIT LICENSE**, polished **README**
    with results, figures, methodology, and honest limitations.

Bugs found & fixed autonomously: isotonic-calibration log-loss regression (→ chose plain
logistic, ADR-003); Poisson GLM numeric overflow (→ standardized features); `np.math`
removal in numpy 2; script import path.

### Suggested next (v2, for when Matías is back)
- Dixon-Coles draw correction + host-nation home advantage.
- Bookmaker-odds comparison + Kelly staking (the quant capstone we deferred — needs odds data).
- Streamlit dashboard for a clickable CV demo.

### Setup
- Created the decision-team protocol (`docs/decisions/README.md`) and retroactive ADRs
  001–004 capturing the journey so far.
- Created this log. Next: official bracket research → simulation engine.
