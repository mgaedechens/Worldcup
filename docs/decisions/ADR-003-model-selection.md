# ADR-003 — Logistic regression over gradient boosting (parsimony)

**Date:** 2026-06-11 · **Status:** Accepted

## Context
We need a calibrated 3-class (home/draw/away) model. The original plan favored a gradient
boosting classifier ("hybrid: Elo feature + ML").

## Options considered
Benchmarked on a strict out-of-time test (train < 2022, test ≥ 2022, includes WC2022) using
proper scoring rules (log loss, multiclass Brier, RPS):

| Model | log_loss | brier | rps |
|---|---|---|---|
| Baseline (base rates) | 1.050 | 0.633 | 0.228 |
| Elo-only logistic | 0.879 | 0.517 | 0.173 |
| GBDT (all features) | 0.877 | 0.516 | 0.172 |
| **Logistic (all features)** | **0.8765** | **0.515** | **0.1718** |

## Decision
Select the **multinomial Logistic Regression** on standardized features
(`elo_diff`, `form_diff`, `rest_diff`, `is_neutral`).

## Consequences
- ✅ It ties/beats GBDT on every proper score while being simpler, faster, naturally
  well-calibrated, and interpretable (coefficients). Occam's razor wins on evidence.
- ✅ Still honors the "Elo-as-feature + ML classifier" hybrid philosophy — we just let the
  data choose the classifier.
- ⚠️ Football is low-signal: no model beats the baseline by a huge margin; the value is in
  rigor and calibration, not in a magic edge. Documented openly.
- Note: isotonic calibration *hurt* log loss on the small calibration slice; plain logistic
  needed no post-hoc calibration.
