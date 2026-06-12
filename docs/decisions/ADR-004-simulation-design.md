# ADR-004 — Poisson goals model + official FIFA bracket

**Date:** 2026-06-11 · **Status:** Accepted

## Context
Simulating the tournament needs more than win/draw/loss: group standings use points **and**
goal difference, and the 48-team format ranks the 8 best third-placed teams largely on goal
difference. The knockout draw is a fixed FIFA template keyed to group letters.

## Options considered
**Match outcomes / goals:**
1. Use only the W/D/L classifier + a crude tie-break (points → head-to-head → random).
2. Add an **independent Poisson goals model** driven by Elo → coherent scorelines, real goal
   difference, FIFA-style tie-breaks. Use the classifier as a cross-check.

**Knockout bracket:**
1. A generic re-seeded bracket (approximation, no external data).
2. The **official FIFA Round-of-32 template** with real group labels.

## Decision
Adopt **Poisson goals model (option 2)** and the **official FIFA bracket (option 2)**.
(User chose the higher-fidelity path; "make it really good".)

## Consequences
- ✅ Scorelines, goal difference, and tie-breaks are internally consistent.
- ✅ The bracket matches reality, which is credible for a portfolio.
- ✅ Poisson-implied W/D/L can be validated against the independent logistic classifier.
- ⚠️ Independent Poisson slightly under-predicts draws (ignores score correlation); the
  Dixon–Coles correction is the documented v2 improvement.
- ⚠️ Official group letters must be sourced and mapped to our reconstructed groups; recorded
  in `data/external/` and verified by team-set matching.
