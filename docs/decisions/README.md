# Architecture Decision Records (ADRs)

This folder is the project's **decision memory**. Every non-trivial choice is recorded as
a short, dated ADR so that any contributor — human or AI agent (Claude Code / Gemini CLI) —
can see *what* was decided, *why*, and *what was rejected*, without re-litigating it.

## The "decision team" protocol

To avoid single-perspective blind spots, each significant decision is stress-tested from
four explicit lenses before it is accepted. Think of it as a standing review panel:

| Lens | Question it forces |
|------|--------------------|
| 🧪 **ML Scientist** | Is it statistically sound? Leakage? Calibration? Proper metrics? |
| 🏗️ **Data Engineer** | Is it reproducible, maintainable, and simple to run/regenerate? |
| 📈 **Quant** | Are we benchmarking vs naive? Is risk/uncertainty honest? Out-of-sample? |
| 🕵️ **Skeptic / Reviewer** | What's the simplest thing that breaks this? Are we over-engineering? |

A decision is only recorded once it survives all four. The "Consequences" section of each
ADR notes the residual risks the panel could not fully eliminate.

## Format

Each ADR follows: **Context → Options considered → Decision → Consequences**.
Numbered sequentially, immutable once accepted (supersede with a new ADR rather than editing).

## Index

- [ADR-001](ADR-001-data-source.md) — Data source & acquisition
- [ADR-002](ADR-002-training-window.md) — Training window (2002+) with full-history Elo warm-up
- [ADR-003](ADR-003-model-selection.md) — Logistic regression over gradient boosting (parsimony)
- [ADR-004](ADR-004-simulation-design.md) — Poisson goals model + official FIFA bracket
