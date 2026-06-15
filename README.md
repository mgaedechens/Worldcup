# 🏆 World Cup 2026 Predictor

> A reproducible, **explainable** machine-learning pipeline that estimates each national
> team's probability of winning the **2026 FIFA World Cup**, built from 150+ years of
> international match results and a 10,000-run Monte Carlo simulation of the official bracket.

*[Versión en español más abajo ↓](#-predictor-del-mundial-2026-español)*

![Python](https://img.shields.io/badge/python-3.12+-blue) ![Tests](https://img.shields.io/badge/tests-14%20passing-brightgreen) ![License](https://img.shields.io/badge/license-MIT-green)

---

## 🥇 Headline result

10,000 simulated tournaments → probability of lifting the trophy:

![Title probabilities](reports/figures/10_title_probabilities.png)

| # | Team | Champion | Reaches Final | Reaches Semifinal |
|---|------|---------:|--------------:|------------------:|
| 1 | 🇪🇸 Spain | **15.8%** | 24.6% | 38.3% |
| 2 | 🇫🇷 France | 11.3% | 18.8% | 31.5% |
| 3 | 🇦🇷 Argentina | 10.9% | 18.2% | 30.1% |
| 4 | 🏴 England | 9.2% | 16.4% | 27.3% |
| 5 | 🇵🇹 Portugal | 6.5% | 12.6% | 23.3% |

Team strength is a **composite rating** that fuses three signals on the Elo scale: 150 years of
results (**Elo**), the **de-vigged betting market** (the most accurate public forecast there is),
and **Transfermarkt squad value** (a current-players signal). This is what cools an over-rated
side and warms an under-rated one — Argentina drops from its Elo-only 14.4% as the market and an
ageing squad temper its rating, while France and England climb. The favorite tracks the market
closely (Spain 15.8% vs ~15.8% market) — by **calibration against the market**, not by eye. Each
simulated tournament also redraws every team's strength around its rating (σ=125, see
[ADR-006](docs/decisions/ADR-006-composite-strength.md)), modeling the fact that ratings are
estimates whose errors persist across a whole tournament. Football is **low-signal,
high-variance**, and the model embraces that uncertainty rather than faking confidence.

## Why this project

The goal is **not** a crystal ball. It is to demonstrate a rigorous, end-to-end data-science
process that is **explainable**, **reproducible**, and **honest about uncertainty** — the kind
of work that holds up under scrutiny.

## How it works

```
results.csv (1872–2026)          market odds + squad value (2026 snapshots)
        │  clean + normalize               │
        ▼                                   ▼
   Elo ratings  ──────────►  COMPOSITE STRENGTH (Elo 45% · market 35% · squad 20%)
        │                                   │
        │                                   ├──►  match features ──► Logistic classifier
        │                                   │                         P(win/draw/loss), calibrated
        │                                   ▼                                   ▲
        │                       Poisson goals model ───────────────────────────┘
        │                                   │  scorelines        cross-check (agree out-of-sample)
        ▼                                   ▼
        └──────────►  Monte Carlo × 10,000  (official FIFA 48-team bracket)
                                            ▼
                          P(title) and P(reach each stage) per nation
```

**Two complementary models, cross-validated against each other:**
- A **calibrated logistic classifier** predicts match outcomes (used for evaluation & rigor).
- An **Elo-driven Poisson goals model** generates scorelines (used to drive the simulation,
  giving coherent goal difference for group tie-breaks).

The two are built independently yet agree on out-of-sample skill (log loss 0.8765 vs 0.8768),
strong evidence the pipeline is internally consistent.

### Key design decisions (with rationale)
| Decision | Choice | Why |
|---|---|---|
| Data | [`martj42/international_results`](https://github.com/martj42/international_results) | Free, no auth, reproducible |
| Team strength | **Composite**: Elo + de-vigged market + squad value | Fuses results, expert prior & current players ([ADR-006](docs/decisions/ADR-006-composite-strength.md)) |
| Validation | **Temporal** split (train < 2022, test ≥ 2022) | No look-ahead bias |
| Model | Logistic regression **over** gradient boosting | Parsimony: ties/beats GBDT, simpler & calibrated |
| Scoring | Log loss, Brier, **RPS** | Proper scoring rules, not just accuracy |
| Bracket | **Official** FIFA 2026 structure | Credible, matches reality |
| Tournament realism | Per-tournament rating noise (σ=125) + data-fitted host advantage | Corrects correlated-error overconfidence; σ fit to the full market title distribution |
| Consistency guard | Results sidecar (`simulation_results.meta.json`) | Dashboard verifies cached numbers match the live engine |

Full rationale lives in [`docs/decisions/`](docs/decisions/) as Architecture Decision Records.

## Results & analysis (notebooks)

| Notebook | What it shows |
|---|---|
| [`01_eda.ipynb`](notebooks/01_eda.ipynb) | Data exploration: goals, home advantage, coverage |
| [`02_features.ipynb`](notebooks/02_features.ipynb) | Elo engine + historic power evolution |
| [`03_modeling.ipynb`](notebooks/03_modeling.ipynb) | Benchmark, model selection, calibration |
| [`04_simulation.ipynb`](notebooks/04_simulation.ipynb) | Monte Carlo + title probabilities |
| [`05_robustness.ipynb`](notebooks/05_robustness.ipynb) | Sensitivity: training window & host advantage |

<p align="center">
  <img src="reports/figures/06_elo_evolution.png" width="49%" />
  <img src="reports/figures/08_calibration.png" width="42%" />
</p>

*Left: Elo evolution of historic powers (peaks match real dynasties). Right: the classifier's
probabilities are well-calibrated — when it says 70%, it happens ~70% of the time.*

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate              # Windows  (use: source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt

# Run the whole pipeline (download → clean → features → train → simulate):
python scripts/run_pipeline.py

# Or run the test suite:
pytest

# Or launch the interactive dashboard:
streamlit run streamlit_app.py
```

## 🖥️ Interactive dashboard

`streamlit run streamlit_app.py` opens a polished warm editorial dashboard (real team flags,
no clutter) with seven views:
- **Title race** — championship odds for all 48 teams, with a top-3 podium.
- **Bracket** — the fixed reference simulation: all 104 matches with exact scorelines, full
  group standings with every result, knockout tree and tournament stats. Its seed is chosen
  programmatically so its champion always matches the model's favourite.
- **Simulator** — deal brand-new tournaments from the same model, one click each, to *see*
  what a ~20% favourite really means.
- **Groups** — per-group advance probabilities.
- **Match predictor** — pick any two teams for win/draw/loss probabilities and an expected scoreline.
- **Validation** — the published evidence: out-of-time model benchmark, the reliability
  diagram, and the rating-uncertainty sweep tuned against market benchmarks.
- **How it works** — a plain-language, step-by-step methodology so the model explains itself.

### Deploying your own copy (free)

The repo is deploy-ready: the three artifacts the app needs (`matches_clean.csv` and the two
model files) are committed, so Streamlit Cloud can boot it directly.

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. **Create app** → pick this repository, branch `main`, main file `streamlit_app.py`.
3. Deploy. You get a public, shareable URL like `https://<your-app>.streamlit.app`.

## Project structure

```
.
├── data/                # raw (downloaded), processed (regenerated), external (WC2026 fixture)
├── src/
│   ├── data/            # download + cleaning
│   ├── features/        # Elo + feature engineering
│   ├── models/          # classifier + Poisson goals model
│   └── simulation/      # bracket, match engine, tournament, Monte Carlo
├── notebooks/           # narrative analysis (01 → 04)
├── tests/               # pytest suite (invariants)
├── docs/decisions/      # Architecture Decision Records
├── reports/figures/     # generated visualizations
└── scripts/run_pipeline.py
```

## Validation & rigor
- **No leakage:** every feature is pre-match; the train/test split is strictly chronological.
- **Proper scoring rules:** evaluated with log loss, multiclass Brier, and RPS — not accuracy alone.
- **Calibration:** verified with a reliability diagram.
- **Internal consistency:** the independent classifier and Poisson models agree out-of-sample.
- **Tests:** 21 passing invariants (Elo zero-sum, composite stays on scale, distribution sums, one-champion-per-tournament, …).

## Honest limitations
- Reads squad value, not the team sheet — day-of injuries, suspensions and lineups are invisible.
- Market & squad snapshots are curated (dated, reproducible) and go stale; they are trivially updatable.
- Host advantage applied only in the hosts' own group games; knockout venues are treated as neutral.
- Independent Poisson slightly under-models draws (Dixon-Coles is the planned v2 fix).
- Deep group tie-breaks resolved by lot, as in the real rules.

## Roadmap
- [x] Data pipeline, EDA, Elo, classifier, Poisson model, Monte Carlo, official bracket, tests
- [x] Interactive Streamlit dashboard (`streamlit_app.py`)
- [x] Composite strength: Elo blended with the betting market and squad value ([ADR-006](docs/decisions/ADR-006-composite-strength.md))
- [ ] *(v2)* Dixon-Coles draw correction
- [ ] *(v2)* Bookmaker-implied probabilities + Kelly staking (quant capstone)

---

## 🏆 Predictor del Mundial 2026 (Español)

Pipeline de *machine learning* **reproducible y explicable** que estima la probabilidad de
que cada selección gane el **Mundial 2026**, a partir de 150+ años de resultados y una
simulación **Monte Carlo de 10.000 torneos** sobre el cuadro oficial.

**Resultado principal:** 🇪🇸 España **15.8%** · 🇫🇷 Francia 11.3% · 🇦🇷 Argentina 10.9% ·
🏴 Inglaterra 9.2% · 🇵🇹 Portugal 6.5%. El favorito sigue al mercado de apuestas casi exacto
(España ~15.8%), calibrado contra el mercado y no "al ojo", reflejando con honestidad la alta
varianza del fútbol.

**Cómo funciona:** cada selección recibe una **fuerza compuesta** que mezcla tres señales en la
escala Elo: 150 años de resultados (**Elo**), el **mercado de apuestas** sin margen (el
pronóstico público más preciso) y el **valor de plantel** de Transfermarkt (los jugadores
actuales). Esa fuerza alimenta un **clasificador logístico calibrado** y un **modelo de goles de
Poisson** que genera marcadores → se simula el torneo 10.000 veces con el **cuadro oficial de
FIFA** → se leen las probabilidades de campeón y de llegar a cada fase. Esto enfría a equipos
sobrevalorados (Argentina baja de 14.4%) y calienta a los subvalorados.

**El punto clave:** el objetivo no es adivinar al campeón, sino demostrar un proceso de ciencia
de datos **riguroso, explicable y reproducible**, honesto sobre la incertidumbre. Las decisiones
de diseño están documentadas en [`docs/decisions/`](docs/decisions/).

**Cómo correrlo:**
```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
python scripts/run_pipeline.py
```

---

*Built as a portfolio project demonstrating applied ML, data engineering, statistics, and
reproducible research practices. Co-developed with AI pair programmers (Claude Code / Gemini CLI)
under a documented collaboration protocol — see [`PROJECT_STATUS.md`](PROJECT_STATUS.md).*
