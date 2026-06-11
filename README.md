# 🏆 World Cup 2026 Predictor

> A reproducible, explainable machine-learning pipeline that estimates each national
> team's probability of winning the **2026 FIFA World Cup** — built from 150+ years of
> international match results.

*[Versión en español más abajo ↓](#-predictor-del-mundial-2026-español)*

> ⚠️ **Status: v1 in progress.** This README documents the intended design; sections are
> filled in as the pipeline is built. See [`PROJECT_STATUS.md`](PROJECT_STATUS.md) for the
> live development log.

---

## Why this project

Predicting a World Cup is a genuinely hard problem: football is **low-signal, high-variance**.
The goal here is **not** to build an oracle, but to demonstrate a rigorous, end-to-end
data-science process that is **explainable** and **reproducible** — the kind of work that
holds up under scrutiny.

## Approach

A **hybrid** model that combines domain knowledge with machine learning:

1. **Elo ratings** computed from historical matches capture each team's strength over time.
2. Elo (plus recent form, match context, etc.) becomes a **feature** for a calibrated
   **gradient-boosting classifier** that predicts `P(win / draw / loss)` for any matchup.
3. A **Monte Carlo simulation** plays out the 48-team tournament thousands of times to
   estimate each team's probability of reaching each stage and winning the title.

```
results.csv (1872–2026) → cleaning → Elo + features → calibrated GBDT → Monte Carlo → P(title)
```

### Key design decisions
| Decision | Choice | Rationale |
|---|---|---|
| Model | Elo-as-feature + gradient boosting | Shows full ML pipeline *and* domain knowledge; explainable |
| Classifier | `HistGradientBoostingClassifier` (scikit-learn) | Native to sklearn; avoids dependency/wheel risk |
| Validation | **Temporal** split (past → future) | Prevents leakage; mirrors real predictive use |
| Data | [`martj42/international_results`](https://github.com/martj42/international_results) | Free, no auth, fully reproducible |

## Tech stack

Python 3.12+ · pandas · NumPy · scikit-learn · matplotlib · seaborn · Jupyter

## Project structure

```
.
├── data/
│   ├── raw/          # downloaded source CSVs (git-ignored, reproducible)
│   ├── processed/    # cleaned datasets (regenerated from raw)
│   └── external/     # hand-curated WC2026 teams & fixtures
├── src/
│   ├── data/         # download + cleaning
│   ├── features/     # Elo, form, feature engineering
│   ├── models/       # training + evaluation
│   └── simulation/   # Monte Carlo tournament
├── notebooks/        # narrative analysis (EDA → modeling → simulation)
├── reports/figures/  # generated visualizations
└── tests/
```

## Getting started

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

*(Pipeline run instructions will be added as scripts land.)*

## Roadmap

- [x] Repository scaffolding & development protocol
- [ ] Data download & cleaning
- [ ] Exploratory data analysis (EDA)
- [ ] Feature engineering (Elo, recent form)
- [ ] Model training & temporal evaluation
- [ ] Monte Carlo tournament simulation
- [ ] Visualizations & final documentation
- [ ] *(v2)* Interactive Streamlit dashboard

---

## 🏆 Predictor del Mundial 2026 (Español)

Pipeline de *machine learning* **reproducible y explicable** que estima la probabilidad de
que cada selección gane el **Mundial 2026**, a partir de más de 150 años de resultados
internacionales.

### Por qué este proyecto
Predecir un Mundial es difícil de verdad: el fútbol es de **baja señal y alta varianza**.
El objetivo **no** es construir un oráculo, sino demostrar un proceso de ciencia de datos
**riguroso, explicable y reproducible** de principio a fin.

### Enfoque
Un modelo **híbrido** que combina conocimiento de dominio con *machine learning*:
1. **Rating Elo** calculado desde los partidos históricos para medir la fuerza de cada selección.
2. El Elo (más forma reciente y contexto) se usa como **feature** de un **clasificador de
   gradient boosting calibrado** que predice `P(victoria / empate / derrota)`.
3. Una **simulación Monte Carlo** juega el torneo de 48 equipos miles de veces para estimar
   la probabilidad de cada selección de avanzar y de ser campeón.

### Cómo empezar
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

El registro de desarrollo vive en [`PROJECT_STATUS.md`](PROJECT_STATUS.md).

---

*Built as a portfolio project demonstrating applied ML, data engineering, and reproducible
research practices.*
