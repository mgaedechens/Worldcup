# Estado del proyecto — World Cup 2026 Predictor

> **Archivo de continuidad entre agentes (Claude Code ↔ Gemini CLI).**
> Regla del protocolo: **ningún agente asume que recuerda el contexto.** Antes de
> tocar cualquier archivo, lee este documento, el `README.md` y revisa la estructura.
> Al terminar una tarea, **agrega una nueva entrada arriba** (orden cronológico inverso).

---

## Resumen rápido (leer primero)

- **Objetivo:** modelo predictivo explicable para estimar el ganador del Mundial 2026.
- **Enfoque elegido:** Híbrido → Elo como *feature* + clasificador ML calibrado
  (`HistGradientBoostingClassifier`) → simulación Monte Carlo del torneo.
- **Idioma:** código y docstrings en **inglés**; `README.md` y `PROJECT_STATUS.md` bilingües (es/en).
- **Fuente de datos:** repo GitHub `martj42/international_results` (CSV libre, sin login).
- **Entorno:** Windows 11, Python 3.14.3, `venv` nativo. Sin conda/uv.
- **Entrega v1:** notebooks narrativos + módulos `src/` + figuras. El repo es la cara
  visible (irá como link en el CV del usuario). v2 candidata: dashboard Streamlit desplegado.

### Decisiones tomadas y su porqué (no re-litigar sin motivo)
| Decisión | Elección | Por qué |
|---|---|---|
| Modelo | Elo-feature + GBDT calibrado | Muestra pipeline ML completo + dominio; explicable |
| Clasificador | `HistGradientBoostingClassifier` (sklearn) | Nativo de sklearn; evita riesgo de wheels de XGBoost/LightGBM en Py3.14 |
| Datos | `martj42/international_results` | Gratis, sin auth, reproducible al clonar |
| Validación | **Temporal** (train pasado → test futuro) | Evita data leakage; refleja el uso real predictivo |
| Idioma código | Inglés | Alcance internacional del portfolio |

### Riesgos / advertencias activas
- ⚠️ **Python 3.14 es muy nuevo:** algunas librerías podrían no tener wheels. Si la
  instalación de `scikit-learn`/`pandas` falla, considerar bajar a Python 3.12/3.13.
- ⚠️ El fútbol es un fenómeno de **baja señal / alta varianza**: el modelo NO será
  un oráculo. El valor del proyecto está en el *proceso* riguroso, no en acertar el campeón.
- ⚠️ Hay que evitar **leakage temporal** en features (forma, Elo) y en el split.

---

## Últimos cambios

### Fecha: 2026-06-12 (entrada 7 — dashboard final, CIERRE DE FASE) ✅
**Agente:** Claude Code

**Estado: proyecto COMPLETO y pulido.** Dashboard profesional de 6 pestañas, 15 tests, ruff
limpio, todo verificado con AppTest y pusheado.

**Cambios de esta entrada:**
- `src/simulation/scenario.py`: ahora registra los **72 marcadores de fase de grupos**
  (`GroupMatch`) además de los 31 de eliminatorias.
- `streamlit_app.py` rediseño final:
  - Pestaña **Bracket** = UNA simulación de referencia FIJA (seed 2, campeón = España,
    coincide con el favorito modal). Reproducible.
  - Pestaña **Simulator** (nueva, separada) = botón para generar torneos nuevos (contador
    de simulación visible).
  - Tablas de grupo ahora incluyen **todos los resultados con marcador** (filas `gm`),
    ganador resaltado, dentro de tarjetas `gblock`.
  - **How it works reescrito**: 6 pasos en tono humano y profundo (datos → Elo → clasificador
    → Poisson → Monte Carlo → validación), sin em-dashes en todo el copy visible.
  - Facts actualizados (15/15 tests).
- README actualizado (6 pestañas).

**Verificación:** AppTest 6 tabs sin excepción (incl. click del botón Simulator), 15/15
pytest, ruff limpio, 12 bloques de grupo × 72 marcadores confirmados en HTML.

**Próximos pasos sugeridos (cuando se retome):**
1. **Deploy a Streamlit Community Cloud** (el repo está listo; falta resolver que
   `data/processed` y `models/` se regeneren en el arranque o se incluyan: opción simple =
   script de setup que corre `run_pipeline.py` si faltan artefactos, o commitear los 3
   artefactos pequeños necesarios).
2. Dixon-Coles, cuotas de mercado + Kelly, sensibilidad adicional (ver entradas previas).

**Notas para el siguiente agente:**
- El dashboard usa SOLO builders HTML puros (testeables sin servidor) + `AppTest` para
  integración. Mantener ese patrón.
- Tras editar módulos en `src/`, REINICIAR streamlit (Ctrl+C y relanzar), no basta Rerun.

---

### Fecha: 2026-06-11 (entrada 6 — simulación + v1 COMPLETA) ✅
**Agente:** Claude Code (Opus 4.8) — sesión autónoma nocturna

**Estado: v1 funcional completa.** `python scripts/run_pipeline.py` corre todo de punta a punta.

**Archivos creados:**
- `src/simulation/{engine,bracket,tournament,montecarlo}.py` — simulación Monte Carlo completa.
- `src/models/goals.py` — modelo Poisson de goles (ya existía de entrada 5b, integrado).
- `notebooks/04_simulation.ipynb` — gráficos finales + verificación cruzada.
- `tests/` (3 archivos, 14 tests, todos pasan) + `pytest.ini`.
- `docs/decisions/` (protocolo + ADR 001–004), `OVERNIGHT_LOG.md`.
- `scripts/run_pipeline.py`, `LICENSE` (MIT), `README.md` reescrito.
- `reports/simulation_results.csv` + figuras 05–11.

**Bracket oficial:** se obtuvieron los 12 grupos oficiales y el cruce 32avos→final (matches
73–104). Mapeo a etiquetas oficiales por equipo "ancla" único (verificado). Asignación de
mejores terceros vía matching con restricciones (`scipy.linear_sum_assignment`). ⚠️ RESUELTO
el pendiente de "etiquetas de grupo arbitrarias".

**Resultado (10.000 sims, seed 42):** España 27.7% · Argentina 19.7% · Francia 10.3% ·
Inglaterra 7.3% · Brasil 5.0%. Probabilidades de campeón suman 1.0; reach R32 suma 32. ✓

**Verificación cruzada:** clasificador logístico vs Poisson-implícito en test ≥2022 →
log loss 0.8765 vs 0.8768, RPS 0.1718 vs 0.1720. Coinciden → consistencia interna. ✓

**Bugs encontrados y corregidos autónomamente:**
- Calibración isotónica empeoraba log loss → se eligió logística simple (ADR-003).
- PoissonRegressor con overflow numérico → estandarizar features.
- `np.math.factorial` eliminado en numpy 2 → `scipy.stats.poisson`.
- Import path de `scripts/run_pipeline.py`.

**Decisiones de simulación (ADR-004):** Mundial modelado en cancha neutral (sin ventaja de
local para anfitriones) — simplificación documentada. Poisson independiente sub-modela
empates ligeramente (Dixon-Coles = v2). Desempates profundos de grupo por sorteo (random).

**Próximos pasos sugeridos (v2):**
1. Corrección Dixon-Coles + ventaja de anfitrión (USA/CAN/MEX en sus sedes).
2. Comparación vs cuotas de mercado + Kelly (capstone quant; requiere datos de cuotas).
3. Dashboard Streamlit (link clickeable para el CV).
4. Posible: análisis de sensibilidad a la ventana de entrenamiento (2002 vs otras).

**Notas para el siguiente agente:**
- Todo regenerable: `python scripts/run_pipeline.py`. Tests: `pytest`.
- El repo está limpio, commiteado y pusheado a GitHub (`mgaedechens/Worldcup`).

---

### Fecha: 2026-06-11 (entrada 5 — modelado)
**Agente:** Claude Code (Opus 4.8)

**Archivos creados:**
- `src/models/train.py` — benchmark temporal + selección + modelo final.
- `notebooks/03_modeling.ipynb` — historia completa con gráficos (ejecutado).
- `reports/figures/07_model_benchmark.png`, `08_calibration.png`, `09_coefficients.png`.
- `models/wc_model.joblib` — modelo final (gitignored, regenerable con `python -m src.models.train`).

**Resultados (test out-of-time ≥2022, incl. Mundial 2022):**
| Modelo | accuracy | log_loss | brier | rps |
|---|---|---|---|---|
| Baseline (base rates) | 0.478 | 1.050 | 0.633 | 0.228 |
| Elo-only logistic | 0.596 | 0.879 | 0.517 | 0.173 |
| GBDT (all features) | 0.598 | 0.877 | 0.516 | 0.172 |
| **Logistic (all) ★ELEGIDO** | **0.599** | **0.8765** | **0.515** | **0.1718** |

**Decisión clave (parsimonia / Occam):** el GBDT NO supera a la regresión logística →
se elige la **logística** (más simple, mejor en todas las métricas propias, interpretable,
bien calibrada). El `elo_diff` domina como predictor. Calibración validada (reliability
diagram pega en la diagonal). Modelo final reentrenado con TODOS los datos para deployment.

**Métodos quant aplicados:** split temporal (no look-ahead), benchmark vs naïve,
proper scoring rules (log loss, Brier, **RPS**), reliability diagram (≈ backtesting VaR),
parsimonia. Elo=EWMA. PENDIENTE v2: comparación vs cuotas + Kelly.

**Próximos pasos sugeridos (Day 3 — la simulación, el payoff):**
1. `src/simulation/` Monte Carlo: cargar `wc_model.joblib` + Elo actual de las 48 selecciones
   + `wc2026_fixtures.csv` + grupos. Jugar el torneo N veces → P(título), P(llegar a cada fase).
   - `predict.py`: dado dos equipos (sus Elo, neutral) → P(H/D/A). Mapear a P(local/empate/visita)
     en cancha NEUTRAL (is_neutral=1, salvo sede USA/CAN/MEX como matiz opcional).
   - Fase de grupos: 3 pts victoria, 1 empate; resolver desempates (pts, dif goles...). OJO:
     necesitamos GOLES simulados, no solo W/D/L → opción: muestrear marcador con Poisson
     calibrado por Elo, O resolver grupos por puntos con tie-break aleatorio simple (v1).
2. `notebooks/04_simulation.ipynb`: tabla y gráfico de probabilidades de campeón (el entregable).
3. README final con resultados + figuras.

**Advertencias para el siguiente agente:**
- ⚠️ **BRACKET DE OCTAVOS:** las etiquetas A–L siguen siendo arbitrarias. El formato 48 es
  complejo (clasifican 1º, 2º + 8 mejores 3º; la estructura del cruce está predefinida por FIFA).
  DECISIÓN PENDIENTE: (a) mapear a letras oficiales y usar el bracket oficial, o (b) v1
  simplificada con un bracket plausible (documentando que es aproximado). Discutir con el usuario.
- ⚠️ El modelo predice W/D/L, no goles. Para desempates de grupo necesitamos goles →
  decidir método (Poisson por Elo recomendado) antes de codificar.

---

### Fecha: 2026-06-11 (entrada 4 — Elo + features)
**Agente:** Claude Code (Opus 4.8)

**Archivos creados:**
- `src/features/elo.py` — motor Elo (World Football Elo: ventaja local, MOV, peso de torneo).
  Framing quant: Elo = estimador EWMA. Devuelve Elo PRE-partido (sin leakage).
- `src/features/build_features.py` — tabla de features a nivel partido.
- `notebooks/02_features.ipynb` — validación Elo + gráficos (ejecutado).
- `reports/figures/05_top20_elo.png`, `06_elo_evolution.png`.
- `data/processed/features.csv` (23.271 filas, 2002+; gitignored, regenerable).

**Validaciones (pasadas):**
- Top Elo actual = potencias reales (España, Argentina, Francia, Inglaterra, Brasil...). ✓
- Evolución histórica coincide con dinastías (España 2008-12, Francia 2018-22). ✓
- Sanity features: mean `elo_diff` por outcome → H:+141, D:−10, A:−157 (monótono). ✓
- Class balance target: H 48.0% / A 28.7% / D 23.3%.

**Features incluidas:** `elo_diff` (estrella), `form_diff` (forma PPG últimos 10),
`rest_diff` (días de descanso), `is_neutral`. Target {H,D,A}.

**Decisión quant integrada (a pedido del usuario — quiere aprender métodos CFA/quant):**
- Elo = EWMA (memoria larga) + form = señal momentum corta (como combinar MA lenta/rápida).
- PLAN de modelado quant: walk-forward backtesting (evitar look-ahead bias), proper scoring
  rules (log loss, Brier), reliability/calibration (como backtesting de VaR), benchmark vs
  baseline naïve. v2 opcional: comparación vs cuotas de mercado + Kelly (requiere datos de
  cuotas, sin fuente gratuita limpia — pendiente).

**Próximos pasos sugeridos:**
1. `notebooks/03_modeling.ipynb` + `src/models/train.py`: HistGradientBoostingClassifier
   con CALIBRACIÓN. Split TEMPORAL (train ≤2021, test 2022+ incl. Mundial 2022). Baselines:
   Elo-logístico y "siempre local". Métricas: log loss, Brier multiclase, accuracy, calibración.
2. Guardar modelo (`models/*.joblib`) + `src/models/predict.py` (probabilidades H/D/A).
3. Day 3: `src/simulation/` Monte Carlo del torneo.

**Advertencias para el siguiente agente:**
- ⚠️ NO usar `train_test_split` aleatorio — DEBE ser temporal (cronológico) o hay leakage.
- ⚠️ Calibrar probabilidades (`CalibratedClassifierCV` o `method='sigmoid/isotonic'`),
  porque el Monte Carlo depende de probabilidades bien calibradas, no solo de accuracy.
- ⚠️ (Sigue) Etiquetas de grupo A–L arbitrarias → bracket de octavos.

---

### Fecha: 2026-06-11 (entrada 3 — EDA)
**Agente:** Claude Code (Opus 4.8)

**Archivos modificados / creados:**
- `notebooks/01_eda.ipynb` — EDA ejecutado con outputs incrustados (4 secciones).
- `reports/figures/01..04_*.png` — figuras generadas.
- Kernel del venv registrado (`.venv/share/jupyter/kernels/python3`, vía `--sys-prefix`).

**Hallazgos clave (con evidencia):**
- **Outcomes globales:** Home 49.0% / Draw 22.7% / Away 28.3%. Avg goles/partido ≈ 2.94.
- **Ventaja local REAL:** con local → victoria local 50.7% (visita 26.4%); en neutral →
  44.2% (visita 33.4%). El flag `neutral` es informativo → será feature.
- **Densidad temporal:** 65% de los partidos son ≥1990, 47% ≥2002, 32% ≥2010. Pre-1990 disperso.
- **Cobertura de equipos WC2026:** algunos clasificados tienen poca historia reciente (ratings
  más ruidosos) — caveat anotado.

**Decisiones tomadas:**
- ✅ **Ventana de entrenamiento = partidos desde 2002.** Justificación: densidad alta y era
  moderna relevante, manteniendo ~20+ años de muestra. IMPORTANTE: el **Elo se calienta con
  TODA la historia** (no cold-start); solo el *dataset de entrenamiento del clasificador* se
  filtra a ≥2002.
- ✅ Narrativa de notebooks en **inglés** (consistencia portfolio); explicaciones al usuario
  en español por chat.

**Próximos pasos sugeridos:**
1. `src/features/elo.py`: Elo cronológico sobre TODA la historia. K-factor base ~32,
   ponderado por importancia de torneo (Mundial > clasificatorio > amistoso) y margen de
   goles. Ventaja local en la expectativa (~+65-100 pts Elo al local no-neutral).
2. `src/features/build_features.py`: features a nivel partido (diff Elo pre-partido, forma
   reciente últimos N, días de descanso, flag neutral) + target {H,D,A}. Filtrar a ≥2002.
3. `notebooks/02_features.ipynb`: validar Elo (top teams históricos, evolución) y features.

**Problemas pendientes / advertencias:**
- ⚠️ (Sigue) Etiquetas de grupo A–L arbitrarias, no oficiales FIFA → afecta bracket de octavos.
- ⚠️ CUIDADO con leakage temporal: el Elo de un partido debe ser el **PRE-partido** (estado
  antes de jugarse), nunca incluir el resultado del propio partido.

**Notas para el siguiente agente:**
- Regenerar datos: `python -m src.data.download && python -m src.data.clean`.
- Re-ejecutar EDA: `.venv/Scripts/python.exe -m jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb`.

---

### Fecha: 2026-06-11 (entrada 2 — datos)
**Agente:** Claude Code (Opus 4.8)

**Archivos modificados / creados:**
- `src/data/download.py` — descarga `results.csv` + `shootouts.csv` (idempotente).
- `src/data/clean.py` — limpieza, normalización de nombres, split jugados/futuros,
  reconstrucción de grupos.
- Generados: `data/processed/matches_clean.csv` (49.405 partidos jugados),
  `data/external/wc2026_fixtures.csv` (72), `data/external/wc2026_groups.csv` (12×4=48).
- `requirements.lock.txt` — versiones exactas (Py3.14: pandas 3.0.3, numpy 2.4.6, sklearn 1.9.0).

**Cambios realizados:**
- Entorno `venv` creado e instalado; **wheels de Py3.14 confirmadas** (riesgo cerrado).
- Descarga y limpieza de datos ejecutadas y validadas.
- **Hallazgo clave:** el dataset YA contiene el fixture oficial WC2026 (72 partidos con
  marcador NaN). De ahí extrajimos los 48 equipos y los 12 grupos automáticamente.

**Motivo:**
- Separar descarga/limpieza en módulos reutilizables; tener un dataset de entrenamiento
  limpio y la entrada de simulación listos antes de feature engineering.

**Próximos pasos sugeridos:**
1. `notebooks/01_eda.ipynb`: EDA sobre `matches_clean.csv` (goles, ventaja local,
   cobertura temporal, decidir ventana de entrenamiento moderna p.ej. ≥1990/2002).
2. `src/features/elo.py`: implementar Elo con K-factor ponderado por importancia de
   torneo y diferencia de goles; backfill cronológico sobre `matches_clean.csv`.
3. `src/features/build_features.py`: a nivel partido → diff de Elo, forma reciente
   (últimos N), reposo/contexto, flag neutral. Encodear outcome {H,D,A}.

**Problemas pendientes / advertencias:**
- ⚠️ **Etiquetas de grupo A–L son arbitrarias** (orden alfabético), NO las oficiales FIFA.
  La composición es correcta. Para el bracket de octavos (Día 3) hay que mapear a las
  letras oficiales o definir la estructura del bracket de 48 equipos (32avos: 12 primeros
  + 12 segundos + 8 mejores terceros). Decisión pendiente.
- ⚠️ `matches_clean.csv` incluye historia muy antigua y posibles "selecciones" no-FIFA
  (regiones). Filtrar ventana temporal y/o a miembros FIFA en la etapa de features.
- ⚠️ Normalización de nombres es mínima/debatible (ver `TEAM_NAME_MAP` en `clean.py`).

**Notas para el siguiente agente:**
- Ejecuta `python -m src.data.download && python -m src.data.clean` para regenerar datos
  (no están commiteados los `data/raw` ni `processed`; sí los `external`... revisar .gitignore).
- Usa el intérprete del venv: `.venv/Scripts/python.exe`.

---

### Fecha: 2026-06-11 (entrada 1 — scaffolding)
**Agente:** Claude Code (Opus 4.8)

**Archivos modificados / creados:**
- Estructura de carpetas: `data/{raw,processed,external}`, `src/{data,features,models,simulation}`,
  `notebooks/`, `reports/figures/`, `tests/`, `config/`.
- `.gitignore`, `requirements.txt`, `README.md`, `PROJECT_STATUS.md` (este archivo).

**Cambios realizados:**
- Scaffolding inicial del repositorio (sesión cero — el directorio estaba vacío).
- Definición de arquitectura, stack y plan de 3 días.

**Motivo:**
- Establecer la base reproducible y el archivo de continuidad antes de escribir código,
  según el protocolo de colaboración entre agentes.

**Próximos pasos sugeridos:**
1. Inicializar git (`git init`) y crear el `venv`; instalar `requirements.txt`
   (de-riesgar Python 3.14 cuanto antes).
2. Escribir `src/data/download.py`: descargar `results.csv` (y opcionalmente
   `shootouts.csv`) del repo `martj42/international_results` a `data/raw/`.
3. Escribir `src/data/clean.py`: tipado de fechas, normalización de nombres de
   selecciones, manejo de torneos/amistosos, guardar en `data/processed/`.
4. Notebook `01_eda.ipynb`: exploración (distribución de goles, ventaja local,
   cobertura temporal por selección).

**Problemas pendientes:**
- Confirmar que las wheels de sklearn/pandas existen para Python 3.14 (pendiente de probar).
- Definir la lista oficial de 48 selecciones del Mundial 2026 y el fixture/formato
  (48 equipos, 12 grupos de 4 — formato nuevo). Irá en `data/external/`.

**Notas para el siguiente agente:**
- Aún no hay código ejecutable ni datos descargados. Empieza por los próximos pasos 1–2.
- Mantén los nombres de funciones/variables en inglés. Documenta el *porqué* en commits.
