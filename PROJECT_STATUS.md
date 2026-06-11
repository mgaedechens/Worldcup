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
