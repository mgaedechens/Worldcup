"""Train, benchmark, and select the match-outcome classifier.

Modeling philosophy (quant-flavored):

* **No look-ahead bias.** We split strictly by time: fit on the past, evaluate on the
  future. A random train/test split would leak future information and inflate scores.
* **Always benchmark.** A model is only "good" relative to a naive baseline. We compare
  against "always predict the base rates", an Elo-only logistic regression, and a
  gradient-boosting model.
* **Score probabilities, not labels.** Accuracy hides whether the *probabilities* are
  any good. We use proper scoring rules — log loss, multiclass Brier, and the
  Ranked Probability Score (RPS), the football-forecasting standard (an ordinal cousin
  of the CRPS used in finance/meteorology).
* **Parsimony (Occam's razor).** When models tie, prefer the simpler, more interpretable
  one. Here a multinomial Logistic Regression matches/beats gradient boosting on every
  proper score, so it is the selected model. Bonus: it is naturally well-calibrated and
  its coefficients are directly explainable.

Run:
    python -m src.models.train
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

FEATURES = ["elo_diff", "form_diff", "rest_diff", "is_neutral"]
# Class order is the natural ORDINAL scale (away win < draw < home win), which also
# happens to be alphabetical, so sklearn's sorted classes match it.
CLASSES = ["A", "D", "H"]

TEST_START_YEAR = 2022   # out-of-time test set (includes the 2022 World Cup)


def build_model() -> Pipeline:
    """The selected model: standardized features -> multinomial logistic regression.

    Scaling matters for logistic regression (gradient-based, regularized); it is a
    no-op for tree models but harmless here.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, C=1.0)),
    ])


# --------------------------------------------------------------------------- #
# Proper scoring rules
# --------------------------------------------------------------------------- #
def _onehot(y_idx: np.ndarray, n_classes: int = 3) -> np.ndarray:
    return np.eye(n_classes)[y_idx]


def multiclass_brier(y_idx: np.ndarray, probs: np.ndarray) -> float:
    """Mean squared error between predicted prob vector and the one-hot truth."""
    return float(np.mean(np.sum((probs - _onehot(y_idx)) ** 2, axis=1)))


def ranked_probability_score(y_idx: np.ndarray, probs: np.ndarray) -> float:
    """RPS: ordinal-aware proper scoring rule (lower is better).

    Compares the *cumulative* predicted and observed distributions, so predicting a
    draw when the truth is an away win is penalized less than predicting a home win.
    """
    cum_pred = np.cumsum(probs, axis=1)
    cum_obs = np.cumsum(_onehot(y_idx), axis=1)
    r = probs.shape[1]
    return float(np.mean(np.sum((cum_pred - cum_obs) ** 2, axis=1) / (r - 1)))


def evaluate(name: str, y_idx: np.ndarray, probs: np.ndarray) -> dict:
    """Bundle the metrics for one model on one dataset."""
    preds = probs.argmax(axis=1)
    return {
        "model": name,
        "accuracy": accuracy_score(y_idx, preds),
        "log_loss": log_loss(y_idx, probs, labels=[0, 1, 2]),
        "brier": multiclass_brier(y_idx, probs),
        "rps": ranked_probability_score(y_idx, probs),
    }


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_features() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DIR / "features.csv", parse_dates=["date"])
    df["year"] = df["date"].dt.year
    df["target_idx"] = df["target"].map({c: i for i, c in enumerate(CLASSES)})
    return df


def temporal_split(df: pd.DataFrame):
    """Two chronological folds: train < 2022, test >= 2022 (out-of-time)."""
    train = df[df["year"] < TEST_START_YEAR]
    test = df[df["year"] >= TEST_START_YEAR]
    return train, test


# --------------------------------------------------------------------------- #
# Benchmarking & selection
# --------------------------------------------------------------------------- #
def benchmark(df: pd.DataFrame) -> pd.DataFrame:
    """Train candidates on the past, score them on the untouched future test set.

    This is what justifies the model choice; it is *evaluation*, not the final fit.
    """
    train, test = temporal_split(df)
    X_tr, y_tr = train[FEATURES], train["target_idx"].to_numpy()
    X_te, y_te = test[FEATURES], test["target_idx"].to_numpy()

    # (0) Base rates: predict the training class frequencies for every match.
    base_rates = np.bincount(y_tr, minlength=3) / len(y_tr)
    probs_base = np.tile(base_rates, (len(y_te), 1))

    # (1) Elo-only logistic regression (domain baseline).
    elo_lr = build_model().fit(train[["elo_diff"]], y_tr)
    probs_elo = elo_lr.predict_proba(test[["elo_diff"]])

    # (2) Gradient-boosted trees (the "complex" challenger).
    gbdt = HistGradientBoostingClassifier(
        learning_rate=0.05, max_depth=3, max_iter=400,
        l2_regularization=1.0, random_state=42,
    ).fit(X_tr, y_tr)
    probs_gbdt = gbdt.predict_proba(X_te)

    # (3) Selected model: multinomial logistic on all features.
    model = build_model().fit(X_tr, y_tr)
    probs_model = model.predict_proba(X_te)

    return pd.DataFrame([
        evaluate("Baseline (base rates)", y_te, probs_base),
        evaluate("Elo-only logistic", y_te, probs_elo),
        evaluate("GBDT (all features)", y_te, probs_gbdt),
        evaluate("Logistic (all features) *selected*", y_te, probs_model),
    ])


def fit_final_model(df: pd.DataFrame) -> Pipeline:
    """Refit the selected model on ALL available data for deployment.

    Standard practice: once a model is *chosen* via the temporal holdout, retrain it on
    every match so the World Cup predictions use the maximum amount of information.
    """
    X, y = df[FEATURES], df["target_idx"].to_numpy()
    return build_model().fit(X, y)


def run() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_features()

    results = benchmark(df)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print("Out-of-time test (matches from 2022 onward):\n")
    print(results.to_string(index=False))
    print("\nLower is better for log_loss / brier / rps. Higher for accuracy.")

    # Persist the benchmark as evidence (consumed by the dashboard's Validation tab).
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(reports_dir / "model_benchmark.csv", index=False)

    model = fit_final_model(df)
    joblib.dump({"model": model, "features": FEATURES, "classes": CLASSES},
                MODELS_DIR / "wc_model.joblib")
    print(f"\n[ok] selected model refit on all data -> {MODELS_DIR / 'wc_model.joblib'}")


if __name__ == "__main__":
    run()
