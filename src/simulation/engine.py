"""Match simulation engine: turn two teams' Elo into a scoreline.

Speed matters (we play ~1M matches across a Monte Carlo run), so instead of calling the
sklearn pipeline per match we extract the Poisson GLM into closed-form coefficients in the
ORIGINAL feature space and evaluate lambda with plain numpy:

    lambda = exp(b0_eff + b_elo_eff * elo_adv + b_home_eff * is_home)

World Cup matches are modeled on neutral ground (is_home = 0 for both sides). Giving hosts
a home edge is a documented future refinement.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import poisson

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"


@dataclass(frozen=True)
class GoalsParams:
    """Closed-form Poisson coefficients in the original (unscaled) feature space."""

    b0: float          # effective intercept
    b_elo: float       # per Elo-point effect on log-goals
    b_home: float      # home-side effect on log-goals

    def lam(self, elo_adv: float, is_home: int = 0) -> float:
        return float(np.exp(self.b0 + self.b_elo * elo_adv + self.b_home * is_home))


def load_goals_params(path: Path | None = None) -> GoalsParams:
    """Load the saved Poisson pipeline and collapse scaler+GLM into raw-space coefficients.

    The pipeline is StandardScaler -> PoissonRegressor, so
        eta = intercept + sum_i coef_i * (x_i - mean_i) / scale_i
            = [intercept - sum_i coef_i*mean_i/scale_i]  +  sum_i (coef_i/scale_i) * x_i
    """
    path = path or (MODELS_DIR / "goals_model.joblib")
    pipe = joblib.load(path)
    scaler = pipe.named_steps["scaler"]
    glm = pipe.named_steps["poisson"]

    mean = scaler.mean_
    scale = scaler.scale_
    coef = glm.coef_           # order: [elo_adv, is_home]
    intercept = glm.intercept_

    b0_eff = float(intercept - np.sum(coef * mean / scale))
    b_elo_eff = float(coef[0] / scale[0])
    b_home_eff = float(coef[1] / scale[1])
    return GoalsParams(b0=b0_eff, b_elo=b_elo_eff, b_home=b_home_eff)


def simulate_scoreline(
    elo_a: float, elo_b: float, params: GoalsParams, rng: np.random.Generator,
    *, a_is_home: int = 0, b_is_home: int = 0,
) -> tuple[int, int]:
    """Sample (goals_a, goals_b) from two independent Poissons driven by the Elo gap."""
    lam_a = params.lam(elo_a - elo_b, a_is_home)
    lam_b = params.lam(elo_b - elo_a, b_is_home)
    return int(rng.poisson(lam_a)), int(rng.poisson(lam_b))


def knockout_winner(
    team_a: str, team_b: str, ratings: dict[str, float],
    params: GoalsParams, rng: np.random.Generator,
) -> str:
    """Decide a knockout match. A regulation draw is settled by a penalty shootout whose
    probability is the teams' relative regulation win-strength (slight edge to the favorite,
    but close to a coin flip — as real shootouts are)."""
    ga, gb = simulate_scoreline(ratings[team_a], ratings[team_b], params, rng)
    if ga > gb:
        return team_a
    if gb > ga:
        return team_b
    # Shootout: weight by each side's Poisson win probability so the favorite is mildly
    # favored; if both are equal it is a true coin flip.
    lam_a = params.lam(ratings[team_a] - ratings[team_b])
    lam_b = params.lam(ratings[team_b] - ratings[team_a])
    # Quick implied win probabilities via small Poisson grids.
    g = np.arange(11)
    joint = np.outer(poisson.pmf(g, lam_a), poisson.pmf(g, lam_b))
    p_a_win = np.tril(joint, -1).sum()
    p_b_win = np.triu(joint, 1).sum()
    prob_a = 0.5 if (p_a_win + p_b_win) == 0 else p_a_win / (p_a_win + p_b_win)
    return team_a if rng.random() < prob_a else team_b
