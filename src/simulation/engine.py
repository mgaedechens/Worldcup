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


def _shootout_winner(
    team_a: str, team_b: str, ratings: dict[str, float],
    params: GoalsParams, rng: np.random.Generator,
) -> str:
    """Settle a level knockout match by a penalty shootout.

    The probability is the teams' relative regulation win-strength, so the favorite is mildly
    favored; if both are equal it is a true coin flip (as real shootouts nearly are)."""
    lam_a = params.lam(ratings[team_a] - ratings[team_b])
    lam_b = params.lam(ratings[team_b] - ratings[team_a])
    g = np.arange(11)
    joint = np.outer(poisson.pmf(g, lam_a), poisson.pmf(g, lam_b))
    p_a_win = np.tril(joint, -1).sum()
    p_b_win = np.triu(joint, 1).sum()
    prob_a = 0.5 if (p_a_win + p_b_win) == 0 else p_a_win / (p_a_win + p_b_win)
    return team_a if rng.random() < prob_a else team_b


def knockout_win_prob(
    team_a: str, team_b: str, ratings: dict[str, float], params: GoalsParams,
) -> float:
    """Probability that ``team_a`` advances from a knockout against ``team_b``.

    Closed-form (no sampling): regulation is two independent Poissons; a level game goes to a
    shootout won in proportion to each side's regulation win-strength (the same rule the sampler
    in ``_shootout_winner`` uses). So
        P(a advances) = P(a wins) + P(draw) * P(a wins) / [P(a wins) + P(b wins)].
    """
    lam_a = params.lam(ratings[team_a] - ratings[team_b])
    lam_b = params.lam(ratings[team_b] - ratings[team_a])
    g = np.arange(11)
    joint = np.outer(poisson.pmf(g, lam_a), poisson.pmf(g, lam_b))
    p_a = float(np.tril(joint, -1).sum())   # a scores more
    p_b = float(np.triu(joint, 1).sum())    # b scores more
    p_draw = float(np.trace(joint))
    shoot_a = 0.5 if (p_a + p_b) == 0 else p_a / (p_a + p_b)
    return p_a + p_draw * shoot_a


def knockout_winner(
    team_a: str, team_b: str, ratings: dict[str, float],
    params: GoalsParams, rng: np.random.Generator,
) -> str:
    """Decide a knockout match (winner only)."""
    ga, gb = simulate_scoreline(ratings[team_a], ratings[team_b], params, rng)
    if ga > gb:
        return team_a
    if gb > ga:
        return team_b
    return _shootout_winner(team_a, team_b, ratings, params, rng)


def knockout_result(
    team_a: str, team_b: str, ratings: dict[str, float],
    params: GoalsParams, rng: np.random.Generator,
) -> tuple[str, int, int, bool]:
    """Decide a knockout match and report detail: (winner, goals_a, goals_b, went_to_pens)."""
    ga, gb = simulate_scoreline(ratings[team_a], ratings[team_b], params, rng)
    if ga != gb:
        return (team_a if ga > gb else team_b, ga, gb, False)
    return (_shootout_winner(team_a, team_b, ratings, params, rng), ga, gb, True)
