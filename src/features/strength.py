"""Composite team-strength rating: blend Elo with the betting market and squad value.

Why this exists
---------------
The original engine drove every simulated match from a single number — each team's Elo,
learned purely from past results. Elo is a strong, leakage-free backbone, but on its own it
lags two things that professionals price in:

* **the betting market** — de-vigged outright-winner odds are the single most accurate public
  forecast of football outcomes (they fold in injuries, form, squad news and sharp money);
* **squad value** — Transfermarkt aggregate value is a "current players" signal that match
  results can lag (a young core on the rise, or an ageing side coasting on reputation).

We fuse the three into ONE rating, expressed on the Elo scale, and feed that to the existing
Monte Carlo. Because the rating drives every match, the whole bracket becomes consistent with
the market instead of Elo-only — which is what makes the simulated knockouts believable.

Method (transparent, no black box)
----------------------------------
1. Each auxiliary signal is reduced to a *z-score* over the teams it covers, then mapped onto
   the Elo scale using Elo's own mean/standard deviation **over that same set of teams**. So a
   signal that puts a team 1 sd above its peers places that team ~1 Elo-sd above its peers.
   - squad value: log-transformed first (heavy-tailed), covers all 48 teams.
   - market: from the de-vigged title probability, log-transformed (title odds are roughly
     log-linear in strength), covers only the ~13 teams the books price individually.
2. The composite is a weighted average of whichever signals are present for a team, with the
   weights renormalised when a signal is missing (most minnows have no market quote, so they
   fall back to Elo + squad value automatically).

Weights are a documented modelling choice (the market gets the most weight because it is the
most accurate; Elo is the dynamic backbone; squad value is a smaller nudge). They are exposed
as ``StrengthWeights`` so the dashboard and sensitivity analyses can vary them.

Run:
    python -m src.features.strength
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

MARKET_ODDS_CSV = EXTERNAL_DIR / "market_odds_2026.csv"
SQUAD_VALUE_CSV = EXTERNAL_DIR / "squad_value_2026.csv"

# Share of true title probability assumed to be held by the teams the books price
# individually (the long tail of unpriced minnows holds the rest). Used to de-vig.
MARKET_TITLE_MASS = 0.90


@dataclass(frozen=True)
class StrengthWeights:
    """Relative weight of each signal in the composite (renormalised per team)."""

    elo: float = 0.45
    market: float = 0.35
    squad: float = 0.20


def market_implied_probs(
    odds: dict[str, float], *, title_mass: float = MARKET_TITLE_MASS
) -> dict[str, float]:
    """Turn decimal outright odds into de-vigged title probabilities.

    Raw implied probability is ``1 / decimal_odds``; summed across the quoted teams it exceeds
    the true mass by the bookmaker's overround (vig) *plus* the share held by unpriced teams.
    We remove both by normalising the quoted teams to sum to ``title_mass``.
    """
    inv = {t: 1.0 / o for t, o in odds.items()}
    total = sum(inv.values())
    return {t: title_mass * v / total for t, v in inv.items()}


def _to_elo_scale(
    values: dict[str, float], elo: dict[str, float]
) -> dict[str, float]:
    """Map a raw signal onto the Elo scale via z-score, anchored on the covered teams' Elo.

    Standardise ``values`` over the teams it covers, then rescale to the mean/std of those same
    teams' Elo. This keeps the signal on Elo's absolute scale (so it can be averaged with Elo)
    while expressing it as a peer-relative adjustment.
    """
    teams = [t for t in values if t in elo]
    if len(teams) < 2:
        return {}
    v = np.array([values[t] for t in teams], dtype=float)
    e = np.array([elo[t] for t in teams], dtype=float)
    v_std = v.std()
    if v_std == 0:
        return {t: float(e.mean()) for t in teams}
    z = (v - v.mean()) / v_std
    scaled = e.mean() + e.std() * z
    return {t: float(s) for t, s in zip(teams, scaled)}


def load_market_ratings(elo: dict[str, float], path: Path | None = None) -> dict[str, float]:
    """Elo-scaled rating implied by the betting market (only for priced teams)."""
    odds_df = pd.read_csv(path or MARKET_ODDS_CSV, comment="#")
    odds = dict(zip(odds_df["team"], odds_df["decimal_odds"]))
    probs = market_implied_probs(odds)
    return _to_elo_scale({t: np.log(p) for t, p in probs.items()}, elo)


def load_squad_ratings(elo: dict[str, float], path: Path | None = None) -> dict[str, float]:
    """Elo-scaled rating implied by squad market value (all 48 teams)."""
    sv_df = pd.read_csv(path or SQUAD_VALUE_CSV, comment="#")
    values = dict(zip(sv_df["team"], sv_df["value_eur_m"]))
    return _to_elo_scale({t: np.log(v) for t, v in values.items()}, elo)


def compose_strength(
    elo: dict[str, float],
    *,
    weights: StrengthWeights | None = None,
    market: dict[str, float] | None = None,
    squad: dict[str, float] | None = None,
) -> dict[str, float]:
    """Blend Elo, market and squad ratings into one composite per team (Elo scale).

    Weights are renormalised over whichever signals are present for each team, so a team with
    no market quote is rated from Elo + squad value alone.
    """
    weights = weights or StrengthWeights()
    market = load_market_ratings(elo) if market is None else market
    squad = load_squad_ratings(elo) if squad is None else squad

    out: dict[str, float] = {}
    for team, r_elo in elo.items():
        parts = [(weights.elo, r_elo)]
        if team in market:
            parts.append((weights.market, market[team]))
        if team in squad:
            parts.append((weights.squad, squad[team]))
        wsum = sum(w for w, _ in parts)
        out[team] = sum(w * r for w, r in parts) / wsum
    return out


def build_strength_table(
    elo: dict[str, float], *, weights: StrengthWeights | None = None
) -> pd.DataFrame:
    """Per-team breakdown of every signal and the composite (for inspection / the dashboard)."""
    market = load_market_ratings(elo)
    squad = load_squad_ratings(elo)
    composite = compose_strength(elo, weights=weights, market=market, squad=squad)
    rows = [
        {
            "team": t,
            "elo": round(elo[t]),
            "market_rating": round(market[t]) if t in market else np.nan,
            "squad_rating": round(squad[t]) if t in squad else np.nan,
            "composite": round(composite[t]),
            "shift": round(composite[t] - elo[t]),
        }
        for t in elo
    ]
    return pd.DataFrame(rows).sort_values("composite", ascending=False).reset_index(drop=True)


def run() -> None:
    from src.features.elo import compute_elo

    matches = pd.read_csv(PROCESSED_DIR / "matches_clean.csv", parse_dates=["date"])
    elo = compute_elo(matches).final_ratings
    # Restrict to the 48 qualified teams for a readable report.
    groups = pd.read_csv(EXTERNAL_DIR / "wc2026_groups.csv")
    elo48 = {t: elo[t] for t in groups["team"] if t in elo}

    table = build_strength_table(elo48)
    pd.set_option("display.max_rows", None)
    print("Composite strength (Elo scale) — biggest movers vs Elo-only:\n")
    print(table.to_string(index=False))


if __name__ == "__main__":
    run()
