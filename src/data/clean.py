"""Clean the raw match data and extract the World Cup 2026 fixture.

Responsibilities (kept narrow on purpose -- feature engineering lives elsewhere):

1. Parse/validate types.
2. Normalize team names so a country has ONE identity across 150 years of history
   (important for Elo continuity later).
3. Split the data into:
     - played matches  -> data/processed/matches_clean.csv   (training history)
     - WC2026 fixture  -> data/external/wc2026_fixtures.csv   (to simulate)
4. Reconstruct the 12 groups from the fixture graph and the list of 48 teams.

Run:
    python -m src.data.clean
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"

# --------------------------------------------------------------------------- #
# Team name normalization.
#
# WHY: Elo ratings carry strength forward through time, so a country must keep a
# single identity. We map defunct/renamed entities to their commonly accepted FIFA
# successor. This is a SIMPLIFICATION and is debatable (e.g. should Yugoslavia map
# to Serbia?). We document it openly; the effect on 2026 predictions is small
# because these entities stopped playing decades ago, but it keeps history coherent.
# --------------------------------------------------------------------------- #
TEAM_NAME_MAP: dict[str, str] = {
    "West Germany": "Germany",
    "Soviet Union": "Russia",
    "Czechoslovakia": "Czechia",
    "Czech Republic": "Czechia",
    "Yugoslavia": "Serbia",
    "Serbia and Montenegro": "Serbia",
    "FR Yugoslavia": "Serbia",
    "Zaïre": "DR Congo",
    "Zaire": "DR Congo",
    "Republic of Ireland": "Ireland",
}


def normalize_team_names(df: pd.DataFrame) -> pd.DataFrame:
    """Apply :data:`TEAM_NAME_MAP` to both team columns (and the shootout winner)."""
    df = df.copy()
    for col in ("home_team", "away_team"):
        df[col] = df[col].replace(TEAM_NAME_MAP)
    return df


def load_raw() -> pd.DataFrame:
    """Load ``results.csv`` with proper dtypes."""
    df = pd.read_csv(RAW_DIR / "results.csv", parse_dates=["date"])
    return df


def split_played_and_future(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separate matches with a recorded score from scheduled (NaN-score) fixtures.

    A missing score means the match has not been played yet. As of the project
    start date (2026-06-11) these are exactly the 72 World Cup 2026 group games.
    """
    played = df[df["home_score"].notna() & df["away_score"].notna()].copy()
    future = df[df["home_score"].isna() | df["away_score"].isna()].copy()

    # Scores are integers once we know the match was played.
    played["home_score"] = played["home_score"].astype(int)
    played["away_score"] = played["away_score"].astype(int)
    return played, future


def reconstruct_groups(fixtures: pd.DataFrame) -> pd.DataFrame:
    """Recover the 12 groups (4 teams each) from the round-robin fixture graph.

    In a group of 4, every team plays the other 3, so the four teams form a
    connected component. We grow each component greedily from the matchup list.
    Returns a tidy frame: columns ``group`` (A..L) and ``team``.
    """
    # Build adjacency: which teams have a scheduled match against each other.
    adjacency: dict[str, set[str]] = {}
    for home, away in zip(fixtures["home_team"], fixtures["away_team"]):
        adjacency.setdefault(home, set()).add(away)
        adjacency.setdefault(away, set()).add(home)

    seen: set[str] = set()
    groups: list[list[str]] = []
    # Sort for deterministic output across runs/agents.
    for team in sorted(adjacency):
        if team in seen:
            continue
        # A group is the team plus everyone it plays (round-robin clique of 4).
        component = sorted({team} | adjacency[team])
        seen.update(component)
        groups.append(component)

    rows = []
    for label, teams in zip("ABCDEFGHIJKL", groups):
        for team in teams:
            rows.append({"group": label, "team": team})
    return pd.DataFrame(rows)


def run() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)

    df = normalize_team_names(load_raw())
    played, future = split_played_and_future(df)

    # Persist the cleaned training history.
    played_out = PROCESSED_DIR / "matches_clean.csv"
    played.to_csv(played_out, index=False)
    print(f"[ok] played matches  -> {played_out.name}  ({len(played):,} rows)")

    # Persist the WC2026 fixture (the games we will simulate).
    fixtures = future.sort_values("date").reset_index(drop=True)
    fixtures_out = EXTERNAL_DIR / "wc2026_fixtures.csv"
    fixtures.to_csv(fixtures_out, index=False)
    print(f"[ok] WC2026 fixtures -> {fixtures_out.name}  ({len(fixtures):,} rows)")

    # Derive the 48-team list and the 12 groups.
    groups = reconstruct_groups(fixtures)
    groups_out = EXTERNAL_DIR / "wc2026_groups.csv"
    groups.to_csv(groups_out, index=False)
    n_groups = groups["group"].nunique()
    n_teams = groups["team"].nunique()
    print(f"[ok] groups          -> {groups_out.name}  ({n_groups} groups, {n_teams} teams)")

    # Sanity checks: surface problems loudly instead of failing silently later.
    if n_teams != 48:
        print(f"[WARN] expected 48 teams, got {n_teams} -- inspect the fixture.")
    sizes = groups.groupby("group").size()
    if not (sizes == 4).all():
        print(f"[WARN] not all groups have 4 teams:\n{sizes.to_string()}")


if __name__ == "__main__":
    run()
