"""Interactive dashboard for the World Cup 2026 Predictor.

Run locally:
    streamlit run streamlit_app.py

Three views:
- 🏆 Title odds   — Monte Carlo championship probabilities for all 48 teams.
- 📊 Groups       — per-group advance probabilities.
- ⚔️ Head-to-head — pick any two teams and see the model's match prediction.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.features.elo import compute_elo
from src.models.goals import implied_outcome_probs
from src.simulation.engine import load_goals_params

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "reports" / "simulation_results.csv"
MODEL = ROOT / "models" / "wc_model.joblib"
CLASSES = ["A", "D", "H"]  # away win, draw, home win

st.set_page_config(page_title="World Cup 2026 Predictor", page_icon="🏆", layout="wide")


@st.cache_data
def load_results() -> pd.DataFrame:
    return pd.read_csv(RESULTS)


@st.cache_resource
def load_ratings() -> dict[str, float]:
    matches = pd.read_csv(ROOT / "data" / "processed" / "matches_clean.csv", parse_dates=["date"])
    return compute_elo(matches).final_ratings


@st.cache_resource
def load_classifier():
    return joblib.load(MODEL)


@st.cache_resource
def load_goals():
    return load_goals_params()


def main() -> None:
    st.title("🏆 World Cup 2026 Predictor")
    st.caption("Elo + calibrated ML + Poisson goals, simulated 10,000 times over the official "
               "FIFA bracket. Probabilities, not predictions — football is high-variance.")

    if not RESULTS.exists():
        st.warning("Run `python scripts/run_pipeline.py` first to generate results.")
        st.stop()

    results = load_results()
    tab_odds, tab_groups, tab_h2h = st.tabs(["🏆 Title odds", "📊 Groups", "⚔️ Head-to-head"])

    # --- Title odds -------------------------------------------------------- #
    with tab_odds:
        n = st.slider("Show top N teams", 5, 48, 16)
        top = results.head(n)
        st.bar_chart(top.set_index("team")["Champion"], horizontal=True, height=28 * n)
        st.dataframe(
            results[["team", "group", "elo", "Champion", "Final", "Semifinal", "Quarterfinal"]]
            .style.format({c: "{:.1%}" for c in
                           ["Champion", "Final", "Semifinal", "Quarterfinal"]}),
            use_container_width=True, hide_index=True,
        )

    # --- Groups ------------------------------------------------------------ #
    with tab_groups:
        g = st.selectbox("Group", sorted(results["group"].unique()))
        sub = results[results["group"] == g].sort_values("Round of 16", ascending=False)
        st.subheader(f"Group {g}")
        cols = ["team", "elo", "Round of 16", "Quarterfinal", "Semifinal", "Champion"]
        st.dataframe(
            sub[cols].style.format({c: "{:.1%}" for c in cols[2:]}),
            use_container_width=True, hide_index=True,
        )
        st.caption("‘Round of 16’ ≈ probability of advancing from the group.")

    # --- Head-to-head ------------------------------------------------------ #
    with tab_h2h:
        ratings = load_ratings()
        clf = load_classifier()["model"]
        params = load_goals()
        teams = sorted(results["team"])

        c1, c2 = st.columns(2)
        team_a = c1.selectbox("Team A", teams, index=teams.index("Spain") if "Spain" in teams else 0)
        team_b = c2.selectbox("Team B", teams, index=teams.index("Brazil") if "Brazil" in teams else 1)

        if team_a == team_b:
            st.info("Pick two different teams.")
            st.stop()

        elo_diff = ratings[team_a] - ratings[team_b]
        # Neutral venue, no form/rest info for a hypothetical match.
        probs = clf.predict_proba([[elo_diff, 0.0, 0.0, 1]])[0]
        p_a, p_draw, p_b = probs[CLASSES.index("H")], probs[CLASSES.index("D")], probs[CLASSES.index("A")]

        lam_a, lam_b = params.lam(elo_diff), params.lam(-elo_diff)

        m1, m2, m3 = st.columns(3)
        m1.metric(f"{team_a} win", f"{p_a:.0%}")
        m2.metric("Draw", f"{p_draw:.0%}")
        m3.metric(f"{team_b} win", f"{p_b:.0%}")
        st.write(f"**Expected scoreline:** {team_a} {lam_a:.1f} – {lam_b:.1f} {team_b}  ·  "
                 f"Elo {ratings[team_a]:.0f} vs {ratings[team_b]:.0f}")
        st.caption("P(win/draw/loss) from the calibrated logistic classifier; expected goals "
                   "from the Poisson model. Both share the same Elo input.")


if __name__ == "__main__":
    main()
