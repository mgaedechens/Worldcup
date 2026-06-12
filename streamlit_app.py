"""World Cup 2026 Predictor — interactive analytics dashboard.

A dark, editorial sports-analytics interface (custom HTML/CSS, real team flags) built on top
of the trained models and Monte Carlo results.

Run locally:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.features.elo import compute_elo
from src.simulation.engine import load_goals_params

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "reports" / "simulation_results.csv"
MODEL = ROOT / "models" / "wc_model.joblib"
CLASSES = ["A", "D", "H"]  # away win, draw, home win

# ISO 3166-1 alpha-2 codes (flagcdn) for every qualified nation, keyed by dataset name.
FLAGS: dict[str, str] = {
    "Mexico": "mx", "South Korea": "kr", "South Africa": "za", "Czechia": "cz",
    "Canada": "ca", "Switzerland": "ch", "Qatar": "qa", "Bosnia and Herzegovina": "ba",
    "Brazil": "br", "Morocco": "ma", "Scotland": "gb-sct", "Haiti": "ht",
    "United States": "us", "Paraguay": "py", "Australia": "au", "Turkey": "tr",
    "Germany": "de", "Ecuador": "ec", "Ivory Coast": "ci", "Curaçao": "cw",
    "Netherlands": "nl", "Japan": "jp", "Tunisia": "tn", "Sweden": "se",
    "Belgium": "be", "Iran": "ir", "Egypt": "eg", "New Zealand": "nz",
    "Spain": "es", "Uruguay": "uy", "Saudi Arabia": "sa", "Cape Verde": "cv",
    "France": "fr", "Senegal": "sn", "Norway": "no", "Iraq": "iq",
    "Argentina": "ar", "Austria": "at", "Algeria": "dz", "Jordan": "jo",
    "Portugal": "pt", "Colombia": "co", "Uzbekistan": "uz", "DR Congo": "cd",
    "England": "gb-eng", "Croatia": "hr", "Panama": "pa", "Ghana": "gh",
}

st.set_page_config(page_title="World Cup 2026 — Forecast", layout="wide")


# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Oswald:wght@500;600;700&family=Manrope:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@500;600&display=swap');

:root{
  --bg:#0B0F14; --elev:#131A22; --elev2:#1A232E; --border:#26313D;
  --text:#EAEFF4; --muted:#8593A1; --faint:#5A6673;
  --accent:#00C46B; --accent2:#22E39B; --gold:#F2C14E; --silver:#C7D0DA; --bronze:#C08552;
  --red:#E0573E;
}
.stApp{ background: radial-gradient(1100px 560px at 50% -12%, #18242F 0%, var(--bg) 58%) fixed; }
html, body, [class*="css"], p, span, div, label{ font-family:'Manrope', sans-serif; color:var(--text); }
#MainMenu, [data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"], footer{ display:none !important; }
[data-testid="stHeader"]{ background:transparent; height:0; }
.block-container{ max-width:1100px; padding-top:1.2rem; padding-bottom:3rem; }
h1,h2,h3,h4{ font-family:'Oswald', sans-serif; letter-spacing:.4px; font-weight:600; }
.mono{ font-family:'IBM Plex Mono', monospace; font-variant-numeric:tabular-nums; }

/* Tabs */
div[data-baseweb="tab-list"]{ gap:6px; border-bottom:1px solid var(--border); }
button[data-baseweb="tab"]{ font-family:'Oswald',sans-serif!important; text-transform:uppercase;
  letter-spacing:2px; font-size:.78rem!important; color:var(--muted)!important; padding:6px 4px!important; }
button[data-baseweb="tab"][aria-selected="true"]{ color:var(--text)!important; }
div[data-baseweb="tab-highlight"]{ background-color:var(--accent)!important; height:2px!important; }

/* Hero */
.hero{ padding:14px 0 22px; border-bottom:1px solid var(--border); margin-bottom:26px; }
.hero-kicker{ font-family:'Oswald'; text-transform:uppercase; letter-spacing:4px; font-size:.72rem;
  color:var(--accent); margin-bottom:8px; }
.hero-title{ font-family:'Oswald'; font-weight:700; font-size:3.1rem; line-height:1; margin:0;
  text-transform:uppercase; letter-spacing:1px; }
.hero-sub{ color:var(--muted); font-size:.95rem; max-width:680px; margin-top:12px; line-height:1.55; }

/* Podium */
.podium{ display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin:8px 0 30px; }
.pcard{ background:linear-gradient(180deg,var(--elev2),var(--elev)); border:1px solid var(--border);
  border-radius:14px; padding:18px; position:relative; overflow:hidden; }
.pcard .seed{ font-family:'Oswald'; font-size:.7rem; letter-spacing:2px; text-transform:uppercase; color:var(--faint); }
.pcard.gold{ border-color:rgba(242,193,78,.45); box-shadow:0 0 0 1px rgba(242,193,78,.12) inset, 0 12px 30px -18px rgba(242,193,78,.5); }
.pcard .topline{ height:3px; width:46px; border-radius:3px; margin-bottom:14px; }
.pcard.gold .topline{ background:var(--gold);} .pcard.silver .topline{ background:var(--silver);} .pcard.bronze .topline{ background:var(--bronze);}
.pcard .pteam{ display:flex; align-items:center; gap:12px; margin:6px 0 4px; }
.pcard .pteam .name{ font-family:'Oswald'; font-size:1.5rem; font-weight:600; }
.pcard .pval{ font-family:'IBM Plex Mono'; font-size:2.2rem; font-weight:600; margin-top:6px; }
.pcard.gold .pval{ color:var(--gold);}
.pcard .plabel{ color:var(--muted); font-size:.78rem; }

/* Leaderboard rows */
.lb{ display:flex; flex-direction:column; gap:2px; }
.row{ display:grid; grid-template-columns:34px 30px 1fr 64px; align-items:center; gap:14px;
  padding:9px 6px; border-bottom:1px solid rgba(38,49,61,.55); }
.row .rk{ font-family:'IBM Plex Mono'; color:var(--faint); font-size:.9rem; text-align:right; }
.row .nm{ font-weight:600; font-size:.98rem; display:flex; align-items:center; gap:10px; }
.row .nm small{ color:var(--faint); font-family:'IBM Plex Mono'; font-weight:500; font-size:.72rem; }
.row .pc{ font-family:'IBM Plex Mono'; font-weight:600; text-align:right; font-size:.95rem; }
.track{ height:8px; background:var(--elev2); border-radius:6px; overflow:hidden; }
.fill{ height:100%; border-radius:6px; background:linear-gradient(90deg,var(--accent),var(--accent2)); }
.fill.lead{ background:linear-gradient(90deg,#B8902E,var(--gold)); }
.flag{ width:26px; height:18px; border-radius:3px; object-fit:cover; box-shadow:0 0 0 1px rgba(255,255,255,.08); }

/* Group cards */
.gwrap{ display:grid; grid-template-columns:repeat(2,1fr); gap:12px; }
.gcard{ background:var(--elev); border:1px solid var(--border); border-radius:12px; padding:14px 16px;
  display:flex; align-items:center; gap:14px; }
.gcard .gname{ font-weight:700; font-size:1.05rem; flex:1; }
.gcard .gpc{ font-family:'IBM Plex Mono'; font-weight:600; color:var(--accent2); font-size:1.05rem; }
.gcard .glabel{ color:var(--faint); font-size:.68rem; text-transform:uppercase; letter-spacing:1px; }
.gcard.qual{ border-color:rgba(0,196,107,.4); }

/* Match predictor */
.vs{ background:linear-gradient(180deg,var(--elev2),var(--elev)); border:1px solid var(--border);
  border-radius:16px; padding:26px; margin-top:8px; }
.vs-head{ display:grid; grid-template-columns:1fr auto 1fr; align-items:center; gap:18px; }
.vs-team{ display:flex; flex-direction:column; align-items:center; gap:10px; }
.vs-team img{ width:64px; height:44px; border-radius:5px; box-shadow:0 0 0 1px rgba(255,255,255,.1); }
.vs-team .vn{ font-family:'Oswald'; font-size:1.35rem; font-weight:600; text-align:center; }
.vs-team .elo{ font-family:'IBM Plex Mono'; color:var(--faint); font-size:.74rem; }
.vs-score{ font-family:'IBM Plex Mono'; font-size:2.6rem; font-weight:600; letter-spacing:1px; }
.seg{ display:flex; height:40px; border-radius:9px; overflow:hidden; margin-top:24px; border:1px solid var(--border); }
.seg div{ display:flex; align-items:center; justify-content:center; font-family:'IBM Plex Mono';
  font-size:.8rem; font-weight:600; color:#06120C; }
.seg .sa{ background:linear-gradient(90deg,var(--accent),var(--accent2)); }
.seg .sd{ background:#39444F; color:var(--text); }
.seg .sb{ background:linear-gradient(90deg,#C75B45,var(--red)); color:#1a0a06; }
.seglabels{ display:flex; justify-content:space-between; margin-top:8px; color:var(--muted); font-size:.74rem; }

/* Footer */
.foot{ margin-top:34px; padding-top:16px; border-top:1px solid var(--border); color:var(--faint);
  font-size:.74rem; line-height:1.6; }
.foot b{ color:var(--muted); }
</style>
"""


def flag(team: str, cls: str = "flag") -> str:
    code = FLAGS.get(team)
    if not code:
        return ""
    return f'<img class="{cls}" src="https://flagcdn.com/w80/{code}.png" alt="{team}">'


# --------------------------------------------------------------------------- #
# Data loaders (cached)
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# HTML builders (pure — easy to test)
# --------------------------------------------------------------------------- #
def podium_html(df: pd.DataFrame) -> str:
    tiers = ["gold", "silver", "bronze"]
    seeds = ["Projected winner", "2nd favourite", "3rd favourite"]
    cards = []
    for (_, r), tier, seed in zip(df.head(3).iterrows(), tiers, seeds):
        cards.append(
            f'<div class="pcard {tier}"><div class="topline"></div>'
            f'<div class="seed">{seed}</div>'
            f'<div class="pteam">{flag(r["team"])}<span class="name">{r["team"]}</span></div>'
            f'<div class="pval">{r["Champion"] * 100:.1f}%</div>'
            f'<div class="plabel">to lift the trophy &middot; Elo {r["elo"]:.0f}</div></div>'
        )
    return f'<div class="podium">{"".join(cards)}</div>'


def leaderboard_html(df: pd.DataFrame, value_col: str = "Champion") -> str:
    top = df.sort_values(value_col, ascending=False).reset_index(drop=True)
    vmax = top[value_col].iloc[0] or 1.0
    rows = []
    for i, r in top.iterrows():
        width = max(1.0, r[value_col] / vmax * 100)
        lead = " lead" if i == 0 else ""
        rows.append(
            f'<div class="row"><div class="rk">{i + 1:02d}</div>{flag(r["team"])}'
            f'<div class="nm">{r["team"]}<small>{r["group"]}</small></div>'
            f'<div class="track"><div class="fill{lead}" style="width:{width:.1f}%"></div></div>'
            f'<div class="pc">{r[value_col] * 100:.1f}%</div></div>'
        )
    return f'<div class="lb">{"".join(rows)}</div>'


def group_html(sub: pd.DataFrame) -> str:
    sub = sub.sort_values("Round of 16", ascending=False).reset_index(drop=True)
    cards = []
    for i, r in sub.iterrows():
        qual = " qual" if i < 2 else ""
        cards.append(
            f'<div class="gcard{qual}">{flag(r["team"])}'
            f'<span class="gname">{r["team"]}</span>'
            f'<div style="text-align:right"><div class="gpc">{r["Round of 16"] * 100:.0f}%</div>'
            f'<div class="glabel">advance</div></div></div>'
        )
    return f'<div class="gwrap">{"".join(cards)}</div>'


def vs_flag(team: str) -> str:
    """Higher-resolution flag for the large VS header (sized by .vs-team img CSS)."""
    code = FLAGS.get(team, "")
    return f'<img src="https://flagcdn.com/w160/{code}.png" alt="{team}">' if code else ""


def h2h_html(a: str, b: str, p_a: float, p_d: float, p_b: float,
             lam_a: float, lam_b: float, elo_a: float, elo_b: float) -> str:
    wa, wd, wb = p_a * 100, p_d * 100, p_b * 100
    return (
        f'<div class="vs"><div class="vs-head">'
        f'<div class="vs-team">{vs_flag(a)}<div class="vn">{a}</div>'
        f'<div class="elo">ELO {elo_a:.0f}</div></div>'
        f'<div class="vs-score">{lam_a:.1f} &ndash; {lam_b:.1f}</div>'
        f'<div class="vs-team">{vs_flag(b)}<div class="vn">{b}</div>'
        f'<div class="elo">ELO {elo_b:.0f}</div></div></div>'
        f'<div class="seg"><div class="sa" style="width:{wa:.1f}%">{wa:.0f}%</div>'
        f'<div class="sd" style="width:{wd:.1f}%">{wd:.0f}%</div>'
        f'<div class="sb" style="width:{wb:.1f}%">{wb:.0f}%</div></div>'
        f'<div class="seglabels"><span>{a} win</span><span>Draw</span><span>{b} win</span></div></div>'
    )


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(
        '<div class="hero"><div class="hero-kicker">Predictive Model &middot; 2026 FIFA World Cup</div>'
        '<h1 class="hero-title">The Title Race</h1>'
        '<div class="hero-sub">Championship probabilities from 10,000 Monte Carlo simulations of the '
        'official 48-team bracket — driven by 150 years of results, Elo ratings, and a calibrated '
        'machine-learning model.</div></div>',
        unsafe_allow_html=True,
    )

    if not RESULTS.exists():
        st.warning("Run `python scripts/run_pipeline.py` first to generate results.")
        st.stop()

    results = load_results()
    tab_race, tab_groups, tab_match = st.tabs(["Title race", "Groups", "Match predictor"])

    with tab_race:
        st.markdown(podium_html(results), unsafe_allow_html=True)
        st.markdown('<h4 style="margin:6px 0 4px">Full championship odds</h4>', unsafe_allow_html=True)
        n = st.slider("Teams shown", 8, 48, 20, label_visibility="collapsed")
        st.markdown(leaderboard_html(results.head(n)), unsafe_allow_html=True)

    with tab_groups:
        g = st.selectbox("Group", sorted(results["group"].unique()),
                         format_func=lambda x: f"Group {x}")
        sub = results[results["group"] == g]
        st.markdown(f'<h3 style="margin:6px 0 14px">Group {g}</h3>', unsafe_allow_html=True)
        st.markdown(group_html(sub), unsafe_allow_html=True)
        st.markdown('<div style="color:#8593A1;font-size:.8rem;margin-top:12px">'
                    'Top two advance automatically; "advance" combines a top-two finish and the '
                    'best-third-place path.</div>', unsafe_allow_html=True)

    with tab_match:
        ratings = load_ratings()
        clf = load_classifier()["model"]
        params = load_goals()
        teams = sorted(results["team"])
        c1, c2 = st.columns(2)
        a = c1.selectbox("Team A", teams, index=teams.index("Spain") if "Spain" in teams else 0)
        b = c2.selectbox("Team B", teams, index=teams.index("Brazil") if "Brazil" in teams else 1)
        if a == b:
            st.info("Select two different teams.")
        else:
            elo_diff = ratings[a] - ratings[b]
            probs = clf.predict_proba([[elo_diff, 0.0, 0.0, 1]])[0]
            p_a, p_d, p_b = probs[CLASSES.index("H")], probs[CLASSES.index("D")], probs[CLASSES.index("A")]
            lam_a, lam_b = params.lam(elo_diff), params.lam(-elo_diff)
            html = h2h_html(a, b, p_a, p_d, p_b, lam_a, lam_b, ratings[a], ratings[b])
            st.markdown(html, unsafe_allow_html=True)

    st.markdown(
        '<div class="foot"><b>Methodology.</b> Elo ratings (full history) feed a calibrated logistic '
        'classifier and an Elo-driven Poisson goals model; the tournament is simulated 10,000 times '
        'over the official FIFA bracket. Probabilities, not predictions — football is high-variance.<br>'
        'Neutral venues assumed. Flags via flagcdn.com. Not affiliated with FIFA.</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
