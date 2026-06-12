"""World Cup 2026 Predictor — interactive analytics dashboard.

A dark, editorial sports-analytics interface (custom HTML/CSS, real team flags) built on top
of the trained models and Monte Carlo results.

Run locally:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.features.elo import compute_elo
from src.simulation.engine import load_goals_params
from src.simulation.scenario import simulate_scenario
from src.simulation.tournament import STRENGTH_NOISE, build_context

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "reports" / "simulation_results.csv"
META = ROOT / "reports" / "simulation_results.meta.json"
BENCHMARK = ROOT / "reports" / "model_benchmark.csv"
SIGMA_EVIDENCE = ROOT / "reports" / "sigma_calibration.csv"
CALIBRATION_FIG = ROOT / "reports" / "figures" / "08_calibration.png"
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

st.set_page_config(page_title="World Cup 2026 Forecast", layout="wide")


# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Oswald:wght@500;600;700&family=Manrope:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@500;600&display=swap');

:root{
  --bg:#F7F0E6; --wash:#FBF7F0; --elev:#FFFDF8; --elev2:#EFE4D4; --border:#DDCCB8;
  --text:#31271F; --muted:#76685A; --faint:#A59483;
  --accent:#A65F3D; --accent2:#C9794A; --accent3:#6F7B52;
  --gold:#B9852E; --silver:#9E958A; --bronze:#B77755; --red:#A84A3A;
  --shadow:0 18px 44px -34px rgba(70,45,24,.62);
}
*{ box-sizing:border-box; }
.stApp{ background:
  radial-gradient(920px 420px at 18% -12%, rgba(203,144,91,.18) 0%, rgba(203,144,91,0) 62%),
  linear-gradient(180deg,var(--wash) 0%,var(--bg) 58%,#EFE2D0 100%) fixed; }
html, body, [class*="css"], p, span, div, label{ font-family:'Manrope', sans-serif; color:var(--text); }
body{ overflow-x:hidden; }
#MainMenu, [data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"], footer{ display:none !important; }
[data-testid="stHeader"]{ background:transparent; height:0; }
.block-container{ max-width:1160px; padding-top:1.25rem; padding-bottom:3.2rem; }
h1,h2,h3,h4{ font-family:'Oswald', sans-serif; letter-spacing:0; font-weight:600; color:var(--text); }
.mono{ font-family:'IBM Plex Mono', monospace; font-variant-numeric:tabular-nums; }
a{ color:var(--accent); }

/* Tabs */
div[data-baseweb="tab-list"]{ gap:12px; border-bottom:1px solid var(--border); overflow-x:auto; scrollbar-width:none; padding-bottom:2px; }
div[data-baseweb="tab-list"]::-webkit-scrollbar{ display:none; }
button[data-baseweb="tab"]{ font-family:'Oswald',sans-serif!important; text-transform:uppercase;
  letter-spacing:1.2px; font-size:.8rem!important; color:var(--muted)!important; padding:10px 4px!important;
  white-space:nowrap; transition:color 0.2s ease; }
button[data-baseweb="tab"][aria-selected="true"]{ color:var(--text)!important; }
div[data-baseweb="tab-highlight"]{ background-color:var(--accent)!important; height:2px!important; transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1)!important; }
div[data-baseweb="tab-border"]{ background:transparent!important; }

/* Hero */
.hero{ padding:18px 0 24px; border-bottom:1px solid var(--border); margin-bottom:26px; }
.hero-kicker{ font-family:'Oswald'; text-transform:uppercase; letter-spacing:4px; font-size:.72rem;
  color:var(--accent); margin-bottom:8px; }
.hero-title{ font-family:'Oswald'; font-weight:700; font-size:clamp(2.2rem,6vw,4.25rem); line-height:.96; margin:0;
  text-transform:uppercase; letter-spacing:0; color:var(--text); }
.hero-sub{ color:var(--muted); font-size:1rem; max-width:720px; margin-top:14px; line-height:1.65; }

/* Podium */
.podium{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:16px; margin:10px 0 32px; }
.pcard{ background:linear-gradient(180deg,var(--elev),#F8F0E5); border:1px solid var(--border);
  border-radius:14px; padding:22px 20px; position:relative; overflow:hidden; box-shadow:var(--shadow); min-width:0;
  transition:transform 0.2s cubic-bezier(0.175, 0.885, 0.32, 1.275), box-shadow 0.2s ease; }
.pcard:hover{ transform:translateY(-4px); box-shadow:0 24px 52px -30px rgba(70,45,24,.7); }
.pcard .seed{ font-family:'Oswald'; font-size:.72rem; letter-spacing:1.8px; text-transform:uppercase; color:var(--muted); font-weight:500; }
.pcard.gold{ border-color:rgba(185,133,46,.45); box-shadow:0 0 0 1px rgba(185,133,46,.12) inset, var(--shadow); }
.pcard .topline{ height:3px; width:46px; border-radius:3px; margin-bottom:16px; }
.pcard.gold .topline{ background:var(--gold);} .pcard.silver .topline{ background:var(--silver);} .pcard.bronze .topline{ background:var(--bronze);}
.pcard .pteam{ display:flex; align-items:center; gap:12px; margin:10px 0 4px; min-width:0; }
.pcard .pteam .name{ font-family:'Oswald'; font-size:clamp(1.25rem,2.2vw,1.65rem); font-weight:600; overflow-wrap:anywhere; line-height:1.1; }
.pcard .pval{ font-family:'IBM Plex Mono'; font-size:clamp(1.8rem,3vw,2.35rem); font-weight:600; margin-top:8px; color:var(--text); }
.pcard.gold .pval{ color:var(--gold);}
.pcard .plabel{ color:var(--muted); font-size:.8rem; line-height:1.45; }

/* Leaderboard rows */
.lb{ display:flex; flex-direction:column; gap:4px; }
.row{ display:grid; grid-template-columns:36px 30px minmax(120px,1fr) minmax(140px,1.15fr) 70px; align-items:center; gap:14px;
  padding:12px 10px; border-bottom:1px solid rgba(117,94,68,.12); min-width:0; transition:all 0.2s ease; border-radius:10px; }
.row:hover{ background:rgba(221,204,184,0.22); transform:translateX(4px); }
.row .rk{ font-family:'IBM Plex Mono'; color:var(--faint); font-size:.92rem; text-align:right; }
.row .nm{ font-weight:700; font-size:.98rem; display:flex; align-items:center; gap:10px; min-width:0; overflow-wrap:anywhere; }
.row .nm small{ color:var(--faint); font-family:'IBM Plex Mono'; font-weight:500; font-size:.72rem; margin-left:2px; }
.row .pc{ font-family:'IBM Plex Mono'; font-weight:600; text-align:right; font-size:.95rem; }
.track{ height:9px; background:#E7D7C5; border-radius:999px; overflow:hidden; min-width:0; }
.fill{ height:100%; border-radius:6px; background:linear-gradient(90deg,var(--accent),var(--accent2)); transition:width 0.8s cubic-bezier(0.16, 1, 0.3, 1); }
.fill.lead{ background:linear-gradient(90deg,#B8902E,var(--gold)); }
.flag{ width:26px; height:18px; border-radius:3px; object-fit:cover; box-shadow:0 0 0 1px rgba(49,39,31,.14); flex:0 0 auto; }

/* Group cards */
.gwrap{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:12px; }
.gcard{ background:var(--elev); border:1px solid var(--border); border-radius:12px; padding:14px 16px;
  display:flex; align-items:center; gap:14px; min-width:0; box-shadow:0 14px 36px -34px rgba(70,45,24,.5);
  transition:transform 0.2s ease; }
.gcard:hover{ transform:translateY(-1px); }
.gcard .gname{ font-weight:700; font-size:1.05rem; flex:1; min-width:0; overflow-wrap:anywhere; line-height:1.25; }
.gcard .gpc{ font-family:'IBM Plex Mono'; font-weight:600; color:var(--accent); font-size:1.05rem; }
.gcard .glabel{ color:var(--faint); font-size:.68rem; text-transform:uppercase; letter-spacing:1px; }
.gcard.qual{ border-color:rgba(111,123,82,.45); background:linear-gradient(180deg,#FFFDF8,#F4EBDD); }

/* Match predictor */
.vs{ background:linear-gradient(180deg,var(--elev),#F4EBDD); border:1px solid var(--border);
  border-radius:16px; padding:26px; margin-top:10px; box-shadow:var(--shadow); overflow:hidden; }
.vs-head{ display:grid; grid-template-columns:minmax(0,1fr) auto minmax(0,1fr); align-items:center; gap:20px; }
.vs-team{ display:flex; flex-direction:column; align-items:center; gap:10px; }
.vs-team img{ width:64px; height:44px; border-radius:5px; box-shadow:0 0 0 1px rgba(49,39,31,.14); }
.vs-team .vn{ font-family:'Oswald'; font-size:clamp(1.05rem,2vw,1.35rem); font-weight:600; text-align:center; overflow-wrap:anywhere; line-height:1.1; }
.vs-team .elo{ font-family:'IBM Plex Mono'; color:var(--faint); font-size:.74rem; }
.vs-score{ font-family:'IBM Plex Mono'; font-size:clamp(1.75rem,4vw,2.6rem); font-weight:600; letter-spacing:0; white-space:nowrap; color:var(--text); }
.seg{ display:flex; height:42px; border-radius:10px; overflow:hidden; margin-top:24px; border:1px solid var(--border); background:#E7D7C5; }
.seg div{ display:flex; align-items:center; justify-content:center; font-family:'IBM Plex Mono';
  font-size:.8rem; font-weight:600; color:#FFFDF8; min-width:0; overflow:hidden; text-overflow:clip; white-space:nowrap; }
.seg .sa{ background:linear-gradient(90deg,var(--accent),var(--accent2)); }
.seg .sd{ background:#CDBBA5; color:var(--text); }
.seg .sb{ background:linear-gradient(90deg,#B77755,var(--red)); color:#FFFDF8; }
.seglabels{ display:flex; justify-content:space-between; gap:10px; margin-top:9px; color:var(--muted); font-size:.74rem; }
.seglabels span{ min-width:0; overflow-wrap:anywhere; }

/* Explanations / methodology */
.lead{ color:var(--muted); font-size:.9rem; line-height:1.65; margin:-4px 0 20px; max-width:800px; }
.prose{ color:var(--muted); line-height:1.75; font-size:.96rem; max-width:780px; }
.prose b{ color:var(--text); }
.sec{ font-family:'Oswald'; text-transform:uppercase; letter-spacing:2.5px; font-size:.76rem;
  color:var(--accent); margin:32px 0 14px; }
.steps{ display:flex; flex-direction:column; gap:10px; }
.step{ display:grid; grid-template-columns:44px 1fr; gap:16px; align-items:start; background:var(--elev);
  border:1px solid var(--border); border-radius:12px; padding:16px 18px; box-shadow:0 16px 42px -38px rgba(70,45,24,.46); min-width:0; }
.step .num{ font-family:'Oswald'; font-size:1.25rem; color:var(--accent); border:1px solid var(--border);
  border-radius:9px; width:44px; height:44px; display:flex; align-items:center; justify-content:center; background:#F7F0E6; }
.step .st-t{ font-family:'Oswald'; font-size:1.05rem; letter-spacing:.5px; }
.step .st-d{ color:var(--muted); font-size:.87rem; margin-top:3px; line-height:1.55; }
.step .st-d b{ color:var(--text); }
.facts{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin-bottom:8px; }
.fact{ background:var(--elev); border:1px solid var(--border); border-radius:12px; padding:16px; min-width:0; box-shadow:0 14px 38px -36px rgba(70,45,24,.45); }
.fact .fv{ font-family:'IBM Plex Mono'; font-size:1.5rem; font-weight:600; color:var(--text); }
.fact .fv.acc{ color:var(--accent); }
.fact .fl{ color:var(--muted); font-size:.75rem; margin-top:5px; line-height:1.45; }
.limit{ border-left:2px solid var(--border); padding-left:16px; color:var(--muted); font-size:.9rem;
  line-height:1.75; max-width:740px; }
.limit b{ color:var(--text); }
.repo{ display:inline-block; margin-top:6px; font-family:'Oswald'; letter-spacing:1.5px;
  text-transform:uppercase; font-size:.76rem; color:var(--accent); border:1px solid var(--border);
  border-radius:8px; padding:11px 20px; text-decoration:none; background:#FFFDF8; transition:all 0.2s ease; }
.repo:hover{ border-color:var(--accent); background:#F4EBDD; transform:translateY(-1px); }

/* Scenario: champion banner + stats + bracket */
.champ{ background:linear-gradient(180deg,rgba(185,133,46,.15),var(--elev)); border:1px solid rgba(185,133,46,.4);
  border-radius:16px; padding:20px 24px; display:flex; align-items:center; gap:20px; margin:8px 0 18px; box-shadow:var(--shadow); min-width:0; }
.champ img{ width:74px; height:50px; border-radius:6px; box-shadow:0 0 0 1px rgba(49,39,31,.14); flex:0 0 auto; }
.champ .cl{ font-family:'Oswald'; text-transform:uppercase; letter-spacing:3px; font-size:.72rem; color:var(--gold); }
.champ .ct{ font-family:'Oswald'; font-size:clamp(1.7rem,4vw,2.1rem); font-weight:700; line-height:1.05; overflow-wrap:anywhere; }
.kflag{ width:22px; height:15px; border-radius:2px; object-fit:cover; box-shadow:0 0 0 1px rgba(49,39,31,.14); flex:0 0 auto; }
.ko-grid{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:10px; }
.ko-grid.one{ grid-template-columns:minmax(280px,460px); justify-content:center; }
.ko-card{ display:grid; grid-template-columns:1fr auto 1fr; align-items:center; gap:10px;
  background:var(--elev); border:1px solid var(--border); border-radius:12px; padding:12px 16px; min-width:0;
  transition:all 0.2s ease; box-shadow:0 4px 12px -8px rgba(70,45,24,0.1); }
.ko-card:hover{ border-color:var(--muted); box-shadow:0 8px 24px -12px rgba(70,45,24,0.2); }
.ko-card.final{ border-color:rgba(185,133,46,.44); padding:16px 20px; background:linear-gradient(180deg,#FFFDF8,#F4EBDD); }
.ko-a{ display:flex; align-items:center; justify-content:flex-end; gap:8px; text-align:right; min-width:0; }
.ko-b{ display:flex; align-items:center; gap:8px; min-width:0; }
.ko-nm{ font-size:.9rem; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.ko-nm.win{ color:var(--accent3); font-weight:800; }
.ko-sc{ font-family:'IBM Plex Mono'; font-weight:600; min-width:52px; text-align:center; font-size:.94rem; }
.ko-pens{ grid-column:1/-1; text-align:center; color:var(--faint); font-size:.66rem; letter-spacing:.5px; margin-top:4px; }

/* Group match scorelines */
.gm{ display:grid; grid-template-columns:1fr auto 1fr; gap:8px; align-items:center;
  padding:6px 2px; border-bottom:1px solid rgba(117,94,68,.12); min-width:0; }
.gm:last-child{ border-bottom:none; }
.gm .a{ display:flex; justify-content:flex-end; gap:7px; align-items:center;
  font-size:.8rem; color:var(--muted); min-width:0; }
.gm .b{ display:flex; gap:7px; align-items:center; font-size:.8rem; color:var(--muted); min-width:0; }
.gm .a span, .gm .b span{ min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.gm .w{ color:var(--text); font-weight:600; }
.gm .s{ font-family:'IBM Plex Mono'; font-size:.78rem; color:var(--text); font-weight:600;
  min-width:38px; text-align:center; background:var(--elev2); border-radius:5px; padding:2px 4px; }
.gm-h{ font-family:'Oswald'; font-weight:500; color:var(--faint); text-transform:uppercase;
  font-size:.62rem; letter-spacing:1.5px; margin:12px 0 6px; }
.gblock{ background:var(--elev); border:1px solid var(--border); border-radius:12px; padding:14px 16px; min-width:0; overflow-x:auto; box-shadow:0 18px 44px -38px rgba(70,45,24,.5); }

/* Group standings tables */
.gst-wrap{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:18px 24px; }
.gst-h{ font-family:'Oswald'; letter-spacing:1px; font-size:.92rem; margin:0 0 6px; color:var(--text); }
.gst{ width:100%; min-width:380px; border-collapse:collapse; }
.gst th{ font-family:'Oswald'; font-weight:500; color:var(--faint); text-transform:uppercase;
  font-size:.6rem; letter-spacing:1px; text-align:right; padding:4px 5px; border-bottom:1px solid var(--border); }
.gst th.tm{ text-align:left; }
.gst td{ padding:8px 5px; text-align:right; border-bottom:1px solid rgba(117,94,68,.14);
  font-family:'IBM Plex Mono'; font-size:.78rem; color:var(--muted); }
.gst tr:hover td{ background:rgba(221,204,184,0.1); color:var(--text); }
.gst td.tm{ text-align:left; font-family:'Manrope'; font-weight:600; color:var(--text);
  display:flex; align-items:center; gap:8px; min-width:128px; }
.gst td.pts{ color:var(--text); font-weight:600; }
.gst tr.qual td.pts{ color:var(--accent3); }
.gst tr.qual td.tm{ position:relative; }

/* Footer */
.foot{ margin-top:34px; padding-top:16px; border-top:1px solid var(--border); color:var(--faint);
  font-size:.74rem; line-height:1.6; }
.foot b{ color:var(--muted); }

/* Streamlit controls */
.stButton button{ background:var(--accent)!important; color:#FFFDF8!important; border:1px solid var(--accent)!important;
  border-radius:10px!important; box-shadow:0 8px 24px -12px rgba(166,95,61,.8); font-weight:700!important; 
  padding:0.5rem 1.5rem!important; transition:all 0.2s ease!important; }
.stButton button:hover{ background:var(--accent2)!important; border-color:var(--accent2)!important; color:#FFFDF8!important; transform:translateY(-1px); box-shadow:0 12px 28px -12px rgba(166,95,61,.9); }
.stButton button:active{ transform:translateY(0); }
[data-testid="stSlider"] [role="slider"]{ background:var(--accent)!important; border-color:var(--accent)!important; }
[data-testid="stSlider"] div[data-testid="stTickBar"]{ color:var(--muted)!important; }
div[data-baseweb="select"] > div{ background:var(--elev)!important; border-color:var(--border)!important; border-radius:10px!important; }
div[data-baseweb="select"] span{ color:var(--text)!important; }
[data-testid="stAlert"]{ background:#FFF7E7!important; border:1px solid var(--border)!important; color:var(--text)!important; }

@media (max-width: 900px){
  .podium,.facts,.gst-wrap{ grid-template-columns:1fr; }
  .ko-grid{ grid-template-columns:1fr; }
  .gwrap{ grid-template-columns:1fr; }
}

@media (max-width: 640px){
  .block-container{ padding-left:1.1rem; padding-right:1.1rem; }
  .hero-kicker{ letter-spacing:2.5px; line-height:1.35; }
  div[data-baseweb="tab-list"]{ gap:16px; margin-bottom:12px; }
  button[data-baseweb="tab"]{ font-size:.75rem!important; padding:8px 2px!important; }
  .row{ grid-template-columns:32px 30px minmax(0,1fr) 64px; gap:8px 10px; padding:12px 6px; }
  .row .track{ grid-column:3 / -1; grid-row:2; margin-top:4px; }
  .row .pc{ grid-column:4; grid-row:1; }
  .vs{ padding:20px 14px; }
  .vs-head{ grid-template-columns:1fr; gap:16px; }
  .vs-score{ order:-1; text-align:center; }
  .seg{ height:38px; }
  .seg div{ font-size:.68rem; }
  .seglabels{ font-size:.68rem; }
  .step{ grid-template-columns:36px 1fr; gap:12px; padding:14px; }
  .step .num{ width:36px; height:36px; font-size:1rem; }
  .champ{ padding:16px; align-items:flex-start; gap:14px; }
  .champ img{ width:58px; height:40px; }
  .ko-grid.one{ grid-template-columns:1fr; }
  .ko-card{ padding:10px 12px; }
}
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


@st.cache_data
def load_meta() -> dict | None:
    """Sidecar written together with the results CSV; binds it to the engine settings."""
    if not META.exists():
        return None
    return json.loads(META.read_text(encoding="utf-8"))


@st.cache_data
def load_csv(path: str) -> pd.DataFrame | None:
    p = Path(path)
    return pd.read_csv(p) if p.exists() else None


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


@st.cache_resource
def load_context():
    return build_context()


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


_ROUND_TITLES = [("R32", "Round of 32"), ("R16", "Round of 16"),
                 ("QF", "Quarterfinals"), ("SF", "Semifinals"), ("F", "Final")]


def _kflag(team: str) -> str:
    code = FLAGS.get(team, "")
    return f'<img class="kflag" src="https://flagcdn.com/w40/{code}.png" alt="{team}">' if code else ""


def _ko_card(m, final: bool = False) -> str:
    aw = " win" if m.winner == m.team_a else ""
    bw = " win" if m.winner == m.team_b else ""
    pens = '<div class="ko-pens">decided on penalties</div>' if m.pens else ""
    cls = "ko-card final" if final else "ko-card"
    return (
        f'<div class="{cls}">'
        f'<div class="ko-a"><span class="ko-nm{aw}">{m.team_a}</span>{_kflag(m.team_a)}</div>'
        f'<div class="ko-sc">{m.goals_a}&ndash;{m.goals_b}</div>'
        f'<div class="ko-b">{_kflag(m.team_b)}<span class="ko-nm{bw}">{m.team_b}</span></div>'
        f'{pens}</div>'
    )


def champion_banner_html(scen) -> str:
    code = FLAGS.get(scen.champion, "")
    return (
        f'<div class="champ"><img src="https://flagcdn.com/w160/{code}.png" alt="{scen.champion}">'
        f'<div><div class="cl">Simulated champion</div>'
        f'<div class="ct">{scen.champion}</div></div></div>'
    )


def scenario_stats_html(scen) -> str:
    s = scen.stats
    bw = s["biggest_win"]
    facts = [
        ("fv", str(s["total_goals"]), "goals scored"),
        ("fv", str(s["avg_goals"]), "goals per match"),
        ("fv", str(s["shootouts"]), "penalty shootouts"),
        ("fv acc", f'{bw.goals_a}&ndash;{bw.goals_b}', f'biggest win &middot; {bw.team_a} v {bw.team_b}'),
    ]
    cells = "".join(
        f'<div class="fact"><div class="{c}">{v}</div><div class="fl">{lbl}</div></div>'
        for c, v, lbl in facts
    )
    return f'<div class="facts">{cells}</div>'


def bracket_html(scen) -> str:
    out = []
    for key, title in _ROUND_TITLES:
        ms = [m for m in scen.knockouts if m.round_key == key]
        if not ms:
            continue
        grid_cls = "ko-grid one" if key == "F" else "ko-grid"
        cards = "".join(_ko_card(m, final=(key == "F")) for m in ms)
        out.append(f'<div class="sec">{title}</div><div class="{grid_cls}">{cards}</div>')
    return "".join(out)


def _group_match_row(m) -> str:
    hw = "w" if m.goals_h > m.goals_a else ""
    aw = "w" if m.goals_a > m.goals_h else ""
    return (
        f'<div class="gm"><div class="a"><span class="{hw}">{m.home}</span>{_kflag(m.home)}</div>'
        f'<div class="s">{m.goals_h}&ndash;{m.goals_a}</div>'
        f'<div class="b">{_kflag(m.away)}<span class="{aw}">{m.away}</span></div></div>'
    )


def group_tables_html(scen) -> str:
    blocks = []
    for g, rows in scen.group_tables.items():
        trs = []
        for i, r in enumerate(rows):
            q = "qual" if i < 2 else ""
            trs.append(
                f'<tr class="{q}"><td class="tm">{_kflag(r["team"])}{r["team"]}</td>'
                f'<td>{r["pld"]}</td><td>{r["w"]}</td><td>{r["d"]}</td><td>{r["l"]}</td>'
                f'<td>{r["gf"]}</td><td>{r["ga"]}</td><td>{r["gd"]:+d}</td>'
                f'<td class="pts">{r["pts"]}</td></tr>'
            )
        matches = "".join(_group_match_row(m) for m in scen.group_matches[g])
        blocks.append(
            f'<div class="gblock"><div class="gst-h">Group {g}</div><table class="gst">'
            f'<tr><th class="tm">Team</th><th>P</th><th>W</th><th>D</th><th>L</th>'
            f'<th>GF</th><th>GA</th><th>GD</th><th>Pts</th></tr>'
            f'{"".join(trs)}</table>'
            f'<div class="gm-h">Results</div>{matches}</div>'
        )
    return f'<div class="gst-wrap">{"".join(blocks)}</div>'


_STEPS = [
    ("Collect 150 years of football",
     "Everything starts with a public dataset of every official international match since "
     "1872, roughly <b>49,000 games</b>. We clean it carefully: dates are typed, country names "
     "are unified across history (West Germany becomes Germany, Zaire becomes DR Congo) so each "
     "nation keeps a single identity through time. A nice surprise hidden in the data: the 72 "
     "scheduled group games of 2026 are already there, so the 48 qualified teams and the 12 "
     "official groups come straight from the source instead of being typed by hand."),
    ("Give every team a strength rating (Elo)",
     "Each nation carries an <b>Elo rating</b>, the same idea chess uses. After every match the "
     "winner takes points from the loser. The clever part is the asymmetry: beating a far "
     "stronger side earns a lot, while beating a minnow earns almost nothing, so upsets move "
     "the table and routine wins do not. Our version also weighs matches by importance (a World "
     "Cup game moves ratings three times more than a friendly), by margin of victory (a 5-0 says "
     "more than a 1-0) and gives home sides a bonus. One strict rule protects everything: when a "
     "match is used for training, we only look at the ratings <b>before</b> kickoff. The model "
     "never gets to peek at the result it is trying to predict."),
    ("Learn how strength turns into results",
     "With ratings in hand, a <b>logistic regression</b> learns the pattern from about 23,000 "
     "modern matches (2002 onwards): given an Elo gap, recent form, rest days and venue, what is "
     "the probability of a win, a draw or a loss? We also trained a fancier gradient boosting "
     "model. It did <b>not</b> beat the simple one on any honest metric, so we kept the simple "
     "one. That choice is deliberate: when two models score the same, the one you can explain "
     "wins. Its probabilities are calibrated, meaning when it says 70%, history shows that "
     "outcome really happens about 70% of the time."),
    ("Turn probabilities into scorelines (Poisson)",
     "Group standings need actual goals, not just results, because ties are broken on goal "
     "difference. So a second model estimates how many goals each side should score, using the "
     "<b>Poisson distribution</b>, the classic statistical model for rare counting events like "
     "goals. A team's expected goals rise with its Elo advantage. Two evenly matched sides "
     "average about 1.1 goals each; a heavy favourite pushes 2.3 while its rival drops to 0.5. "
     "This model was built independently from the classifier, and the fact that both agree on "
     "unseen matches is one of our strongest sanity checks."),
    ("Respect what we do not know",
     "An Elo rating is an estimate, not a fact, and if it is slightly wrong for a team, that "
     "same error follows the team through its entire tournament. So before each simulated "
     "tournament we redraw every team's underlying strength around its rating. The width of "
     "that redraw was not chosen by eye: we swept candidate values and checked the favourite's "
     "title odds against the betting market and the historical record of World Cup favourites, "
     "both of which sit around 15 to 20%. The full sweep is published in the Validation tab. "
     "The hosts also get their data-fitted home boost in the group games they play at home, "
     "because the fixture list says those are real home matches."),
    ("Play the World Cup 10,000 times (Monte Carlo)",
     "One run plays all 104 matches: 72 group games with sampled scorelines, FIFA tie-breakers, "
     "the ranking of best third-placed teams, then the official knockout bracket from the round "
     "of 32 to the final, penalties included. Then we do it again. And again. <b>10,000 times.</b> "
     "Counting how often each nation lifts the trophy gives its title probability: {fav} at "
     "{pct} means {fav} won {wins} of those 10,000 tournaments. The Bracket tab shows one of "
     "these stories in full detail; the Simulator lets you deal new ones."),
    ("Check the work",
     "Before trusting any number we tested the pipeline the way a quant tests a trading model. "
     "Trained on the past, evaluated only on later matches it had never seen (2022 onwards, "
     "including the last World Cup). Scored with proper probability metrics, not just accuracy. "
     "Stress-tested: changing the training window from 1990 to 2014 barely moves the skill. "
     "A suite of <b>automated tests</b> guards the internals, from Elo being zero-sum to every "
     "simulated tournament producing exactly one champion, and a metadata file ties the "
     "published numbers to the exact engine settings that produced them."),
]

_FACTS = [
    ("fv", "10,000", "tournaments simulated"),
    ("fv", "49,000", "matches since 1872"),
    ("fv acc", "0.172", "forecast skill (RPS) vs 0.228 naive baseline"),
    ("fv acc", "16/16", "automated tests passing"),
]


def methodology_html(fav: str, fav_prob: float) -> str:
    pct = f"{fav_prob:.1%}"
    wins = f"{round(fav_prob * 10_000):,}"
    steps = "".join(
        f'<div class="step"><div class="num">{i}</div><div><div class="st-t">{t}</div>'
        f'<div class="st-d">{d.format(fav=fav, pct=pct, wins=wins)}</div></div></div>'
        for i, (t, d) in enumerate(_STEPS, 1)
    )
    facts = "".join(
        f'<div class="fact"><div class="{cls}">{v}</div><div class="fl">{lbl}</div></div>'
        for cls, v, lbl in _FACTS
    )
    return (
        '<div class="prose">This project answers one question: <b>who is most likely to win the '
        '2026 World Cup?</b> It does not claim to know the future. Football is famously '
        'unpredictable, and any tool that hands you a single guaranteed winner is lying to you. '
        'What it offers instead is an honest forecast, built step by step from public data, '
        'where every number on this dashboard can be traced back to its source. Here is the '
        'whole journey, in plain words.</div>'
        '<div class="sec">The pipeline, step by step</div>'
        f'<div class="steps">{steps}</div>'
        '<div class="sec">Why you can trust it</div>'
        f'<div class="facts">{facts}</div>'
        '<div class="prose" style="margin-top:14px">'
        'The test that matters most is simple: predict matches the model has never seen. On '
        'every metric we tracked, it clearly beats naive guessing, and its probabilities mean '
        'what they say. When two independently built models (the classifier and the goals model) '
        'arrive at the same answer on unseen data, that agreement is hard to fake.</div>'
        '<div class="sec">What it cannot do</div>'
        '<div class="limit">It knows teams, not players: injuries, suspensions and squad form are '
        'invisible to it.<br>'
        'Host advantage is applied only where the fixture list confirms a real home game (the '
        'hosts\' own group matches); knockout venues are treated as neutral.<br>'
        f'And above all, it outputs <b>probabilities, not certainties</b>. A {pct} favourite '
        f'still loses the tournament about {1 - fav_prob:.0%} of the time. That is not a '
        'weakness of the model. That is football.</div>'
        '<div class="sec">Under the hood</div>'
        '<div class="prose">Python, pandas, scikit-learn, NumPy, SciPy and Streamlit. Elo ratings '
        'feed a calibrated logistic regression and a Poisson goals model, which drive a Monte '
        'Carlo simulation of the official FIFA bracket. The whole pipeline is open source, '
        'reproducible with one command, and documented decision by decision.</div>'
        '<a class="repo" href="https://github.com/mgaedechens/Worldcup" target="_blank">View the code on GitHub</a>'
    )


def benchmark_table_html(df: pd.DataFrame) -> str:
    """Out-of-time model benchmark (the evidence behind the model choice)."""
    rows = []
    for _, r in df.iterrows():
        selected = "*selected*" in r["model"]
        name = r["model"].replace(" *selected*", "")
        badge = (' <span style="color:var(--accent);font-weight:800;font-size:.68rem;'
                 'text-transform:uppercase;letter-spacing:1px">selected</span>') if selected else ""
        cls = ' class="qual"' if selected else ""
        rows.append(
            f'<tr{cls}><td class="tm">{name}{badge}</td>'
            f'<td>{r["accuracy"]:.3f}</td><td>{r["log_loss"]:.4f}</td>'
            f'<td>{r["brier"]:.4f}</td><td class="pts">{r["rps"]:.4f}</td></tr>'
        )
    return (
        '<div class="gblock"><table class="gst">'
        '<tr><th class="tm">Model</th><th>Accuracy</th><th>Log loss</th>'
        '<th>Brier</th><th>RPS</th></tr>'
        f'{"".join(rows)}</table>'
        '<div class="gm-h">Test set: every match from 2022 onward, never seen in training. '
        'Lower is better for log loss, Brier and RPS.</div></div>'
    )


def sigma_table_html(df: pd.DataFrame, chosen: float) -> str:
    """Rating-uncertainty sweep (how sigma was chosen against the market benchmark)."""
    rows = []
    for _, r in df.iterrows():
        sel = float(r["sigma"]) == float(chosen)
        badge = (' <span style="color:var(--accent);font-weight:800;font-size:.68rem;'
                 'text-transform:uppercase;letter-spacing:1px">chosen</span>') if sel else ""
        cls = ' class="qual"' if sel else ""
        rows.append(
            f'<tr{cls}><td class="tm">&sigma; = {int(r["sigma"])}{badge}</td>'
            f'<td>{r["favorite"]}</td><td>{r["favorite_prob"]:.1%}</td>'
            f'<td>{r["top4_prob_sum"]:.1%}</td><td class="pts">{r["mexico_prob"]:.1%}</td></tr>'
        )
    return (
        '<div class="gblock"><table class="gst">'
        '<tr><th class="tm">Rating uncertainty</th><th>Favourite</th>'
        '<th>Title prob</th><th>Top-4 share</th><th>Mexico</th></tr>'
        f'{"".join(rows)}</table>'
        '<div class="gm-h">Benchmark: market-implied favourite probability and the historical '
        'record of World Cup favourites both sit around 15 to 20%. 4,000 simulations per row.'
        '</div></div>'
    )


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(
        '<div class="hero"><div class="hero-kicker">Predictive Model &middot; 2026 FIFA World Cup</div>'
        '<h1 class="hero-title">The Title Race</h1>'
        '<div class="hero-sub">Championship probabilities from 10,000 Monte Carlo simulations of the '
        'official 48-team bracket, driven by 150 years of results, Elo ratings and a calibrated '
        'machine-learning model.</div></div>',
        unsafe_allow_html=True,
    )

    if not RESULTS.exists():
        st.warning("Run `python scripts/run_pipeline.py` first to generate results.")
        st.stop()

    results = load_results()
    meta = load_meta()

    # Single-source-of-truth guard: the published numbers must come from THIS engine.
    if meta is None or meta.get("strength_noise") != STRENGTH_NOISE or meta.get("host_bonus", 0.0) != 0.0:
        st.warning("The cached results were generated with different engine settings. "
                   "Run `python -m src.simulation.montecarlo` to refresh them.")

    fav_team = results.iloc[0]["team"]
    fav_prob = float(results.iloc[0]["Champion"])
    ref_seed = (meta or {}).get("reference_seed", 2)

    tab_race, tab_bracket, tab_sim, tab_groups, tab_match, tab_val, tab_how = st.tabs(
        ["Title race", "Bracket", "Simulator", "Groups", "Match predictor",
         "Validation", "How it works"])

    with tab_race:
        st.markdown('<div class="lead">Each figure is the share of <b>10,000 simulated '
                    'tournaments</b> that team won. Notice the favourite still sits below 50%: '
                    'in football, nobody is ever safe. The <b>How it works</b> tab explains the '
                    'full method.</div>', unsafe_allow_html=True)
        st.markdown(podium_html(results), unsafe_allow_html=True)
        st.markdown('<div class="sec">Full championship odds</div>', unsafe_allow_html=True)
        n = st.slider("Teams shown", 8, 48, 20, label_visibility="collapsed")
        st.markdown(leaderboard_html(results.head(n)), unsafe_allow_html=True)

    def render_scenario(scen) -> None:
        st.markdown(champion_banner_html(scen), unsafe_allow_html=True)
        st.markdown(scenario_stats_html(scen), unsafe_allow_html=True)
        st.markdown(bracket_html(scen), unsafe_allow_html=True)
        st.markdown('<div class="sec">Group stage: standings and every result</div>',
                    unsafe_allow_html=True)
        st.markdown(group_tables_html(scen), unsafe_allow_html=True)

    with tab_bracket:
        st.markdown(f'<div class="lead">The reference simulation: one complete tournament played '
                    f'out match by match, with the exact score of all 104 games, from the group '
                    f'stage to the final. It is fixed and reproducible (seed {ref_seed}), picked '
                    f'automatically so its champion coincides with the model\'s favourite, '
                    f'{fav_team}. Remember it is one plausible story among 10,000, not a '
                    f'certainty. Want to see how differently it can unfold? Open the '
                    f'<b>Simulator</b> tab.</div>', unsafe_allow_html=True)
        render_scenario(simulate_scenario(load_context(), np.random.default_rng(ref_seed)))

    with tab_sim:
        st.markdown(f'<div class="lead">Here you can replay the World Cup as many times as you '
                    f'like. Every click deals a brand-new tournament from the same probability '
                    f'model: sometimes the favourite cruises, sometimes a dark horse lifts the '
                    f'trophy. That spread is not a bug, it is exactly what a {fav_prob:.1%} '
                    f'favourite means.</div>', unsafe_allow_html=True)
        if "sim_seed" not in st.session_state:
            st.session_state.sim_seed = 100
        if st.button("Simulate a new tournament"):
            st.session_state.sim_seed += 1
        st.markdown(f'<div class="gm-h" style="margin:2px 0 10px">Simulation '
                    f'#{st.session_state.sim_seed - 99}</div>', unsafe_allow_html=True)
        render_scenario(simulate_scenario(load_context(),
                                          np.random.default_rng(st.session_state.sim_seed)))

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
        st.markdown('<div class="lead">A hypothetical neutral-venue match. Win / draw / loss come '
                    'from the calibrated classifier; the scoreline is the Poisson model\'s expected '
                    'goals. Both read from the same Elo ratings.</div>', unsafe_allow_html=True)
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

    with tab_val:
        st.markdown('<div class="lead">Every number on this dashboard traces back to a measured, '
                    'reproducible decision. This tab publishes the evidence: how the model was '
                    'chosen, whether its probabilities can be trusted, and how the simulation\'s '
                    'realism was tuned against external benchmarks rather than by eye.</div>',
                    unsafe_allow_html=True)

        st.markdown('<div class="sec">1. Model selection, tested on the future</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="prose">Four candidates were trained on matches before 2022 and '
                    'scored on everything after, including the 2022 World Cup. Accuracy alone '
                    'hides probability quality, so we use proper scoring rules: log loss, the '
                    'Brier score and the Ranked Probability Score (RPS), the standard in '
                    'football forecasting. The simple logistic regression beat the fancier '
                    'gradient boosting on every one of them, which is why it was selected.</div>',
                    unsafe_allow_html=True)
        bench = load_csv(str(BENCHMARK))
        if bench is not None:
            st.markdown(benchmark_table_html(bench), unsafe_allow_html=True)
        else:
            st.info("Run `python -m src.models.train` to generate the benchmark table.")

        st.markdown('<div class="sec">2. Are the probabilities honest?</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="prose">A reliability diagram bins the model\'s predictions and '
                    'checks them against reality on the unseen test matches: when it says 70%, '
                    'does that happen about 70% of the time? The curve hugging the diagonal '
                    'means yes. This matters because the Monte Carlo consumes these '
                    'probabilities ten thousand times; a biased input would compound.</div>',
                    unsafe_allow_html=True)
        if CALIBRATION_FIG.exists():
            st.image(str(CALIBRATION_FIG), width=520)

        st.markdown('<div class="sec">3. Tournament realism, tuned against the market</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="prose">Treating Elo ratings as exact truths makes a tournament '
                    'simulation overconfident, because a rating error follows a team through '
                    'all seven of its matches. We correct this by redrawing each team\'s '
                    'strength around its rating once per simulated tournament. The width of '
                    'that redraw was swept and compared against the betting market and the '
                    'historical record of World Cup favourites, both around 15 to 20% for the '
                    'top team. The sweep below is the published evidence.</div>',
                    unsafe_allow_html=True)
        sigma = load_csv(str(SIGMA_EVIDENCE))
        if sigma is not None:
            st.markdown(sigma_table_html(sigma, STRENGTH_NOISE), unsafe_allow_html=True)
        else:
            st.info("Run `python scripts/calibrate_sigma.py` to generate the sweep.")
        st.markdown('<div class="prose" style="margin-top:12px">Full decision records live in '
                    'the repository under <span class="mono">docs/decisions/</span>, and the '
                    'engine settings that produced the published numbers are pinned in a '
                    'metadata file the app checks on every load.</div>', unsafe_allow_html=True)

    with tab_how:
        st.markdown(methodology_html(fav_team, fav_prob), unsafe_allow_html=True)

    st.markdown(
        '<div class="foot"><b>Methodology.</b> Elo ratings built from the full match history feed a '
        'calibrated logistic classifier and a Poisson goals model; the tournament is then simulated '
        '10,000 times over the official FIFA bracket with per-tournament rating uncertainty. '
        'The output is probabilities, never certainties.<br>'
        'Host group games use the data-fitted home advantage; knockout venues are neutral. '
        'Flags via flagcdn.com. Not affiliated with FIFA.</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
