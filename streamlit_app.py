"""World Cup 2026 Predictor — interactive analytics dashboard.

A dark, editorial sports-analytics interface (custom HTML/CSS, real team flags) built on top
of the trained models and Monte Carlo results.

Run locally:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.features.elo import compute_elo
from src.simulation.engine import load_goals_params
from src.simulation.scenario import simulate_scenario
from src.simulation.tournament import build_context

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

/* Explanations / methodology */
.lead{ color:var(--muted); font-size:.86rem; line-height:1.55; margin:-6px 0 18px; max-width:760px; }
.prose{ color:var(--muted); line-height:1.7; font-size:.94rem; max-width:740px; }
.prose b{ color:var(--text); }
.sec{ font-family:'Oswald'; text-transform:uppercase; letter-spacing:2.5px; font-size:.76rem;
  color:var(--accent); margin:30px 0 14px; }
.steps{ display:flex; flex-direction:column; gap:10px; }
.step{ display:grid; grid-template-columns:44px 1fr; gap:16px; align-items:start; background:var(--elev);
  border:1px solid var(--border); border-radius:12px; padding:15px 18px; }
.step .num{ font-family:'Oswald'; font-size:1.25rem; color:var(--accent); border:1px solid var(--border);
  border-radius:9px; width:44px; height:44px; display:flex; align-items:center; justify-content:center; }
.step .st-t{ font-family:'Oswald'; font-size:1.05rem; letter-spacing:.5px; }
.step .st-d{ color:var(--muted); font-size:.87rem; margin-top:3px; line-height:1.55; }
.step .st-d b{ color:var(--text); }
.facts{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-bottom:6px; }
.fact{ background:var(--elev); border:1px solid var(--border); border-radius:12px; padding:16px; }
.fact .fv{ font-family:'IBM Plex Mono'; font-size:1.5rem; font-weight:600; color:var(--text); }
.fact .fv.acc{ color:var(--accent2); }
.fact .fl{ color:var(--muted); font-size:.75rem; margin-top:5px; line-height:1.45; }
.limit{ border-left:2px solid var(--border); padding-left:16px; color:var(--muted); font-size:.9rem;
  line-height:1.75; max-width:740px; }
.limit b{ color:var(--text); }
.repo{ display:inline-block; margin-top:6px; font-family:'Oswald'; letter-spacing:1.5px;
  text-transform:uppercase; font-size:.76rem; color:var(--accent); border:1px solid var(--border);
  border-radius:8px; padding:9px 16px; text-decoration:none; }
.repo:hover{ border-color:var(--accent); }

/* Scenario: champion banner + stats + bracket */
.champ{ background:linear-gradient(180deg,rgba(242,193,78,.10),var(--elev)); border:1px solid rgba(242,193,78,.4);
  border-radius:16px; padding:20px 24px; display:flex; align-items:center; gap:20px; margin:6px 0 18px; }
.champ img{ width:74px; height:50px; border-radius:6px; box-shadow:0 0 0 1px rgba(255,255,255,.12); }
.champ .cl{ font-family:'Oswald'; text-transform:uppercase; letter-spacing:3px; font-size:.72rem; color:var(--gold); }
.champ .ct{ font-family:'Oswald'; font-size:2.1rem; font-weight:700; line-height:1.05; }
.kflag{ width:22px; height:15px; border-radius:2px; object-fit:cover; box-shadow:0 0 0 1px rgba(255,255,255,.08); }
.ko-grid{ display:grid; grid-template-columns:repeat(2,1fr); gap:8px; }
.ko-grid.one{ grid-template-columns:minmax(280px,440px); justify-content:center; }
.ko-card{ display:grid; grid-template-columns:1fr auto 1fr; align-items:center; gap:8px;
  background:var(--elev); border:1px solid var(--border); border-radius:10px; padding:8px 12px; }
.ko-card.final{ border-color:rgba(242,193,78,.4); padding:12px 16px; }
.ko-a{ display:flex; align-items:center; justify-content:flex-end; gap:8px; text-align:right; min-width:0; }
.ko-b{ display:flex; align-items:center; gap:8px; min-width:0; }
.ko-nm{ font-size:.86rem; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.ko-nm.win{ color:var(--accent2); font-weight:700; }
.ko-sc{ font-family:'IBM Plex Mono'; font-weight:600; min-width:46px; text-align:center; font-size:.9rem; }
.ko-pens{ grid-column:1/-1; text-align:center; color:var(--faint); font-size:.64rem; letter-spacing:.5px; }

/* Group standings tables */
.gst-wrap{ display:grid; grid-template-columns:repeat(2,1fr); gap:18px 24px; }
.gst-h{ font-family:'Oswald'; letter-spacing:1px; font-size:.92rem; margin:0 0 6px; color:var(--text); }
.gst{ width:100%; border-collapse:collapse; }
.gst th{ font-family:'Oswald'; font-weight:500; color:var(--faint); text-transform:uppercase;
  font-size:.6rem; letter-spacing:1px; text-align:right; padding:3px 5px; border-bottom:1px solid var(--border); }
.gst th.tm{ text-align:left; }
.gst td{ padding:5px; text-align:right; border-bottom:1px solid rgba(38,49,61,.4);
  font-family:'IBM Plex Mono'; font-size:.78rem; color:var(--muted); }
.gst td.tm{ text-align:left; font-family:'Manrope'; font-weight:600; color:var(--text);
  display:flex; align-items:center; gap:8px; }
.gst td.pts{ color:var(--text); font-weight:600; }
.gst tr.qual td.pts{ color:var(--accent2); }
.gst tr.qual td.tm{ position:relative; }

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
        blocks.append(
            f'<div><div class="gst-h">Group {g}</div><table class="gst">'
            f'<tr><th class="tm">Team</th><th>P</th><th>W</th><th>D</th><th>L</th>'
            f'<th>GF</th><th>GA</th><th>GD</th><th>Pts</th></tr>'
            f'{"".join(trs)}</table></div>'
        )
    return f'<div class="gst-wrap">{"".join(blocks)}</div>'


_STEPS = [
    ("Data", "Every <b>international match since 1872</b> (≈49,000 games) is downloaded and "
             "cleaned. The official 2026 fixture, the 48 teams and the 12 groups are extracted "
             "straight from the data."),
    ("Team strength — Elo", "Each team carries an <b>Elo rating</b> (like chess) updated after "
             "every match: beating a strong side earns a lot, losing to a weak one costs a lot. "
             "Only the rating <b>before</b> a match is ever used, so the model can't peek at the future."),
    ("Match model", "A <b>calibrated logistic regression</b> turns the Elo gap (plus recent form, "
             "rest and venue) into win / draw / loss probabilities. It was benchmarked against "
             "gradient boosting and <b>won</b> — so we kept the simpler, more transparent model."),
    ("Goals model — Poisson", "An Elo-driven <b>Poisson model</b> simulates realistic scorelines. "
             "This is what gives coherent goal difference for the group-stage tie-breaks."),
    ("Monte Carlo", "The 48-team tournament is replayed <b>10,000 times</b> over the official FIFA "
             "bracket. How often each nation wins becomes its title probability — 27.7% means it "
             "won 2,770 of the 10,000 simulated tournaments."),
]

_FACTS = [
    ("fv", "10,000", "tournaments simulated"),
    ("fv", "≈49,000", "matches since 1872"),
    ("fv acc", "0.172", "out-of-sample RPS (vs 0.228 baseline)"),
    ("fv acc", "14/14", "automated tests passing"),
]


def methodology_html() -> str:
    steps = "".join(
        f'<div class="step"><div class="num">{i}</div><div><div class="st-t">{t}</div>'
        f'<div class="st-d">{d}</div></div></div>'
        for i, (t, d) in enumerate(_STEPS, 1)
    )
    facts = "".join(
        f'<div class="fact"><div class="{cls}">{v}</div><div class="fl">{lbl}</div></div>'
        for cls, v, lbl in _FACTS
    )
    return (
        '<div class="prose">This project answers one question — <b>who will win the 2026 World '
        'Cup, and with what probability?</b> — with a rigorous, reproducible pipeline rather than '
        'a hot take. The aim is not a crystal ball (football is famously unpredictable) but an '
        '<b>honest, explainable</b> forecast where every number traces back to data.</div>'
        '<div class="sec">How the forecast is built</div>'
        f'<div class="steps">{steps}</div>'
        '<div class="sec">Why you can trust it</div>'
        f'<div class="facts">{facts}</div>'
        '<div class="prose" style="margin-top:14px">'
        '<b>No look-ahead bias</b> — the model is trained on the past and tested on later, unseen '
        'matches (2022+, including the last World Cup). <b>Well-calibrated</b> — when it says "70%", '
        'that happens about 70% of the time. <b>Cross-checked</b> — two independent models (the '
        'classifier and the Poisson goals model) agree out-of-sample. <b>Robust</b> — the result '
        'barely moves when the training window or home-field assumptions are changed.</div>'
        '<div class="sec">What it can\'t do</div>'
        '<div class="limit">No player-level, squad or injury data — strength is purely team-level.<br>'
        'World Cup matches are modelled on neutral ground (no host-crowd boost).<br>'
        'It outputs <b>probabilities, not certainties</b>: a 28% favourite still loses most of the time.</div>'
        '<div class="sec">Under the hood</div>'
        '<div class="prose">Python · pandas · scikit-learn · NumPy/SciPy · Streamlit. Elo + calibrated '
        'logistic regression + Poisson goals model + Monte Carlo. Fully documented and tested.</div>'
        '<a class="repo" href="https://github.com/mgaedechens/Worldcup" target="_blank">View the code on GitHub</a>'
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
    tab_race, tab_bracket, tab_groups, tab_match, tab_how = st.tabs(
        ["Title race", "Bracket", "Groups", "Match predictor", "How it works"])

    with tab_race:
        st.markdown('<div class="lead">Each figure is the share of <b>10,000 simulated '
                    'tournaments</b> that team won. The favourite still sits below 50% — football '
                    'is high-variance. See the <b>How it works</b> tab for the full method.</div>',
                    unsafe_allow_html=True)
        st.markdown(podium_html(results), unsafe_allow_html=True)
        st.markdown('<div class="sec">Full championship odds</div>', unsafe_allow_html=True)
        n = st.slider("Teams shown", 8, 48, 20, label_visibility="collapsed")
        st.markdown(leaderboard_html(results.head(n)), unsafe_allow_html=True)

    with tab_bracket:
        st.markdown('<div class="lead">A single simulated tournament with the exact scoreline of '
                    'every match. The model is probabilistic, so this is <b>one of 10,000 possible '
                    'outcomes</b> — re-roll to draw another. For each team\'s overall chances, see '
                    'the <b>Title race</b> tab.</div>', unsafe_allow_html=True)
        if "scenario_seed" not in st.session_state:
            st.session_state.scenario_seed = 2
        if st.button("Simulate another tournament"):
            st.session_state.scenario_seed += 1
        scen = simulate_scenario(load_context(),
                                 np.random.default_rng(st.session_state.scenario_seed))
        st.markdown(champion_banner_html(scen), unsafe_allow_html=True)
        st.markdown(scenario_stats_html(scen), unsafe_allow_html=True)
        st.markdown(bracket_html(scen), unsafe_allow_html=True)
        st.markdown('<div class="sec">Group stage — final standings</div>', unsafe_allow_html=True)
        st.markdown(group_tables_html(scen), unsafe_allow_html=True)

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

    with tab_how:
        st.markdown(methodology_html(), unsafe_allow_html=True)

    st.markdown(
        '<div class="foot"><b>Methodology.</b> Elo ratings (full history) feed a calibrated logistic '
        'classifier and an Elo-driven Poisson goals model; the tournament is simulated 10,000 times '
        'over the official FIFA bracket. Probabilities, not predictions — football is high-variance.<br>'
        'Neutral venues assumed. Flags via flagcdn.com. Not affiliated with FIFA.</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
