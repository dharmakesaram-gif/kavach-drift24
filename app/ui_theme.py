"""
ui_theme.py - Mission-control look for the Space Screening dashboard.

Usage in streamlit_app.py:
    from ui_theme import apply_theme, style_fig, hero, pill
    apply_theme()                       # once, right after st.set_page_config
    st.plotly_chart(style_fig(fig), use_container_width=True)
    hero("Title", "Subtitle")           # top of the overview page
"""
import random
import streamlit as st

# ---- Tokens -----------------------------------------------------------
INK = "#070b14"        # page
PANEL = "#0d1424"      # cards
EDGE = "#1c2a44"       # hairlines
TEXT = "#e6edf7"
MUTED = "#8aa0c0"
ACCENT = "#7dd3fc"     # ice cyan: the only chrome accent
ACCEPT, REVIEW, REJECT = "#4ade80", "#fbbf24", "#f87171"
COLORWAY = [ACCENT, "#818cf8", "#5eead4", REVIEW, REJECT, ACCEPT]

_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {{
  --ink:{INK}; --panel:{PANEL}; --edge:{EDGE}; --text:{TEXT};
  --muted:{MUTED}; --accent:{ACCENT};
  --accept:{ACCEPT}; --review:{REVIEW}; --reject:{REJECT};
}}

html, body, [class*="css"], .stApp {{ font-family:'IBM Plex Sans',sans-serif; }}
.stApp {{
  background:
    radial-gradient(1200px 500px at 70% -10%, rgba(125,211,252,.07), transparent 60%),
    var(--ink);
  color:var(--text);
}}
header[data-testid="stHeader"] {{ background:transparent; }}
#MainMenu, footer {{ visibility:hidden; }}
.block-container {{ padding-top:2.2rem; max-width:1400px; }}

/* Headings: solid, so emoji stay visible */
h1,h2,h3,h4 {{
  font-family:'Space Grotesk',sans-serif !important;
  color:var(--text) !important; letter-spacing:-0.015em;
}}
h1 {{ font-weight:700; }}
h2,h3 {{ font-weight:600; }}
h3 {{ margin-top:.6rem; }}
hr {{ border-color:var(--edge) !important; margin:1.6rem 0 !important; }}
p, li {{ line-height:1.6; }}

/* Metrics: flat instrument panels */
div[data-testid="stMetric"], div[data-testid="metric-container"] {{
  background:var(--panel);
  border:1px solid var(--edge);
  border-left:3px solid var(--accent);
  border-radius:6px;
  padding:16px 18px;
}}
div[data-testid="stMetricLabel"] p {{ color:var(--muted); font-size:.82rem; }}
div[data-testid="stMetricValue"] {{
  font-family:'Space Grotesk',sans-serif; font-weight:600;
  font-size:1.9rem; font-variant-numeric:tabular-nums;
}}

/* Sidebar */
section[data-testid="stSidebar"] {{
  background:#060a12; border-right:1px solid var(--edge);
}}
section[data-testid="stSidebar"] h1 {{ font-size:1.25rem; }}
section[data-testid="stSidebar"] div[role="radiogroup"] {{ gap:2px; }}
section[data-testid="stSidebar"] div[role="radiogroup"] > label {{
  padding:7px 10px; border-radius:5px; border-left:2px solid transparent;
  transition:background .15s;
}}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {{
  background:rgba(125,211,252,.06);
}}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {{
  background:rgba(125,211,252,.10); border-left-color:var(--accent);
}}
section[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child {{ display:none; }}

/* Tabs */
button[data-baseweb="tab"] {{ font-family:'Space Grotesk',sans-serif; font-weight:500; }}
div[data-baseweb="tab-highlight"] {{ background:var(--accent) !important; }}

/* Buttons */
.stButton > button, .stDownloadButton > button, a[data-testid="stBaseLinkButton-secondary"] {{
  background:var(--panel); border:1px solid var(--edge); color:var(--text);
  border-radius:6px; font-weight:500; transition:border-color .15s, background .15s;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{
  border-color:var(--accent); background:rgba(125,211,252,.08); color:var(--text);
}}
.stButton > button[kind="primary"] {{ background:var(--accent); color:#04121c; border-color:var(--accent); }}

/* Alerts, tables, expanders */
div[data-testid="stAlert"] {{ border-radius:6px; border:1px solid var(--edge); }}
div[data-testid="stDataFrame"] {{
  border:1px solid var(--edge); border-radius:6px; overflow:hidden;
}}
details {{ border:1px solid var(--edge) !important; border-radius:6px !important; background:var(--panel); }}

/* Plotly container */
div[data-testid="stPlotlyChart"] {{
  background:var(--panel); border:1px solid var(--edge); border-radius:6px; padding:6px;
}}

/* Disposition text + report box (kept for existing markup) */
.status-accept {{ color:var(--accept); font-weight:600; }}
.status-review {{ color:var(--review); font-weight:600; }}
.status-reject {{ color:var(--reject); font-weight:600; }}
.report-box {{
  background:var(--panel); border:1px solid var(--edge);
  border-left:3px solid var(--accent); border-radius:6px; padding:22px;
}}
.pill {{
  display:inline-block; padding:2px 10px; border-radius:999px; font-size:.78rem;
  font-weight:600; border:1px solid currentColor;
}}
.pill.accept {{ color:var(--accept); background:rgba(74,222,128,.10); }}
.pill.review {{ color:var(--review); background:rgba(251,191,36,.10); }}
.pill.reject {{ color:var(--reject); background:rgba(248,113,113,.10); }}

/* Hero */
.hero {{
  display:grid; grid-template-columns:minmax(0,5fr) minmax(0,6fr); gap:32px; align-items:center;
  border:1px solid var(--edge); border-radius:8px; padding:32px 36px; margin-bottom:24px;
  background:linear-gradient(160deg, rgba(125,211,252,.05), transparent 55%), var(--panel);
}}
.hero h1 {{ font-size:2.6rem; line-height:1.08; margin:0 0 12px 0; }}
.hero p.sub {{ color:var(--muted); font-size:1.05rem; margin:0 0 18px 0; max-width:46ch; }}
.hero .facts {{ display:flex; gap:22px; flex-wrap:wrap; color:var(--muted); font-size:.9rem; }}
.hero .facts b {{ color:var(--text); font-family:'Space Grotesk',sans-serif; font-size:1.25rem; display:block; }}
.hero svg {{ width:100%; height:auto; }}
.hero .pulse {{ transform-origin:center; transform-box:fill-box; animation:ping 2.4s ease-out infinite; }}
@keyframes ping {{ 0%{{transform:scale(1);opacity:.7}} 100%{{transform:scale(3.2);opacity:0}} }}
@media (max-width:900px) {{ .hero {{ grid-template-columns:1fr; padding:22px; }} .hero h1 {{ font-size:2rem; }} }}
@media (prefers-reduced-motion:reduce) {{ .hero .pulse {{ animation:none; opacity:.35; }} }}
</style>
"""


def apply_theme():
    st.markdown(_CSS, unsafe_allow_html=True)


def pill(decision: str) -> str:
    """Return HTML for a coloured ACCEPT / REVIEW / REJECT badge."""
    d = str(decision).upper()
    return f'<span class="pill {d.lower()}">{d}</span>'


def _hero_svg() -> str:
    """The problem statement in one picture: lot cluster, static limit, dynamic limit, one outlier."""
    rnd = random.Random(7)
    x0, x1, w = 30, 610, 640           # axis 0..60 uA mapped to x0..x1
    sx = lambda v: x0 + (v / 60.0) * (x1 - x0)
    dots = "".join(
        f'<circle cx="{sx(max(2, rnd.gauss(10, 1.6))):.1f}" cy="{rnd.uniform(70, 150):.1f}" r="3.2" fill="{ACCENT}" opacity=".75"/>'
        for _ in range(90)
    )
    static_x, dyn_x, out_x = sx(50), sx(16), sx(45)
    return f"""
<svg viewBox="0 0 {w} 230" role="img" aria-label="A 45 microamp part passes the 50 microamp static limit but sits far outside the lot distribution">
  <line x1="{x0}" y1="190" x2="{x1}" y2="190" stroke="{EDGE}" stroke-width="1"/>
  <g fill="{MUTED}" font-size="11" font-family="IBM Plex Sans">
    <text x="{sx(0)}" y="208" text-anchor="middle">0</text>
    <text x="{sx(20)}" y="208" text-anchor="middle">20</text>
    <text x="{sx(40)}" y="208" text-anchor="middle">40</text>
    <text x="{sx(60)}" y="208" text-anchor="middle">60 µA</text>
  </g>
  {dots}
  <line x1="{dyn_x}" y1="40" x2="{dyn_x}" y2="190" stroke="{ACCEPT}" stroke-width="1.5" stroke-dasharray="4 4"/>
  <text x="{dyn_x+6}" y="52" fill="{ACCEPT}" font-size="12" font-family="IBM Plex Sans">Dynamic limit</text>
  <line x1="{static_x}" y1="40" x2="{static_x}" y2="190" stroke="{MUTED}" stroke-width="1.5" stroke-dasharray="4 4"/>
  <text x="{static_x-6}" y="52" fill="{MUTED}" font-size="12" font-family="IBM Plex Sans" text-anchor="end">Static limit 50</text>
  <circle class="pulse" cx="{out_x}" cy="110" r="6" fill="{REJECT}"/>
  <circle cx="{out_x}" cy="110" r="6" fill="{REJECT}"/>
  <text x="{out_x-12}" y="106" fill="{REJECT}" font-size="12" font-weight="600" font-family="IBM Plex Sans" text-anchor="end">45 µA: passes static, rejected here</text>
</svg>"""


def hero(title: str, subtitle: str):
    st.markdown(
        f"""
<div class="hero">
  <div>
    <h1>{title}</h1>
    <p class="sub">{subtitle}</p>
    <div class="facts">
      <div><b>4.5×</b>lot median, yet under datasheet limit</div>
      <div><b>8.1σ</b>dynamic PAT score</div>
      <div><b>24 h</b>to reject, not 168 h</div>
    </div>
  </div>
  <div>{_hero_svg()}</div>
</div>""",
        unsafe_allow_html=True,
    )


def style_fig(fig, height=None):
    """Apply the dashboard look to any Plotly figure. Per-trace colours you set stay untouched."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Sans, sans-serif", color="#cbd5e1", size=13),
        title_font=dict(family="Space Grotesk, sans-serif", size=18, color=TEXT),
        colorway=COLORWAY,
        margin=dict(l=24, r=24, t=64, b=24),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=12)),
        hoverlabel=dict(bgcolor=PANEL, bordercolor=EDGE, font=dict(family="IBM Plex Sans")),
    )
    fig.update_xaxes(gridcolor="rgba(138,160,192,.10)", zerolinecolor=EDGE, linecolor=EDGE)
    fig.update_yaxes(gridcolor="rgba(138,160,192,.10)", zerolinecolor=EDGE, linecolor=EDGE)
    if height:
        fig.update_layout(height=height)
    return fig
