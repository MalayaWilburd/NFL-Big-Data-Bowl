"""
NFL Big Data Bowl - Player Win Impact Dashboard (v3)
=====================================================

Enhanced Dash app with coach view, player comparison, search/highlight,
fixed stat cards, improved radar, and consistent slider layout.

Run with:  python src/dashboard.py
Open:      http://127.0.0.1:8050
"""

import os
import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

# --------------------------------------------------------------------------- #
# Paths & data loading
# --------------------------------------------------------------------------- #
DERIVED = os.path.join(os.path.dirname(__file__), "..", "data", "derived")

game_outcomes  = pd.read_csv(os.path.join(DERIVED, "game_outcomes.csv"))
player_game    = pd.read_csv(os.path.join(DERIVED, "player_game_stats.csv"))
player_season  = pd.read_csv(os.path.join(DERIVED, "player_season_stats.csv"))
play_epa       = pd.read_csv(os.path.join(DERIVED, "play_epa.csv"))
win_features   = pd.read_csv(os.path.join(DERIVED, "win_impact_features.csv"))

# ---- Data enrichment ------------------------------------------------------- #

# Add team to player_season for display (most common team per player)
_team_map = (
    player_game.groupby("nflId")["team"]
    .agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0] if len(x) > 0 else "")
    .reset_index()
)
_team_map.columns = ["nflId", "team"]
player_season = player_season.merge(_team_map, on="nflId", how="left")

# Add position group
POS_GROUP_MAP = {}
for p in ["QB"]:
    POS_GROUP_MAP[p] = "QB"
for p in ["RB", "FB"]:
    POS_GROUP_MAP[p] = "RB"
for p in ["WR"]:
    POS_GROUP_MAP[p] = "WR"
for p in ["TE"]:
    POS_GROUP_MAP[p] = "TE"
for p in ["C", "G", "T"]:
    POS_GROUP_MAP[p] = "OL"
for p in ["DE", "DT", "NT"]:
    POS_GROUP_MAP[p] = "DL"
for p in ["LB", "MLB", "OLB", "ILB"]:
    POS_GROUP_MAP[p] = "LB"
for p in ["CB", "S", "SS", "FS", "DB"]:
    POS_GROUP_MAP[p] = "DB"

player_season["posGroup"] = player_season["officialPosition"].map(POS_GROUP_MAP).fillna("Other")
player_game["posGroup"]   = player_game["officialPosition"].map(POS_GROUP_MAP).fillna("Other")

# Percentile ranks within position for stat cards
for stat in ["totalEpa", "epaPerSnap", "winRate"]:
    player_season[f"{stat}_pct"] = player_season.groupby("officialPosition")[stat].rank(pct=True)
    player_season[f"{stat}_rank"] = player_season.groupby("officialPosition")[stat].rank(ascending=False).astype(int)
player_season["n_at_pos"] = player_season.groupby("officialPosition")["nflId"].transform("count")

# Enrich player_game with opponent + week for game labels
pg_enriched = player_game.merge(
    game_outcomes[["gameId", "week", "homeTeamAbbr", "visitorTeamAbbr"]],
    on="gameId", how="left",
)
pg_enriched["opponent"] = np.where(
    pg_enriched["team"] == pg_enriched["homeTeamAbbr"],
    pg_enriched["visitorTeamAbbr"],
    np.where(
        pg_enriched["team"] == pg_enriched["visitorTeamAbbr"],
        pg_enriched["homeTeamAbbr"],
        "???",
    ),
)
pg_enriched["gameLabel"] = pg_enriched.apply(
    lambda r: f"Wk {int(r['week'])} vs {r['opponent']}"
    if pd.notna(r["week"]) else f"Game {r['gameId']}",
    axis=1,
)

# Team records
_team_rows = []
for _, r in game_outcomes.iterrows():
    _team_rows.append({"team": r["homeTeamAbbr"], "won": int(r["homeWin"] == 1)})
    _team_rows.append({"team": r["visitorTeamAbbr"], "won": int(r["homeWin"] == 0)})
_team_df = pd.DataFrame(_team_rows)
team_summary = (
    _team_df.groupby("team")
    .agg(games=("team", "count"), wins=("won", "sum"))
    .reset_index()
)
team_summary["losses"] = team_summary["games"] - team_summary["wins"]
team_summary["winRate"] = team_summary["wins"] / team_summary["games"]
team_summary["record"] = (
    team_summary["wins"].astype(int).astype(str) + "-"
    + team_summary["losses"].astype(int).astype(str)
)
team_summary = team_summary.sort_values("winRate", ascending=False)

# ---- Win-impact logistic regression --------------------------------------- #
_WIN_FEATURES = ["totalEpa", "avgEpa", "snaps", "sacks", "hurries", "hits",
                 "beaten", "hitsAllowed", "hurriesAllowed", "sacksAllowed"]
_win_df = player_game.dropna(subset=_WIN_FEATURES + ["teamWon"]).copy()
_win_df = _win_df[_win_df["snaps"] >= 10]
_win_X = _win_df[_WIN_FEATURES].fillna(0)
_win_y = _win_df["teamWon"].astype(int)
_win_scaler = StandardScaler()
_win_X_s = _win_scaler.fit_transform(_win_X)
_win_model = LogisticRegression(max_iter=1000, random_state=42)
_win_model.fit(_win_X_s, _win_y)
_win_acc = _win_model.score(_win_X_s, _win_y)
_win_auc = roc_auc_score(_win_y, _win_model.predict_proba(_win_X_s)[:, 1])

# Baseline win probability (at league-median values) for delta display
_baseline_row = pd.DataFrame(
    [[float(player_game[f].median()) for f in _WIN_FEATURES]],
    columns=_WIN_FEATURES,
)
_baseline_proba = float(
    _win_model.predict_proba(_win_scaler.transform(_baseline_row))[0, 1]
)

# Slider specs
_slider_specs = []
for feat, label, short in [
    ("avgEpa",         "Avg Impact per Play",    "Avg EPA/Play"),
    ("totalEpa",       "Total Impact (Game)",    "Total EPA"),
    ("hurries",        "Pressures Generated",     "Hurries"),
    ("hurriesAllowed", "Pressures Allowed",       "Hurries Allowed"),
    ("beaten",         "Times Beaten",            "Beaten"),
]:
    if feat in player_game.columns:
        col = player_game[feat].dropna()
        _slider_specs.append({
            "key": feat, "label": label, "short": short,
            "min": round(float(np.percentile(col, 5)), 2),
            "max": round(float(np.percentile(col, 95)), 2),
            "med": round(float(col.median()), 2),
            "step": max(round(float(col.std() / 10), 3), 0.01),
        })

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
BG       = "#0d1117"
CARD_BG  = "#161b22"
TEXT     = "#e6edf3"
ACCENT   = "#58a6ff"   # blue
ACCENT2  = "#3fb950"   # green = good
ACCENT3  = "#f78166"   # orange = bad
YELLOW   = "#d29922"   # gold = borderline
GRID     = "#30363d"

TEAM_COLORS = {
    "ARI": "#97233F", "ATL": "#A71930", "BAL": "#241773", "BUF": "#00338D",
    "CAR": "#0085CA", "CHI": "#0B162A", "CIN": "#FB4F14", "CLE": "#FF3C00",
    "DAL": "#003594", "DEN": "#FB4F14", "DET": "#0076B6", "GB": "#203731",
    "HOU": "#03202F", "IND": "#002C5F", "JAX": "#101820", "KC": "#E31837",
    "LV": "#000000", "LAC": "#0080C5", "LAR": "#003594", "MIA": "#008E97",
    "MIN": "#4F2683", "NE": "#002244", "NO": "#D3BC8D", "NYG": "#0B2265",
    "NYJ": "#125740", "PHI": "#004C54", "PIT": "#FFB612", "SF": "#AA0000",
    "SEA": "#002244", "TB": "#D50A0A", "TEN": "#0C2340", "WAS": "#5A1414",
}

FEATURE_LABELS = {
    "avgEpa": "Avg Impact per Play",
    "totalEpa": "Total Impact",
    "hurriesAllowed": "Pressures Allowed",
    "beaten": "Times Beaten",
    "hurries": "Pressures Generated",
    "hitsAllowed": "Hits Allowed",
    "sacksAllowed": "Sacks Allowed",
    "hits": "Times Hit",
    "sacks": "Sacks",
    "snaps": "Snaps Played",
}

RADAR_AXES = {
    "sacks": "Sacks",
    "hurries": "Hurries",
    "hits": "Times Hit",
    "beaten": "Times Beaten",
    "hitsAllowed": "Hits Allowed",
    "hurriesAllowed": "Hurries Allowed",
    "sacksAllowed": "Sacks Taken",
}
INVERTED_STATS = {"hitsAllowed", "hurriesAllowed", "sacksAllowed", "beaten"}

CHART_LAYOUT = dict(
    paper_bgcolor=BG, plot_bgcolor=BG,
    font=dict(color=TEXT, family="Segoe UI, sans-serif", size=13),
    margin=dict(l=40, r=60, t=50, b=40),
)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def card(title, children):
    return html.Div(
        style=dict(backgroundColor=CARD_BG, borderRadius=12,
                   padding="20px 24px", marginBottom=16,
                   border=f"1px solid {GRID}"),
        children=[
            html.H3(title, style=dict(color=TEXT, fontSize=16, fontWeight=600,
                                      marginBottom=12)),
            *children,
        ],
    )

def stat_card(label, value, subtitle=None, color=ACCENT):
    children = [
        html.Div(label, style=dict(color="#8b949e", fontSize=12, marginBottom=6)),
        html.Div(value, style=dict(color=color, fontSize=26, fontWeight=700)),
    ]
    if subtitle:
        children.append(
            html.Div(subtitle, style=dict(color="#8b949e", fontSize=10, marginTop=4))
        )
    return html.Div(
        style=dict(backgroundColor=CARD_BG, borderRadius=12, padding="16px 20px",
                   textAlign="center", border=f"1px solid {GRID}"),
        children=children,
    )

def color_for(value, baseline):
    return ACCENT2 if value >= baseline else ACCENT3

def winprob_color(proba):
    """3-zone color: red below 45%, gold 45-50%, green above 50%."""
    if proba < 0.45:
        return ACCENT3
    elif proba < 0.50:
        return YELLOW
    else:
        return ACCENT2

# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
app = dash.Dash(__name__, title="NFL Player Win Impact Dashboard")
app.config.suppress_callback_exceptions = True
server = app.server  # expose Flask server for gunicorn / Render

player_options = sorted(player_season["displayName"].dropna().unique())

POS_FILTER_OPTS = [
    {"label": "All Positions", "value": "All"},
    {"label": "QB", "value": "QB"},
    {"label": "RB", "value": "RB"},
    {"label": "WR", "value": "WR"},
    {"label": "TE", "value": "TE"},
    {"label": "OL (C, G, T)", "value": "OL"},
    {"label": "DL (DE, DT, NT)", "value": "DL"},
    {"label": "LB", "value": "LB"},
    {"label": "DB (CB, S)", "value": "DB"},
]

app.layout = html.Div(
    style=dict(backgroundColor=BG, minHeight="100vh", color=TEXT,
               fontFamily="Segoe UI, sans-serif"),
    children=[
        dcc.Store(id="coach-view-store", data=False),

        # Header
        html.Div(
            style=dict(padding="24px 32px 16px", borderBottom=f"1px solid {GRID}",
                       display="flex", justifyContent="space-between",
                       alignItems="center"),
            children=[
                html.Div(children=[
                    html.H1("NFL Player Win Impact Dashboard",
                            style=dict(fontSize=24, fontWeight=700, marginBottom=4)),
                    html.P("EPA-based analysis of how individual player performance "
                           "correlates with game outcomes — 2021 season tracking data",
                           style=dict(color="#8b949e", fontSize=13)),
                ]),
                dcc.Checklist(
                    id="coach-view-toggle",
                    options=[{"label": " Coach View", "value": "coach"}],
                    value=[],
                    labelStyle=dict(color=TEXT, fontSize=14, cursor="pointer"),
                    inputStyle=dict(marginRight=6, cursor="pointer",
                                    accentColor=ACCENT2),
                ),
            ],
        ),

        dcc.Tabs(id="tabs", value="overview", style=dict(
            backgroundColor=BG, color="#8b949e", fontWeight=600,
        ), colors=dict(border=GRID, primary=ACCENT), children=[
            dcc.Tab(label="Overview",          value="overview"),
            dcc.Tab(label="Player Impact",     value="player"),
            dcc.Tab(label="Position Analysis",  value="position"),
            dcc.Tab(label="Win Impact Model",   value="winmodel"),
        ]),

        html.Div(id="tab-content", style=dict(padding="24px 32px")),
    ],
)


# --------------------------------------------------------------------------- #
# Coach-view toggle → store
# --------------------------------------------------------------------------- #
@app.callback(Output("coach-view-store", "data"), Input("coach-view-toggle", "value"))
def toggle_coach_view(value):
    return "coach" in (value or [])


# --------------------------------------------------------------------------- #
# Tab routing
# --------------------------------------------------------------------------- #
@app.callback(Output("tab-content", "children"),
              [Input("tabs", "value"), Input("coach-view-store", "data")])
def render_tab(tab, coach_view):
    coach_view = bool(coach_view)
    if tab == "overview":
        return overview_tab(coach_view)
    elif tab == "player":
        return player_tab(coach_view)
    elif tab == "position":
        return position_tab(coach_view)
    elif tab == "winmodel":
        return winmodel_tab(coach_view)


# ============================================================================ #
# TAB: OVERVIEW
# ============================================================================ #
def overview_tab(coach_view):
    n_games   = len(game_outcomes)
    n_plays   = len(play_epa)
    n_players = len(player_season)
    home_wr   = game_outcomes["homeWin"].mean()

    fig_top = make_top_epa_figure("All", coach_view)

    # Team win rates — all 32 teams with records + 50% line, full width
    tc = [TEAM_COLORS.get(t, ACCENT) for t in team_summary["team"]]
    fig_wr = go.Figure(go.Bar(
        x=team_summary["team"], y=team_summary["winRate"],
        marker_color=tc, text=team_summary["record"],
        textposition="outside", textfont=dict(size=10, color="#8b949e"),
    ))
    fig_wr.update_layout(**CHART_LAYOUT, height=420,
        xaxis=dict(gridcolor=GRID, tickangle=-45, title=""),
        yaxis=dict(gridcolor=GRID, tickformat=".0%", range=[0, 1],
                   title="Win %" if coach_view else "Win Rate"),
    )
    fig_wr.add_hline(y=0.5, line_dash="dash", line_color=TEXT, opacity=0.4,
                     annotation_text="50% baseline", annotation_font_size=10,
                     annotation_font_color="#8b949e")
    fig_wr.update_traces(marker_line_width=0, opacity=0.9)

    # EPA distribution with threshold lines (non-overlapping annotations)
    epa_label = "Impact per Play" if coach_view else "EPA per Play"
    fig_dist = px.histogram(play_epa, x="epa", nbins=80,
                            color_discrete_sequence=[ACCENT2],
                            labels=dict(epa=epa_label))
    fig_dist.update_layout(**CHART_LAYOUT, height=300)
    fig_dist.update_traces(marker_line_width=0, opacity=0.8)
    fig_dist.update_xaxes(gridcolor=GRID, zerolinecolor=ACCENT3, title=epa_label)
    fig_dist.update_yaxes(gridcolor=GRID)
    # Threshold lines without inline annotations
    fig_dist.add_vline(x=3, line_dash="dash", line_color=ACCENT2, opacity=0.6)
    fig_dist.add_vline(x=-1, line_dash="dash", line_color=ACCENT3, opacity=0.6)
    # Separate annotations at top corners to avoid overlap
    fig_dist.add_annotation(x=3, y=0.95, yref="paper", text="Elite (+3)",
                            font=dict(size=10, color=ACCENT2),
                            showarrow=False, xanchor="left")
    fig_dist.add_annotation(x=-1, y=0.95, yref="paper", text="Negative (−1)",
                            font=dict(size=10, color=ACCENT3),
                            showarrow=False, xanchor="right")

    title_top = "Top 20 Most Impactful Players" if coach_view else "Top 20 Players by Total EPA"
    title_wr  = "Team Records" if coach_view else "Team Win Rates"
    title_dist = "Play Impact Distribution" if coach_view else "EPA Distribution"
    dist_subtitle = ("Most plays have near-zero EPA — big plays on either end are rare"
                     if coach_view else "")

    return [
        html.Div(style=dict(display="flex", gap=16, marginBottom=24), children=[
            stat_card("Games", f"{n_games}"),
            stat_card("Plays Analyzed", f"{n_plays:,}"),
            stat_card("Players", f"{n_players:,}"),
            stat_card("Home Win Rate", f"{home_wr:.1%}",
                      subtitle=f"League baseline: 50.0%",
                      color=ACCENT2),
        ]),
        card(title_top, [
            html.Div([
                html.Span("Filter: ", style=dict(color="#8b949e", fontSize=12)),
                dcc.Dropdown(
                    id="overview-pos-filter",
                    options=POS_FILTER_OPTS,
                    value="All",
                    clearable=False,
                    style=dict(width=220, display="inline-block",
                               color="#0d1117", backgroundColor=CARD_BG),
                ),
            ], style=dict(marginBottom=12)),
            dcc.Graph(id="overview-top-epa", figure=fig_top),
        ]),
        card(title_wr, [
            html.P("All 32 teams — 2021 season sample (Weeks 1-8)",
                   style=dict(color="#8b949e", fontSize=11, marginBottom=8)),
            dcc.Graph(figure=fig_wr),
        ]),
        card(title_dist, [
            html.P(dist_subtitle, style=dict(color="#8b949e",
                    fontSize=11, marginBottom=8)) if dist_subtitle else html.Div(),
            dcc.Graph(figure=fig_dist),
        ]),
    ]


def make_top_epa_figure(pos_filter, coach_view):
    df = player_season.copy()
    if pos_filter and pos_filter != "All":
        df = df[df["posGroup"] == pos_filter]
    top = df.nlargest(20, "totalEpa").sort_values("totalEpa", ascending=True)

    epa_lbl = "Total Impact" if coach_view else "Total EPA"
    top["displayLabel"] = top["displayName"] + " (" + top["team"].fillna("?") + ")"
    top["hoverEpaPerSnap"] = top["epaPerSnap"].round(3)

    fig = px.bar(
        top, x="totalEpa", y="displayLabel", orientation="h",
        color="officialPosition",
        color_discrete_sequence=px.colors.qualitative.Set2,
        custom_data=["hoverEpaPerSnap"],
        labels=dict(totalEpa=epa_lbl, displayLabel=""),
    )
    # Increase right margin + extend x-axis range to prevent label clipping
    fig.update_layout(**{**CHART_LAYOUT, "margin": dict(l=40, r=80, t=50, b=40)},
                      height=520, legend=dict(bgcolor=CARD_BG, title="Position"))
    x_max = top["totalEpa"].max()
    x_min = min(0, top["totalEpa"].min())
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, title=epa_lbl,
                    range=[x_min - 2, x_max * 1.18])
    fig.update_yaxes(gridcolor=GRID)
    fig.update_traces(
        marker_line_width=0,
        texttemplate=[f"{v:+.1f}" for v in top["totalEpa"]],
        textposition="outside",
        textfont=dict(size=10, color="#8b949e"),
    )
    fig.update_traces(
        hovertemplate="<b>%{y}</b><br>" + epa_lbl + ": %{x:+.1f}<br>"
                      "EPA/Snap: %{customdata[0]}<extra></extra>"
    )
    return fig


@app.callback(
    Output("overview-top-epa", "figure"),
    [Input("overview-pos-filter", "value"), Input("coach-view-store", "data")],
)
def update_top_epa(pos_filter, coach_view):
    if pos_filter is None:
        pos_filter = "All"
    return make_top_epa_figure(pos_filter, bool(coach_view))


# ============================================================================ #
# TAB: PLAYER IMPACT  (with comparison mode)
# ============================================================================ #
def player_tab(coach_view):
    return [
        card("Select Players", [
            html.Div(style=dict(display="flex", gap=12), children=[
                html.Div(style=dict(flex=1), children=[
                    html.Label("Player",
                               style=dict(color="#8b949e", fontSize=12, marginBottom=4, display="block")),
                    dcc.Dropdown(
                        id="player-dropdown",
                        options=[{"label": p, "value": p} for p in player_options],
                        value=player_season.loc[
                            player_season["totalEpa"].idxmax(), "displayName"
                        ],
                        style=dict(color="#0d1117", backgroundColor=CARD_BG),
                        clearable=False,
                    ),
                ]),
                html.Div(style=dict(flex=1), children=[
                    html.Label("Compare with (optional)",
                               style=dict(color="#8b949e", fontSize=12, marginBottom=4, display="block")),
                    dcc.Dropdown(
                        id="compare-dropdown",
                        options=[{"label": "— None —", "value": "none"}]
                                + [{"label": p, "value": p} for p in player_options],
                        value="none",
                        style=dict(color="#0d1117", backgroundColor=CARD_BG),
                        clearable=False,
                    ),
                ]),
            ]),
        ]),
        html.Div(id="player-output"),
    ]


def make_player_cards(name, coach_view):
    """Build stat cards for a single player with percentile context."""
    season = player_season[player_season["displayName"] == name]
    if season.empty:
        return [html.P("Not found", style=dict(color=TEXT))]
    s = season.iloc[0]
    pos = s["officialPosition"]
    wr  = s["winRate"]
    epa = s["totalEpa"]
    eps = s["epaPerSnap"]
    snaps_per_game = int(s["totalSnaps"]) / max(1, int(s["games"]))
    gms = int(s["games"])

    pos_df = player_season[player_season["officialPosition"] == pos]
    pos_avg_epa = pos_df["totalEpa"].mean()
    pos_avg_wr  = pos_df["winRate"].mean()
    pos_avg_eps = pos_df["epaPerSnap"].mean()

    def pct_label(pct, rank, n):
        if n == 0:
            return ""
        return f"#{rank} of {n} {pos}s · Top {100 - int(pct * 100)}%"

    epa_lbl  = "Total Impact" if coach_view else "Total EPA"
    wr_lbl   = "Win %" if coach_view else "Win Rate"
    eps_lbl  = "Impact / Play" if coach_view else "EPA / Snap"
    snap_lbl = "Snaps / Game" if coach_view else "Snaps per Game"

    return [
        stat_card("Position", pos or "—"),
        stat_card("Games", f"{gms}"),
        stat_card(snap_lbl, f"{snaps_per_game:.0f}"),
        stat_card(epa_lbl, f"{epa:+.1f}",
                  subtitle=pct_label(s.get("totalEpa_pct", 0.5),
                                     int(s.get("totalEpa_rank", 0)),
                                     int(s.get("n_at_pos", 0))),
                  color=color_for(epa, pos_avg_epa)),
        stat_card(wr_lbl, f"{wr:.0%}",
                  subtitle=pct_label(s.get("winRate_pct", 0.5),
                                     int(s.get("winRate_rank", 0)),
                                     int(s.get("n_at_pos", 0))),
                  color=color_for(wr, pos_avg_wr)),
        stat_card(eps_lbl, f"{eps:+.3f}",
                  subtitle=pct_label(s.get("epaPerSnap_pct", 0.5),
                                     int(s.get("epaPerSnap_rank", 0)),
                                     int(s.get("n_at_pos", 0))),
                  color=color_for(eps, pos_avg_eps)),
    ]


def make_epa_chart(name, coach_view):
    """Build the EPA per game bar chart for a player."""
    pg = pg_enriched[pg_enriched["displayName"] == name].copy()
    pg = pg.sort_values("gameId")

    fig = go.Figure()
    for won, color, suffix in [(1, ACCENT2, "Win"), (0, ACCENT3, "Loss")]:
        subset = pg[pg["teamWon"] == won]
        if len(subset) == 0:
            continue
        fig.add_trace(go.Bar(
            x=subset["gameLabel"], y=subset["totalEpa"],
            marker_color=color, name=suffix,
            text=[f"{v:+.1f}" for v in subset["totalEpa"]],
            textposition="outside", textfont=dict(size=10, color="#8b949e"),
        ))
    avg_epa = pg["totalEpa"].mean()
    fig.add_hline(y=avg_epa, line_dash="dot", line_color=TEXT, opacity=0.5,
                 annotation_text=f"Avg: {avg_epa:+.1f}",
                 annotation_font_size=10, annotation_font_color="#8b949e")
    fig.update_layout(**CHART_LAYOUT, height=360, barmode="relative",
        xaxis=dict(gridcolor=GRID, tickangle=-35, type="category", title=""),
        yaxis=dict(gridcolor=GRID, zerolinecolor=ACCENT3,
                   title="Impact" if coach_view else "EPA"),
        legend=dict(bgcolor=CARD_BG, title="Result"),
    )
    fig.update_traces(marker_line_width=0)
    return fig


def make_radar_data(name):
    """Return (scout_cols, player_vals, pos, avg_vals) for a player."""
    pg = player_game[player_game["displayName"] == name]
    season = player_season[player_season["displayName"] == name]
    pos = season.iloc[0]["officialPosition"] if not season.empty else "?"

    scout_cols = ["sacks", "hurries", "hits", "beaten",
                 "hitsAllowed", "hurriesAllowed", "sacksAllowed"]
    scout_cols = [c for c in scout_cols if c in pg.columns]

    player_vals = [pg[c].sum() for c in scout_cols]
    pos_avg = player_game[player_game["officialPosition"] == pos]
    avg_vals = [pos_avg[c].sum() / max(1, pos_avg["nflId"].nunique())
               for c in scout_cols]
    return scout_cols, player_vals, pos, avg_vals


def _radar_traces(norm_vals, axis_labels, name, color, fill_alpha, line_width, dash=None):
    """Build a single Scatterpolar trace for the radar."""
    kwargs = dict(
        r=norm_vals + [norm_vals[0]],
        theta=axis_labels + [axis_labels[0]],
        fill="toself", name=name,
        line=dict(color=color, width=line_width, dash=dash) if dash else dict(color=color, width=line_width),
        fillcolor=color.replace(")", f", {fill_alpha})").replace("rgb", "rgba")
                       if color.startswith("rgb") else
                       f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},{fill_alpha})",
    )
    return go.Scatterpolar(**kwargs)


def make_radar_figure(name, coach_view):
    """Single-player radar with position average — bold player, visible avg."""
    scout_cols, player_vals, pos, avg_vals = make_radar_data(name)

    # Flip inverted stats
    all_raw = player_vals + avg_vals
    max_val = max(all_raw) if max(all_raw) > 0 else 1
    player_flip = [(max_val - v) if c in INVERTED_STATS else v
                   for v, c in zip(player_vals, scout_cols)]
    avg_flip    = [(max_val - v) if c in INVERTED_STATS else v
                   for v, c in zip(avg_vals, scout_cols)]

    all_flip = player_flip + avg_flip
    norm_max = max(all_flip) if max(all_flip) > 0 else 1
    player_norm = [v / norm_max for v in player_flip]
    avg_norm   = [v / norm_max for v in avg_flip]

    # Axis labels with inversion indicators
    axis_labels = [f"{RADAR_AXES.get(c, c)} {'↓' if c in INVERTED_STATS else '↑'}"
                   for c in scout_cols]

    fig = go.Figure()
    # Player: bold blue fill
    fig.add_trace(go.Scatterpolar(
        r=player_norm + [player_norm[0]],
        theta=axis_labels + [axis_labels[0]],
        fill="toself", name=name,
        line=dict(color=ACCENT, width=2.5),
        fillcolor="rgba(88,166,255,0.25)",
        hovertext=[f"{RADAR_AXES.get(c,c)}: {v:.1f}"
                   for c, v in zip(scout_cols, player_vals)],
        hoverinfo="text",
    ))
    # Position avg: visible orange fill behind
    fig.add_trace(go.Scatterpolar(
        r=avg_norm + [avg_norm[0]],
        theta=axis_labels + [axis_labels[0]],
        fill="toself", name=f"Avg {pos}", opacity=0.35,
        line=dict(color=ACCENT3, width=2.5, dash="dash"),
        fillcolor="rgba(247,129,102,0.20)",
        hovertext=[f"Avg {RADAR_AXES.get(c,c)}: {v:.1f}"
                   for c, v in zip(scout_cols, avg_vals)],
        hoverinfo="text",
    ))
    fig.update_layout(**CHART_LAYOUT, height=480,
        polar=dict(
            bgcolor=BG,
            radialaxis=dict(gridcolor=GRID, linecolor=GRID, color="#8b949e",
                            range=[0, 1.1], tickvals=[]),
            angularaxis=dict(gridcolor=GRID, linecolor=GRID, color="#8b949e",
                             tickfont=dict(size=12)),
        ),
        legend=dict(bgcolor=CARD_BG, orientation="h", y=-0.08, x=0.3, font=dict(size=12)),
    )
    return fig, pos


def make_comparison_radar(name1, name2, coach_view):
    """Overlay two players on one radar chart."""
    cols1, vals1, pos1, _ = make_radar_data(name1)
    cols2, vals2, pos2, _ = make_radar_data(name2)
    scout_cols = cols1

    # Flip inverted stats
    all_raw = vals1 + vals2
    max_val = max(all_raw) if max(all_raw) > 0 else 1
    flip1 = [(max_val - v) if c in INVERTED_STATS else v for v, c in zip(vals1, scout_cols)]
    flip2 = [(max_val - v) if c in INVERTED_STATS else v for v, c in zip(vals2, scout_cols)]

    all_flip = flip1 + flip2
    norm_max = max(all_flip) if max(all_flip) > 0 else 1
    norm1 = [v / norm_max for v in flip1]
    norm2 = [v / norm_max for v in flip2]

    axis_labels = [f"{RADAR_AXES.get(c, c)} {'↓' if c in INVERTED_STATS else '↑'}"
                   for c in scout_cols]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=norm1 + [norm1[0]], theta=axis_labels + [axis_labels[0]],
        fill="toself", name=f"{name1} ({pos1})",
        line=dict(color=ACCENT, width=2.5),
        fillcolor="rgba(88,166,255,0.20)",
    ))
    fig.add_trace(go.Scatterpolar(
        r=norm2 + [norm2[0]], theta=axis_labels + [axis_labels[0]],
        fill="toself", name=f"{name2} ({pos2})",
        line=dict(color=ACCENT3, width=2.5),
        fillcolor="rgba(247,129,102,0.15)",
    ))
    fig.update_layout(**CHART_LAYOUT, height=480,
        polar=dict(
            bgcolor=BG,
            radialaxis=dict(gridcolor=GRID, linecolor=GRID, color="#8b949e",
                            range=[0, 1.1], tickvals=[]),
            angularaxis=dict(gridcolor=GRID, linecolor=GRID, color="#8b949e",
                             tickfont=dict(size=12)),
        ),
        legend=dict(bgcolor=CARD_BG, orientation="h", y=-0.08, x=0.2, font=dict(size=12)),
    )
    return fig


@app.callback(
    Output("player-output", "children"),
    [Input("player-dropdown", "value"),
     Input("compare-dropdown", "value"),
     Input("coach-view-store", "data")],
)
def update_player(name, compare_name, coach_view):
    coach_view = bool(coach_view)
    if not name:
        return html.P("Select a player", style=dict(color=TEXT))

    # Determine if we're in comparison mode
    comparing = (compare_name and compare_name != "none" and compare_name != name)

    epa_lbl = "Impact" if coach_view else "EPA"

    if not comparing:
        # Single player view
        cards = make_player_cards(name, coach_view)
        fig_epa = make_epa_chart(name, coach_view)
        fig_radar, pos = make_radar_figure(name, coach_view)

        epa_title = f"{name} — Game-by-Game Impact" if coach_view else f"{name} — EPA per Game"
        radar_title = f"{name} — Performance vs Avg {pos}" if coach_view else f"{name} — Scouting Stats vs Avg {pos}"

        return [
            html.Div(style=dict(display="flex", gap=12, marginBottom=24, flexWrap="wrap"),
                     children=cards),
            card(epa_title, [dcc.Graph(figure=fig_epa)]),
            card(radar_title, [
                html.P("Outward = better on all axes. ↓ = less is better (inverted).",
                       style=dict(color="#8b949e", fontSize=11, marginBottom=8)),
                dcc.Graph(figure=fig_radar),
            ]),
        ]
    else:
        # Comparison view — side by side
        cards_a = make_player_cards(name, coach_view)
        cards_b = make_player_cards(compare_name, coach_view)
        fig_epa_a = make_epa_chart(name, coach_view)
        fig_epa_b = make_epa_chart(compare_name, coach_view)
        fig_radar = make_comparison_radar(name, compare_name, coach_view)

        return [
            # Side by side stat cards
            html.Div(style=dict(display="flex", gap=16, marginBottom=24), children=[
                html.Div(style=dict(flex=1, display="flex", flexWrap="wrap", gap=8),
                         children=cards_a),
                html.Div(style=dict(flex=1, display="flex", flexWrap="wrap", gap=8),
                         children=cards_b),
            ]),
            # Side by side EPA charts
            html.Div(style=dict(display="flex", gap=16), children=[
                html.Div(style=dict(flex=1), children=[
                    card(f"{name} — {epa_lbl} per Game", [dcc.Graph(figure=fig_epa_a)]),
                ]),
                html.Div(style=dict(flex=1), children=[
                    card(f"{compare_name} — {epa_lbl} per Game", [dcc.Graph(figure=fig_epa_b)]),
                ]),
            ]),
            # Comparison radar
            card(f"{name} vs {compare_name} — Scouting Comparison", [
                html.P("Outward = better on all axes. ↓ = less is better (inverted).",
                       style=dict(color="#8b949e", fontSize=11, marginBottom=8)),
                dcc.Graph(figure=fig_radar),
            ]),
        ]


# ============================================================================ #
# TAB: POSITION ANALYSIS  (with player search)
# ============================================================================ #
def position_tab(coach_view):
    df = player_season.dropna(subset=["officialPosition", "totalEpa"]).copy()
    df = df[df["totalSnaps"] >= 50]

    fig_scatter = make_scatter_figure("All", None, coach_view)

    # Box plot — sorted by median, grouped by position group, with 0 line
    median_order = (
        df.groupby("posGroup")["epaPerSnap"].median()
        .sort_values(ascending=False)
    )
    epa_snap_lbl = "Impact / Play" if coach_view else "EPA / Snap"
    fig_box = px.box(
        df.sort_values("posGroupSorted") if "posGroupSorted" in df.columns else df,
        x="posGroup", y="epaPerSnap",
        color="posGroup",
        color_discrete_sequence=px.colors.qualitative.Set2,
        category_orders={"posGroup": list(median_order.index)},
        labels=dict(posGroup="Position Group", epaPerSnap=epa_snap_lbl),
    )
    fig_box.update_layout(**CHART_LAYOUT, height=350, showlegend=False)
    fig_box.update_xaxes(gridcolor=GRID, title="Position Group",
                         categoryorder="array", categoryarray=list(median_order.index))
    fig_box.update_yaxes(gridcolor=GRID, zerolinecolor=ACCENT3, zerolinewidth=1,
                         title=epa_snap_lbl)
    fig_box.add_hline(y=0, line_dash="dash", line_color=ACCENT3, opacity=0.5,
                      annotation_text="League Avg", annotation_font_size=10,
                      annotation_font_color="#8b949e")
    fig_box.update_traces(marker_line_width=0)

    # Top 5 per position — HTML tables with column headers
    top5_html = build_top5_tables(df, coach_view)

    scatter_title = "Player Impact vs Win %" if coach_view else "Player EPA vs Win Rate"
    box_title     = "Impact per Play by Position" if coach_view else "EPA per Snap Distribution"
    top5_title    = "Top 5 Players by Position"

    return [
        card(scatter_title, [
            html.Div(style=dict(display="flex", gap=12, marginBottom=12, alignItems="center"), children=[
                html.Div([
                    html.Span("Filter: ", style=dict(color="#8b949e", fontSize=12)),
                    dcc.Dropdown(
                        id="position-filter",
                        options=POS_FILTER_OPTS,
                        value="All",
                        clearable=False,
                        style=dict(width=200, display="inline-block",
                                   color="#0d1117", backgroundColor=CARD_BG),
                    ),
                ]),
                html.Div([
                    html.Span("Search: ", style=dict(color="#8b949e", fontSize=12)),
                    dcc.Input(
                        id="player-search",
                        type="text",
                        placeholder="Player name...",
                        debounce=True,
                        style=dict(width=200, display="inline-block",
                                   backgroundColor=CARD_BG, color=TEXT,
                                   border=f"1px solid {GRID}", borderRadius=6,
                                   padding="6px 10px", fontSize=13),
                    ),
                ]),
            ]),
            dcc.Graph(id="position-scatter", figure=fig_scatter),
        ]),
        html.Div(style=dict(display="flex", gap=16), children=[
            html.Div(style=dict(flex=1), children=[
                card(box_title, [dcc.Graph(figure=fig_box)]),
            ]),
            html.Div(style=dict(flex=1), children=[
                card(top5_title, [top5_html]),
            ]),
        ]),
    ]


def make_scatter_figure(pos_filter, search_text, coach_view):
    df = player_season.dropna(subset=["officialPosition", "totalEpa"]).copy()
    df = df[df["totalSnaps"] >= 50]

    if pos_filter and pos_filter != "All":
        df = df[df["posGroup"] == pos_filter]

    epa_lbl = "Total Impact" if coach_view else "Total EPA"
    wr_lbl  = "Win %" if coach_view else "Win Rate"

    # Colorblind-friendly palette (Okabe-Ito adapted for dark background)
    SCATTER_COLORS = [
        "#56B4E9",  # sky blue (QB)
        "#009E73",  # bluish green (RB)
        "#E69F00",  # orange (WR)
        "#CC79A7",  # reddish purple (TE)
        "#0072B2",  # blue (OL)
        "#F0E442",  # yellow (DL)
        "#D55E00",  # vermillion (LB)
        "#FF6DB1",  # pink (DB)
    ]

    fig = px.scatter(
        df, x="totalEpa", y="winRate",
        color="posGroup",
        size="totalSnaps",
        hover_name="displayName",
        custom_data=["team", "epaPerSnap"],
        labels=dict(totalEpa=epa_lbl, winRate=wr_lbl, posGroup="Position Group",
                     totalSnaps="Snaps"),
        color_discrete_sequence=SCATTER_COLORS,
    )
    fig.update_layout(**CHART_LAYOUT, height=520, legend=dict(bgcolor=CARD_BG, title="Position"))
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, title=epa_lbl)
    fig.update_yaxes(gridcolor=GRID, tickformat=".0%", range=[0, 1], title=wr_lbl)

    # Quadrant lines — 50% win rate line now has a label
    med_epa = df["totalEpa"].median()
    fig.add_hline(y=0.5, line_dash="dash", line_color=GRID, opacity=0.6,
                  annotation_text="50% Win Rate",
                  annotation_font_size=10, annotation_font_color="#8b949e",
                  annotation_position="top left")
    fig.add_vline(x=med_epa, line_dash="dash", line_color=GRID, opacity=0.6,
                  annotation_text="Median EPA",
                  annotation_font_size=10, annotation_font_color="#8b949e",
                  annotation_position="bottom right")

    # Quadrant labels — increased 50% (11 → 16)
    x_lo, x_hi = df["totalEpa"].min(), df["totalEpa"].max()
    fig.add_annotation(x=x_hi * 0.85, y=0.88, text="High Impact Winners",
                       font=dict(size=16, color=ACCENT2), showarrow=False, opacity=0.7)
    fig.add_annotation(x=x_lo * 0.85 if x_lo < 0 else x_lo * 0.5, y=0.88,
                       text="Carried by Team",
                       font=dict(size=16, color="#8b949e"), showarrow=False, opacity=0.7)
    fig.add_annotation(x=x_hi * 0.85, y=0.15, text="High Impact, Low Wins",
                       font=dict(size=16, color=ACCENT3), showarrow=False, opacity=0.7)
    fig.add_annotation(x=x_lo * 0.85 if x_lo < 0 else x_lo * 0.5, y=0.15,
                       text="Low Impact, Low Wins",
                       font=dict(size=16, color="#8b949e"), showarrow=False, opacity=0.7)

    fig.update_traces(
        hovertemplate="<b>%{hovertext}</b><br>Team: %{customdata[0]}<br>"
                      + epa_lbl + ": %{x:+.1f}<br>" + wr_lbl + ": %{y:.0%}<br>"
                      "EPA/Snap: %{customdata[1]:.3f}<br>Snaps: %{marker.size:,}<extra></extra>",
        marker=dict(opacity=0.7, line=dict(width=0)),
    )

    # Top 10 player labels with leader lines (staggered to avoid collision)
    top_10 = df.nlargest(10, "totalEpa").reset_index(drop=True)
    for i, r in top_10.iterrows():
        # Alternate left/right, stagger vertically in pairs
        ax = 45 if i % 2 == 0 else -45
        row = i // 2
        ay = -30 - row * 22
        # If near top of chart, flip label below to avoid going off-chart
        if r["winRate"] > 0.75:
            ay = 30 + row * 22
        fig.add_annotation(
            x=r["totalEpa"], y=r["winRate"],
            text=r["displayName"],
            showarrow=True,
            arrowhead=2, arrowsize=0.5, arrowwidth=0.5,
            arrowcolor="#8b949e",
            ax=ax, ay=ay,
            font=dict(size=10, color=TEXT),
            bgcolor=CARD_BG, bordercolor=GRID, borderpad=3, opacity=0.9,
        )

    # Bubble size legend
    fig.add_annotation(
        x=0.99, y=-0.18, xref="paper", yref="paper",
        text="Bubble size = Snaps Played",
        font=dict(size=11, color="#8b949e"),
        showarrow=False, xanchor="right",
    )

    # Search highlight overlay + zoom to player
    if search_text and len(search_text) >= 2:
        mask = player_season["displayName"].str.contains(search_text, case=False, na=False)
        highlighted = player_season[mask & (player_season["totalSnaps"] >= 50)]
        if pos_filter and pos_filter != "All":
            highlighted = highlighted[highlighted["posGroup"] == pos_filter]
        if len(highlighted) > 0:
            fig.add_trace(go.Scatter(
                x=highlighted["totalEpa"], y=highlighted["winRate"],
                mode="markers+text",
                text=highlighted["displayName"],
                textposition="top center",
                textfont=dict(size=12, color=TEXT),
                marker=dict(size=18, color=ACCENT,
                            line=dict(width=2, color="#ffffff"),
                            symbol="star"),
                name="Search Match",
                showlegend=False,
                hoverinfo="skip",
            ))
            # Zoom to the first matched player
            p = highlighted.iloc[0]
            x_pad = max(abs(df["totalEpa"].std()) * 2, 50)
            y_pad = 0.25
            fig.update_xaxes(range=[p["totalEpa"] - x_pad, p["totalEpa"] + x_pad])
            fig.update_yaxes(range=[max(0, p["winRate"] - y_pad),
                                   min(1, p["winRate"] + y_pad)])

    return fig


@app.callback(
    Output("position-scatter", "figure"),
    [Input("position-filter", "value"),
     Input("player-search", "value"),
     Input("coach-view-store", "data")],
)
def update_position_scatter(pos_filter, search_text, coach_view):
    if pos_filter is None:
        pos_filter = "All"
    return make_scatter_figure(pos_filter, search_text, bool(coach_view))


def build_top5_tables(df, coach_view):
    """Position-grouped tables with column headers."""
    positions = sorted(df["officialPosition"].dropna().unique())
    pos_colors = px.colors.qualitative.Set2
    tables = []
    for i, pos in enumerate(positions):
        top5 = df[df["officialPosition"] == pos].nlargest(5, "totalEpa")
        if len(top5) == 0:
            continue
        color = pos_colors[i % len(pos_colors)]
        # Header row
        rows = [html.Tr([
            html.Td("Player", style=dict(color="#8b949e", fontSize=9, padding="2px 6px",
                                         fontWeight=600)),
            html.Td("EPA", style=dict(color="#8b949e", fontSize=9, padding="2px 6px",
                                      fontWeight=600, textAlign="right")),
            html.Td("EPA/Snap", style=dict(color="#8b949e", fontSize=9, padding="2px 6px",
                                            fontWeight=600, textAlign="right")),
        ], style=dict(borderBottom=f"1px solid {GRID}"))]
        # Data rows
        for j, (_, r) in enumerate(top5.iterrows()):
            epa_color = ACCENT2 if r["totalEpa"] >= 0 else ACCENT3
            rows.append(html.Tr([
                html.Td(f"{j+1}. {r['displayName']}",
                        style=dict(color=TEXT, fontSize=11, padding="3px 6px",
                                   whiteSpace="nowrap", overflow="hidden",
                                   textOverflow="ellipsis", maxWidth=110)),
                html.Td(f"{r['totalEpa']:+.1f}",
                        style=dict(color=epa_color, fontSize=11, padding="3px 6px",
                                   textAlign="right", fontWeight=600)),
                html.Td(f"{r['epaPerSnap']:.3f}",
                        style=dict(color="#8b949e", fontSize=10, padding="3px 6px",
                                   textAlign="right")),
            ], style=dict(borderBottom=f"1px solid {GRID}")))
        tables.append(html.Div(
            style=dict(flex="1 1 200px", minWidth=180),
            children=[
                html.Div(pos,
                         style=dict(color=color, fontSize=13, fontWeight=600,
                                    marginBottom=4, paddingBottom=4,
                                    borderBottom=f"2px solid {color}")),
                html.Table(rows, style=dict(width="100%", borderCollapse="collapse")),
            ],
        ))
    return html.Div(style=dict(display="flex", flexWrap="wrap", gap=12),
                    children=tables)


# ============================================================================ #
# TAB: WIN IMPACT MODEL
# ============================================================================ #
def winmodel_tab(coach_view):
    # Feature importance with plain English labels
    wf = win_features.sort_values("abs_coef", ascending=True).copy()
    wf["label"] = wf["feature"].map(FEATURE_LABELS).fillna(wf["feature"])
    colors = [ACCENT2 if c > 0 else ACCENT3 for c in wf["coefficient"]]

    x_label = "Win Impact Score" if coach_view else "Impact on Winning"
    takeaway = ("Average impact per play is the single strongest predictor of winning."
                if coach_view else
                "Avg EPA per play is the single strongest predictor of winning.")

    fig_imp = go.Figure(go.Bar(
        x=wf["coefficient"], y=wf["label"],
        orientation="h", marker_color=colors,
    ))
    fig_imp.update_layout(**CHART_LAYOUT, height=400,
        xaxis=dict(title=x_label, gridcolor=GRID,
                   zerolinecolor=TEXT, zerolinewidth=1),
        yaxis=dict(gridcolor=GRID),
    )
    fig_imp.update_traces(marker_line_width=0)

    # Per-stat binned win rate (top 4 features, plain English labels)
    top4 = win_features.nlargest(4, "abs_coef")["feature"].tolist()
    bin_charts = []
    for feat in top4:
        if feat not in player_game.columns:
            continue
        d = player_game.dropna(subset=[feat, "teamWon"]).copy()
        d = d[d["snaps"] >= 10]
        try:
            d["bin"] = pd.qcut(d[feat], q=5, duplicates="drop",
                               labels=["Very Low", "Low", "Avg", "High", "Very High"]
                                     [:len(pd.qcut(d[feat], q=5, duplicates="drop").unique())])
        except Exception:
            d["bin"] = pd.cut(d[feat], bins=5,
                              labels=["Very Low", "Low", "Avg", "High", "Very High"])
        wr_by_bin = d.groupby("bin", observed=True)["teamWon"].mean().reset_index()
        wr_by_bin.columns = [feat, "winRate"]
        feat_label = FEATURE_LABELS.get(feat, feat)
        sign = "+" if win_features[win_features["feature"] == feat]["coefficient"].values[0] > 0 else "−"

        fig = px.bar(wr_by_bin, x=feat, y="winRate",
                     color_discrete_sequence=[ACCENT])
        fig.update_layout(**CHART_LAYOUT, height=240,
                          title=dict(text=f"{feat_label}  ({sign})",
                                     font=dict(size=12, color=TEXT),
                                     x=0.02, xanchor="left"))
        fig.update_traces(marker_line_width=0, opacity=0.85)
        fig.update_xaxes(gridcolor=GRID, title="")
        fig.update_yaxes(gridcolor=GRID, tickformat=".0%", range=[0, 1],
                         title="Win %" if coach_view else "Win Rate")
        bin_charts.append(html.Div(style=dict(flex=1, minWidth=280),
                                   children=[dcc.Graph(figure=fig)]))

    # Slider rows — consistent layout: range in label, min/max marks, tooltip on top
    slider_rows = []
    for spec in _slider_specs:
        def fmt(v):
            return str(int(v)) if v == int(v) else f"{v:.1f}"
        marks = {
            spec["min"]: {"label": fmt(spec["min"]), "style": {"color": "#8b949e", "fontSize": 10}},
            spec["max"]: {"label": fmt(spec["max"]), "style": {"color": "#8b949e", "fontSize": 10}},
        }
        slider_rows.append(html.Div(style=dict(marginBottom=20), children=[
            html.Div([
                html.Span(spec["label"], style=dict(color="#8b949e", fontSize=12)),
                html.Span(f"  ({fmt(spec['min'])} to {fmt(spec['max'])})",
                         style=dict(color="#8b949e", fontSize=10, marginLeft=8)),
            ], style=dict(marginBottom=6)),
            dcc.Slider(
                id=f"slider-{spec['key']}",
                min=spec["min"], max=spec["max"],
                step=spec["step"], value=spec["med"],
                marks=marks,
                tooltip=dict(always_visible=True, placement="top"),
            ),
        ]))

    n_games = len(game_outcomes)
    n_pg = len(player_game)

    return [
        html.Div(style=dict(display="flex", gap=16, marginBottom=16), children=[
            stat_card("Model Accuracy", f"{_win_acc:.1%}",
                      subtitle="vs 50% baseline", color=ACCENT2),
            stat_card("Model AUC", f"{_win_auc:.2f}",
                      subtitle="0.5=random, 1.0=perfect", color=ACCENT),
            stat_card("Player-Games", f"{n_pg:,}"),
            stat_card("Games", f"{n_games}"),
        ]),
        card("What Drives Winning", [
            html.P(takeaway, style=dict(color=ACCENT2, fontSize=13,
                    marginBottom=10, fontWeight=600)),
            dcc.Graph(figure=fig_imp),
            html.P("Green = helps winning, Orange = hurts winning. "
                   "Longer bar means stronger effect." if coach_view else
                   "Green = positive impact on win probability, "
                   "Orange = negative impact. Longer bar means stronger effect.",
                   style=dict(color="#8b949e", fontSize=11, marginTop=8)),
        ]),
        card("How Stats Translate to Wins" if coach_view else "Win Probability by Stat Level", [
            html.P("How each stat level maps to actual win rate — the model's "
                   "most influential features." if not coach_view else
                   "Each chart shows how win rate changes as the stat improves.",
                   style=dict(color="#8b949e", fontSize=11, marginBottom=12)),
            html.Div(
                style=dict(display="flex", gap=16, flexWrap="wrap"),
                children=bin_charts,
            ),
        ]),
        card("Win Predictor" if coach_view else "Win Probability Calculator", [
            html.P("Adjust player performance stats to see the model's "
                   "predicted win probability. Starts at league-median values.",
                   style=dict(color="#8b949e", fontSize=11, marginBottom=16)),
            html.Div(style=dict(display="flex", gap=32), children=[
                html.Div(style=dict(flex=1), children=slider_rows + [
                    html.Button("Reset to League Average", id="reset-button",
                                n_clicks=0,
                                style=dict(backgroundColor=ACCENT, color="#fff",
                                           border="none", borderRadius=8,
                                           padding="8px 16px", cursor="pointer",
                                           fontSize=12, fontWeight=600, marginTop=8)),
                ]),
                html.Div(id="winprob-output", style=dict(
                    flex=1, display="flex", alignItems="center",
                    justifyContent="center",
                )),
            ]),
        ]),
    ]


@app.callback(
    Output("winprob-output", "children"),
    [Input(f"slider-{s['key']}", "value") for s in _slider_specs],
)
def update_winprob(*slider_vals):
    feature_vals = {}
    for spec, val in zip(_slider_specs, slider_vals):
        feature_vals[spec["key"]] = val if val is not None else spec["med"]
    for feat in _WIN_FEATURES:
        if feat not in feature_vals:
            feature_vals[feat] = float(player_game[feat].median())

    row = pd.DataFrame([[feature_vals[f] for f in _WIN_FEATURES]],
                       columns=_WIN_FEATURES)
    row_s = _win_scaler.transform(row)
    proba = float(_win_model.predict_proba(row_s)[0, 1])

    color = winprob_color(proba)

    # Delta from league-average baseline
    delta = proba - _baseline_proba
    delta_str = f"{'↑' if delta >= 0 else '↓'} {abs(delta):.1%} from league avg"

    return html.Div(
        style=dict(textAlign="center"),
        children=[
            html.Div("Predicted Win Probability",
                     style=dict(color="#8b949e", fontSize=13, marginBottom=8)),
            html.Div(f"{proba:.1%}",
                     style=dict(color=color, fontSize=48, fontWeight=700)),
            html.Div(delta_str,
                     style=dict(color=color, fontSize=13, marginTop=6)),
            html.Div(
                ("↑ Model favours this team" if proba >= 0.50
                 else "↓ Model favours the opponent" if proba < 0.45
                 else "≈ Coin flip territory"),
                style=dict(color=color, fontSize=11, marginTop=4),
            ),
        ],
    )


@app.callback(
    [Output(f"slider-{s['key']}", "value") for s in _slider_specs],
    Input("reset-button", "n_clicks"),
    prevent_initial_call=True,
)
def reset_sliders(n):
    return [s["med"] for s in _slider_specs]


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8050)
