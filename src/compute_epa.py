"""
NFL Big Data Bowl - EPA Pipeline & Player Win Impact Analysis
=============================================================

Computes Expected Points Added (EPA) for every play, attributes it to the
players on the field, and produces player-level statistics correlated with
game outcomes.

Pipeline:
  1. Load plays, games, players, PFF scouting data.
  2. Derive final scores & winners from score progression + play descriptions.
  3. Compute "next score" per play (target for EP model).
  4. Train a gradient-boosting EP model on (down, distance, yardline).
  5. Compute EPA per play.
  6. Attribute EPA to players via pffScoutingData.
  7. Aggregate to player-game and player-season level.
  8. Logistic regression: which player stats predict winning?
  9. Save derived CSVs.
"""

import os
import re
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUT_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "derived")
os.makedirs(OUT_DIR, exist_ok=True)


# --------------------------------------------------------------------------- #
# 1.  Load data
# --------------------------------------------------------------------------- #
def load_data():
    print("Loading data ...")
    games   = pd.read_csv(os.path.join(DATA_DIR, "games.csv"))
    players = pd.read_csv(os.path.join(DATA_DIR, "players.csv"))
    plays   = pd.read_csv(os.path.join(DATA_DIR, "plays.csv"))
    pff     = pd.read_csv(os.path.join(DATA_DIR, "pffScoutingData.csv"))
    print(f"  games:   {len(games):>8,} rows")
    print(f"  players: {len(players):>8,} rows")
    print(f"  plays:   {len(plays):>8,} rows")
    print(f"  pff:     {len(pff):>8,} rows")
    return games, players, plays, pff


# --------------------------------------------------------------------------- #
# 2.  Derive final scores & game winners
# --------------------------------------------------------------------------- #
def derive_final_scores(plays, games):
    """Reconstruct the final score of every game from the progression of
    preSnapHomeScore / preSnapVisitorScore across plays, plus parsing the
    last play's description for any scoring not captured in pre-snap scores."""
    print("Deriving final scores ...")

    team_map = games[["gameId", "homeTeamAbbr", "visitorTeamAbbr"]].copy()
    plays = plays.merge(team_map, on="gameId", how="left")
    plays = plays.sort_values(["gameId", "playId"]).reset_index(drop=True)

    # For each game, compute score deltas between consecutive plays
    plays["homeScoreAfter"] = plays.groupby("gameId")["preSnapHomeScore"].shift(-1)
    plays["visScoreAfter"]  = plays.groupby("gameId")["preSnapVisitorScore"].shift(-1)

    # Points scored by each team on each play (from score delta)
    plays["homePts"] = (plays["homeScoreAfter"] - plays["preSnapHomeScore"].fillna(0)).fillna(0)
    plays["visPts"]  = (plays["visScoreAfter"]  - plays["preSnapVisitorScore"].fillna(0)).fillna(0)

    # For the LAST play of each game, the score-after is unknown (no next play).
    # Parse the play description to detect scoring plays.
    is_last = plays.groupby("gameId")["playId"].transform("max") == plays["playId"]

    def parse_score(desc, possession_team, home_team, visitor_team):
        """Parse a play description for TOUCHDOWN / FIELD GOAL / SAFETY.
        Returns (home_pts, vis_pts) for the play."""
        if pd.isna(desc):
            return 0.0, 0.0
        desc_upper = str(desc).upper()
        home_pts, vis_pts = 0.0, 0.0

        if "TOUCHDOWN" in desc_upper:
            pts = 7.0  # 6 + XP (approximate; 2PT conversions are rare)
            if possession_team == home_team:
                home_pts = pts
            elif possession_team == visitor_team:
                vis_pts = pts
            elif "SAFETY" in desc_upper:
                # Safety = 2 points for the defensive team
                if possession_team == home_team:
                    vis_pts = 2.0
                else:
                    home_pts = 2.0
        elif "FIELD GOAL" in desc_upper:
            pts = 3.0
            if possession_team == home_team:
                home_pts = pts
            else:
                vis_pts = pts
        elif "SAFETY" in desc_upper:
            if possession_team == home_team:
                vis_pts = 2.0
            else:
                home_pts = 2.0
        return home_pts, vis_pts

    # Apply parsing to last plays
    for idx in plays[is_last].index:
        hp, vp = parse_score(
            plays.loc[idx, "playDescription"],
            plays.loc[idx, "possessionTeam"],
            plays.loc[idx, "homeTeamAbbr"],
            plays.loc[idx, "visitorTeamAbbr"],
        )
        plays.loc[idx, "homePts"] = hp
        plays.loc[idx, "visPts"] = vp

    # Final scores and winners come from real results, NOT from plays.csv.
    # plays.csv only holds pass plays (~70 per game), so runs, field goals and
    # kneel-downs are missing and the score progression cannot be rebuilt.
    # data/final_scores.csv holds the official 2021 regular-season scores
    # (source: nflverse/nfldata games.csv, joined on week + home + away team).
    real = pd.read_csv(os.path.join(DATA_DIR, "final_scores.csv"))
    games = games.merge(real, on="gameId", how="left")
    if games[["finalHomeScore", "finalVisitorScore"]].isna().any().any():
        raise ValueError("final_scores.csv is missing scores for some games")

    games["tie"] = (games["finalHomeScore"] == games["finalVisitorScore"]).astype(int)
    # homeWin: 1 = home won, 0 = visitor won, 0.5 = tie (counts as half a win for each team)
    games["homeWin"] = np.select(
        [games["finalHomeScore"] > games["finalVisitorScore"],
         games["finalHomeScore"] < games["finalVisitorScore"]],
        [1.0, 0.0], default=0.5,
    )

    hw = (games["homeWin"] == 1).sum()
    aw = (games["homeWin"] == 0).sum()
    ti = (games["tie"] == 1).sum()
    print(f"  {len(games)} games  |  home wins: {hw}  away wins: {aw}  ties: {ti}")

    return plays, games


# --------------------------------------------------------------------------- #
# 3.  Compute next-score value for every play
# --------------------------------------------------------------------------- #
def compute_next_score(plays):
    """For every play, find the next scoring event in the same game and
    record its value from the possession team's perspective
    (+7 for own TD, -7 for opponent TD, +3 for own FG, etc.)."""
    print("Computing next-score values ...")

    # Determine if possession team is home or visitor
    plays["possIsHome"] = plays["possessionTeam"] == plays["homeTeamAbbr"]

    # Net points from possession team's perspective for each play
    plays["possPts"] = np.where(plays["possIsHome"], plays["homePts"], plays["visPts"])
    plays["defPts"]  = np.where(plays["possIsHome"], plays["visPts"], plays["homePts"])
    plays["netPts"]  = plays["possPts"] - plays["defPts"]

    # For each play, find the NEXT scoring play in the same game
    # A scoring play is one where any points were scored (homePts or visPts != 0)
    plays["anyScore"] = (plays["homePts"].abs() + plays["visPts"].abs()) > 0

    # Efficient vectorized approach: within each game, carry the next
    # scoring play's netPts forward to all preceding plays
    next_score_vals = []
    for gid, g in plays.groupby("gameId"):
        g = g.sort_values("playId").copy()
        nsv = np.zeros(len(g))
        scores = g[g["anyScore"]]
        if len(scores) > 0:
            score_idx_list = list(scores.index)
            score_vals_list = scores["netPts"].values
            all_idx = list(g.index)
            # For each play, find the first scoring play at or after it
            si = 0  # pointer into score_idx_list
            for i in all_idx:
                # Advance si to the first score >= i
                while si < len(score_idx_list) and score_idx_list[si] < i:
                    si += 1
                if si < len(score_idx_list):
                    nsv[all_idx.index(i)] = score_vals_list[si]
                # if si past end, nsv stays 0 (no more scores in game)
        g["nextScoreVal"] = nsv
        next_score_vals.append(g)

    plays = pd.concat(next_score_vals, ignore_index=True)

    # Compute yardline from possession team's perspective (1-99)
    plays["yardline"] = np.where(
        plays["yardlineSide"] == plays["possessionTeam"],
        plays["yardlineNumber"],
        100 - plays["yardlineNumber"],
    )
    plays["yardline"] = plays["yardline"].fillna(50).clip(1, 99)

    print(f"  nextScoreVal: mean={plays['nextScoreVal'].mean():.3f}  "
          f"std={plays['nextScoreVal'].std():.3f}")
    return plays


# --------------------------------------------------------------------------- #
# 4.  Train the EP model
# --------------------------------------------------------------------------- #
def train_ep_model(plays):
    print("Training EP model ...")

    train = plays.dropna(subset=["down", "yardsToGo"]).copy()
    train = train[train["down"].between(1, 4)]
    train["yardline"] = train["yardline"].clip(1, 99)

    X = train[["down", "yardsToGo", "yardline"]].astype(float)
    y = train["nextScoreVal"].astype(float)

    model = GradientBoostingRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=42,
    )
    model.fit(X, y)

    r2 = model.score(X, y)
    print(f"  EP model R² (in-sample): {r2:.4f}")
    return model


# --------------------------------------------------------------------------- #
# 5.  Compute EPA per play
# --------------------------------------------------------------------------- #
def compute_epa(plays, ep_model):
    print("Computing EPA per play ...")

    plays = plays.sort_values(["gameId", "playId"]).reset_index(drop=True)

    # Pre-play features
    plays["down_f"]      = plays["down"].fillna(1).clip(1, 4).astype(float)
    plays["yardsToGo_f"] = plays["yardsToGo"].fillna(10).clip(1, 20).astype(float)
    plays["yardline_f"]  = plays["yardline"].clip(1, 99).astype(float)

    X_before = plays[["down_f", "yardsToGo_f", "yardline_f"]].rename(columns={
        "down_f": "down", "yardsToGo_f": "yardsToGo", "yardline_f": "yardline"})
    plays["epBefore"] = ep_model.predict(X_before)

    # Post-play state: next play's down/distance/yardline
    plays["nextDown"]      = plays.groupby("gameId")["down"].shift(-1)
    plays["nextYardsToGo"] = plays.groupby("gameId")["yardsToGo"].shift(-1)
    plays["nextYardline"]  = plays.groupby("gameId")["yardline"].shift(-1)

    next_poss = plays.groupby("gameId")["possessionTeam"].shift(-1)
    plays["possChanged"] = next_poss != plays["possessionTeam"]

    # If possession changed (turnover), flip yardline perspective
    plays.loc[plays["possChanged"] == True, "nextYardline"] = (
        100 - plays.loc[plays["possChanged"] == True, "nextYardline"]
    )

    has_next = plays.groupby("gameId")["down"].shift(-1).notna()

    plays["nextDown_f"]      = plays["nextDown"].fillna(4).clip(1, 4).astype(float)
    plays["nextYardsToGo_f"] = plays["nextYardsToGo"].fillna(10).clip(1, 20).astype(float)
    plays["nextYardline_f"]  = plays["nextYardline"].clip(1, 99).fillna(50).astype(float)

    X_after = plays[["nextDown_f", "nextYardsToGo_f", "nextYardline_f"]].rename(columns={
        "nextDown_f": "down", "nextYardsToGo_f": "yardsToGo", "nextYardline_f": "yardline"})
    plays["epAfter"] = ep_model.predict(X_after)
    plays.loc[~has_next, "epAfter"] = 0.0  # end of half/game → no remaining value

    # EPA = EP(after) - EP(before)
    plays["epa"] = plays["epAfter"] - plays["epBefore"]

    # Turnovers: EPA is negative for the original possession team
    plays.loc[plays["possChanged"] == True, "epa"] = -abs(plays.loc[plays["possChanged"] == True, "epa"])

    # Scoring plays: override with actual net points
    scored = plays["netPts"].abs() > 0
    plays.loc[scored, "epa"] = plays.loc[scored, "netPts"]

    print(f"  EPA: mean={plays['epa'].mean():.4f}  std={plays['epa'].std():.4f}")
    return plays


# --------------------------------------------------------------------------- #
# 6.  Attribute EPA to players on the field
# --------------------------------------------------------------------------- #
def attribute_player_epa(plays, pff):
    print("Attributing EPA to players ...")

    epa_cols = ["gameId", "playId", "epa", "possessionTeam", "defensiveTeam",
                "playResult", "passResult", "down", "yardsToGo", "yardline"]
    player_plays = pff.merge(plays[epa_cols], on=["gameId", "playId"], how="left")

    # Offensive roles get raw EPA; defensive roles (Pass Rush, Coverage) get
    # negated EPA. Every play in plays.csv is a pass play, so the offensive
    # roles are the passer, blockers and route runners.
    offensive_roles = {"Pass", "Pass Block", "Pass Route"}
    known_roles = offensive_roles | {"Pass Rush", "Coverage"}
    unknown = set(player_plays["pff_role"].dropna().unique()) - known_roles
    if unknown:
        raise ValueError(f"Unmapped pff_role values: {unknown}")
    player_plays["isOffense"] = player_plays["pff_role"].isin(offensive_roles)
    player_plays["playerEpa"] = np.where(
        player_plays["isOffense"],
        player_plays["epa"],
        -player_plays["epa"],
    )

    print(f"  {len(player_plays):,} player-play rows attributed")
    return player_plays


# --------------------------------------------------------------------------- #
# 7.  Aggregate to player-game and player-season
# --------------------------------------------------------------------------- #
def aggregate_player_stats(player_plays, games, players):
    print("Aggregating player stats ...")

    game_outcome = games[["gameId", "homeTeamAbbr", "visitorTeamAbbr",
                          "finalHomeScore", "finalVisitorScore", "homeWin"]]
    player_plays = player_plays.merge(game_outcome, on="gameId", how="left",
                                       suffixes=("", "_g"))

    # Determine player's team
    player_plays["playerTeam"] = np.where(
        player_plays["isOffense"],
        player_plays["possessionTeam"],
        player_plays["defensiveTeam"],
    )
    player_plays["teamIsHome"] = player_plays["playerTeam"] == player_plays["homeTeamAbbr"]
    player_plays["teamWon"] = np.where(
        player_plays["teamIsHome"],
        player_plays["homeWin"],
        1 - player_plays["homeWin"],
    )

    # --- Player-game ---
    pg = (
        player_plays.groupby(["gameId", "nflId"])
        .agg(
            totalEpa=("playerEpa", "sum"),
            avgEpa=("playerEpa", "mean"),
            snaps=("playerEpa", "count"),
            sacks=("pff_sack", "sum"),
            hurries=("pff_hurry", "sum"),
            hits=("pff_hit", "sum"),
            beaten=("pff_beatenByDefender", "sum"),
            hitsAllowed=("pff_hitAllowed", "sum"),
            hurriesAllowed=("pff_hurryAllowed", "sum"),
            sacksAllowed=("pff_sackAllowed", "sum"),
            teamWon=("teamWon", "first"),
            team=("playerTeam", "first"),
            primaryRole=("pff_role", "first"),
            positionLinedUp=("pff_positionLinedUp", "first"),
        )
        .reset_index()
    )

    pg = pg.merge(
        players[["nflId", "displayName", "officialPosition", "height", "weight"]],
        on="nflId", how="left",
    )

    # --- Player-season ---
    ps = (
        pg.groupby("nflId")
        .agg(
            games=("gameId", "count"),
            totalEpa=("totalEpa", "sum"),
            avgEpa=("avgEpa", "mean"),
            totalSnaps=("snaps", "sum"),
            avgSnaps=("snaps", "mean"),
            totalSacks=("sacks", "sum"),
            totalHurries=("hurries", "sum"),
            totalHits=("hits", "sum"),
            totalBeaten=("beaten", "sum"),
            totalHitsAllowed=("hitsAllowed", "sum"),
            totalHurriesAllowed=("hurriesAllowed", "sum"),
            totalSacksAllowed=("sacksAllowed", "sum"),
            winRate=("teamWon", "mean"),
        )
        .reset_index()
    )
    ps = ps.merge(
        players[["nflId", "displayName", "officialPosition", "height", "weight"]],
        on="nflId", how="left",
    )
    ps["epaPerSnap"] = ps["totalEpa"] / ps["totalSnaps"]

    print(f"  player-game:   {len(pg):,} rows")
    print(f"  player-season: {len(ps):,} rows")
    return pg, ps


# --------------------------------------------------------------------------- #
# 8.  Logistic regression: player stats → win prediction
# --------------------------------------------------------------------------- #
def win_impact_model(pg):
    print("Running win-impact logistic regression ...")

    features = ["totalEpa", "avgEpa", "snaps", "sacks", "hurries", "hits",
                "beaten", "hitsAllowed", "hurriesAllowed", "sacksAllowed"]

    df = pg.dropna(subset=features + ["teamWon"]).copy()
    df = df[df["snaps"] >= 10]

    X = df[features].fillna(0).to_numpy()
    y = df["teamWon"].astype(int).to_numpy()
    groups = df["gameId"].to_numpy()

    # Held-out evaluation: 5-fold cross-validation grouped by game, so every
    # player from a given game is either all in training or all in testing.
    accs, aucs = [], []
    for train_idx, test_idx in GroupKFold(n_splits=5).split(X, y, groups):
        fold_model = make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
        fold_model.fit(X[train_idx], y[train_idx])
        fold_proba = fold_model.predict_proba(X[test_idx])[:, 1]
        accs.append(accuracy_score(y[test_idx], fold_proba >= 0.5))
        aucs.append(roc_auc_score(y[test_idx], fold_proba))
    acc, auc = float(np.mean(accs)), float(np.mean(aucs))
    baseline = float(max(y.mean(), 1 - y.mean()))

    # Final model on all data, used only to read off the coefficients.
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_s, y)

    importance = pd.DataFrame({
        "feature": features,
        "coefficient": model.coef_[0],
        "abs_coef": np.abs(model.coef_[0]),
    }).sort_values("abs_coef", ascending=False)

    metrics = pd.DataFrame([{
        "heldOutAccuracy": acc, "heldOutAccuracyStd": float(np.std(accs)),
        "heldOutAuc": auc, "heldOutAucStd": float(np.std(aucs)),
        "baselineAccuracy": baseline,
        "rows": len(df), "games": int(df["gameId"].nunique()), "folds": 5,
    }])

    print(f"  Held-out (5-fold, grouped by game): accuracy {acc:.4f} "
          f"(+/-{np.std(accs):.4f})  AUC {auc:.4f} (+/-{np.std(aucs):.4f})")
    print(f"  Baseline (always guess the majority outcome): {baseline:.4f}")
    for _, row in importance.head(5).iterrows():
        print(f"    {row['feature']:>20s}  coef={row['coefficient']:+.4f}")

    return importance, metrics


# --------------------------------------------------------------------------- #
# 9.  Save outputs
# --------------------------------------------------------------------------- #
def save_outputs(games, plays, pg, ps, importance, metrics):
    print("Saving derived CSVs ...")
    games[["gameId", "season", "week", "gameDate", "homeTeamAbbr", "visitorTeamAbbr",
           "finalHomeScore", "finalVisitorScore", "homeWin", "tie"]].to_csv(
        os.path.join(OUT_DIR, "game_outcomes.csv"), index=False)

    plays[["gameId", "playId", "down", "yardsToGo", "yardline", "possessionTeam",
           "defensiveTeam", "epa", "epBefore", "epAfter", "playResult",
           "passResult", "netPts"]].to_csv(
        os.path.join(OUT_DIR, "play_epa.csv"), index=False)

    pg.to_csv(os.path.join(OUT_DIR, "player_game_stats.csv"), index=False)
    ps.to_csv(os.path.join(OUT_DIR, "player_season_stats.csv"), index=False)
    importance.to_csv(os.path.join(OUT_DIR, "win_impact_features.csv"), index=False)
    metrics.to_csv(os.path.join(OUT_DIR, "model_metrics.csv"), index=False)
    print(f"  Saved 6 CSVs to {OUT_DIR}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    games, players, plays, pff = load_data()
    plays, games = derive_final_scores(plays, games)
    plays = compute_next_score(plays)
    ep_model = train_ep_model(plays)
    plays = compute_epa(plays, ep_model)
    player_plays = attribute_player_epa(plays, pff)
    pg, ps = aggregate_player_stats(player_plays, games, players)
    importance, metrics = win_impact_model(pg)
    save_outputs(games, plays, pg, ps, importance, metrics)
    print("\nPipeline complete.  Derived data saved to data/derived/")


if __name__ == "__main__":
    main()
