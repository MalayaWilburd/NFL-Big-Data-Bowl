"""
Build data/final_scores.csv: official final scores for the 122 games in games.csv.

Why this exists: games.csv has no scores, and plays.csv contains only pass
plays (~70 per game), so final scores cannot be rebuilt from play data.
Scores come from the public nflverse schedule dataset, joined to our games on
week + home team + away team.

Run once (needs internet):  python src/build_final_scores.py
"""

import os
import pandas as pd

SOURCE = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def main():
    games = pd.read_csv(os.path.join(DATA_DIR, "games.csv"))
    nv = pd.read_csv(SOURCE)
    nv = nv[(nv["season"] == 2021) & (nv["game_type"] == "REG")]
    nv = nv[["week", "home_team", "away_team", "home_score", "away_score"]]

    merged = games.merge(
        nv,
        left_on=["week", "homeTeamAbbr", "visitorTeamAbbr"],
        right_on=["week", "home_team", "away_team"],
        how="left",
    )
    missing = int(merged["home_score"].isna().sum())
    if missing or merged["gameId"].duplicated().any():
        raise ValueError(f"Could not match every game: {missing} missing")

    out = merged[["gameId", "home_score", "away_score"]].rename(
        columns={"home_score": "finalHomeScore", "away_score": "finalVisitorScore"})
    out = out.astype({"finalHomeScore": int, "finalVisitorScore": int})
    out.to_csv(os.path.join(DATA_DIR, "final_scores.csv"), index=False)
    print(f"Wrote {len(out)} games to data/final_scores.csv")


if __name__ == "__main__":
    main()
