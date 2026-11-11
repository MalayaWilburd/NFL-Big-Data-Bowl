# NFL Player Win Impact Dashboard

An EPA-based analytics tool built on the 2021 NFL Big Data Bowl dataset (Next Gen Stats with PFF scouting data) that relates individual player performance to game outcomes across 1,679 players, 8,557 pass plays and 122 games from Weeks 1-8 of the 2021 season. A gradient-boosting Expected Points model estimates EPA for every play, and a logistic regression finds which player stats are most tied to winning. On games held out of training, the win model reaches 65.9% accuracy (AUC 0.74) against a 50% baseline. Results are presented in four interactive tabs for analysts and coaching staff, with a Coach View toggle that swaps statistical terms for plain English.

![NFL Player Win Impact Dashboard](assets/dashboard_screenshot.png)

## Live Demo

**https://nfl-player-win-impact.onrender.com**

The demo runs on a free host that sleeps when idle, so the first load can take up to a minute.

## Run it locally

Requires Python 3.9+.

```bash
pip install -r requirements.txt
python src/compute_epa.py     # builds data/derived/ (about 2 minutes)
python src/dashboard.py       # opens at http://127.0.0.1:8050
```

## What's inside

| Tab | Purpose |
|---|---|
| **Overview** | Top 20 players by EPA, win rate for all 32 teams, play impact distribution |
| **Player Impact** | Stat cards with position rank, game-by-game EPA, scouting radar vs position average, side-by-side player comparison |
| **Position Analysis** | EPA vs win rate chart with quadrant labels, position filter, player search, top 5 per position |
| **Win Impact Model** | Which stats are most tied to winning, win rate by stat level, interactive win probability calculator |

## Method and limitations

- **Held-out evaluation:** accuracy and AUC come from 5-fold cross-validation grouped by game (65.9% ± 3.1%, AUC 0.74 ± 0.03), so no game appears in both training and testing. The code is in `src/compute_epa.py`.
- **Final scores:** `data/games.csv` has no scores, and `plays.csv` holds only pass plays, so official scores come from the public [nflverse schedule data](https://github.com/nflverse/nfldata) (`src/build_final_scores.py`).
- **Shared credit:** every player on the field receives the play's full EPA. Teammates on the same unit score almost the same, so player numbers reflect the unit, not individual impact.
- **Association, not prediction:** EPA and the game result come from the same game, so the model shows which stats go with winning, not which ones forecast it.
- **Simple EP model:** expected points use only down, distance and field position, and the data contains only pass plays, so EPA is approximate.
- **Tracking files unused:** the 122 per-game tracking files are in the repo but this analysis does not use them yet.

## Data

Source: [NFL Big Data Bowl regional event data](https://github.com/ThompsonJamesBliss/nfl-big-data-bowl-regional-event-data): `games.csv`, `players.csv`, `plays.csv`, `pffScoutingData.csv` and 122 per-game tracking files (about 833 MB total). Final scores: nflverse.
