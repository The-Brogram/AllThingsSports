"""Slate settings. The slate date comes from NFL_DATE (slate.py sets it) or slate.json; season and week are
looked up in the schedule, so a new week needs no editing.

slate.json: "date" is the default slate when none is given; "slates" holds news per slate date (YYYYMMDD):
  title (optional page name),
  notes: {team: text} warnings shown on that team's games; a team key flags all its props,
         a key ending in "_" (e.g. "SEA_") only shows the note,
  hold_rushing: teams whose rushing lines are held out of picks (backfield role news),
  qb: {team: gsis_id} confirmed starter when the depth chart lags the news,
  out: {team: [gsis_id, ...]} players confirmed out that the injury feed has not caught.
"""
import json
import os
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
_C = json.load(open(os.path.join(ROOT, "slate.json")))
DATE = os.environ.get("NFL_DATE", _C["date"])
DAY = datetime.strptime(DATE, "%Y%m%d")


def schedule():
    import pandas as pd
    g = pd.read_csv(os.path.join(ROOT, "data", "games.csv"), usecols=["season", "week", "gameday", "gametime", "away_team", "home_team", "result"])
    return g.assign(date=g.gameday.str.replace("-", ""))


_G = schedule()
_S = _G[_G.date == DATE]
_N = _C.get("slates", {}).get(DATE, {})
if len(_S):
    SEASON, WEEK = int(_S.season.iloc[0]), int(_S.week.iloc[0])
else:
    SEASON, WEEK = DAY.year, 0
_auto = f"{_S.away_team.iloc[0]} at {_S.home_team.iloc[0]} Game Book" if len(_S) == 1 else f"Week {WEEK} {DAY:%A} Game Book"
TITLE = _N.get("title") or _auto
NOTES = _N.get("notes", {})
HOLD_RUSH = set(_N.get("hold_rushing", []))
QB_START = _N.get("qb", {})
OUT_IDS = _N.get("out", {})
PLAY_IDS = _N.get("play", {})  # {team: {gsis_id: chance he plays}}: confirmed news that overrides a stale injury-feed status
OUT = os.path.join(ROOT, "output", f"nfl_{DATE}_report.html")
PRED = os.path.join(ROOT, "output", f"predictions_{DATE}.csv")
PICKS = os.path.join(ROOT, "output", f"picks_{DATE}.csv")  # first run's picks; reruns write picks_<date>_final.csv
ESPN = f"espn_{DATE}.json"
CACHE = os.environ.get("NFL_CACHE", os.path.join(ROOT, "data", f"slate_{DATE}.pkl"))  # simulation output, large, not versioned
# stadium (or city) coordinates for NWS forecasts and travel distance
COORD = {"ARI": (33.53, -112.26), "ATL": (33.76, -84.40), "BAL": (39.28, -76.62), "BUF": (42.77, -78.79), "CAR": (35.23, -80.85), "CHI": (41.86, -87.62), "CIN": (39.10, -84.52), "CLE": (41.51, -81.70),
         "DAL": (32.75, -97.09), "DEN": (39.74, -105.02), "DET": (42.34, -83.05), "GB": (44.50, -88.06), "HOU": (29.68, -95.41), "IND": (39.76, -86.16), "JAX": (30.32, -81.64), "KC": (39.05, -94.48),
         "LA": (33.95, -118.34), "LAC": (33.95, -118.34), "LV": (36.09, -115.18), "MIA": (25.96, -80.24), "MIN": (44.97, -93.26), "NE": (42.09, -71.26), "NO": (29.95, -90.08), "NYG": (40.81, -74.07),
         "NYJ": (40.81, -74.07), "PHI": (39.90, -75.17), "PIT": (40.45, -80.02), "SEA": (47.60, -122.33), "SF": (37.40, -121.97), "TB": (27.98, -82.50), "TEN": (36.17, -86.77), "WAS": (38.91, -76.86)}
