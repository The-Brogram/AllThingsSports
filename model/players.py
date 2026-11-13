"""Player Lookup data: every QB, RB, WR and TE game from 2022 through the current 2026 week, regular season and playoffs.

Writes output/hub/players.json, which the hub's Player Lookup page loads on demand. Rows are arrays in the order of
COLS to keep the file small. Home/away and the final score come from games.csv by game_id.
Run on its own (venv/bin/python players.py) or through season.py, which calls build().
"""
import json
import os
import numpy as np
import pandas as pd
from engine import DATA

SEASONS = list(range(2022, 2027))
POS = ["QB", "RB", "WR", "TE"]
COLS = ["s", "w", "post", "tm", "opp", "home", "pf", "pa", "cmp", "att", "py", "ptd", "int", "sk", "car", "ry", "rtd", "rec", "tgt", "ryd", "rctd", "fl", "fp", "ts"]
SRC = {"cmp": "completions", "att": "attempts", "py": "passing_yards", "ptd": "passing_tds", "int": "passing_interceptions", "sk": "sacks_suffered",
       "car": "carries", "ry": "rushing_yards", "rtd": "rushing_tds", "rec": "receptions", "tgt": "targets", "ryd": "receiving_yards", "rctd": "receiving_tds",
       "fp": "fantasy_points_ppr", "ts": "target_share"}
ROOT = os.path.dirname(os.path.abspath(__file__))


def build(out=os.path.join(ROOT, "output", "hub", "players.json")):
    st = pd.concat([pd.read_csv(f"{DATA}/stats_player_week_{y}.csv", low_memory=False) for y in SEASONS])
    st = st[st.position_group.isin(POS) & st.season_type.isin(["REG", "POST"]) & st.team.notna()]
    st = st.copy()
    fum = [c for c in ("rushing_fumbles_lost", "receiving_fumbles_lost", "sack_fumbles_lost") if c in st]
    st["fl"] = st[fum].fillna(0).sum(axis=1)
    g = pd.read_csv(f"{DATA}/games.csv", low_memory=False, usecols=["game_id", "home_team", "home_score", "away_score"])
    st = st.merge(g, on="game_id", how="left")
    st["home"] = (st.team == st.home_team).astype(int)
    st["pf"] = np.where(st.home == 1, st.home_score, st.away_score)
    st["pa"] = np.where(st.home == 1, st.away_score, st.home_score)
    st["post"] = (st.season_type == "POST").astype(int)
    # a player who touched the ball in no game is not worth a search entry
    act = st[[SRC[k] for k in ("att", "car", "tgt")]].fillna(0).sum(axis=1) > 0
    st = st[act].sort_values(["player_id", "season", "week"])
    players = []
    for pid, x in st.groupby("player_id", sort=False):
        last = x.iloc[-1]
        rows = []
        for r in x.itertuples(index=False):
            d = r._asdict()
            row = [int(d["season"]), int(d["week"]), int(d["post"]), d["team"], d["opponent_team"], int(d["home"]),
                   None if pd.isna(d["pf"]) else int(d["pf"]), None if pd.isna(d["pa"]) else int(d["pa"])]
            for k in COLS[8:]:
                v = d["fl"] if k == "fl" else d.get(SRC[k])
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    v = 0
                row.append(round(float(v), 3) if k == "ts" else (round(float(v), 1) if k == "fp" else int(round(float(v)))))
            rows.append(row)
        players.append(dict(id=pid, n=last.player_display_name, pos=last.position_group, tm=last.team, g=rows))
    players.sort(key=lambda p: (-sum(r[COLS.index("fp")] for r in p["g"] if r[0] == SEASONS[-1]), p["n"]))
    data = dict(cols=COLS, seasons=SEASONS, built=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
                through=dict(season=int(st.season.max()), week=int(st[st.season == st.season.max()].week.max())), players=players)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    print(f"Players: {len(players)} offensive players, {sum(len(p['g']) for p in players)} games, {os.path.getsize(out) / 1e6:.2f} MB -> {out}")
    return out


if __name__ == "__main__":
    build()
