"""Post-game review for every finished NFL game of the season -> output/hub/nflrev.json, keyed "AWAY@HOME" (called from season.py).

Each review is the same card the live score job writes (cloud_scores.review): final, the winner's price, spread and total result,
and each team's leaders from ESPN's game summary. A game is fetched once and kept in output/nfl_reviews.json, so a rebuild only
asks ESPN for games that finished since the last one. Where ESPN no longer carries a closing line, the nflverse schedule's line is used.
"""
import json
import os
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import cloud_scores as CS

ROOT = os.path.dirname(os.path.abspath(__file__))
KEEP = os.path.join(ROOT, "output", "nfl_reviews.json")


def num(v):
    return None if pd.isna(v) else float(v)


def one(r):
    g = dict(id=str(int(r.espn)), date=f"{r.gameday}T{r.gametime}", home=r.home_team, away=r.away_team, hs=int(r.home_score), as_=int(r.away_score), detail="Final")
    v = CS.review("nfl", g, CS.NFL_FIX)
    if not v:
        return None
    hm = g["hs"] - g["as_"]
    if v.get("spread_home") is None and num(r.spread_line) is not None:  # nflverse: positive = home favored; reviews use the home team's line
        sp = -num(r.spread_line)
        ats = hm + sp
        v.update(spread_home=sp, book="nflverse schedule", cover=None if ats == 0 else (g["home"] if ats > 0 else g["away"]), fav=None if sp == 0 else (g["home"] if sp < 0 else g["away"]))
    if v.get("total") is None and num(r.total_line) is not None:
        t, ou = g["hs"] + g["as_"], num(r.total_line)
        v.update(total=ou, ou="push" if t == ou else ("over" if t > ou else "under"))
    if v.get("ml_h") is None and num(r.home_moneyline) is not None:
        v.update(ml_h=int(num(r.home_moneyline)), ml_a=None if num(r.away_moneyline) is None else int(num(r.away_moneyline)))
    v["week"] = int(r.week)
    return v


def build(out, season=2026):
    g = pd.read_csv(os.path.join(ROOT, "data", "games.csv"), low_memory=False)
    g = g[(g.season == season) & g.result.notna() & g.espn.notna()]
    try:
        R = json.load(open(KEEP))
    except (OSError, ValueError):
        R = {}
    todo = [r for r in g.itertuples() if f"{r.away_team}@{r.home_team}" not in R]
    with ThreadPoolExecutor(8) as ex:
        for r, v in zip(todo, ex.map(one, todo)):
            if v:
                R[f"{r.away_team}@{r.home_team}"] = v
    if todo:
        json.dump(R, open(KEEP, "w"), separators=(",", ":"))
    json.dump(R, open(out, "w"), separators=(",", ":"))
    print(f"NFL reviews: {len(R)} of {len(g)} finished games ({len(todo)} fetched) -> {out}")
    return out


if __name__ == "__main__":
    build(os.path.join(ROOT, "output", "hub", "nflrev.json"))
