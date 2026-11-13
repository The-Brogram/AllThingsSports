"""Fantasy page data: every player's fantasy-point range for the current week, straight from the game simulations.

Each simulated game already draws a full stat line for every player (catches, rushing + receiving yards, TDs, and for the
starting QB the team's passing yards, passing TDs and interceptions). Scoring each draw gives a full distribution, so the
page can show the middle outcome, floor/ceiling, boom/bust chances and, for start/sit, the chance one player outscores another.

Scoring: 0.1 per rushing/receiving yard, 6 per rushing/receiving TD, 0.04 per pass yard, 4 or 6 per pass TD, -2 per INT,
plus 0 / 0.5 / 1 per catch. Fumbles and two-point conversions are not simulated, so they are left out.

Writes output/hub/fantasy.json. Run on its own (venv/bin/python fantasy.py) or through season.py.
"""
import base64
import glob
import json
import os
import pickle
import numpy as np
import pandas as pd
from engine import DATA

ROOT = os.path.dirname(os.path.abspath(__file__))
PPR = {"std": 0.0, "half": 0.5, "ppr": 1.0}
PTD = {"4": 4, "6": 6}
BOOM = {"QB": 25, "RB": 20, "WR": 20, "TE": 15}  # points, in whatever scoring is chosen
BUST = {"QB": 12, "RB": 8, "WR": 8, "TE": 5}
KEEP = 400  # draws kept per player for start/sit; the same draw index is the same simulated game, so teammates stay linked
MIN_PPR = 2.0  # players projected under this many PPR points are left off the page


def comps(df, sim, j):
    """Stat-line components for player j of one team's simulation, one value per draw."""
    p = df.iloc[j]
    c = dict(rec=sim["rec"][:, j], yd=sim["rec_y"][:, j] + sim["rush_y"][:, j], td=sim["ptds"][:, j] + sim["rtds"][:, j])
    if p.pos == "QB":
        if not p.starter_qb:
            return None
        c.update(py=sim["pass_y"], ptd=sim["ptd"], int=sim["ints"])
    return c


def points(c, ppr=1.0, ptd=4):
    f = 0.1 * c["yd"] + 6 * c["td"] + ppr * c["rec"]
    if "py" in c:
        f = f + 0.04 * c["py"] + ptd * c["ptd"] - 2 * c["int"]
    return f


def calib(p, w):
    """Logistic recalibration fit by fantasy_backtest.py (raw sims run low on boom chances and high on bust chances)."""
    if not w:
        return p
    p = min(max(p, 1e-3), 1 - 1e-3)
    return 1 / (1 + np.exp(-(w[0] + w[1] * np.log(p / (1 - p)))))


def summary(f, pos, cal=None):
    q = np.percentile(f, [10, 50, 90])
    cal = cal or {}
    return [round(float(f.mean()), 2), round(float(q[0]), 1), round(float(q[1]), 1), round(float(q[2]), 1),
            round(float(calib((f >= BOOM[pos]).mean(), cal.get("boom"))), 3), round(float(calib((f < BUST[pos]).mean(), cal.get("bust"))), 3)]


def enc(a):
    return base64.b64encode(np.clip(np.rint(a), -32000, 32000).astype("<i2").tobytes()).decode()


def actual_lines(season, week):
    s = pd.read_csv(f"{DATA}/stats_player_week_{season}.csv", low_memory=False)
    s = s[(s.week == week) & (s.season_type == "REG")]
    num = ["receptions", "receiving_yards", "rushing_yards", "receiving_tds", "rushing_tds", "passing_yards", "passing_tds", "passing_interceptions"]
    s = s.assign(**{c: s[c].fillna(0) for c in num})
    return {r.player_id: [int(r.receptions), int(r.receiving_yards + r.rushing_yards), int(r.receiving_tds + r.rushing_tds),
                          int(r.passing_yards), int(r.passing_tds), int(r.passing_interceptions)] for r in s.itertuples()}


def pick_week():
    """Dates to show: every cached slate of the latest week that has one."""
    sch = pd.read_csv(f"{DATA}/games.csv", low_memory=False, usecols=["season", "week", "gameday", "game_type", "result"])
    sch = sch[(sch.season == sch.season.max()) & (sch.game_type == "REG")].assign(date=lambda x: x.gameday.str.replace("-", ""))
    have = {os.path.basename(f)[6:14] for f in glob.glob(f"{DATA}/slate_*.pkl")}
    sch = sch[sch.date.isin(have)]
    if sch.empty:
        return None, None, []
    season, week = int(sch.season.iloc[0]), int(sch.week.max())
    return season, week, sorted(sch[sch.week == week].date.unique())


def build(out=os.path.join(ROOT, "output", "hub", "fantasy.json")):
    season, week, dates = pick_week()
    bt = os.path.join(ROOT, "output", "fantasy_backtest.json")
    bt = json.load(open(bt)) if os.path.exists(bt) else None
    cal = (bt or {}).get("cal")
    players, slates = [], []
    acts = actual_lines(season, week) if week else {}
    for d in dates:
        c = pickle.load(open(f"{DATA}/slate_{d}.pkl", "rb"))
        stat = c.get("inj_status", {})
        games = []
        for s in c["res"]:
            g = s["game"]
            gid = f"{g['away']}@{g['home']}"
            games.append(dict(id=gid, kick=g.get("kick"), total=g.get("total"), spread_home=g.get("spread_home"),
                              pts_h=round(float(s["model_h"]), 1), pts_a=round(float(s["model_a"]), 1)))
            for side, df, sim, team, opp in (("h", s["dfh"], s["sh"], g["home"], g["away"]), ("a", s["dfa"], s["sa"], g["away"], g["home"])):
                pl = {x["player_id"]: x for x in s["pl_" + side]}
                for j in range(len(df)):
                    p = df.iloc[j]
                    cc = comps(df, sim, j)
                    if cc is None or p.pos not in BOOM:
                        continue
                    if points(cc).mean() < MIN_PPR:
                        continue
                    sm = {f"{k}{t}": summary(points(cc, v, pt), p.pos, cal) for k, v in PPR.items() for t, pt in PTD.items()
                          if p.pos == "QB" or t == "4"}
                    x = pl.get(p.player_id, {})
                    keep = {k: enc(v[:KEEP]) for k, v in cc.items()}
                    players.append(dict(id=p.player_id, n=p["name"], pos=p.pos, tm=team, opp=opp, home=int(side == "h"), game=gid, date=d,
                                        st=stat.get(x.get("espn")) if x.get("espn") else None, pp=round(float(p.play_p), 2),
                                        tdp=round(float(x.get("atd", 0)), 3), tg=round(float(x.get("tg", 0)), 1), car=round(float(x.get("car", 0)), 1),
                                        s=sm, d=keep, act=acts.get(p.player_id)))
        slates.append(dict(date=d, games=games))
    # actual lines only for games that are final
    sch = pd.read_csv(f"{DATA}/games.csv", low_memory=False, usecols=["season", "week", "away_team", "home_team", "result"])
    fin = set((sch[(sch.season == season) & (sch.week == week) & sch.result.notna()].pipe(lambda x: x.away_team + "@" + x.home_team)))
    for p in players:
        if p["game"] not in fin:
            p["act"] = None
        elif p["act"] is None:
            p["act"] = [0, 0, 0, 0, 0, 0]  # projected but recorded no stat line (inactive or no touches)
    data = dict(built=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"), season=season, week=week, keep=KEEP, boom=BOOM, bust=BUST,
                final=sorted(fin), slates=slates, players=players, backtest=bt)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    print(f"Fantasy: {season} week {week}, {len(dates)} slate(s), {len(players)} players, {os.path.getsize(out) / 1e6:.2f} MB -> {out}")
    return out


if __name__ == "__main__":
    build()
