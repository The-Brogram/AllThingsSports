"""Build output/hub/nba.json for the season hub's NBA pages (Tonight, Track record, Player lookup). Called from season.py.

Tonight: the latest forecast night (games, win chance and margin range, every player's lines with our probability).
Track record: every graded night (review_<date>.json) plus the walk-forward backtest calibration (output/bt_summary.json).
Player lookup: season averages and the last 10 games for everyone who has played this season.
"""
import glob
import json
import os
import pickle
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = f"{HERE}/output"
D = f"{HERE}/data"
ET = ZoneInfo("America/New_York")


def r(x, k=3):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), k)


def teams():
    t = {}
    for f in sorted(glob.glob(f"{D}/sb/*.json"))[-60:]:
        for e in json.load(open(f)):
            for x in e["teams"]:
                t[x["ab"]] = dict(name=x["name"], short=x["short"], color=x.get("color"), rec=x.get("rec") or t.get(x["ab"], {}).get("rec"))
    return t


def tonight():
    fs = sorted(glob.glob(f"{OUT}/nba_*.pkl"))
    if not fs:
        return None
    m = pickle.load(open(fs[-1], "rb"))
    date = m["date"]
    pr = pd.read_csv(f"{OUT}/predictions_{date}.csv", dtype={"pid": str, "game_id": str})
    try:  # forecasts made before the parser learned to reject wrong-market rebound prices (Nov 12, 2026) may still carry them: hide those
        ok = pd.read_csv(f"{D}/props.csv.gz", dtype={"pid": str, "game_id": str})
        ok = ok[(ok.kind == "milestone") & (ok.market == "reb")][["game_id", "pid", "line"]].drop_duplicates().assign(_ok=1)
        pr = pr.merge(ok, on=["game_id", "pid", "line"], how="left")
        pr = pr[~((pr.kind == "milestone") & (pr.market == "reb") & pr._ok.isna())].drop(columns="_ok")
    except (OSError, KeyError):
        pass
    sb = {e["id"]: e for e in json.load(open(f"{D}/sb/{date.replace('-', '')}.json"))}
    rv = f"{OUT}/review_{date}.json"
    rv = json.load(open(rv)) if os.path.exists(rv) else None
    games = []
    for g in m["games"]:
        e = sb.get(g["game_id"], {})
        sc = {x["ha"]: x.get("score") for x in e.get("teams", [])}
        games.append(dict(id=g["game_id"], t=g["date"], home=g["home"], away=g["away"], spread=g["spread_home"], total=g["total"], ph=r(g["p_home"]),
                          mq=g["margin_q"], mh=g["margin_hist"], tm=r(g["total_mean"], 1), tv=e.get("tv"), state=e.get("state"),
                          hs=sc.get("home") if e.get("state") != "pre" else None, as_=sc.get("away") if e.get("state") != "pre" else None))
    pl = []
    for (gid, pid), x in pr.groupby(["game_id", "pid"], sort=False):
        f = x.iloc[0]
        ln = [[y.market, r(y.line, 1), {"line": 0, "milestone": 1, "model": 2, "alt": 3}[y.kind], r(y.p_over), r(y.dk_over), None if pd.isna(y.price_over) else int(y.price_over),
               None if pd.isna(y.price_under) else int(y.price_under), r(y.mean, 1), r(y.p10, 0), r(y.p90, 0)] for y in x.itertuples()]
        pl.append(dict(g=gid, id=pid, n=f["name"], t=f.team, o=f.opp, pos=f.pos if isinstance(f.pos, str) else "", pp=r(f.play_p, 2), mn=r(f.proj_min, 1),
                       inj=f.inj if isinstance(f.inj, str) else None, note=f.inj_note if isinstance(f.inj_note, str) else None, l=ln))
    built = datetime.fromisoformat(m["built"]).astimezone(ET)
    return dict(date=date, built=f"{built:%a %b} {built.day}, {built:%-I:%M %p}", games=games, players=pl, n=m["n"],
                graded=None if not rv else dict(games=rv["games"], bands=rv["bands"]))


def record():
    days = []
    for f in sorted(glob.glob(f"{OUT}/review_*.json")):
        v = json.load(open(f))
        days.append(dict(date=v["date"], gr=sum(x["right"] for x in v["games"]), gn=len(v["games"]), bands=v["bands"], n=v["n"],
                         top=[{k: x[k] for k in ("name", "team", "market", "kind", "line", "side", "p", "actual", "hit")} for x in v["lines"][:12]]))
    bt = f"{OUT}/bt_summary.json"
    ld = lambda p: json.load(open(p)) if os.path.exists(p) else None
    edge, live = ld(f"{OUT}/edge.json"), ld(f"{OUT}/edge_live.json")  # results at real DraftKings prices (edge_nba.py)
    if live:
        live = {k: v for k, v in live.items() if k != "picks"} | dict(picks=live.get("picks", [])[:30])
    return dict(days=days, bt=ld(bt), edge=edge, live=live)


def lookup():
    p = pd.read_csv(f"{D}/players.csv.gz", dtype={"pid": str, "game_id": str})
    g = pd.read_csv(f"{D}/games.csv", dtype={"game_id": str})
    p = p.merge(g[["game_id", "date_et", "season"]], on="game_id")
    p = p[(p.season == p.season.max()) & (p["min"] > 0)].sort_values("date_et")
    out = []
    for pid, x in p.groupby("pid"):
        s = x[["min", "pts", "reb", "ast", "fg3", "stl", "blk"]].mean()
        last = x.tail(10)
        out.append(dict(id=pid, n=x.name.iloc[-1], t=x.team.iloc[-1], pos=x.pos.iloc[-1] if isinstance(x.pos.iloc[-1], str) else "", gp=len(x),
                        avg=[r(s[k], 1) for k in ("min", "pts", "reb", "ast", "fg3", "stl", "blk")],
                        log=[[y.date_et[5:], y.opp, int(y.home), int(y.min), int(y.pts), int(y.reb), int(y.ast), int(y.fg3)] for y in last.itertuples()][::-1]))
    return sorted(out, key=lambda z: -z["avg"][1])


# team comparison: (key, label, better: 1 higher, -1 lower, 0 neither, format)
TS = [("ppg", "Points scored", 1, 1), ("oppg", "Points allowed", -1, 1), ("net", "Point differential", 1, 1),
      ("ortg", "Offensive rating", 1, 1), ("drtg", "Defensive rating", -1, 1), ("pace", "Pace", 0, 1),
      ("efg", "Effective FG%", 1, -1), ("oefg", "Opponent effective FG%", -1, -1), ("fgp", "FG%", 1, -1), ("p3", "3-point %", 1, -1),
      ("a3", "3-pointers attempted", 0, 1), ("op3", "Opponent 3-point %", -1, -1), ("ftr", "Free throw rate", 1, 3),
      ("reb", "Rebounds", 1, 1), ("oreb", "Offensive rebounds", 1, 1), ("orebp", "Offensive rebound %", 1, -1), ("ast", "Assists", 1, 1),
      ("tov", "Turnovers", -1, 1), ("otov", "Turnovers forced", 1, 1), ("stl", "Steals", 1, 1), ("blk", "Blocks", 1, 1)]


def teamstats(season=None, reg_only=False):
    """Season per-game team stats from the box scores (completed games; current season by default), each with its league rank (1 = best)."""
    p = pd.read_csv(f"{D}/players.csv.gz", dtype={"game_id": str})
    g = pd.read_csv(f"{D}/games.csv", dtype={"game_id": str})
    g = g[g.completed & (g.season == (season or g.season.max()))]
    g = g[~g.home.isin(EXHIB) & ~g.away.isin(EXHIB)]
    if reg_only:
        g = g[g.stype == 2]
    cols = ["pts", "reb", "ast", "oreb", "dreb", "stl", "blk", "tov", "fgm", "fga", "fg3", "fg3a", "ftm", "fta"]
    t = p[p.game_id.isin(g.game_id)].groupby(["game_id", "team"])[cols].sum().reset_index()
    o = t.rename(columns={c: "o_" + c for c in cols + ["team"]})
    t = t.merge(o, on="game_id")
    t = t[t.team != t.o_team].merge(g[["game_id", "date_et", "home", "away", "hs", "as", "spread_home"]], on="game_id")
    t["poss"] = ((t.fga - t.oreb + t.tov + .44 * t.fta) + (t.o_fga - t.o_oreb + t.o_tov + .44 * t.o_fta)) / 2
    t["is_home"] = t.team == t.home
    t["margin"] = np.where(t.is_home, t.hs - t["as"], t["as"] - t.hs)
    t["spread"] = np.where(t.is_home, t.spread_home, -t.spread_home)  # positive = this team favored by that much
    t = t.sort_values("date_et")
    out = {}
    for team, x in t.groupby("team"):
        s = x[cols + ["o_" + c for c in cols] + ["poss"]].sum()
        n = len(x)
        v = dict(ppg=s.pts / n, oppg=s.o_pts / n, net=(s.pts - s.o_pts) / n, ortg=100 * s.pts / s.poss, drtg=100 * s.o_pts / s.poss, pace=s.poss / n,
                 efg=(s.fgm + .5 * s.fg3) / s.fga, oefg=(s.o_fgm + .5 * s.o_fg3) / s.o_fga, fgp=s.fgm / s.fga, p3=s.fg3 / s.fg3a, a3=s.fg3a / n,
                 op3=s.o_fg3 / s.o_fg3a, ftr=s.fta / s.fga, reb=s.reb / n, oreb=s.oreb / n, orebp=s.oreb / (s.oreb + s.o_dreb), ast=s.ast / n,
                 tov=s.tov / n, otov=s.o_tov / n, stl=s.stl / n, blk=s.blk / n)
        rec = lambda z: [int((z.margin > 0).sum()), int((z.margin < 0).sum())]
        h, a, l10 = x[x.is_home], x[~x.is_home], x.tail(10)
        ats = x[x.spread.notna() & (x.margin + (-x.spread) != 0)]
        cov = ats.margin - ats.spread
        out[team] = dict(gp=n, v=v, rec=rec(x), home=rec(h), away=rec(a), l10=rec(l10), l10net=r(l10.margin.mean(), 1),
                         ats=[int((cov > 0).sum()), int((cov < 0).sum())], fav=rec(x[x.spread > 0]), dog=rec(x[x.spread < 0]),
                         last5=[[y.date_et[5:], y.o_team, int(y.is_home), int(y.margin)] for y in x.tail(5).itertuples()][::-1])
    for k, _, better, _ in TS:
        vals = pd.Series({tm: d["v"][k] for tm, d in out.items()})
        rk = vals.rank(ascending=better < 0 if better else False, method="min")
        for tm in out:
            out[tm]["v"][k] = [r(out[tm]["v"][k], 4), int(rk[tm])]
    return dict(n=len(out), through=t.date_et.max() if len(t) else None, metrics=[list(m) for m in TS], teams=out)


LOGK = ["d", "opp", "home", "ts", "os", "min", "pts", "reb", "ast", "fg3", "stl", "blk", "tov", "pf", "pm", "usg", "fgm", "fga", "fg3a", "ftm", "fta", "out", "season"]


def profiles(hub):
    """nbap.json: for every player in tonight's forecast, his game log (this season, the end of last season, and every game
    against tonight's opponent), the sim's full outcome range for each stat, and calibration to price any line.
    Each log row also lists teammates who sat that game (players averaging 8+ points who were on the team then)."""
    fs = sorted(glob.glob(f"{OUT}/nba_*.pkl"))
    if not fs:
        return {}
    m = pickle.load(open(fs[-1], "rb"))
    pr = pd.read_csv(f"{OUT}/predictions_{m['date']}.csv", dtype={"pid": str, "game_id": str})
    tp = pr.drop_duplicates("pid").set_index("pid")
    p = pd.read_csv(f"{D}/players.csv.gz", dtype={"pid": str, "game_id": str})
    g = pd.read_csv(f"{D}/games.csv", dtype={"game_id": str})
    g = g[g.completed & (g.date_et < m["date"])]
    p = p.merge(g[["game_id", "date_et", "season", "home", "hs", "as"]].rename(columns={"home": "home_team"}), on="game_id")
    p["hm"] = (p.team == p.home_team).astype(int)
    p["ts"] = np.where(p.hm == 1, p.hs, p["as"])
    p["os"] = np.where(p.hm == 1, p["as"], p.hs)
    # usage %: share of team plays (FGA + .44 FTA + TOV) used while on the floor
    tt = p.groupby(["game_id", "team"])[["fga", "fta", "tov", "min"]].sum().add_prefix("t_").reset_index()
    p = p.merge(tt, on=["game_id", "team"])
    p["usg"] = np.where(p["min"] > 0, 100 * (p.fga + .44 * p.fta + p.tov) * (p.t_min / 5) / (p["min"] * (p.t_fga + .44 * p.t_fta + p.t_tov)).replace(0, np.nan), np.nan)
    # who sat: team regulars (8+ ppg that season) between their first and last game for the team, with no minutes that night
    pl = p[p["min"] > 0]
    reg = pl.groupby(["team", "season", "pid"]).agg(ppg=("pts", "mean"), gp=("pts", "size"), d0=("date_et", "min"), d1=("date_et", "max"), name=("name", "last")).reset_index()
    reg = reg[(reg.ppg >= 8) & (reg.gp >= 3)]
    played = set(zip(pl.game_id, pl.pid))
    tg = p.drop_duplicates(["game_id", "team"])[["game_id", "team", "season", "date_et"]]
    out = {}
    for (team, seas), x in tg.groupby(["team", "season"]):
        rr = reg[(reg.team == team) & (reg.season == seas)]
        for y in x.itertuples():
            out[(y.game_id, team)] = [q.pid for q in rr.itertuples() if q.d0 < y.date_et < q.d1 and (y.game_id, q.pid) not in played]
    names = {q.pid: [q.name, round(float(q.ppg), 1)] for q in reg.sort_values("season").itertuples()}
    res, used = {}, set()
    for pid, x in pl[pl.pid.isin(tp.index)].sort_values("date_et").groupby("pid"):
        opp = tp.loc[pid, "opp"]
        cur = x[x.season == x.season.max()] if x.season.max() == g.season.max() else x.iloc[0:0]
        prev = x[x.season < g.season.max()]
        keep = pd.concat([prev.tail(max(0, 20 - len(cur))), prev[prev.opp == opp], cur]).drop_duplicates("game_id").sort_values("date_et")
        log = []
        for y in keep.itertuples():
            o = out.get((y.game_id, y.team), [])
            used.update(o)
            log.append([y.date_et, y.opp, y.hm, int(y.ts), int(y.os)] + [None if pd.isna(getattr(y, k)) else int(getattr(y, k)) for k in ("min", "pts", "reb", "ast", "fg3", "stl", "blk", "tov", "pf", "pm")]
                       + [None if pd.isna(y.usg) else round(float(y.usg), 1)] + [None if pd.isna(getattr(y, k)) else int(getattr(y, k)) for k in ("fgm", "fga", "fg3a", "ftm", "fta")] + [o, int(y.season)])
        res[pid] = dict(log=log, dist=m["dist"].get(pid))
    for pid in tp.index:
        res.setdefault(pid, dict(log=[], dist=m["dist"].get(pid)))
    cal = json.load(open(f"{OUT}/calibration.json")) if os.path.exists(f"{OUT}/calibration.json") else None
    j = dict(date=m["date"], keys=LOGK, names={k: names[k] for k in used if k in names}, cal=cal and dict(alt=cal["alt"]), p=res)
    f = os.path.join(hub, "nbap.json")
    json.dump(j, open(f, "w"), separators=(",", ":"), allow_nan=False)
    import heads_nba
    pids = sorted(set(tp.index))
    heads_nba.build(pids, os.path.join(hub, "heads.json"))
    return {"nbap.json": f, "heads.json": os.path.join(hub, "heads.json")}


EXHIB = {"STARS", "STRIPES", "WORLD", "USA", "TEAM LEBRON", "LEBRON", "GIANNIS", "DURANT"}  # All-Star / exhibition sides ESPN files as regular season
PLK = ["id", "n", "t", "pos", "gp", "min", "pts", "reb", "ast", "fg3", "stl", "blk", "tov", "oreb", "fgm", "fga", "fg3a", "ftm", "fta", "dd", "td", "pm", "tg"]


def leaders(hub):
    """nbl.json for the Season leaders page: every player's season totals and every team's per-game stats, for this season
    and last season (regular season only). The page does per-game math, combos, ranks and qualifying minimums."""
    p = pd.read_csv(f"{D}/players.csv.gz", dtype={"pid": str, "game_id": str})
    g = pd.read_csv(f"{D}/games.csv", dtype={"game_id": str})
    g = g[g.completed & (g.stype == 2) & ~g.home.isin(EXHIB) & ~g.away.isin(EXHIB)]
    p = p.merge(g[["game_id", "season", "date_et"]], on="game_id")
    p = p[p["min"] > 0].sort_values("date_et")
    tg = p.drop_duplicates(["game_id", "team"]).groupby(["season", "team"]).size()
    out = []
    for seas in sorted(p.season.unique(), reverse=True)[:2]:
        x = p[p.season == seas].copy()
        x["dd"] = ((x[["pts", "reb", "ast", "stl", "blk"]] >= 10).sum(axis=1) >= 2).astype(int)
        x["td"] = ((x[["pts", "reb", "ast", "stl", "blk"]] >= 10).sum(axis=1) >= 3).astype(int)
        agg = x.groupby("pid").agg(n=("name", "last"), t=("team", "last"), pos=("pos", "last"), gp=("game_id", "size"),
                                   **{k: (k, "sum") for k in ["min", "pts", "reb", "ast", "fg3", "stl", "blk", "tov", "oreb", "fgm", "fga", "fg3a", "ftm", "fta", "dd", "td", "pm"]}).reset_index()
        agg["tg"] = [int(tg.get((seas, t), 0)) for t in agg.t]
        rows = [[y.pid, y.n, y.t, y.pos if isinstance(y.pos, str) else ""] + [int(getattr(y, k)) for k in PLK[4:]] for y in agg.itertuples()]
        ts = teamstats(int(seas), reg_only=True)
        out.append(dict(season=int(seas), label=f"{int(seas) - 1}-{str(int(seas))[2:]}", through=x.date_et.max(), players=rows, teams=ts["teams"], tmetrics=ts["metrics"],
                        maxg=int(tg[seas].max())))
    f = os.path.join(hub, "nbl.json")
    json.dump(dict(keys=PLK, seasons=out), open(f, "w"), separators=(",", ":"), allow_nan=False)
    # photos for the leaders table: rotation players (15+ min a night, 5+ games) from either season, loaded only on that page
    import heads_nba
    pids = sorted({r[0] for x in out for r in x["players"] if r[4] >= 5 and r[5] >= 15 * r[4]})
    heads_nba.build(pids, os.path.join(hub, "nblh.json"))
    return {"nbl.json": f, "nblh.json": os.path.join(hub, "nblh.json")}



DVK = [["pts", "Points", 1], ["reb", "Rebounds", 1], ["ast", "Assists", 1], ["stl", "Steals", 1], ["blk", "Blocks", 1], ["fg3", "3-pointers", 1],
       ["tov", "Turnovers", -1], ["fgp", "Field goal %", 1], ["ftp", "Free throw %", 1]]
ZONES = [["RA", "Restricted area"], ["PNT", "In the paint (non-RA)"], ["MID", "Mid-range"], ["LC3", "Left corner 3"], ["RC3", "Right corner 3"], ["AB3", "Above the break 3"]]
SIMK = ["pts", "reb", "ast", "fg3", "stl", "blk", "tov", "pr", "pa", "ra", "pra"]


def matchups(hub):
    """nbm.json for the matchup sections under a player's profile (this regular season):
    - dvp: what each defense allows per game to guards / forwards / centers (season, last 7, last 15), ranked 1-30 where 30 = best for the shooter
    - tz: each defense's points allowed per game from each shot zone and FG% there, ranked the same way
    - pz: each of tonight's players' shots by zone (attempts, makes, points) plus free-throw points
    - og: every player game against each team (for "similar players vs this team"), with DraftKings' main line when there was one
    - avg: season averages per player for the similarity match"""
    p = pd.read_csv(f"{D}/players.csv.gz", dtype={"pid": str, "game_id": str})
    g = pd.read_csv(f"{D}/games.csv", dtype={"game_id": str})
    seas = g.season.max()
    g = g[g.completed & (g.season == seas) & (g.stype == 2) & ~g.home.isin(EXHIB) & ~g.away.isin(EXHIB)]
    p = p.merge(g[["game_id", "date_et"]], on="game_id")
    p = p[p["min"] > 0].copy()
    p["pg"] = p.pos.fillna("").str[-1].map({"G": "G", "F": "F", "C": "C"}).fillna("F")
    # --- defense vs position
    tg = p.groupby(["game_id", "opp", "pg"])[["pts", "reb", "ast", "stl", "blk", "fg3", "tov", "fgm", "fga", "ftm", "fta"]].sum().reset_index()
    tg = tg.merge(g[["game_id", "date_et"]], on="game_id")
    dvp = {}
    for pos in "GFC":
        x = tg[tg.pg == pos]
        out = {}
        for lab, n in (("s", None), ("l7", 7), ("l15", 15)):
            rows = {}
            for t, y in x.sort_values("date_et").groupby("opp"):
                y = y.tail(n) if n else y
                v = {k: y[k].mean() for k in ["pts", "reb", "ast", "stl", "blk", "fg3", "tov"]}
                v["fgp"] = y.fgm.sum() / y.fga.sum() if y.fga.sum() else None
                v["ftp"] = y.ftm.sum() / y.fta.sum() if y.fta.sum() else None
                rows[t] = v
            for k, _, d in DVK:
                order = sorted([t for t in rows if rows[t][k] is not None], key=lambda t: rows[t][k] * d)
                for i, t in enumerate(order):
                    rows[t][k + "_r"] = i + 1
            out[lab] = rows
        dvp[pos] = {t: {k: [round(out["s"][t][k], 3 if k in ("fgp", "ftp") else 1), out["s"][t][k + "_r"], out["l7"].get(t, {}).get(k + "_r"), out["l15"].get(t, {}).get(k + "_r")]
                        for k, _, _ in DVK if out["s"][t].get(k) is not None} for t in out["s"]}
    # --- shot zones
    tz, pz = {}, {}
    try:
        sh = pd.read_csv(f"{D}/shots.csv.gz", dtype={"pid": str, "game_id": str})
        sh = sh[sh.game_id.isin(g.game_id)]
        home = dict(zip(g.game_id, g.home)); away = dict(zip(g.game_id, g.away))
        sh["dteam"] = [away[i] if t == home[i] else home[i] for i, t in zip(sh.game_id, sh.team)]
        ng = sh.groupby("dteam").game_id.nunique()
        z = sh[sh.ft == 0].groupby(["dteam", "zone"]).agg(a=("made", "size"), m=("made", "sum"), pts=("pts", "sum")).reset_index()
        rows = {}
        for r in z.itertuples():
            rows.setdefault(r.dteam, {})[r.zone] = [round(r.pts / ng[r.dteam], 1), round(r.m / r.a, 3) if r.a else None, round(r.a / ng[r.dteam], 1)]
        for zk, _ in ZONES:
            order = sorted([t for t in rows if zk in rows[t]], key=lambda t: rows[t][zk][0])
            for i, t in enumerate(order):
                rows[t][zk].append(i + 1)
        tz = rows
        fs = sorted(glob.glob(f"{OUT}/predictions_*.csv"))
        tonight = set(pd.read_csv(fs[-1], dtype={"pid": str}).pid) if fs else set()
        q = sh[sh.pid.isin(tonight)]
        for pid, y in q.groupby("pid"):
            d = {r.zone: [int(r.a), int(r.m), int(r.pts)] for r in y.groupby("zone").agg(a=("made", "size"), m=("made", "sum"), pts=("pts", "sum")).reset_index().itertuples()}
            d["g"] = int(y.game_id.nunique())
            pz[pid] = d
    except FileNotFoundError:
        pass
    # --- every player game vs each team, with DraftKings' main line
    for k, parts in (("pr", ("pts", "reb")), ("pa", ("pts", "ast")), ("ra", ("reb", "ast")), ("pra", ("pts", "reb", "ast"))):
        p[k] = sum(p[c] for c in parts)
    ln = {}
    try:
        pp = pd.read_csv(f"{D}/props.csv.gz", dtype={"pid": str, "game_id": str})
        pp = pp[(pp.kind == "line") & pp.game_id.isin(g.game_id)]
        o = pp[pp.side == "over"][["game_id", "pid", "market", "line", "price"]].merge(pp[pp.side == "under"][["game_id", "pid", "market", "line", "price"]], on=["game_id", "pid", "market", "line"], suffixes=("_o", "_u"))
        o["gap"] = (o.price_o - o.price_u).abs()
        o = o.sort_values("gap").drop_duplicates(["game_id", "pid", "market"])
        ln = {(a, b, c): float(l) for a, b, c, l in zip(o.game_id, o.pid, o.market, o.line)}
    except FileNotFoundError:
        pass
    avg = p.groupby("pid").agg(n=("name", "last"), pg=("pg", "last"), t=("team", "last"), gp=("game_id", "size"), mn=("min", "mean"), **{k: (k, "mean") for k in SIMK}).reset_index()
    og = {}
    for r in p.sort_values("date_et", ascending=False).itertuples():
        og.setdefault(r.opp, []).append([r.pid, r.date_et[5:], r.team, int(r.min)] + [int(getattr(r, k)) for k in SIMK] + [ln.get((r.game_id, r.pid, k)) for k in SIMK])
    out = dict(season=int(seas), through=g.date_et.max(), dvk=DVK, zones=ZONES, simk=SIMK, dvp=dvp, tz=tz, pz=pz, og=og,
               avg={r.pid: [r.n, r.pg, r.t, int(r.gp), round(r.mn, 1)] + [round(getattr(r, k), 2) for k in SIMK] for r in avg.itertuples()})
    f = os.path.join(hub, "nbm.json")
    json.dump(out, open(f, "w"), separators=(",", ":"), allow_nan=False)
    return {"nbm.json": f}


# arena city, latitude, longitude, hours behind Eastern (straight-line travel distances; LAL and LAC share a city)
ARENA = {"ATL": ("Atlanta", 33.757, -84.396, 0), "BOS": ("Boston", 42.366, -71.062, 0), "BKN": ("Brooklyn", 40.683, -73.975, 0), "CHA": ("Charlotte", 35.225, -80.839, 0),
         "CHI": ("Chicago", 41.881, -87.674, 1), "CLE": ("Cleveland", 41.497, -81.688, 0), "DAL": ("Dallas", 32.790, -96.810, 1), "DEN": ("Denver", 39.749, -105.008, 2),
         "DET": ("Detroit", 42.341, -83.055, 0), "GS": ("San Francisco", 37.768, -122.388, 3), "HOU": ("Houston", 29.751, -95.362, 1), "IND": ("Indianapolis", 39.764, -86.156, 0),
         "LAC": ("Los Angeles", 33.945, -118.341, 3), "LAL": ("Los Angeles", 34.043, -118.267, 3), "MEM": ("Memphis", 35.138, -90.051, 1), "MIA": ("Miami", 25.781, -80.187, 0),
         "MIL": ("Milwaukee", 43.045, -87.917, 1), "MIN": ("Minneapolis", 44.980, -93.276, 1), "NO": ("New Orleans", 29.949, -90.082, 1), "NY": ("New York", 40.751, -73.994, 0),
         "OKC": ("Oklahoma City", 35.463, -97.515, 1), "ORL": ("Orlando", 28.539, -81.384, 0), "PHI": ("Philadelphia", 39.901, -75.172, 0), "PHX": ("Phoenix", 33.446, -112.071, 2),
         "POR": ("Portland", 45.532, -122.667, 3), "SA": ("San Antonio", 29.427, -98.438, 1), "SAC": ("Sacramento", 38.580, -121.500, 3), "TOR": ("Toronto", 43.643, -79.379, 0),
         "UTAH": ("Salt Lake City", 40.768, -111.901, 2), "WSH": ("Washington", 38.898, -77.021, 0)}


def miles(a, b):
    """Straight-line miles between two teams' arenas; 0 inside one city, None when a site is unknown."""
    if a not in ARENA or b not in ARENA:
        return None
    if ARENA[a][0] == ARENA[b][0]:
        return 0
    la1, lo1, la2, lo2 = map(np.radians, (ARENA[a][1], ARENA[a][2], ARENA[b][1], ARENA[b][2]))
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return int(round(3958.8 * 2 * np.arcsin(np.sqrt(h)) / 10) * 10)


def travel(tn):
    """For each of tonight's games, each team's travel and rest: where it played last and how far it came, back-to-backs,
    games and miles in the last week, the length of the road trip or home stand, and where it goes next.
    A game's site is the home team's arena; a neutral-site game has no known city, so legs in and out of it carry no distance."""
    g = pd.read_csv(f"{D}/games.csv", dtype={"game_id": str})
    g = g[g.stype.isin([2, 3, 5]) & g.home.isin(ARENA) & g.away.isin(ARENA)].sort_values("date")
    g["day"] = pd.to_datetime(g.date_et)
    today = pd.Timestamp(tn["date"])
    g = g[g.day >= today - pd.Timedelta(days=45)]
    city = lambda site: ARENA[site][0] if site else "a neutral site"

    def team(t):
        x = g[(g.home == t) | (g.away == t)]
        rows = [dict(day=r.day, site=None if r.neutral else r.home, ha="N" if r.neutral else ("H" if r.home == t else "A"), opp=r.away if r.home == t else r.home)
                for _, r in x.iterrows()]
        i = next((k for k, r in enumerate(rows) if r["day"] == today), None)
        if i is None:
            return None
        cur, past = rows[i], rows[:i]
        leg = lambda a, b: None if a["site"] is None or b["site"] is None else miles(a["site"], b["site"])
        out = dict(to=city(cur["site"]), ha=cur["ha"], flags=[])
        trail, seq = [], past[-5:] + [cur]
        for k, r in enumerate(seq):
            prev = seq[k - 1] if k else (past[-6] if len(past) > 5 else None)
            trail.append([f"{r['day']:%Y-%m-%d}", city(r["site"]), r["ha"], r["opp"], None if prev is None else leg(prev, r), None if prev is None else int((r["day"] - prev["day"]).days)])
        out["trail"] = trail
        if past:
            last = past[-1]
            gap = int((cur["day"] - last["day"]).days)
            mi = leg(last, cur)
            tz = None if last["site"] is None or cur["site"] is None else ARENA[last["site"]][3] - ARENA[cur["site"]][3]  # + = moved east
            out.update(frm=city(last["site"]), last=f"{last['day']:%Y-%m-%d}", last_opp=last["opp"], last_ha=last["ha"], gap=gap, mi=mi, tz=tz, b2b=gap == 1)
            if gap == 1:
                out["flags"].append("Second night of a back-to-back" + ("" if mi is None else ", no travel" if mi == 0 else f", after a {mi:,}-mile trip"))
            for n, d in ((3, 4), (4, 6), (5, 7)):
                if len(past) >= n - 1 and (span := (cur["day"] - past[-(n - 1)]["day"]).days + 1) <= d:
                    out["flags"].append(f"{n}{'rd' if n == 3 else 'th'} game in {span} nights")
            if tz and abs(tz) >= 2:
                out["flags"].append(f"{abs(tz)} time zones {'east' if tz > 0 else 'west'} of the last game")
        # road trip or home stand: consecutive games at the same kind of site, ending tonight
        n = 1
        for r in reversed(past):
            if r["ha"] != cur["ha"]:
                break
            n += 1
        out["run"] = n
        if cur["ha"] == "A" and n >= 3:
            out["flags"].append(f"Game {n} of a road trip")
        wk = [r for r in past if (cur["day"] - r["day"]).days <= 6] + [cur]
        legs = [leg(a, b) for a, b in zip(wk, wk[1:])]
        out["g7"] = len(wk)
        out["mi7"] = int(sum(x for x in legs if x)) if legs else 0
        out["cities7"] = len({r["site"] for r in wk if r["site"]})
        if i + 1 < len(rows):
            nx = rows[i + 1]
            out["next"] = [f"{nx['day']:%Y-%m-%d}", city(nx["site"]), nx["ha"], nx["opp"], leg(cur, nx), int((nx["day"] - cur["day"]).days)]
            if out["next"][5] == 1:
                out["flags"].append("First night of a back-to-back: plays " + ("at home" if nx["ha"] == "H" else f"in {city(nx['site'])}") + " tomorrow")
        return out

    for gm in tn["games"]:
        gm["trv"] = {t: team(t) for t in (gm["away"], gm["home"])}


def old_games():
    """Final scores from the three seasons before our own data starts, from ESPN team schedules (fetched once, then cached)."""
    f = f"{D}/h2h_old.json"
    if os.path.exists(f):
        return json.load(open(f))
    import subprocess
    from concurrent.futures import ThreadPoolExecutor
    ids = {}
    for sb in sorted(glob.glob(f"{D}/sb/*.json"))[-40:]:
        for e in json.load(open(sb)):
            for t in e.get("teams", []):
                ids[t["ab"]] = t["id"]
    def one(a):
        tid, yr = a
        r = subprocess.run(["curl", "-s", "--max-time", "40", f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{tid}/schedule?season={yr}&seasontype=2"], capture_output=True, text=True)
        try:
            return json.loads(r.stdout).get("events", [])
        except ValueError:
            return []
    seen = {}
    with ThreadPoolExecutor(8) as ex:
        for ev in ex.map(one, [(i, y) for i in sorted(set(ids.values())) for y in (2023, 2024, 2025)]):
            for e in ev:
                c = e["competitions"][0]
                if not c.get("status", {}).get("type", {}).get("completed"):
                    continue
                t = {x["homeAway"]: x for x in c["competitors"]}
                sc = lambda x: (x.get("score") or {}).get("value") if isinstance(x.get("score"), dict) else x.get("score")
                try:
                    seen[e["id"]] = [e["date"][:10], t["home"]["team"]["abbreviation"], t["away"]["team"]["abbreviation"], int(float(sc(t["home"]))), int(float(sc(t["away"])))]
                except (KeyError, TypeError, ValueError):
                    continue
    out = sorted(seen.values())
    if len(out) > 2000:  # only cache a complete pull
        json.dump(out, open(f, "w"), separators=(",", ":"))
    return out


def matchup_history(tn):
    """For each of tonight's games: the last 10 meetings, and what has changed on each roster since the last one we have box scores for."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import nba_model as M
    d = M.load()
    g = d["games"]
    g = g[g.completed & g.hs.notna()]
    rows = [[r.date_et, r.home, r.away, int(r.hs), int(r["as"]), r.game_id] for _, r in g.iterrows()] + [x + [None] for x in old_games()]
    p = d["players"]
    p = p[p["min"] > 0]
    seas0 = p[p.season == p.season.max()].date_et.min()
    pr = pd.read_csv(f"{OUT}/predictions_{tn['date']}.csv", dtype={"pid": str})
    for gm in tn["games"]:
        H, A = gm["home"], gm["away"]
        h2h = sorted([x for x in rows if {x[1], x[2]} == {H, A} and x[0] < tn["date"]], key=lambda x: x[0], reverse=True)  # meetings before this slate
        gm["h2h"] = [x[:5] for x in h2h[:10]]
        last = next((x for x in h2h if x[5]), None)
        if not last:
            gm["chg"] = None
            continue
        chg = dict(since=last[0])
        for t in (H, A):
            was = p[(p.game_id == last[5]) & (p.team == t)].set_index("pid")["min"].to_dict()
            tg = p[p.team == t].drop_duplicates("game_id").game_id.tolist()[-5:]
            cur = p[(p.team == t) & p.game_id.isin(tg)].groupby("pid").agg(m=("min", "mean"), n=("name", "last"))
            new, gone = [], []
            for pid, r in cur[cur.m >= 12].sort_values("m", ascending=False).iterrows():
                if pid in was:
                    continue
                hist = p[(p.pid == pid) & (p.date_et < last[0])]
                other = hist[hist.team != t]
                new.append(f"{r.n}: " + (f"joined from {other.team.iloc[-1]}" if len(other) and not len(hist[hist.team == t]) else
                                         ("first season in our data" if not len(p[(p.pid == pid) & (p.date_et < seas0)]) else "did not play in the last meeting")) + f" ({r.m:.0f} min a game now)")
            for pid, m in sorted(was.items(), key=lambda z: -z[1]):
                if m < 12 or pid in cur.index:
                    continue
                me = p[p.pid == pid]
                nm, now = me.name.iloc[-1], me.team.iloc[-1]
                gone.append(f"{nm}: " + (f"now with {now}" if now != t else "has not played in the team's last 5 games") + f" (played {m:.0f} min in the last meeting)")
            inj = pr[(pr.team == t) & pr.inj.notna()].drop_duplicates("pid")
            chg[t] = dict(new=new[:6], gone=gone[:6], inj=[f"{r['name']}: {r.inj}" + (f", plays {r.play_p:.0%}" if r.play_p < .97 else "") for _, r in inj.iterrows()][:8])
        gm["chg"] = chg


def past_weeks(hub):
    """nbw.json: every finished game this season with what a post-game review needs (score, closing line, each team's leaders),
    for the Previous weeks page. Weeks run Monday to Sunday, counted from the week of opening night."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import nba_model as M
    d = M.load()
    g = d["games"]
    g = g[(g.season == g.season.max()) & g.completed & g.hs.notna() & g.stype.isin([2, 3, 5])].sort_values("date")
    p = d["players"]
    p = p[p.game_id.isin(set(g.game_id)) & (p["min"] > 0)]
    lead = {}
    for (gid, tm), x in p.groupby(["game_id", "team"]):
        rows = []
        for k, lab in (("pts", "Points"), ("reb", "Rebounds"), ("ast", "Assists")):
            r = x.sort_values([k, "min"], ascending=False).iloc[0]
            rows.append(dict(cat=lab, name=r["name"], line=f"{int(r[k])}" + (f" ({int(r.pts)} pts, {int(r.reb)} reb, {int(r.ast)} ast)" if k == "pts" else "")))
        lead.setdefault(gid, []).append(dict(team=tm, rows=rows))
    day0 = pd.Timestamp(g.date_et.min())
    wk0 = day0 - pd.Timedelta(days=day0.weekday())
    num = lambda v: None if pd.isna(v) else float(v)
    out = []
    for _, r in g.iterrows():
        hs, as_ = int(r.hs), int(r["as"])
        sp = num(r.spread_home)                      # ours: positive = home favored; reviews use the home team's line (negative = favored)
        x = dict(sport="nba", id=r.game_id, date=r.date_et, wk=int((pd.Timestamp(r.date_et) - wk0).days // 7) + 1, home=r.home, away=r.away, hs=hs, as_=as_, detail="Final",
                 winner=r.home if hs > as_ else r.away, ml_h=num(r.ml_h), ml_a=num(r.ml_a), spread_home=None if sp is None else -sp, total=num(r.total), book="DraftKings",
                 leaders=sorted(lead.get(r.game_id, []), key=lambda z: z["team"] != r.home))
        if sp is not None:
            ats = (hs - as_) - sp
            x["cover"] = None if ats == 0 else (r.home if ats > 0 else r.away)
        if x["total"] is not None:
            x["ou"] = "push" if hs + as_ == x["total"] else ("over" if hs + as_ > x["total"] else "under")
        out.append(x)
    f = os.path.join(hub, "nbw.json")
    json.dump(dict(start=f"{wk0:%Y-%m-%d}", games=out), open(f, "w"), separators=(",", ":"))
    return {"nbw.json": f}


def build(hub):
    j = dict(season="2026-27", teams=teams(), tonight=tonight(), record=record(), players=lookup(), ts=teamstats())
    try:
        if j["tonight"]:
            matchup_history(j["tonight"])
    except Exception as e:  # noqa: BLE001  history is an extra: never block the hub
        print("NBA matchup history skipped:", repr(e))
    try:
        if j["tonight"]:
            travel(j["tonight"])
    except Exception as e:  # noqa: BLE001  travel is an extra: never block the hub
        print("NBA travel skipped:", repr(e))
    f = os.path.join(hub, "nba.json")
    json.dump(j, open(f, "w"), separators=(",", ":"), default=lambda x: None if x != x else float(x), allow_nan=False)
    files = {"nba.json": f}
    files.update(profiles(hub))
    files.update(leaders(hub))
    files.update(matchups(hub))
    try:
        files.update(past_weeks(hub))
    except Exception as e:  # noqa: BLE001
        print("NBA previous weeks skipped:", repr(e))
    return files


if __name__ == "__main__":
    import sys
    print(build(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "output", "hub")))
