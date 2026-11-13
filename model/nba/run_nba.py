"""Tonight's NBA slate: sim every game and write per-player probabilities for every DraftKings prop line.

Run (after fetch_nba.py --today and parse_nba.py): ../venv/bin/python run_nba.py [--date 2026-11-10] [--n 10000]
Writes output/nba_<date>.pkl (sims summary) and output/predictions_<date>.csv (one row per player, market, line).

Availability: ESPN's injury feed has only "Out" and "Day-To-Day"; the blurb says more ("ruled out", "questionable", "probable").
A blurb counts only if the team has not played since it was written (otherwise it was about an earlier game).
"""
import argparse
import json
import os
import pickle
import re
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import nba_model as M

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = f"{HERE}/output"
MKTS = ["pts", "reb", "ast", "fg3", "pr", "pa", "ra", "pra", "stl", "blk"]
ALT = ("pts", "reb", "ast", "fg3", "pr", "pa", "ra", "pra")
ALT_MIN = {"pts": 10, "reb": 4, "ast": 3, "fg3": 2, "pr": 12, "pa": 12, "ra": 6, "pra": 15}  # smallest "N+" worth showing
NAMES = {"pts": "Points", "reb": "Rebounds", "ast": "Assists", "fg3": "3-pointers", "pr": "Pts + Reb", "pa": "Pts + Ast", "ra": "Reb + Ast",
         "pra": "Pts + Reb + Ast", "stl": "Steals", "blk": "Blocks", "dd": "Double-double"}
RULES = [(r"ruled out|won't play|will not play|will miss|is out|remain out|sidelined|out for|isn't expected to play|not expected to play|out indefinitely", 0.0),
         (r"doubtful", .2), (r"questionable|game-time decision|uncertain", .5),
         (r"probable|expected to play|will play|is available|cleared to play|plans to play|will return|set to return|in the starting lineup", .9)]


def calib():
    try:
        return json.load(open(f"{OUT}/calibration.json"))
    except (OSError, ValueError):
        return None


def lg(p):
    p = np.clip(np.asarray(p, float), 1e-3, 1 - 1e-3)
    return np.log(p / (1 - p))


def apply_cal(p, c, dk=np.nan, milestone=False, kind="line"):
    """Final probability (fit in calibrate_nba.py): the sim's probability, blended with DraftKings' price when the line is priced."""
    if not c:
        return float(p)
    if dk == dk and dk is not None and 0 < dk < 1:
        w = c["blend"]
        z = w["a"] + w["b"] * lg(p) + w["c"] * lg(dk) + (w["cm"] * lg(dk) if milestone else 0)
    else:
        w = c["alt"] if kind == "alt" else c["model"]
        z = w["a"] + w["b"] * lg(p)
    return float(1 / (1 + np.exp(-z)))


def play_prob(row, team_last):
    st = (row.get("status") or "").lower()
    txt = (row.get("short") or "").lower()
    when = (row.get("date") or "")[:10]
    fresh = not team_last or when >= team_last  # no team game since the blurb
    if fresh:
        for pat, p in RULES:
            if re.search(pat, txt):
                return p, txt
    if st == "out":
        ret = row.get("ret") or ""
        return (0.0 if (fresh or not ret or ret > team_last) else .5), txt
    return .75, txt


def availability(ctx, slate_date):
    inj = json.load(open(f"{M.D}/injuries.json"))["rows"]
    last = ctx.p.groupby("team").date_et.max().to_dict()
    av, notes = {}, {}
    for r in inj:
        if not r.get("id"):
            continue
        p, txt = play_prob(r, last.get(r["team"]))
        av.setdefault(r["team"], {})[r["id"]] = p
        notes[r["id"]] = dict(status=r.get("status"), p=p, note=r.get("short"), date=(r.get("date") or "")[:10])
    return av, notes


def american_p(a):
    a = np.asarray(a, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(a < 0, -a / (-a + 100), 100 / (a + 100))


def main():
    ap = argparse.ArgumentParser()
    now = datetime.now(timezone.utc) - timedelta(hours=5)
    ap.add_argument("--date", default=f"{now:%Y-%m-%d}")
    ap.add_argument("--n", type=int, default=10000)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    d = M.load()
    g = d["games"]
    slate = g[(g.date_et == a.date) & (g.stype.isin([2, 3, 5]))]
    ctx = M.Ctx(d, a.date)
    res = M.residuals(ctx.g)
    av, notes = availability(ctx, a.date)
    props = pd.read_csv(f"{M.D}/props.csv.gz", dtype={"pid": str, "game_id": str})
    props = props[props.game_id.isin(slate.game_id)]
    cal = calib()
    rng = np.random.default_rng(int(a.date.replace("-", "")))
    games, rows, dist = [], [], {}
    # a rerun keeps the forecast for any game that has already tipped (its lines turn live after tip-off)
    sbf = f"{M.D}/sb/{a.date.replace('-', '')}.json"
    state = {e["id"]: e.get("state") for e in json.load(open(sbf))} if os.path.exists(sbf) else {}
    old_pkl, old_csv = f"{OUT}/nba_{a.date}.pkl", f"{OUT}/predictions_{a.date}.csv"
    old = pickle.load(open(old_pkl, "rb")) if os.path.exists(old_pkl) and os.path.exists(old_csv) else None
    old_pr = pd.read_csv(old_csv, dtype={"pid": str, "game_id": str}) if old else None
    for gm in slate.itertuples():
        og = next((x for x in old["games"] if x["game_id"] == gm.game_id), None) if old else None
        if og and state.get(gm.game_id) not in (None, "pre"):
            games.append(og)
            rows += old_pr[old_pr.game_id == gm.game_id].to_dict("records")
            dist.update({k: v for k, v in old.get("dist", {}).items() if k in set(old_pr[old_pr.game_id == gm.game_id].pid)})
            print(f"{gm.away} @ {gm.home}: already started, kept the earlier forecast")
            continue
        sh = json.load(open(f"{M.D}/games/{gm.game_id}.json"))
        if pd.isna(gm.spread_home) or pd.isna(gm.total):
            print("no line yet:", gm.away, "@", gm.home)
            continue
        s = M.sim_game(rng, ctx, res, gm.home, gm.away, gm.spread_home, gm.total, a.n, av.get(gm.home), av.get(gm.away))
        if s is None:
            continue
        mg = s["margin"]
        games.append(dict(game_id=gm.game_id, date=gm.date, home=gm.home, away=gm.away, spread_home=gm.spread_home, total=gm.total, ml_h=gm.ml_h, ml_a=gm.ml_a,
                          p_home=float((mg > 0).mean() + .5 * (mg == 0).mean()), margin_q=np.percentile(mg, [10, 25, 50, 75, 90]).round(1).tolist(),
                          margin_hist=np.histogram(np.clip(mg, -30, 30), bins=np.arange(-30, 31, 3))[0].tolist(), total_mean=float(s["total"].mean()),
                          status=sh.get("completed")))
        pg = props[props.game_id == gm.game_id]
        for side, df, sm, opp in (("home", s["dfh"], s["sh"], gm.away), ("away", s["dfa"], s["sa"], gm.home)):
            team = gm.home if side == "home" else gm.away
            for i, r in df.iterrows():
                played = sm["min"][:, i] > 0
                if played.mean() < .05:
                    continue
                pp = pg[pg.pid == r.pid]
                base = dict(game_id=gm.game_id, team=team, opp=opp, pid=r.pid, name=r["name"], pos=r.pos, play_p=float(played.mean()), proj_min=float(sm["min"][played, i].mean()),
                            inj=notes.get(r.pid, {}).get("status"), inj_note=notes.get(r.pid, {}).get("note"))
                # full outcome range for the hub's player profile: per-mille chance of at least k, k = 0, 1, 2, ...
                dist[r.pid] = dict(min=[round(float(x), 1) for x in np.percentile(sm["min"][played, i], [10, 50, 90])] if played.any() else None)
                for mk in MKTS:
                    v = sm[mk][played, i].astype(int)
                    if len(v):
                        surv = np.cumsum(np.bincount(v)[::-1])[::-1] / len(v)
                        keep = np.nonzero(surv >= .002)[0]
                        dist[r.pid][mk] = [int(round(1000 * x)) for x in surv[: keep[-1] + 1]]
                for mk in MKTS + ["dd"]:
                    v = sm[mk][played, i]
                    if not len(v):
                        continue
                    q = pp[pp.market == mk]
                    lines = sorted(set(q.line.dropna()))
                    st = dict(mean=float(v.mean()), med=float(np.median(v)), p10=float(np.percentile(v, 10)), p90=float(np.percentile(v, 90)))
                    for L in lines:
                        po = float((v > L).mean())
                        qq = q[q.line == L]
                        o = qq[qq.side == "over"]
                        u = qq[(qq.side == "under") & (qq.kind == "line")]
                        dko = float(american_p(o.price.iloc[0])) if len(o) else np.nan
                        dku = float(american_p(u.price.iloc[0])) if len(u) else np.nan
                        kind = o.kind.iloc[0] if len(o) else "line"
                        dk = dko / (dko + dku) if (dku == dku and dko == dko) else dko
                        rows.append(dict(base, market=mk, line=L, kind=kind, p_over_raw=po, p_over=apply_cal(po, cal, dk, kind == "milestone") if mk != "dd" else po,
                                         price_over=o.price.iloc[0] if len(o) else np.nan, price_under=u.price.iloc[0] if len(u) else np.nan, dk_over=dk, **st))
                    if not lines and mk in ("pts", "reb", "ast", "pra"):
                        L = np.floor(st["med"]) + .5 if st["med"] >= 1 else .5
                        po = float((v > L).mean())
                        rows.append(dict(base, market=mk, line=L, kind="model", p_over_raw=po, p_over=apply_cal(po, cal), **st))
                        lines = [L]
                    if mk in ALT and base["proj_min"] >= 18:
                        # alt thresholds below the market line: the high-probability "N+" lines, priced by the sim alone
                        for t in sorted({np.floor(np.percentile(v, q)) - .5 for q in (5, 12, 20, 30)}):
                            if t + .5 >= ALT_MIN[mk] and all(abs(t - x) > .01 for x in lines):
                                po = float((v > t).mean())
                                rows.append(dict(base, market=mk, line=float(t), kind="alt", p_over_raw=po, p_over=apply_cal(po, cal, kind="alt"), **st))
        print(f"{gm.away} @ {gm.home}: home {games[-1]['p_home']:.0%}, {len(pg)} prop prices")
    pr = pd.DataFrame(rows)
    pr.to_csv(f"{OUT}/predictions_{a.date}.csv", index=False)
    with open(f"{OUT}/nba_{a.date}.pkl", "wb") as h:
        pickle.dump(dict(date=a.date, built=datetime.now(timezone.utc).isoformat(timespec="seconds"), games=games, notes=notes, n=a.n, dist=dist), h)
    print(f"{len(games)} games, {len(pr)} player lines -> output/predictions_{a.date}.csv")
    try:  # the proof layer: record every priced side at the price on offer right now (first price seen is kept)
        import edge_nba
        edge_nba.log(a.date)
    except Exception as ex:  # noqa: BLE001  a logging problem must never stop the forecast
        print("pick log skipped:", repr(ex))


if __name__ == "__main__":
    main()
