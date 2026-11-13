"""Walk-forward backtest: for each game date, fit only on earlier games, sim every game, and record P(over) for each player market.

Lines: the DraftKings prop line where one was priced (plus the de-vigged DraftKings over probability), otherwise a stand-in line of
the projected mean rounded to .5, so every player with 12+ projected minutes is graded.
Availability: we have no archived pre-game injury reports, so a player counts as out when he did not appear or sat with a listed
injury/illness reason (what a pre-game report would mostly show). Coach's-decision DNPs are NOT known in advance: those players keep
their normal projection, so the backtest pays for surprise scratches like the live model will.

Run: ../venv/bin/python backtest_nba.py [--from 2025-11-15] [--to 2026-11-09] [--every 1] [--n 2000] [--grid]
Writes output/bt_rows.csv.gz and prints calibration.
"""
import argparse
import os
import numpy as np
import pandas as pd
import nba_model as M

HERE = os.path.dirname(os.path.abspath(__file__))
MKTS = ["pts", "reb", "ast", "fg3", "pr", "pa", "ra", "pra", "stl", "blk"]
ALT = ("pts", "reb", "ast", "fg3", "pr", "pa", "ra", "pra")


def american_p(a):
    a = np.asarray(a, float)
    a = np.where(a == 0, np.nan, a)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(a < 0, -a / (-a + 100), 100 / (a + 100))


def dk_lines(props):
    """One row per (game, pid, market, line) for over/under pairs, with de-vigged over probability."""
    x = props[(props.kind == "line") & props.market.isin(MKTS)]
    o = x[x.side == "over"][["game_id", "pid", "market", "line", "price"]]
    u = x[x.side == "under"][["game_id", "pid", "market", "line", "price"]]
    j = o.merge(u, on=["game_id", "pid", "market", "line"], suffixes=("_o", "_u"))
    po, pu = american_p(j.price_o), american_p(j.price_u)
    j["dk_p"] = po / (po + pu)
    j["kind"] = "line"
    m = props[(props.kind == "milestone") & props.market.isin(MKTS)][["game_id", "pid", "market", "line", "price"]].copy()
    m["dk_p"] = american_p(m.price)  # one-sided, vig left in
    m["kind"] = "milestone"
    cols = ["game_id", "pid", "market", "line", "dk_p", "kind"]
    return pd.concat([j.drop_duplicates(["game_id", "pid", "market", "line"])[cols], m.drop_duplicates(["game_id", "pid", "market", "line"])[cols]])


def out_set(day_players):
    r = day_players.reason.fillna("").str.upper()
    sat = (day_players["min"] <= 0) & ~r.str.contains("COACH")
    return set(day_players[sat].pid)


def run(d, start, end, every, n, P=M.PARAMS, seed=7, quiet=False):
    g, p = d["games"], d["players"]
    try:
        lines = dk_lines(pd.read_csv(f"{M.D}/props.csv.gz", dtype={"pid": str, "game_id": str}))
    except FileNotFoundError:
        lines = pd.DataFrame(columns=["game_id", "pid", "market", "line", "dk_p"])
    lk = {}
    for r in lines.itertuples():
        lk.setdefault((r.game_id, r.pid, r.market), []).append((r.line, r.dk_p, r.kind))
    gg = g[g.completed & g.spread_home.notna() & g.total.notna() & (g.date_et >= start) & (g.date_et <= end)]
    dates = sorted(gg.date_et.unique())[::every]
    rng = np.random.default_rng(seed)
    rows = []
    for k, dt in enumerate(dates):
        ctx = M.Ctx(d, dt)
        res = M.residuals(ctx.g)
        if len(res) < 300:
            continue
        for gm in gg[gg.date_et == dt].itertuples():
            act = p[p.game_id == gm.game_id]
            out = out_set(act)
            av = {t: {pid: (0.0 if pid in out else 1.0) for pid in act[act.team == t].pid} for t in (gm.home, gm.away)}
            s = M.sim_game(rng, ctx, res, gm.home, gm.away, gm.spread_home, gm.total, n, av[gm.home], av[gm.away], P, listed_only=True)
            if s is None:
                continue
            a = act.set_index("pid")
            for df, sm in ((s["dfh"], s["sh"]), (s["dfa"], s["sa"])):
                mmin = sm["min"].mean(0)
                for i, r in df.iterrows():
                    if r.pid not in a.index or a.loc[r.pid, "min"] <= 0 or mmin[i] < 8:
                        continue  # graded only if he played (books void DNPs)
                    ar = a.loc[r.pid]
                    for mk in MKTS:
                        v = sm[mk][:, i]
                        played = sm["min"][:, i] > 0
                        v = v[played] if played.sum() > 100 else v
                        actual = sum(ar[x] for x in M.COMBO.get(mk, (mk,)))
                        L = lk.get((gm.game_id, r.pid, mk)) or ([(np.floor(v.mean()) + .5, np.nan, "model")] if mk not in ("stl", "blk") else [])
                        if mk in ALT:
                            # high-probability alt thresholds (what "most likely to hit" leans on); no DraftKings price
                            for q in (5, 12, 20, 30):
                                t = np.percentile(v, q) - .5
                                if t >= .5 and all(abs(t - x[0]) > .01 for x in L):
                                    L = L + [(t, np.nan, "alt")]
                        for line, dkp, kind in L:
                            rows.append((dt, gm.game_id, r.pid, r["name"], mk, kind, line, float((v > line).mean()), float(v.mean()), actual, dkp, mmin[i], ar["min"], df.attrs.get("vac_min", 0.0)))
        if not quiet and k % 20 == 0:
            print(dt, len(rows), flush=True)
    return pd.DataFrame(rows, columns=["date", "game_id", "pid", "name", "market", "kind", "line", "p_over", "mean", "actual", "dk_p", "proj_min", "act_min", "vac_min"])


def report(b):
    b = b.copy()
    b["hit_over"] = (b.actual > b.line).astype(int)
    b["p"] = np.maximum(b.p_over, 1 - b.p_over)
    b["hit"] = np.where(b.p_over >= .5, b.hit_over, 1 - b.hit_over)
    ll = lambda p, y: -np.mean(y * np.log(np.clip(p, 1e-4, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-4, 1)))
    print(f"\n{len(b)} player-markets, {b.game_id.nunique()} games, {b.date.min()}..{b.date.max()}")
    print("minutes MAE", round((b.drop_duplicates(['game_id', 'pid']).eval('proj_min-act_min')).abs().mean(), 2))
    bins = [.5, .55, .6, .65, .7, .75, .8, .9, 1.01]
    t = b.groupby(pd.cut(b.p, bins, right=False), observed=True).agg(n=("hit", "size"), said=("p", "mean"), hit=("hit", "mean"))
    print(t.round(3).to_string())
    print(b.groupby("market").apply(lambda x: pd.Series(dict(n=len(x), mae=(x["mean"] - x.actual).abs().mean(), bias=(x["mean"] - x.actual).mean(),
                                                               logloss=ll(x.p_over, x.hit_over), said=x.p.mean(), hit=x.hit.mean())), include_groups=False).round(3).to_string())
    k = b[b.kind == "line"]
    if len(k):
        print(f"\nvs DraftKings on {len(k)} priced lines: model logloss {ll(k.p_over, k.hit_over):.4f}  DK logloss {ll(k.dk_p, k.hit_over):.4f}")
        k = k.assign(mp=np.maximum(k.p_over, 1 - k.p_over), dkside=np.where(k.p_over >= .5, k.dk_p, 1 - k.dk_p))
        t = k.groupby(pd.cut(k.mp, bins, right=False), observed=True).agg(n=("hit", "size"), said=("mp", "mean"), dk_said=("dkside", "mean"), hit=("hit", "mean"))
        print(t.round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default="2025-11-15")
    ap.add_argument("--to", default="2099-01-01")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--grid", action="store_true")
    a = ap.parse_args()
    d = M.load()
    if a.grid:
        # one multiplier on every stat's dispersion, and minute noise; scored by log loss on stand-in + priced lines
        for dm in (.6, 1.0, 1.6):
            for ms in (.12, .2):
                P = dict(M.PARAMS, disp={k: v * dm for k, v in M.PARAMS["disp"].items()}, min_sd=ms)
                b = run(d, a.start, a.to, a.every, a.n, P, quiet=True)
                b = b[b.kind != "model"]  # score on DraftKings-priced lines only
                y = (b.actual > b.line).astype(int)
                ll = -np.mean(y * np.log(np.clip(b.p_over, 1e-4, 1)) + (1 - y) * np.log(np.clip(1 - b.p_over, 1e-4, 1)))
                print(f"disp x{dm} min_sd {ms}: logloss {ll:.4f}  n {len(b)}", flush=True)
    else:
        b = run(d, a.start, a.to, a.every, a.n)
        os.makedirs(f"{HERE}/output", exist_ok=True)
        b.to_csv(f"{HERE}/output/bt_rows.csv.gz", index=False)
        report(b)
