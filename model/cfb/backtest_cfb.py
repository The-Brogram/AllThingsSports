"""Walk-forward backtest. No hindsight on availability: eligibility uses only prior games, exactly as live."""
import json
import numpy as np
import pandas as pd
import cfb_model as C
import engine as E
from backtest import calib_fit, apply_cal

YD = {"rec": [20, 30, 40, 50, 60, 70, 80, 90, 100, 120], "rush": [20, 30, 40, 50, 60, 70, 80, 90, 100, 120], "pass": [150, 175, 200, 225, 250, 275, 300, 325]}


def run(n=2000, seed=5):
    d = C.load()
    g = d["games"]
    rng = np.random.default_rng(seed)
    # ESPN keeps 2026 closing lines but only ~10% of 2025's, so 2026 Weeks 4-6 train and Weeks 7-9 are the holdout
    res = C.residuals(g[(g.season == 2026) & (g.week <= 6)])
    rows = C.script_rows(d, C.t_of(2026, 4), C.t_of(2026, 6))
    fit = E.fit_script(rows)
    sim = C.Sim(d, fit, res, .10)
    out, gl = [], []
    pl = d["players"]
    weeks = [(2026, w) for w in range(4, 10)]
    for season, wk in weeks:
        t = C.t_of(season, wk)
        r = C.ratings(d, t)
        q = g[(g.season == season) & (g.week == wk) & (g.stype == 2) & g.spread_line.notna() & g.total_line.notna()]
        q = q[q.home.isin(d["fbs"]) & q.away.isin(d["fbs"])]
        for x in q.itertuples():
            ph, pa = C.line(r, x.home, x.away, x.neutral)
            gl.append(dict(season=season, week=wk, result=x.hs - x.aw, total=x.hs + x.aw, spread=x.spread_line, tline=x.total_line, mh=ph - pa, mt=ph + pa))
            s = sim.game(rng, x.home, x.away, x.spread_line, x.total_line, x.neutral, t, r, n)
            if s is None:
                continue
            for df, o, tm in ((s["dfh"], s["sh"], x.home), (s["dfa"], s["sa"], x.away)):
                act = pl[(pl.game_id == x.game_id) & (pl.team == tm)].drop_duplicates("pid").set_index("pid")
                for j, p in df.iterrows():
                    a = act.loc[p.player_id] if p.player_id in act.index else None
                    v = lambda k: float(a[k]) if a is not None else 0.0
                    base = dict(season=2025 if wk <= 6 else 2026, week=wk, pid=p.player_id, pos=p.pos, spread=abs(x.spread_line), absent=a is None)
                    if p.pos == "QB":
                        out.append({**base, "mkt": "pass", "proj": o["pass_y"].mean(), "med": float(np.median(o["pass_y"])), "act": v("pass_y"), "probs": {k: float((o["pass_y"] >= k).mean()) for k in YD["pass"]}})
                    else:
                        out.append({**base, "mkt": "rec", "proj": o["rec_y"][:, j].mean(), "med": float(np.median(o["rec_y"][:, j])), "act": v("rec_y"), "probs": {k: float((o["rec_y"][:, j] >= k).mean()) for k in YD["rec"]}})
                    if p.cshare > .12:
                        out.append({**base, "mkt": "rush", "proj": o["rush_y"][:, j].mean(), "med": float(np.median(o["rush_y"][:, j])), "act": v("rush_y"), "probs": {k: float((o["rush_y"][:, j] >= k).mean()) for k in YD["rush"]}})
                    at = ((o["ptds"][:, j] + o["rtds"][:, j]) > 0).mean()
                    out.append({**base, "mkt": "atd", "proj": at, "med": at, "act": v("rush_td") + v("rec_td"), "probs": {1: float(at)}})
        print(season, wk, len(out), flush=True)
    return pd.DataFrame(out), pd.DataFrame(gl)


if __name__ == "__main__":
    df, gl = run()
    df.to_pickle("output/bt_rows.pkl")
    L = pd.DataFrame([(r.season, r.week, r.mkt, r.pos, k, p, int(r.act >= k), r.pid) for r in df.itertuples() for k, p in r.probs.items()], columns="season week mkt pos k p y pid".split())
    cal = calib_fit(L)
    res = dict(cal=cal, markets={}, n_games=len(gl))
    for m, x in L.groupby("mkt"):
        y = x[x.season == 2026]
        p2 = apply_cal(y.p.values, cal[m])
        b = y.groupby("k").y.transform("mean")
        res["markets"][m] = dict(n=len(y), brier=float(((p2 - y.y) ** 2).mean()), base=float(((b - y.y) ** 2).mean()))
    h = L[L.season == 2026].copy()
    h["pc"] = [apply_cal(np.array([p]), cal[m])[0] for p, m in zip(h.p, h.mkt)]
    h["bin"] = (h.pc * 10).clip(0, 9).astype(int)
    res["reliability"] = h.groupby("bin").agg(n=("y", "size"), pred=("pc", "mean"), hit=("y", "mean")).reset_index().to_dict("records")
    y = df[df.mkt != "atd"]
    res["over_median"] = {f"{m}_{a}": float((x.act > x.med).mean()) for m, g in y.groupby("mkt") for a, x in (("low", g[g.med < g.med.quantile(.6)]), ("high", g[g.med >= g.med.quantile(.6)]))}
    res["mae"] = {m: float((x.proj - x.act).abs().mean()) for m, x in y[y.season == 2026].groupby("mkt")}
    res["absent_rate"] = float(df[df.mkt == "atd"].absent.mean())
    res["bias"] = df[df.mkt != "atd"].groupby(["mkt", "pos"]).agg(n=("act", "size"), proj=("proj", "mean"), act=("act", "mean")).round(1).reset_index().to_dict("records")
    res["blend_margin"] = np.linalg.lstsq(np.column_stack([gl.spread, gl.mh - gl.spread]), gl.result, rcond=None)[0].tolist()
    res["blend_total"] = np.linalg.lstsq(np.column_stack([gl.tline, gl.mt - gl.tline]), gl.total, rcond=None)[0].tolist()
    res["mae_margin"] = dict(market=float((gl.result - gl.spread).abs().mean()), model=float((gl.result - gl.mh).abs().mean()), blend=float((gl.result - (.9 * gl.spread + .1 * gl.mh)).abs().mean()))
    res["mae_total"] = dict(market=float((gl.total - gl.tline).abs().mean()), model=float((gl.total - gl.mt).abs().mean()))
    json.dump(res, open("output/backtest.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "bias"}), res["bias"])
