"""Walk-forward backtest. Train on data before each week, score that week's real results.

Availability uses who actually played that week, so this measures the opportunity and efficiency model
and flatters it slightly versus a live forecast, where inactives are uncertain.
"""
import json
import numpy as np
import pandas as pd
import engine as E

YD = {"rec": [20, 30, 40, 50, 60, 70, 80, 90, 100, 120], "rush": [20, 30, 40, 50, 60, 70, 80, 90, 100, 120],
      "pass": [150, 175, 200, 225, 250, 275, 300, 325]}


WEEKS = [(2025, w) for w in range(5, 19)] + [(2026, w) for w in range(4, 9)]
LOW_T = .15  # receiving calibration is fit separately for players under this target share


def run(n=2500, seed=11, weeks=WEEKS):
    d = E.load()
    rng = np.random.default_rng(seed)
    games = d["games"]
    resid = E.residual_pairs(games[games.season < 2025])
    rows25 = E.script_rows(d, E.tidx(2025, 4), E.tidx(2025, 22))
    fit25 = E.fit_script(rows25)
    allrows = []
    gl = []
    st = d["stats"]
    for season, wk in weeks:
        t = E.tidx(season, wk)
        r = E.ratings(d["team_games"], t)
        for g in games[(games.season == season) & (games.week == wk) & games.result.notna()].itertuples():
            if g.game_type != "REG":
                continue
            ph, pa = E.model_line(r, g.home_team, g.away_team)
            gl.append(dict(season=season, week=wk, game_id=g.game_id, result=g.result, total=g.total, spread=g.spread_line, tline=g.total_line, mh=ph - pa, mt=ph + pa))
            actual = st[st.t == t]
            # active = logged an offensive snap, so players with a zero line stay in the scored sample
            sn = d["snaps"]
            on = sn[(sn.season == season) & (sn.week == wk) & (sn.offense_snaps > 0)]
            def active(tm):
                names = set(on[on.team == tm].player)
                h = st[(st.team == tm) & (st.t < t)]
                return set(h[h.name.isin(names)].player_id) | set(actual[actual.team == tm].player_id)
            ah, aa = active(g.home_team), active(g.away_team)
            # name the starting QB, as the live run does from the depth chart
            qb = {("qb", g.home_team): g.home_qb_id, ("qb", g.away_team): g.away_qb_id}
            s = E.simulate_game(rng, d, g.home_team, g.away_team, g.spread_line, g.total_line, t, fit25, resid, n=n,
                                active_h=ah, active_a=aa, avail=qb, ratings_cache=r)
            for df, sim, tm in ((s["dfh"], s["sh"], g.home_team), (s["dfa"], s["sa"], g.away_team)):
                act = actual[actual.team == tm].set_index("player_id")
                for j, p in df.iterrows():
                    if p.player_id in act.index:
                        a = act.loc[p.player_id]
                        if isinstance(a, pd.DataFrame):
                            a = a.iloc[0]
                    else:
                        a = pd.Series(0, index=["rushing_tds", "receiving_tds", "passing_yards", "receiving_yards", "rushing_yards"])
                    td = int(a.rushing_tds + a.receiving_tds)
                    base = dict(season=season, week=wk, pid=p.player_id, pos=p.pos, game_id=g.game_id, team=tm,
                                opp=g.away_team if tm == g.home_team else g.home_team, tshare=p.tshare, cshare=p.cshare)
                    if p.pos == "QB":
                        allrows.append({**base, "mkt": "pass", "proj": sim["pass_y"].mean(), "med": float(np.median(sim["pass_y"])), "act": a.passing_yards,
                                        "probs": {k: float((sim["pass_y"] >= k).mean()) for k in YD["pass"]}})
                    if p.pos != "QB":
                        allrows.append({**base, "mkt": "rec", "proj": sim["rec_y"][:, j].mean(), "med": float(np.median(sim["rec_y"][:, j])), "act": a.receiving_yards,
                                        "probs": {k: float((sim["rec_y"][:, j] >= k).mean()) for k in YD["rec"]}})
                    if p.cshare > .12:
                        allrows.append({**base, "mkt": "rush", "proj": sim["rush_y"][:, j].mean(), "med": float(np.median(sim["rush_y"][:, j])), "act": a.rushing_yards,
                                        "probs": {k: float((sim["rush_y"][:, j] >= k).mean()) for k in YD["rush"]}})
                    allrows.append({**base, "mkt": "atd", "proj": ((sim["ptds"][:, j] + sim["rtds"][:, j]) > 0).mean(), "act": td,
                                    "probs": {1: float(((sim["ptds"][:, j] + sim["rtds"][:, j]) > 0).mean())}})
        print(season, wk, len(allrows), flush=True)
    return pd.DataFrame(allrows), pd.DataFrame(gl)


def score(df, gl):
    long = []
    for r in df.itertuples():
        for k, p in r.probs.items():
            long.append((r.season, r.week, r.mkt, r.pos, k, p, int(r.act >= k), r.pid, r.tshare))
    L = pd.DataFrame(long, columns="season week mkt pos k p y pid tshare".split())
    L.loc[(L.mkt == "rec") & (L.tshare < LOW_T), "mkt"] = "rec_low"
    return L


def calib_fit(L):
    """Logit-scale recalibration per market on 2025; applied to the 2026 holdout."""
    out = {}
    for m, x in L[L.season == 2025].groupby("mkt"):
        p = x.p.clip(.003, .997).values
        z = np.log(p / (1 - p))
        y = x.y.values
        a, b = 0.0, 1.0
        for _ in range(60):
            q = 1 / (1 + np.exp(-(a + b * z)))
            g = np.array([(y - q).sum(), ((y - q) * z).sum()])
            wt = q * (1 - q)
            H = np.array([[wt.sum(), (wt * z).sum()], [(wt * z).sum(), (wt * z * z).sum()]])
            a, b = np.array([a, b]) + np.linalg.solve(H, g)
        out[m] = (float(a), float(b))
    return out


def cal_key(cal, mk, tshare=None):
    """Depth receivers have their own calibration: the model overrates them by 3-10 points."""
    return "rec_low" if mk == "rec" and tshare is not None and tshare < LOW_T and "rec_low" in cal else mk


def apply_cal(p, c):
    p = np.clip(p, .003, .997)
    return 1 / (1 + np.exp(-(c[0] + c[1] * np.log(p / (1 - p)))))


if __name__ == "__main__":
    df, gl = run()
    df.to_pickle("output/bt_rows.pkl")
    y = df[df.mkt != "atd"]
    res_med = {f"{m}_{lo}": float((x.act > x.med).mean()) for m, g in y.groupby("mkt") for lo, x in (("low", g[g.med < g.med.quantile(.6)]), ("high", g[g.med >= g.med.quantile(.6)]))}
    gl.to_csv("output/bt_games.csv", index=False)
    L = score(df, gl)
    cal = calib_fit(L)
    res = {"cal": cal, "n_games": len(gl), "markets": {}, "over_median": res_med}
    for m, x in L.groupby("mkt"):
        for lab, y in (("2025_in", x[x.season == 2025]), ("2026_holdout", x[x.season == 2026])):
            p2 = apply_cal(y.p.values, cal[m])
            base = y.groupby("k").y.transform("mean")
            res["markets"][f"{m}_{lab}"] = dict(n=len(y), brier=float(((y.p - y.y) ** 2).mean()), brier_cal=float(((p2 - y.y) ** 2).mean()),
                                                brier_base=float(((base - y.y) ** 2).mean()))
    h = L[L.season == 2026].copy()
    h["pc"] = [apply_cal(np.array([p]), cal[m])[0] for p, m in zip(h.p, h.mkt)]
    h["bin"] = (h.pc * 10).clip(0, 9).astype(int)
    res["reliability_2026"] = h.groupby("bin").agg(n=("y", "size"), pred=("pc", "mean"), hit=("y", "mean")).reset_index().to_dict("records")
    h25 = L[L.season == 2025].copy()
    h25["bin"] = (h25.p * 10).clip(0, 9).astype(int)
    res["reliability_2025_raw"] = h25.groupby("bin").agg(n=("y", "size"), pred=("p", "mean"), hit=("y", "mean")).reset_index().to_dict("records")
    res["mae"] = {f"{m}_{s}": float((x.proj - x.act).abs().mean()) for (m, s), x in df[df.mkt != "atd"].groupby(["mkt", "season"])}
    # market blend: how much should our line get versus the closing line?
    X = np.column_stack([gl.spread, gl.mh - gl.spread])
    res["blend_margin"] = np.linalg.lstsq(X, gl.result, rcond=None)[0].tolist()
    X = np.column_stack([gl.tline, gl.mt - gl.tline])
    res["blend_total"] = np.linalg.lstsq(X, gl.total, rcond=None)[0].tolist()
    res["mae_margin"] = dict(market=float((gl.result - gl.spread).abs().mean()), model=float((gl.result - gl.mh).abs().mean()),
                             blend25=float((gl.result - (.75 * gl.spread + .25 * gl.mh)).abs().mean()), blend10=float((gl.result - (.9 * gl.spread + .1 * gl.mh)).abs().mean()))
    res["mae_total"] = dict(market=float((gl.total - gl.tline).abs().mean()), model=float((gl.total - gl.mt).abs().mean()))
    json.dump(res, open("output/backtest.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items()}, indent=1))
