"""Does the WR/CB page know anything? Rebuilds it week by week from data strictly before each game and checks:

1. Are the side estimates stable traits? Split-half (odd vs even weeks) correlation of each receiver's left/middle/right
   target mix and each corner's alignment. Noise would give ~0.
2. Does the corner side predict? Receiver's receiving PPR in the game minus what his own FP per route (shrunk, before
   the game) x his routes that game says. If facing a soft corner helps, that residual should rise with the
   corners' FP-per-coverage-snap allowed (lift) and with the ADVANTAGE call.
2025 weeks 5-18 is the main test (no depth-chart roles that year); 2026 weeks 5-8 is a smaller holdout.
Run: venv/bin/python wrcb_backtest.py -> output/wrcb_backtest.json (the page shows the summary)
"""
import json
import numpy as np
import pandas as pd
import wrcb as W


def reliability(D):
    out = {}
    p = D["p"][D["p"].pass_location.notna() & D["p"].receiver_player_id.notna()]
    p = p.assign(h=p.week % 2, s=p.pass_location.str[0].str.upper())
    x = pd.crosstab([p.receiver_player_id, p.h], p.s).reindex(columns=W.SIDES, fill_value=0)
    n = x.sum(axis=1)
    x = x[n >= 15].div(n[n >= 15], axis=0).unstack("h").dropna()
    out["receiver_side"] = {s: round(float(np.corrcoef(x[(s, 0)], x[(s, 1)])[0, 1]), 3) for s in W.SIDES} | {"players": len(x)}
    cbs = set(W.corners_games(D).gsis_id.dropna())
    cols = ["pass_defense_1_player_id", "pass_defense_2_player_id", "solo_tackle_1_player_id", "assist_tackle_1_player_id", "tackle_with_assist_1_player_id"]
    d = pd.concat([pd.DataFrame(dict(id=p[c], s=p.s, h=p.h)) for c in cols]).dropna()
    d = d[d.id.isin(cbs)]
    y = pd.crosstab([d.id, d.h], d.s).reindex(columns=W.SIDES, fill_value=0)
    n = y.sum(axis=1)
    y = y[n >= 10].div(n[n >= 10], axis=0).unstack("h").dropna()
    out["corner_side"] = {s: round(float(np.corrcoef(y[(s, 0)], y[(s, 1)])[0, 1]), 3) for s in W.SIDES} | {"players": len(y)}
    return out


def rows(D, weeks):
    R, C = W.receivers_games(D), W.corners_games(D)
    g = pd.read_csv(f"{W.DATA}/games.csv", low_memory=False)
    g = g[(g.season == D["season"]) & (g.game_type == "REG")]
    roles = W.roles_2026() if D["season"] == 2026 else None
    res = []
    for wk in weeks:
        B = W.Book(D, wk - 1, roles, RC=(R, C))
        Rw, Cw = R[R.week == wk].set_index("gsis_id"), C[C.week == wk]
        for r in g[g.week == wk].itertuples():
            for off, dfn in ((r.away_team, r.home_team), (r.home_team, r.away_team)):
                played_r = set(Rw[Rw.team == off].index)
                played_c = set(Cw[(Cw.team == dfn) & (Cw.defense_snaps > 0)].pfr_player_id)
                wr, _ = B.score(off, dfn, played_r, played_c)
                for x in wr:
                    a = Rw.loc[[x["id"]]].iloc[0]
                    res.append(dict(season=D["season"], week=wk, game=r.game_id, id=x["id"], pos=x["pos"], adv=x["adv"], score=x["score"], lift=x["face_fprr"] / B.lg_cfpr,
                                    rp=x["rp"], cp=x["cp"], fp=a.fp, tgt=a.targets, routes=a.routes, exp=x["fprr_s"] * a.routes,
                                    exp_t=(B.r.loc[x["id"], "tgt"] + W.K_WR * B.lg_tpr) / (B.r.loc[x["id"], "routes"] + W.K_WR) * a.routes))
    return pd.DataFrame(res)


def summarize(x, seed=7):
    x = x.assign(res=x.fp - x.exp, res_t=x.tgt - x.exp_t, ll=np.log(x.lift))
    r_lift = float(np.corrcoef(x.ll, x.res)[0, 1])
    r_cp = float(np.corrcoef(-x.cp, x.res)[0, 1])
    rng = np.random.default_rng(seed)
    idx = list(x.groupby("game").indices.values())
    ll, rs = x.ll.values, x.res.values
    bs = []
    for _ in range(1000):  # resample whole games
        k = np.concatenate([idx[i] for i in rng.integers(0, len(idx), len(idx))])
        bs.append(np.corrcoef(ll[k], rs[k])[0, 1])
    lo, hi = np.percentile(bs, [5, 95])
    # residual per route by corner-lift fifth: does the soft-corner fifth beat the tough-corner fifth?
    q = pd.qcut(x.lift, 5, labels=False, duplicates="drop")
    by_q = x.groupby(q).apply(lambda z: z.res.sum() / z.routes.sum() * 30, include_groups=False)  # FP per 30 routes above own rate
    by_adv = {k: dict(n=len(z), fp_vs_own_rate=round(float(z.res.mean()), 2), tgt_vs_own_rate=round(float(z.res_t.mean()), 2)) for k, z in x.groupby("adv")}
    return dict(n=len(x), games=int(x.game.nunique()), r_lift_resid=round(r_lift, 3), r_lift_90ci=[round(float(lo), 3), round(float(hi), 3)],
                r_cornerpct_resid=round(r_cp, 3), r_lift_resid_targets=round(float(np.corrcoef(x.ll, x.res_t)[0, 1]), 3),
                fp_per30_by_lift_fifth=[round(float(v), 2) for v in by_q.values], by_adv=by_adv)


def main():
    out = {}
    D25 = W.load(2025)
    out["reliability_2025"] = reliability(D25)
    x25 = rows(D25, range(5, 19))
    out["2025"] = summarize(x25)
    D26 = W.load(2026)
    thru = int(D26["s"].week.max())
    x26 = rows(D26, range(5, thru + 1))
    out["2026"] = summarize(x26)
    a, b = out["2025"], out["2026"]
    rel = out["reliability_2025"]
    out["summary"] = dict(
        seasons="2025 wk 5-18 + 2026 wk 5-%d" % thru, n=a["n"] + b["n"],
        r_2025=a["r_lift_resid"], ci_2025=a["r_lift_90ci"], r_2026=b["r_lift_resid"],
        adv_2025={k: v["fp_vs_own_rate"] for k, v in a["by_adv"].items()}, adv_2026={k: v["fp_vs_own_rate"] for k, v in b["by_adv"].items()},
        wr_side_rel=rel["receiver_side"], cb_side_rel=rel["corner_side"])
    json.dump(out, open("output/wrcb_backtest.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
