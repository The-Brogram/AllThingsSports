"""Do the simulation's fantasy projections beat the obvious baseline? Walk-forward, same setup as backtest.py
(train on data before each week; availability = who actually played, which flatters the model slightly).

For every projected QB/RB/WR/TE (PPR, 4-point pass TDs, no fumbles or two-pointers on either side), compares:
  - model: mean of the simulated fantasy points
  - baseline: the player's average over his last 8 games before that week (needs 2+ games)
and checks the ranges: does the 10th-90th percentile band hold ~80% of outcomes, do boom/bust chances match how often
they happen, and for start/sit, when the sims say A outscores B 70% of the time, does A win about 70%?
Raw boom chances run low and bust chances high, so both get a logistic recalibration (fit on 2025, checked on 2026,
then refit on everything for live use; fantasy.py applies it).
Run: venv/bin/python fantasy_backtest.py -> output/fantasy_backtest.json (the Fantasy page shows the summary)
     venv/bin/python fantasy_backtest.py summary   re-summarize the saved rows without re-simulating
"""
import json
import numpy as np
import pandas as pd
import engine as E
import fantasy as F
from backtest import WEEKS

N = 2000
KEEP = 300


def run(seed=11):
    d = E.load()
    rng = np.random.default_rng(seed)
    games = d["games"]
    resid = E.residual_pairs(games[games.season < 2025])
    fit25 = E.fit_script(E.script_rows(d, E.tidx(2025, 4), E.tidx(2025, 22)))
    st = d["stats"].fillna(0).copy()
    st["fp"] = st.receptions + .1 * (st.receiving_yards + st.rushing_yards) + 6 * (st.receiving_tds + st.rushing_tds) + .04 * st.passing_yards + 4 * st.passing_tds - 2 * st.passing_interceptions
    st = st[st.season_type == "REG"].sort_values("t")
    rows, draws = [], []
    for season, wk in WEEKS:
        t = E.tidx(season, wk)
        r = E.ratings(d["team_games"], t)
        actual = st[st.t == t]
        prior = st[st.t < t].groupby("player_id").tail(8).groupby("player_id").fp.agg(["mean", "size"])
        sn = d["snaps"]
        on = sn[(sn.season == season) & (sn.week == wk) & (sn.offense_snaps > 0)]
        for g in games[(games.season == season) & (games.week == wk) & games.result.notna() & (games.game_type == "REG")].itertuples():
            def active(tm):
                names = set(on[on.team == tm].player)
                h = st[(st.team == tm) & (st.t < t)]
                return set(h[h.name.isin(names)].player_id) | set(actual[actual.team == tm].player_id)
            qb = {("qb", g.home_team): g.home_qb_id, ("qb", g.away_team): g.away_qb_id}
            s = E.simulate_game(rng, d, g.home_team, g.away_team, g.spread_line, g.total_line, t, fit25, resid, n=N,
                                active_h=active(g.home_team), active_a=active(g.away_team), avail=qb, ratings_cache=r)
            for df, sim, tm in ((s["dfh"], s["sh"], g.home_team), (s["dfa"], s["sa"], g.away_team)):
                act = actual[actual.team == tm].groupby("player_id").fp.sum()
                for j in range(len(df)):
                    p = df.iloc[j]
                    c = F.comps(df, sim, j)
                    if c is None or p.pos not in F.BOOM:
                        continue
                    f = F.points(c)
                    if f.mean() < F.MIN_PPR:
                        continue
                    b = prior.loc[p.player_id] if p.player_id in prior.index else None
                    rows.append(dict(season=season, week=wk, game=g.game_id, pid=p.player_id, pos=p.pos, proj=f.mean(), p10=np.percentile(f, 10), p90=np.percentile(f, 90),
                                     boom=(f >= F.BOOM[p.pos]).mean(), bust=(f < F.BUST[p.pos]).mean(), act=float(act.get(p.player_id, 0.0)),
                                     base=float(b["mean"]) if b is not None and b["size"] >= 2 else np.nan))
                    draws.append(f[:KEEP].astype(np.float32))
        print(season, wk, len(rows), flush=True)
    return pd.DataFrame(rows), np.vstack(draws)


def lg(p):
    p = np.clip(p, 1e-3, 1 - 1e-3)
    return np.log(p / (1 - p))


def fit_cal(p, y):
    """Logistic recalibration: calibrated = 1 / (1 + exp(-(a + b * logit(p))))."""
    X = np.c_[np.ones(len(p)), lg(p)]
    w = np.array([0.0, 1.0])
    for _ in range(50):
        q = 1 / (1 + np.exp(-X @ w))
        w -= np.linalg.solve(X.T @ (X * (q * (1 - q))[:, None]), X.T @ (q - y))
    return [round(float(v), 4) for v in w]


def apply_cal(p, w):
    return 1 / (1 + np.exp(-(w[0] + w[1] * lg(np.asarray(p)))))


def summarize(x, D, seed=5):
    out = {"n": len(x), "weeks": f"2025 wk {WEEKS[0][1]}-18, 2026 wk 4-{WEEKS[-1][1]}"}
    y = x.dropna(subset=["base"])
    acc = {}
    for pos, z in [("ALL", y)] + list(y.groupby("pos")):
        acc[pos] = dict(n=len(z), mae_model=round(float((z.proj - z.act).abs().mean()), 2), mae_base=round(float((z.base - z.act).abs().mean()), 2),
                        r_model=round(float(np.corrcoef(z.proj, z.act)[0, 1]), 3), r_base=round(float(np.corrcoef(z.base, z.act)[0, 1]), 3),
                        model_closer=round(float(((z.proj - z.act).abs() < (z.base - z.act).abs()).mean()), 3))
    out["accuracy"] = acc
    # weekly ranking: within each week and position, Spearman rank correlation of projection vs actual, top-36 by projection
    sp = []
    for (s, w, pos), z in y.groupby(["season", "week", "pos"]):
        z = z.nlargest(36 if pos != "QB" else 24, "proj")
        if len(z) >= 10:
            sp.append(dict(pos=pos, model=z.proj.rank().corr(z.act.rank()), base=z.base.rank().corr(z.act.rank())))
    sp = pd.DataFrame(sp)
    out["rank_corr"] = {pos: dict(model=round(float(z.model.mean()), 3), base=round(float(z.base.mean()), 3), weeks=len(z)) for pos, z in sp.groupby("pos")}
    out["band_10_90"] = round(float(((x.act >= x.p10) & (x.act <= x.p90)).mean()), 3)
    out["below_p10"] = round(float((x.act < x.p10).mean()), 3)
    out["above_p90"] = round(float((x.act > x.p90).mean()), 3)
    cal = lambda p, hit: [dict(band=f"{int(lo * 100)}-{int(hi * 100)}%", n=int(m.sum()), pred=round(float(p[m].mean()), 3), hit=round(float(hit[m].mean()), 3))
                          for lo, hi in ((0, .1), (.1, .2), (.2, .3), (.3, .45), (.45, .6), (.6, 1.01)) for m in [(p >= lo) & (p < hi)] if m.sum() >= 30]
    boom_hit = x.act.values >= x.pos.map(F.BOOM).values
    bust_hit = x.act.values < x.pos.map(F.BUST).values
    tr = (x.season == 2025).values
    out["cal"] = {}
    for k, hit in (("boom", boom_hit.astype(float)), ("bust", bust_hit.astype(float))):
        w25 = fit_cal(x[k].values[tr], hit[tr])
        c = apply_cal(x[k].values[~tr], w25)  # 2026 holdout, calibration fit on 2025 only
        out[k + "_holdout"] = [
            dict(band=f"{int(lo * 100)}-{int(hi * 100)}%", n=int(m.sum()), raw=round(float(x[k].values[~tr][m].mean()), 3), pred=round(float(c[m].mean()), 3), hit=round(float(hit[~tr][m].mean()), 3))
            for lo, hi in ((0, .1), (.1, .2), (.2, .3), (.3, .45), (.45, .6), (.6, 1.01)) for m in [(c >= lo) & (c < hi)] if m.sum() >= 15]
        out[k + "_holdout_brier"] = dict(raw=round(float(((x[k].values[~tr] - hit[~tr]) ** 2).mean()), 4), cal=round(float(((c - hit[~tr]) ** 2).mean()), 4), n=int((~tr).sum()))
        out["cal"][k] = fit_cal(x[k].values, hit)  # live calibration, fit on every week
    out["boom_raw"] = cal(x.boom.values, boom_hit)
    out["bust_raw"] = cal(x.bust.values, bust_hit)
    # start/sit: random same-position pairs from the same week, top 36 by projection (24 for QB), different games
    rng = np.random.default_rng(seed)
    pr = []
    x = x.reset_index(drop=True)
    for (s, w, pos), z in x.groupby(["season", "week", "pos"]):
        z = z.nlargest(36 if pos != "QB" else 24, "proj")
        ix = z.index.values
        for _ in range(60):
            a, b = rng.choice(ix, 2, replace=False)
            if x.game[a] == x.game[b] or x.act[a] == x.act[b]:
                continue
            p = float((D[a] > D[b]).mean() + .5 * (D[a] == D[b]).mean())
            pr.append((p, float(x.act[a] > x.act[b])) if p >= .5 else (1 - p, float(x.act[b] > x.act[a])))
    pr = np.array(pr)
    out["start_sit"] = [dict(band=f"{int(lo * 100)}-{int(hi * 100)}%", n=int(m.sum()), pred=round(float(pr[m, 0].mean()), 3), hit=round(float(pr[m, 1].mean()), 3))
                        for lo, hi in ((.5, .6), (.6, .7), (.7, .8), (.8, .9), (.9, 1.01)) for m in [(pr[:, 0] >= lo) & (pr[:, 0] < hi)] if m.sum() >= 30]
    out["start_sit_n"] = len(pr)
    return out


def main():
    import sys
    if sys.argv[1:] == ["summary"]:
        x, D = pd.read_pickle("output/fantasy_bt_rows.pkl"), np.load("output/fantasy_bt_draws.npy")
    else:
        x, D = run()
        x.to_pickle("output/fantasy_bt_rows.pkl")
        np.save("output/fantasy_bt_draws.npy", D)
    out = summarize(x, D)
    json.dump(out, open("output/fantasy_backtest.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
