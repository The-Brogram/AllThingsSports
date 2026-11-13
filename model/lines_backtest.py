"""Does the O-line / D-line matchup tell us anything the model does not already know?

Walk-forward, 2025 weeks 5-18 and 2026 weeks 4 to the last backtested week. Before each game, each line's pressure rate
(allowed / generated) is built from that season's earlier games only and combined into an expected pressure rate.
1. Does the expected rate predict the pressure that actually happened? (vs each side alone)
2. Does it predict passing results beyond the model? QB passing yards minus the model's projection, and the team's
   receiving yards minus projection, both from output/bt_rows.pkl (same walk-forward sims as backtest.py).
   The model already uses each team's sack rate, so only what pressure adds on top shows up here.
Run: venv/bin/python lines_backtest.py -> output/lines_backtest.json (the Game Book's Lines tab shows the summary)
"""
import json
import numpy as np
import pandas as pd
import lines as L

SEASONS = [(2025, range(5, 19)), (2026, range(4, 9))]


def rows():
    out = []
    for season, weeks in SEASONS:
        x = L.team_games(season)
        for w in weeks:
            prev = x[x.week < w]
            if prev.empty:
                continue
            lg = L.league(prev)
            for r in x[x.week == w].itertuples():
                ro, rd = L.rates(prev[prev.off == r.off], lg), L.rates(prev[prev.dfn == r.dfn], lg)
                out.append(dict(season=season, week=w, game=r.game_id, off=r.off, dfn=r.dfn, db=r.db, act=r.pr / r.db, sk=r.sk / r.db,
                                ol=ro["pr"], dl=rd["pr"], lg=lg["pr"], exp=float(L.combine(ro["pr"], rd["pr"], lg["pr"]))))
    return pd.DataFrame(out)


def boot(a, b, games, n=1000, seed=3):
    rng = np.random.default_rng(seed)
    idx = list(pd.Series(range(len(games))).groupby(games.values).indices.values())
    bs = []
    for _ in range(n):
        k = np.concatenate([idx[i] for i in rng.integers(0, len(idx), len(idx))])
        bs.append(np.corrcoef(a[k], b[k])[0, 1])
    return [round(float(v), 3) for v in np.percentile(bs, [5, 95])]


def main():
    x = rows()
    bt = pd.read_pickle("output/bt_rows.pkl")
    qb = bt[bt.mkt == "pass"].assign(res=lambda z: z.act - z.proj).groupby(["season", "week", "team"]).res.sum().rename("qb_res")
    rc = bt[bt.mkt == "rec"].assign(res=lambda z: z.act - z.proj).groupby(["season", "week", "team"]).res.sum().rename("rec_res")
    x = x.merge(qb, left_on=["season", "week", "off"], right_index=True, how="left").merge(rc, left_on=["season", "week", "off"], right_index=True, how="left")
    out = dict(n=len(x), games=int(x.game.nunique()), seasons="2025 wk 5-18, 2026 wk 4-8")
    r = lambda a, b: round(float(np.corrcoef(a, b)[0, 1]), 3)
    w = x.db.values
    mae = lambda pred: round(float(np.average(np.abs(pred - x.act), weights=w)), 4)
    out["pressure"] = dict(r_matchup=r(x.exp, x.act), r_line_only=r(x.ol, x.act), r_rush_only=r(x.dl, x.act),
                           mae_matchup=mae(x.exp), mae_league=mae(x.lg), ci_matchup=boot(x.exp.values, x.act.values, x.game),
                           r_sacks=r(x.exp, x.sk))
    # expected-pressure thirds: what actually happened
    q = pd.qcut(x.exp - x.lg, 3, labels=["Line favored", "Even", "Rush favored"])
    y = x.dropna(subset=["qb_res"])
    qy = pd.qcut(y.exp - y.lg, 3, labels=["Line favored", "Even", "Rush favored"])
    out["thirds"] = [dict(group=str(k), n=int(len(z)), exp=round(float(z.exp.mean()), 3), act=round(float(np.average(z.act, weights=z.db)), 3),
                          sack=round(float(np.average(z.sk, weights=z.db)), 3)) for k, z in x.groupby(q, observed=True)]
    out["thirds_yards"] = [dict(group=str(k), n=int(len(z)), qb_res=round(float(z.qb_res.mean()), 1), rec_res=round(float(z.rec_res.mean()), 1))
                           for k, z in y.groupby(qy, observed=True)]
    d = (y.exp - y.lg).values
    out["passing"] = dict(n=len(y), r_qb=r(d, y.qb_res), ci_qb=boot(d, y.qb_res.values, y.game), r_rec=r(d, y.rec_res.fillna(0)),
                          ci_rec=boot(d, y.rec_res.fillna(0).values, y.game))
    json.dump(out, open("output/lines_backtest.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
