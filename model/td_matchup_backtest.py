"""Does the red-zone matchup factor (Touchdowns tab) make anytime-TD forecasts better?

Uses the stored backtest forecasts (output/bt_rows.pkl, anytime TD rows) and rebuilds the matchup factor for each
player-game from play-by-play strictly before that week, exactly as gamebook.touchdowns does live:
  lift = (team RZ passes/g * player RZ target share * opp shrunk RZ pass-TD rate + team RZ runs/g * carry share * opp shrunk RZ rush-TD rate)
         / (same with league rates), capped 0.7-1.4;  adjusted p = 1 - (1 - p)^lift.
Scores Brier and log loss for the calibrated model alone vs with the factor, fits a strength exponent on 2025 and tests it
on the 2026 holdout, and checks each game's top 5.
Run: venv/bin/python td_matchup_backtest.py   ->  output/td_matchup_backtest.json
"""
import json
import numpy as np
import pandas as pd
from backtest import apply_cal
from engine import DATA

K = 40
COLS = ["season", "season_type", "week", "game_id", "posteam", "defteam", "play_type", "yardline_100", "pass_attempt", "rush_attempt", "pass_touchdown",
        "rush_touchdown", "td_team", "receiver_player_id", "rusher_player_id", "qb_kneel", "qb_spike", "two_point_attempt"]


def rz_plays():
    p = pd.concat([pd.read_csv(f"{DATA}/play_by_play_{y}.csv.gz", usecols=COLS, low_memory=False) for y in (2025, 2026)])
    p = p[(p.season_type == "REG") & (p.yardline_100 <= 20) & p.play_type.isin(["pass", "run"]) & (p.qb_kneel != 1) & (p.qb_spike != 1) & (p.two_point_attempt != 1)]
    return p.assign(ptd=((p.pass_touchdown == 1) & (p.td_team == p.posteam)).astype(int), rtd=((p.rush_touchdown == 1) & (p.td_team == p.posteam)).astype(int))


def lifts(bt, rz):
    out = np.ones(len(bt))
    info = []
    cache = {}
    for i, r in enumerate(bt.itertuples()):
        key = (r.season, r.week)
        if key not in cache:
            x = rz[(rz.season == r.season) & (rz.week < r.week)]
            pa, ru = x[x.pass_attempt == 1], x[x.rush_attempt == 1]
            games = pd.concat([x[["game_id", "posteam"]].rename(columns={"posteam": "t"})]).drop_duplicates().groupby("t").size()
            cache[key] = dict(pa=pa, ru=ru, lg_p=pa.ptd.mean(), lg_r=ru.rtd.mean(), games=games,
                              dp=pa.groupby("defteam").agg(n=("ptd", "size"), td=("ptd", "sum")), dr=ru.groupby("defteam").agg(n=("rtd", "size"), td=("rtd", "sum")),
                              tg=pa.groupby(["posteam", "receiver_player_id"]).size(), car=ru.groupby(["posteam", "rusher_player_id"]).size(),
                              tpa=pa.groupby("posteam").size(), tru=ru.groupby("posteam").size())
        C = cache[key]
        t, o, pid = r.team, r.opp, r.pid
        g = C["games"].get(t, 0)
        if g == 0:
            info.append(1.0)
            continue
        tpa, tru = C["tpa"].get(t, 0), C["tru"].get(t, 0)
        tsh = C["tg"].get((t, pid), 0) / tpa if tpa else 0
        csh = C["car"].get((t, pid), 0) / tru if tru else 0
        dp = C["dp"].loc[o] if o in C["dp"].index else pd.Series(dict(n=0, td=0))
        dr = C["dr"].loc[o] if o in C["dr"].index else pd.Series(dict(n=0, td=0))
        Dp = (dp.td + K * C["lg_p"]) / (dp.n + K)
        Dr = (dr.td + K * C["lg_r"]) / (dr.n + K)
        neu = tpa / g * tsh * C["lg_p"] + tru / g * csh * C["lg_r"]
        vs = tpa / g * tsh * Dp + tru / g * csh * Dr
        out[i] = float(np.clip(vs / neu, .7, 1.4)) if neu > 0 else 1.0
        info.append(out[i])
    return out


def brier(p, y):
    return float(((p - y) ** 2).mean())


def logloss(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def boot(p0, p1, y, n=2000, seed=7):
    """Share of bootstrap resamples (by game) where the factor lowers Brier."""
    rng = np.random.default_rng(seed)
    gid = y.index.get_level_values(0) if isinstance(y.index, pd.MultiIndex) else None
    d0, d1 = (p0 - y.values) ** 2, (p1 - y.values) ** 2
    m = len(y)
    wins = 0
    for _ in range(n):
        idx = rng.integers(0, m, m)
        wins += d1[idx].mean() < d0[idx].mean()
    return wins / n


def main():
    bt = pd.read_pickle("output/bt_rows.pkl")
    bt = bt[bt.mkt == "atd"].reset_index(drop=True)
    cal = json.load(open("output/backtest.json"))["cal"]["atd"]
    bt["p"] = apply_cal(bt.proj.values, cal)
    bt["y"] = (bt.act > 0).astype(int)
    bt["lift"] = lifts(bt, rz_plays())
    res = {}
    # strength exponent fit on 2025 only: adjusted = 1 - (1-p)^(lift^a); a=1 is what the tab uses, a=0 is the model alone
    A = np.round(np.arange(0, 3.01, .1), 2)
    tr = bt[bt.season == 2025]
    fit = {float(a): brier(1 - (1 - tr.p) ** (tr.lift ** a), tr.y) for a in A}
    a_best = min(fit, key=fit.get)
    for lab, x in (("2025 (exponent fit here)", bt[bt.season == 2025]), ("2026 holdout", bt[bt.season == 2026])):
        p0 = x.p.values
        p1 = (1 - (1 - x.p) ** x.lift).values
        pb = (1 - (1 - x.p) ** (x.lift ** a_best)).values
        y = x.y
        res[lab] = dict(n=len(x), td_rate=float(y.mean()), brier_model=brier(p0, y), brier_tab=brier(p1, y), brier_fit=brier(pb, y),
                        logloss_model=logloss(p0, y), logloss_tab=logloss(p1, y), p_tab_better=boot(p0, p1, y),
                        by_pos={pos: dict(n=len(z), brier_model=brier(z.p.values, z.y), brier_tab=brier((1 - (1 - z.p) ** z.lift).values, z.y))
                                for pos, z in x.groupby(x.pos.where(x.pos.isin(["RB", "WR", "TE", "QB"]), "other"))})
        # do players the factor boosts actually score more than the model expects? (calibration by lift bucket)
        b = pd.cut(x.lift, [0, .9, .97, 1.03, 1.1, 2], labels=["<0.90", "0.90-0.97", "0.97-1.03", "1.03-1.10", ">1.10"])
        res[lab]["by_lift"] = {str(k): dict(n=len(z), model_p=float(z.p.mean()), tab_p=float((1 - (1 - z.p) ** z.lift).mean()), actual=float(z.y.mean()))
                               for k, z in x.groupby(b, observed=True)}
        # top 5 per game: TDs scored by the model's top 5 vs the matchup-adjusted top 5
        x = x.assign(adj=1 - (1 - x.p) ** x.lift)
        t0 = x.sort_values("p", ascending=False).groupby("game_id").head(5)
        t1 = x.sort_values("adj", ascending=False).groupby("game_id").head(5)
        res[lab]["top5"] = dict(games=int(x.game_id.nunique()), hits_model=int(t0.y.sum()), hits_tab=int(t1.y.sum()),
                                hit_rate_model=float(t0.y.mean()), hit_rate_tab=float(t1.y.mean()),
                                receivers_model=float(t0.pos.isin(["WR", "TE"]).mean()), receivers_tab=float(t1.pos.isin(["WR", "TE"]).mean()))
    res["exponent_fit_2025"] = dict(best=a_best, brier_by_exponent={k: round(v, 6) for k, v in fit.items() if k in (0.0, .5, 1.0, 1.5, 2.0, 3.0) or k == a_best})
    json.dump(res, open("output/td_matchup_backtest.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
