"""Round two of the five-season study: rating settings, betting-market systems, and a model built to find the market's mistakes.
Run after five_season_study.py: venv/bin/python five_season_round2.py   -> output/five_season_round2.json
Same rules: everything chosen on 2021-2024, then scored once on 2025-2026."""
import itertools
import json
import numpy as np
import pandas as pd
import five_season_study as S

DEV, HOLD, FIVE = S.DEV, S.HOLD, (2021, 2022, 2023, 2024, 2025)
V6 = S.ADJ + S.EXTRA


def ats(x, side_home):
    """side_home: True = we bet the home team, False = away, NaN = no bet. Returns cover rate vs the closing spread."""
    x = x.assign(sh=side_home)
    x = x[x.sh.notna() & (x.result != x.spread)]
    won = np.where(x.sh.astype(bool), x.result > x.spread, x.result < x.spread)
    return S.pct(int(won.sum()), len(x))


def history(g):
    """What each team did before each game: last margin, last cover margin, season cover margin and win rate."""
    rows = []
    for r in g.itertuples():
        m = r.result - r.spread_line
        rows += [(r.game_id, r.gw, r.season, r.home_team, r.result, m), (r.game_id, r.gw, r.season, r.away_team, -r.result, -m)]
    t = pd.DataFrame(rows, columns=["game_id", "gw", "season", "team", "margin", "cover"]).sort_values("gw")
    gb = t.groupby(["team", "season"])
    t["last_margin"] = gb.margin.shift(1)
    t["last_cover"] = gb.cover.shift(1)
    t["szn_cover"] = gb.cover.transform(lambda s: s.shift(1).expanding().mean())
    t["szn_win"] = gb.margin.transform(lambda s: (s.shift(1) > 0).expanding().mean())
    t["gp"] = gb.cumcount()
    return t.set_index(["game_id", "team"])


if __name__ == "__main__":
    g, tg, q = S.load()
    out = {}
    # ---- A. rating settings: recency, carry-over from last season, shrinkage (27 builds), judged on development seasons
    grid = []
    for dec, pri, lam in itertools.product((.90, .95, .98), (.4, .6, .8), (1.5, 3.0, 6.0)):
        S.DECAY, S.PRIOR, S.LAM = dec, pri, lam
        F = S.features(g, tg, q)
        F = F[F.season >= 2016].reset_index(drop=True)
        p = S.walk(F, V6)
        grid.append(dict(decay=dec, prior=pri, lam=lam, dev=S.grade(F, p, DEV), hold=S.grade(F, p, HOLD)))
    grid.sort(key=lambda z: z["dev"]["mae"])
    out["settings"] = dict(best=grid[0], worst=grid[-1], default=next(z for z in grid if (z["decay"], z["prior"], z["lam"]) == (.95, .6, 3.0)))
    b = grid[0]
    S.DECAY, S.PRIOR, S.LAM = b["decay"], b["prior"], b["lam"]
    F = S.features(g, tg, q)
    F = F[F.season >= 2016].reset_index(drop=True)
    p6 = S.walk(F, V6)
    # ---- B. betting-market systems, each a fixed rule
    H = history(g)
    for side, col in (("h", "home"), ("a", "away")):
        j = H.reindex(list(zip(F.game_id, F[col])))
        for c in ("last_margin", "last_cover", "szn_cover", "szn_win", "gp"):
            F[f"{c}_{side}"] = j[c].values
    gi = g.set_index("game_id")
    F["rest_h"], F["rest_a"] = gi.home_rest.reindex(F.game_id).values, gi.away_rest.reindex(F.game_id).values
    dog_home = np.where(F.spread < 0, True, np.where(F.spread > 0, False, np.nan)).astype(object)
    dog_home = pd.Series(pd.to_numeric(pd.Series(dog_home), errors="coerce").values, index=F.index)
    nan = np.nan
    pick = lambda cond_home, cond_away: pd.Series(np.where(cond_home & ~cond_away, 1.0, np.where(cond_away & ~cond_home, 0.0, nan)), index=F.index)
    gap = p6 - F.spread
    systems = {
        "Every underdog": dog_home,
        "Home underdogs": dog_home.where(dog_home == 1),
        "Road underdogs": dog_home.where(dog_home == 0),
        "Underdogs in division games": dog_home.where(F["div"] == 1),
        "Underdogs getting 7 or more": dog_home.where(F.spread.abs() >= 7),
        "Favorites laying 7 or more": (1 - dog_home).where(F.spread.abs() >= 7),
        "Every home team": pd.Series(1.0, index=F.index),
        "Team with 3+ more days of rest": pick(F.rest_h - F.rest_a >= 3, F.rest_a - F.rest_h >= 3),
        "Team that lost its last game by 14+": pick(F.last_margin_h <= -14, F.last_margin_a <= -14),
        "Against a team that won its last game by 14+": pick(F.last_margin_a >= 14, F.last_margin_h >= 14),
        "Team with the worse record against the spread (3+ pts a game, week 5 on)": pick((F.szn_cover_a - F.szn_cover_h >= 3) & (F.gp_h >= 4), (F.szn_cover_h - F.szn_cover_a >= 3) & (F.gp_a >= 4)),
        "Team that failed to cover last week by 10+": pick(F.last_cover_h <= -10, F.last_cover_a <= -10),
        "Team starting a different quarterback than usual": pick(F.qbsame_h < .5, F.qbsame_a < .5),
        "Underdogs in wind of 15+ mph": dog_home.where(F.wind >= 15),
        "Our team model disagrees with the spread by 3+": pd.Series(np.where(gap >= 3, 1.0, np.where(gap <= -3, 0.0, nan)), index=F.index),
        "Our team model disagrees with the spread by 5+": pd.Series(np.where(gap >= 5, 1.0, np.where(gap <= -5, 0.0, nan)), index=F.index),
    }
    sysres = []
    for name, sh in systems.items():
        sysres.append(dict(system=name, **{k: ats(F[F.season.isin(s)], sh[F.season.isin(s)]) for k, s in (("dev", DEV), ("hold", HOLD), ("five", FIVE))}))
    out["systems"] = sysres
    # ---- C. a model for the market's mistakes: predict (result - spread) from market context, overreaction signals and our own line
    X = F.assign(miss=F.result - F.spread, gap=gap, home_dog=(F.spread < 0).astype(float), big=(F.spread.abs() >= 7) * np.sign(F.spread),
                 lm=(F.last_margin_h.fillna(0) - F.last_margin_a.fillna(0)), lc=(F.last_cover_h.fillna(0) - F.last_cover_a.fillna(0)),
                 sc=(F.szn_cover_h.fillna(0) - F.szn_cover_a.fillna(0)), sw=(F.szn_win_h.fillna(.5) - F.szn_win_a.fillna(.5)))
    X["gap"] = X.gap.fillna(0)
    cols9 = ["spread", "total_line", "home_dog", "big", "div", "rest", "wind", "dome", "lm", "lc", "sc", "sw", "gap", "qb_change", "qb_new", "form", "sos"]
    variants = {}
    for lam in (20.0, 100.0, 400.0):
        miss = S.walk(X, cols9, target="miss", lam=lam)
        for cut in (0, .5, 1, 1.5, 2):
            sh = pd.Series(np.where(miss >= cut, 1.0, np.where(miss <= -cut, 0.0, nan)), index=F.index)
            variants[(lam, cut)] = (ats(F[F.season.isin(DEV)], sh[F.season.isin(DEV)]), ats(F[F.season.isin(HOLD)], sh[F.season.isin(HOLD)]), ats(F[F.season.isin(FIVE)], sh[F.season.isin(FIVE)]))
    ok = {k: v for k, v in variants.items() if v[0]["n"] >= 150}
    k = max(ok, key=lambda z: ok[z][0]["pct"])
    out["mistake_model"] = dict(chosen=dict(lam=k[0], cut=k[1]), dev=ok[k][0], hold=ok[k][1], five=ok[k][2],
                                all=[dict(lam=a, cut=c, dev=v[0], hold=v[1]) for (a, c), v in sorted(variants.items())])
    tr = X[X.season <= 2024]
    mu, sd, bcoef, _ = S.ridge_fit(tr[cols9].values, tr.miss.values, 100.0)
    out["mistake_weights"] = dict(sorted(zip(cols9, [float(v) for v in bcoef]), key=lambda z: -abs(z[1])))
    # ---- D. winner picks: how many games can be picked at 70%+, favorite alone vs favorite confirmed by our team model
    wres = []
    for name, cond in (("Favorite of 3+", F.spread.abs() >= 3), ("Favorite of 2.5+", F.spread.abs() >= 2.5),
                       ("Favorite of 3+ that our team model also picks", (F.spread.abs() >= 3) & (np.sign(p6) == np.sign(F.spread))),
                       ("Any favorite our team model also picks", (F.spread != 0) & (np.sign(p6) == np.sign(F.spread))),
                       ("Favorite our team model picks against", (F.spread != 0) & (np.sign(p6) != np.sign(F.spread)))):
        r = {}
        for kk, s in (("dev", DEV), ("hold", HOLD), ("five", FIVE)):
            x = F[F.season.isin(s) & cond & (F.result != 0)]
            r[kk] = S.pct(int(((x.spread > 0) == (x.result > 0)).sum()), len(x))
            r[kk]["share"] = len(x) / max(int((F.season.isin(s) & (F.result != 0)).sum()), 1)
        wres.append(dict(rule=name, **r))
    out["winner_rules"] = wres
    json.dump(out, open("output/five_season_round2.json", "w"), indent=1, default=float)

    f = lambda t: "      –      " if not t or not t.get("n") else f"{t['pct'] * 100:5.1f}% of {t['n']:<4}"
    s = out["settings"]
    print("A. Rating settings (27 tried), version 6:")
    for kk in ("best", "default", "worst"):
        z = s[kk]
        print(f"  {kk:>7}: decay {z['decay']} carry-over {z['prior']} shrink {z['lam']}: miss dev {z['dev']['mae']:.3f} hold {z['hold']['mae']:.3f} | winner dev {f(z['dev']['win'])} hold {f(z['hold']['win'])} | spread dev {f(z['dev']['ats'])} hold {f(z['hold']['ats'])}")
    print("\nB. Betting-market systems (cover rate against the closing spread):")
    print(f"  {'system':<74}{'2021-24':>15}{'2025-26':>15}{'2021-25':>15}")
    for z in sysres:
        print(f"  {z['system']:<74}{f(z['dev']):>15}{f(z['hold']):>15}{f(z['five']):>15}")
    m = out["mistake_model"]
    print(f"\nC. Mistake model: chosen on dev (shrink {m['chosen']['lam']}, gap {m['chosen']['cut']}+): dev {f(m['dev'])}  hold {f(m['hold'])}  five seasons {f(m['five'])}")
    print("   all variants (dev -> hold): " + "; ".join(f"[{z['lam']:.0f},{z['cut']}] {z['dev']['pct'] * 100:.1f}->{z['hold']['pct'] * 100:.1f} (n {z['hold']['n']})" for z in m["all"] if z["dev"].get("n")))
    print("   weights (points of miss per 1 SD): " + ", ".join(f"{a} {v:+.2f}" for a, v in list(out["mistake_weights"].items())[:8]))
    print("\nD. Winner rules:")
    for z in wres:
        print(f"  {z['rule']:<48} dev {f(z['dev'])} ({z['dev'].get('share', 0) * 100:.0f}% of games)  hold {f(z['hold'])} ({z['hold'].get('share', 0) * 100:.0f}%)  five {f(z['five'])}")
