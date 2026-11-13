"""Five-season study: can a team-vs-team model pick NFL winners and beat the spread, and which factors matter?

Run: venv/bin/python five_season_study.py        -> output/five_season_study.json and a printed report

Rules that keep the result honest
  * Walk-forward everywhere: a game's team ratings use only earlier games; each season's model is trained on earlier seasons only.
  * Versions and thresholds are chosen on 2021-2024 (development). 2025 and 2026-to-date are the holdout and are scored once.
  * Spread picks are graded against the closing spread (pushes dropped). At -110 a bettor needs 52.4% to break even.
Data: nflverse games.csv (scores, closing spread/total, rest, roof, wind, starting QBs) and weekly player stats summed to team games.
"""
import json
import numpy as np
import pandas as pd

D = "data"
FIX = {"OAK": "LV", "SD": "LAC", "STL": "LA"}
DEV, HOLD = (2021, 2022, 2023, 2024), (2025, 2026)
# team-game metrics: name -> higher is better for the offense?
METRICS = ["pts", "epa", "pass_epa", "rush_epa", "nya", "ypc", "sack", "to", "fd"]
DECAY, PRIOR, LAM, WINDOW = 0.95, 0.6, 3.0, 44


def team_games():
    use = ["season", "week", "team", "opponent_team", "position", "player_id", "attempts", "passing_yards", "passing_interceptions", "sacks_suffered", "sack_yards_lost",
           "sack_fumbles_lost", "passing_epa", "passing_first_downs", "carries", "rushing_yards", "rushing_fumbles_lost", "rushing_epa", "rushing_first_downs", "receiving_fumbles_lost"]
    st = pd.concat([pd.read_csv(f"{D}/stats_player_week_{y}.csv", usecols=use, low_memory=False) for y in range(2015, 2027)]).fillna(0)
    st["team"] = st.team.replace(FIX)
    num = [c for c in use if c not in ("season", "week", "team", "opponent_team", "position", "player_id")]
    t = st.groupby(["season", "week", "team"])[num].sum().reset_index()
    db = t.attempts + t.sacks_suffered
    plays = db + t.carries
    t["epa"] = (t.passing_epa + t.rushing_epa) / plays.clip(lower=1)
    t["pass_epa"] = t.passing_epa / db.clip(lower=1)
    t["rush_epa"] = t.rushing_epa / t.carries.clip(lower=1)
    t["nya"] = (t.passing_yards - t.sack_yards_lost) / db.clip(lower=1)
    t["ypc"] = t.rushing_yards / t.carries.clip(lower=1)
    t["sack"] = t.sacks_suffered / db.clip(lower=1)
    t["to"] = t.passing_interceptions + t.sack_fumbles_lost + t.rushing_fumbles_lost + t.receiving_fumbles_lost
    t["fd"] = (t.passing_first_downs + t.rushing_first_downs) / plays.clip(lower=1)
    # quarterbacks: EPA per dropback for each QB game, for the "who is starting" factor
    q = st[(st.position == "QB") & (st.attempts + st.sacks_suffered > 0)].copy()
    q["db"] = q.attempts + q.sacks_suffered
    q = q[["season", "week", "player_id", "db", "passing_epa"]]
    return t[["season", "week", "team"] + METRICS[1:]], q


def load():
    g = pd.read_csv(f"{D}/games.csv", low_memory=False)
    g = g[(g.season >= 2015) & g.result.notna() & g.spread_line.notna()].copy()
    g["home_team"], g["away_team"] = g.home_team.replace(FIX), g.away_team.replace(FIX)
    g["gw"] = g.season * 22 + g.week.clip(upper=22)  # a running week count; the offseason gap falls out naturally
    t, q = team_games()
    rows = []
    for r in g.itertuples():
        for tm, op, pf, pa, hm in ((r.home_team, r.away_team, r.home_score, r.away_score, 1), (r.away_team, r.home_team, r.away_score, r.home_score, 0)):
            rows.append((r.game_id, r.season, r.week, r.gw, tm, op, hm if r.location != "Neutral" else .5, pf, pa))
    tg = pd.DataFrame(rows, columns=["game_id", "season", "week", "gw", "team", "opp", "home", "pts", "pa"]).merge(t, on=["season", "week", "team"], how="left")
    tg = tg.dropna(subset=["epa"]).reset_index(drop=True)
    return g.sort_values("gw").reset_index(drop=True), tg, q


def ratings(tg, gw, season):
    """Opponent-adjusted offense and defense for every metric, from games before `gw`: value = mu + off[team] - def[opp] + hfa.
    Returns {metric: (mu, off, deff)}, the raw (unadjusted) averages, each team's last-3 scoring margin and its schedule strength."""
    h = tg[(tg.gw < gw) & (tg.gw >= gw - WINDOW)]
    if len(h) < 200:
        return None
    teams = sorted(set(h.team) | set(h.opp))
    ix = {x: i for i, x in enumerate(teams)}
    m = len(teams)
    w = DECAY ** (gw - h.gw.values) * np.where(h.season.values < season, PRIOR, 1.0)
    X = np.zeros((len(h), 2 * m + 1))
    ti, oi = h.team.map(ix).values, h.opp.map(ix).values
    X[np.arange(len(h)), ti] = 1
    X[np.arange(len(h)), m + oi] = -1
    X[:, 2 * m] = h.home.values - .5
    Y = h[METRICS].values.astype(float)
    mu = (w[:, None] * Y).sum(0) / w.sum()
    A = (X.T * w) @ X + LAM * np.eye(2 * m + 1)
    A[2 * m, 2 * m] -= LAM
    C = np.linalg.solve(A, (X.T * w) @ (Y - mu))
    adj = {k: (mu[j], dict(zip(teams, C[:m, j])), dict(zip(teams, C[m:2 * m, j]))) for j, k in enumerate(METRICS)}
    hw = h.assign(w=w)
    raw_o = hw.groupby("team").apply(lambda x: pd.Series((x[METRICS].values * x.w.values[:, None]).sum(0) / x.w.sum(), index=METRICS), include_groups=False)
    raw_d = hw.groupby("opp").apply(lambda x: pd.Series((x[METRICS].values * x.w.values[:, None]).sum(0) / x.w.sum(), index=METRICS), include_groups=False)
    cur = h[h.season == season].sort_values("gw")
    last3 = cur.assign(md=cur.pts - cur.pa).groupby("team").md.apply(lambda s: s.tail(3).mean()).to_dict() if len(cur) else {}
    net = {x: adj["pts"][1][x] + adj["pts"][2][x] for x in teams}
    sos = hw.assign(on=hw.opp.map(net)).groupby("team").apply(lambda x: float((x.on * x.w).sum() / x.w.sum()), include_groups=False).to_dict()
    return dict(adj=adj, raw_o=raw_o, raw_d=raw_d, mu=dict(zip(METRICS, mu)), last3=last3, net=net, sos=sos)


def qb_table(q):
    """Each QB's EPA per dropback over his previous 16 games, shrunk toward zero by 150 dropbacks."""
    q = q.assign(gw=q.season * 22 + q.week.clip(upper=22)).sort_values("gw")
    out = {}
    for pid, x in q.groupby("player_id"):
        e, d, gws = x.passing_epa.values, x.db.values, x.gw.values
        for i in range(len(x)):
            lo = max(0, i - 16)
            out[(pid, gws[i])] = (e[lo:i].sum() / (d[lo:i].sum() + 150.0), d[lo:i].sum())
    return out, q


def features(g, tg, q):
    qv, _ = qb_table(q)
    starts = {}  # (team, gw) -> QB id who started
    for r in g.itertuples():
        starts[(r.home_team, r.gw)], starts[(r.away_team, r.gw)] = r.home_qb_id, r.away_qb_id
    by_team = {}
    for (tm, gw), pid in starts.items():
        by_team.setdefault(tm, []).append((gw, pid))
    for tm in by_team:
        by_team[tm].sort()
    rows, cache = [], {}
    for r in g.itertuples():
        if r.gw not in cache:
            cache[r.gw] = ratings(tg, r.gw, r.season)
        R = cache[r.gw]
        if R is None or r.home_team not in R["net"] or r.away_team not in R["net"]:
            continue
        H, A = r.home_team, r.away_team
        f = dict(game_id=r.game_id, season=r.season, week=r.week, gw=r.gw, home=H, away=A, result=r.result, spread=r.spread_line, total_line=r.total_line,
                 neutral=float(r.location == "Neutral"))
        for k in METRICS:
            mu, off, de = R["adj"][k]
            f["a_" + k] = (off[H] - de[A]) - (off[A] - de[H])                      # schedule-adjusted: home unit vs away unit, both directions
            ro, rd = R["raw_o"][k], R["raw_d"][k]
            f["r_" + k] = ((ro.get(H, mu) + rd.get(A, mu)) - (ro.get(A, mu) + rd.get(H, mu))) / 2   # same thing from raw averages
        f["form"] = (R["last3"].get(H, 0) - R["net"][H]) - (R["last3"].get(A, 0) - R["net"][A])       # last 3 games vs the team's rating
        f["sos"] = R["sos"].get(H, 0) - R["sos"].get(A, 0)                         # who has faced the tougher schedule
        for side, tm, pid in (("h", H, r.home_qb_id), ("a", A, r.away_qb_id)):
            prev = [p for gw_, p in by_team[tm] if gw_ < r.gw][-4:]
            f["qbsame_" + side] = (sum(p == pid for p in prev) / len(prev)) if prev else 1.0     # how many of the last 4 games this QB started
            usual = max(set(prev), key=prev.count) if prev else pid
            v_new, v_old = qv.get((pid, r.gw), (0.0, 0))[0], qv.get((usual, r.gw), (0.0, 0))[0]
            f["qbd_" + side] = v_new - v_old                                           # tonight's QB vs the usual one (0 when unchanged)
        f["qb_change"] = f["qbd_h"] - f["qbd_a"]
        f["qb_new"] = (1 - f["qbsame_h"]) - (1 - f["qbsame_a"])
        f["rest"] = float(np.clip((r.home_rest or 7) - (r.away_rest or 7), -7, 7))
        f["div"] = float(r.div_game)
        f["dome"] = float(str(r.roof) in ("dome", "closed"))
        f["wind"] = float(r.wind) if r.wind == r.wind else 0.0
        f["hfa"] = 1.0 - f["neutral"]
        rows.append(f)
    return pd.DataFrame(rows)


def ridge_fit(X, y, lam=5.0):
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1
    Z = (X - mu) / sd
    b = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ (y - y.mean()))
    return mu, sd, b, y.mean()


def walk(F, cols, target="result", lam=5.0, start=2021):
    """Predict each season from a model trained on all earlier seasons. Returns the out-of-sample prediction for every game."""
    pred = pd.Series(np.nan, index=F.index)
    for s in range(start, int(F.season.max()) + 1):
        tr, te = F[F.season < s], F[F.season == s]
        if not len(te):
            continue
        mu, sd, b, y0 = ridge_fit(tr[cols].values, tr[target].values, lam)
        pred[te.index] = y0 + ((te[cols].values - mu) / sd) @ b
    return pred


def pct(k, n):
    if not n:
        return dict(n=0)
    p = k / n
    se = (p * (1 - p) / n) ** .5
    return dict(n=int(n), pct=float(p), lo=float(p - 1.28 * se), hi=float(p + 1.28 * se))


def grade(F, pred, seasons, win_cut=0.0, ats_cut=0.0):
    x = F[F.season.isin(seasons)].assign(p=pred)
    x = x[x.p.notna()]
    w = x[(x.result != 0) & (x.p.abs() >= win_cut) & (x.p != 0)]
    a = x[(x.result != x.spread) & ((x.p - x.spread).abs() >= ats_cut) & (x.p != x.spread)]
    return dict(games=int(len(x)), mae=float((x.result - x.p).abs().mean()) if len(x) else None,
                win=pct(int(((w.p > 0) == (w.result > 0)).sum()), len(w)),
                ats=pct(int(((a.p > a.spread) == (a.result > a.spread)).sum()), len(a)))


GROUPS = {"Scoring (points for and against)": ["a_pts"], "Overall efficiency (EPA per play)": ["a_epa"], "Passing matchup": ["a_pass_epa", "a_nya"], "Running matchup": ["a_rush_epa", "a_ypc"],
          "Pass protection vs pass rush": ["a_sack"], "Turnovers": ["a_to"], "Moving the chains (first downs)": ["a_fd"], "Quarterback change": ["qb_change", "qb_new"],
          "Recent form (last 3 games)": ["form"], "Schedule faced": ["sos"], "Rest, division, roof, wind, home field": ["rest", "div", "dome", "wind", "hfa"]}
ADJ = [c for v in list(GROUPS.values())[:7] for c in v]
RAW = [c.replace("a_", "r_") for c in ADJ]
EXTRA = ["qb_change", "qb_new", "form", "sos", "rest", "div", "dome", "wind", "hfa"]
VERSIONS = {
    "1 Sportsbook favorite (the bar to beat)": None,
    "2 Points only, schedule-adjusted": ["a_pts", "hfa"],
    "3 Unit matchups, raw stats (no schedule adjustment)": RAW + ["hfa"],
    "4 Unit matchups, schedule-adjusted": ADJ + ["hfa"],
    "5 Version 4 + quarterback change": ADJ + ["hfa", "qb_change", "qb_new"],
    "6 Version 5 + form, schedule, rest, division, roof, wind": ADJ + EXTRA,
    "7 Version 6 + the sportsbook line as an input": ADJ + EXTRA + ["spread", "total_line"],
}


if __name__ == "__main__":
    g, tg, q = load()
    F = features(g, tg, q)
    F = F[F.season >= 2016].reset_index(drop=True)
    P, res = {}, dict(versions={}, dev=list(DEV), hold=list(HOLD), games=dict(dev=int(F.season.isin(DEV).sum()), hold=int(F.season.isin(HOLD).sum())))
    for name, cols in VERSIONS.items():
        P[name] = F.spread.copy() if cols is None else walk(F, cols)
        res["versions"][name] = dict(dev=grade(F, P[name], DEV), hold=grade(F, P[name], HOLD), five=grade(F, P[name], (2021, 2022, 2023, 2024, 2025)))
    # version 8: predict how far the result lands from the spread, from everything we know (the only way to beat the spread)
    P["8 Predict the miss against the spread"] = F.spread + walk(F.assign(miss=F.result - F.spread), ADJ + EXTRA + ["spread", "total_line"], target="miss", lam=40.0)
    res["versions"]["8 Predict the miss against the spread"] = dict(dev=grade(F, P["8 Predict the miss against the spread"], DEV), hold=grade(F, P["8 Predict the miss against the spread"], HOLD),
                                                                    five=grade(F, P["8 Predict the miss against the spread"], (2021, 2022, 2023, 2024, 2025)))
    # selective picking: thresholds chosen on development seasons only, then scored once on the holdout
    sel = {}
    for name in VERSIONS:
        best = None
        for c in (0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12):
            d = grade(F, P[name], DEV, win_cut=c)["win"]
            if d["n"] >= 100 and d["pct"] >= .70:
                best = c
                break
        w = None if best is None else dict(cut=best, dev=grade(F, P[name], DEV, win_cut=best)["win"], hold=grade(F, P[name], HOLD, win_cut=best)["win"])
        a = None
        if VERSIONS[name] is not None:
            cand = [(grade(F, P[name], DEV, ats_cut=c)["ats"], c) for c in (0, 1, 1.5, 2, 3, 4, 5)]
            cand = [(d, c) for d, c in cand if d["n"] >= 150]
            d, c = max(cand, key=lambda z: z[0]["pct"])
            a = dict(cut=c, dev=d, hold=grade(F, P[name], HOLD, ats_cut=c)["ats"])
        sel[name] = dict(win=w, ats=a)
    n8 = "8 Predict the miss against the spread"
    cand = [(grade(F, P[n8], DEV, ats_cut=c)["ats"], c) for c in (0, .25, .5, .75, 1, 1.5, 2)]
    cand = [(d, c) for d, c in cand if d["n"] >= 150]
    d, c = max(cand, key=lambda z: z[0]["pct"])
    sel[n8] = dict(win=None, ats=dict(cut=c, dev=d, hold=grade(F, P[n8], HOLD, ats_cut=c)["ats"]))
    res["selective"] = sel
    # how many games each winner threshold keeps, for the favorite: 70% by passing on close games
    res["favorite_by_spread"] = [dict(cut=c, **{k: grade(F, F.spread, s, win_cut=c)["win"] for k, s in (("dev", DEV), ("hold", HOLD))}) for c in (0, 1, 2, 3, 4, 5, 6, 7, 10)]
    # factor impact 1: points of margin per one standard deviation, in the full no-market model fit on 2016-2024
    cols = ADJ + EXTRA
    tr = F[F.season <= 2024]
    mu, sd, b, _ = ridge_fit(tr[cols].values, tr.result.values)
    coef = dict(zip(cols, b))
    # factor impact 2: remove one group at a time from version 6 and see what five out-of-sample seasons lose
    base = grade(F, P["6 Version 5 + form, schedule, rest, division, roof, wind"], (2021, 2022, 2023, 2024, 2025))
    imp = []
    for gname, gc in GROUPS.items():
        keep = [c for c in cols if c not in gc]
        gr = grade(F, walk(F, keep), (2021, 2022, 2023, 2024, 2025))
        x = F[F.season.between(2021, 2025)]
        miss = x.result - x.spread
        r_ats = max((abs(float(np.corrcoef(x[c], miss)[0, 1])), float(np.corrcoef(x[c], miss)[0, 1])) for c in gc)[1]
        imp.append(dict(group=gname, weight=float(sum(abs(coef[c]) for c in gc)), mae_loss=gr["mae"] - base["mae"], win_loss=base["win"]["pct"] - gr["win"]["pct"],
                        r_vs_spread=r_ats, n=int(len(x))))
    res["factors"] = sorted(imp, key=lambda z: -z["mae_loss"])
    res["adjust_vs_raw"] = dict(raw=res["versions"]["3 Unit matchups, raw stats (no schedule adjustment)"]["five"], adj=res["versions"]["4 Unit matchups, schedule-adjusted"]["five"])
    json.dump(res, open("output/five_season_study.json", "w"), indent=1)

    f = lambda t: "      –      " if not t or not t.get("n") else f"{t['pct'] * 100:5.1f}% of {t['n']:<4}"
    print(f"Games: development 2021-24 {res['games']['dev']}, holdout 2025-26 {res['games']['hold']}\n")
    print(f"{'version':<58}{'miss':>6}  {'winner dev':>14} {'winner hold':>14}   {'spread dev':>14} {'spread hold':>14}")
    for name, v in res["versions"].items():
        print(f"{name:<58}{v['five']['mae']:6.2f}  {f(v['dev']['win']):>14} {f(v['hold']['win']):>14}   {f(v['dev']['ats']) if 'favorite' not in name else '      –      ':>14} {f(v['hold']['ats']) if 'favorite' not in name else '      –      ':>14}")
    print("\nPicking only confident games (threshold set on 2021-24, then scored on 2025-26):")
    for name, v in sel.items():
        w, a = v["win"], v["ats"]
        print(f"{name:<58} winner: " + (f"gap {w['cut']:>4}+ pts  dev {f(w['dev'])}  hold {f(w['hold'])}" if w else "never reaches 70% on 100+ games") + (f"   | spread: gap {a['cut']}+  dev {f(a['dev'])}  hold {f(a['hold'])}" if a else ""))
    print("\nSportsbook favorite by size of spread (winner %):")
    for t in res["favorite_by_spread"]:
        print(f"  spread {t['cut']:>2}+  dev {f(t['dev'])}  hold {f(t['hold'])}")
    print("\nFactor impact (version 6, five out-of-sample seasons 2021-25):")
    print(f"{'factor':<44}{'weight':>7}{'miss added if removed':>24}{'winner % lost':>15}{'link to beating spread':>24}")
    for t in res["factors"]:
        print(f"{t['group']:<44}{t['weight']:7.2f}{t['mae_loss']:+24.3f}{t['win_loss'] * 100:+15.1f}{t['r_vs_spread']:+24.3f}")
