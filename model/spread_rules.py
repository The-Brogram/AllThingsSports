"""What the Nov 2026 five-season study found worth tracking, applied to the hub.

1. Confident winners: the sportsbook favorite by size of spread. Favorites of 3+ won 70.6% in 2021-25 (69.0% since 2006); our own
   team model did not improve on that, so the tier is the favorite itself, with its historical win rate.
2. Spread rules: three betting-market patterns that covered a little more than half the time (and "every underdog" as the baseline).
   They are LEADS, not proven edges: each is logged before kickoff at the price on offer (output/spread_picks.csv, first line seen is
   kept) and graded at that price, with the closing line for comparison. At -110 a bettor needs 52.4% to break even.

Called by season.py (build()); run alone to print the tables: venv/bin/python spread_rules.py
"""
import json
import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(ROOT, "output", "spread_picks.csv")
SEASON = 2026
RULES = [("div_dog", "Underdogs in division games"), ("big_dog", "Underdogs getting 7 or more"), ("qb_change", "Team starting a different quarterback than usual"),
         ("dog", "Every underdog (baseline)")]
BANDS = [(3, 4, "3 to 3.5"), (4, 7, "4 to 6.5"), (7, 10, "7 to 9.5"), (10, 99, "10 or more")]
COLS = ["logged", "season", "week", "game_id", "gameday", "rule", "team", "opp", "line", "price"]


def pay(a):
    a = float(a) if a == a and a else -110.0
    return 100 / -a if a < 0 else a / 100


def pct(k, n):
    if not n:
        return dict(n=0)
    p = k / n
    se = (p * (1 - p) / n) ** .5
    return dict(n=int(n), w=int(k), pct=float(p), lo=float(max(p - 1.28 * se, 0)), hi=float(min(p + 1.28 * se, 1)))


def games():
    g = pd.read_csv(os.path.join(ROOT, "data", "games.csv"), low_memory=False)
    g = g[g.season >= 2005].copy()
    g["gw"] = g.season * 22 + g.week.clip(upper=22)
    return g.sort_values(["gw", "gameday", "game_id"]).reset_index(drop=True)


def expected_qbs():
    """Who is expected to start for each team now: slate.json overrides first, then the newest depth chart."""
    out = {}
    try:
        dc = pd.read_csv(os.path.join(ROOT, "data", f"depth_charts_{SEASON}.csv"), usecols=["dt", "team", "gsis_id", "pos_abb", "pos_rank"], low_memory=False)
        dc = dc[(dc.pos_abb == "QB") & (dc.pos_rank == 1)]
        dc = dc[dc.dt == dc.groupby("team").dt.transform("max")].drop_duplicates("team")
        out = dict(zip(dc.team, dc.gsis_id))
    except (OSError, ValueError, KeyError):
        pass
    over = {}
    try:
        for date, v in json.load(open(os.path.join(ROOT, "slate.json"))).get("slates", {}).items():
            for t, pid in (v.get("qb") or {}).items():
                over[(date, t)] = pid
    except (OSError, ValueError):
        pass
    return out, over


def rule_picks(g, exp=None, over=None):
    """One row per (game, rule) pick. For finished games the QB rule uses who actually started; for upcoming ones, who is expected to."""
    starts = {}
    rows = []
    for r in g.itertuples():
        H, A = r.home_team, r.away_team
        done = r.result == r.result
        qb = {}
        for tm, pid in ((H, r.home_qb_id), (A, r.away_qb_id)):
            if not (isinstance(pid, str) and pid) and not done and exp is not None:
                pid = (over or {}).get((str(r.gameday).replace("-", ""), tm)) or exp.get(tm)
            qb[tm] = pid if isinstance(pid, str) and pid else None
        sp = r.spread_line
        if sp == sp:
            base = dict(season=int(r.season), week=int(r.week), game_id=r.game_id, gameday=r.gameday, gametime=r.gametime if isinstance(r.gametime, str) else None,
                        gw=r.gw, result=r.result, close=sp, done=done)
            side = lambda tm: dict(base, team=tm, opp=A if tm == H else H, line=-sp if tm == H else sp, price=r.home_spread_odds if tm == H else r.away_spread_odds,
                                   margin=(r.result if tm == H else -r.result) if done else np.nan)
            dog = A if sp > 0 else (H if sp < 0 else None)
            if dog:
                rows.append(dict(side(dog), rule="dog"))
                if r.div_game == 1:
                    rows.append(dict(side(dog), rule="div_dog"))
                if abs(sp) >= 7:
                    rows.append(dict(side(dog), rule="big_dog"))
            chg = []
            for tm in (H, A):
                prev = [p for p in starts.get(tm, [])[-4:]]
                if qb[tm] and prev and sum(p == qb[tm] for p in prev) / len(prev) < .5:
                    chg.append(tm)
            if len(chg) == 1:
                rows.append(dict(side(chg[0]), rule="qb_change"))
        if done:
            for tm in (H, A):
                if qb[tm]:
                    starts.setdefault(tm, []).append(qb[tm])
    return pd.DataFrame(rows)


def settle(p):
    """Grade picks that have a margin: win / loss / push against `line`, profit at `price` (-110 when the price is missing)."""
    p = p[p.margin.notna()].copy()
    cover = p.margin + p.line
    p["won"] = np.where(cover > 0, 1.0, np.where(cover < 0, 0.0, np.nan))
    p["profit"] = [pay(a) if w == 1 else (-1.0 if w == 0 else 0.0) for a, w in zip(p.price, p.won)]
    return p


def summ(p):
    q = p[p.won.notna()]
    if not len(q):
        return dict(n=0)
    o = pct(int(q.won.sum()), len(q))
    roi = float(q.profit.mean())
    se = float(q.profit.std(ddof=0)) / len(q) ** .5 if len(q) > 1 else 0.0
    o.update(units=float(q.profit.sum()), roi=roi, roi_lo=roi - 1.28 * se, roi_hi=roi + 1.28 * se)
    if "clv" in q:
        c = q[q.clv.notna()]
        o["clv"] = dict(n=int(len(c)), beat=float((c.clv > 0).mean()) if len(c) else None, avg=float(c.clv.mean()) if len(c) else None)
    return o


def log_live(g, P):
    """Append this week's rule picks at the line on offer now. A (game, rule) pair already in the log is left alone."""
    now = datetime.now()
    soon = (now + timedelta(days=7)).strftime("%Y-%m-%d")
    kick = pd.to_datetime(P.gameday + " " + P.gametime.fillna("13:00"))          # Eastern, like this machine's clock
    up = P[(~P.done) & (P.season == SEASON) & (kick > now) & (P.gameday <= soon)]   # only games that have not kicked off
    old = pd.read_csv(LOG) if os.path.exists(LOG) else pd.DataFrame(columns=COLS)
    have = set(zip(old.game_id, old.rule))
    new = up[[(a, b) not in have for a, b in zip(up.game_id, up.rule)]].assign(logged=now.strftime("%Y-%m-%d %H:%M"))
    if len(new):
        old = pd.concat([old, new[COLS]], ignore_index=True)
        os.makedirs(os.path.dirname(LOG), exist_ok=True)
        old.to_csv(LOG, index=False)
    return old, len(new)


def build():
    g = games()
    exp, over = expected_qbs()
    P = rule_picks(g, exp, over)
    S = settle(P)
    names = dict(RULES)
    hist = []
    for k, lab in RULES:
        x = S[S.rule == k]
        hist.append(dict(rule=k, label=lab, long=summ(x[x.season.between(2006, 2020)]), five=summ(x[x.season.between(2021, 2025)]), now=summ(x[x.season == SEASON])))
    # live: logged before kickoff, graded at the logged line and price; a QB-change pick is void if the usual QB started after all
    log, added = log_live(g, P)
    live = []
    if len(log):
        gi = g.set_index("game_id")
        L = log[log.game_id.isin(gi.index)].copy()
        res = gi.result.reindex(L.game_id).values
        home = (gi.home_team.reindex(L.game_id).values == L.team.values)
        L["margin"] = np.where(home, res, -res)
        L["clv"] = L.line - np.where(home, -gi.spread_line.reindex(L.game_id).values, gi.spread_line.reindex(L.game_id).values)   # points better than the close
        still = set(zip(P[P.rule == "qb_change"].game_id, P[P.rule == "qb_change"].team))
        L = L[~((L.rule == "qb_change") & L.margin.notna() & ~pd.Series([(a, b) in still for a, b in zip(L.game_id, L.team)], index=L.index))]
        LS = settle(L)
        live = [dict(rule=k, label=lab, **summ(LS[LS.rule == k]), open=int(((L.rule == k) & L.margin.isna()).sum())) for k, lab in RULES]
    up = P[(~P.done) & (P.season == SEASON)]
    wk = int(up.week.min()) if len(up) else None
    week = []
    if wk is not None:
        logged = {(a, b): (c, d) for a, b, c, d in zip(log.game_id, log.rule, log.line, log.price)} if len(log) else {}
        now = datetime.now()
        for (gid, tm), x in up[(up.week == wk) & (up.rule != "dog")].groupby(["game_id", "team"], sort=False):
            f = x.iloc[0]
            if (gid, f.rule) not in logged and pd.Timestamp(f"{f.gameday} {f.gametime or '13:00'}") <= now:
                continue  # already kicked off without being logged: not a pick
            ln, pr = logged.get((gid, f.rule), (f.line, f.price))
            week.append(dict(game_id=gid, team=tm, opp=f.opp, line=float(ln), price=None if pr != pr else float(pr), rules=[names[r] for r in x.rule], gameday=f.gameday))
    # confident winners: the favorite's straight-up record by size of spread
    f = g[g.result.notna() & g.spread_line.notna() & (g.result != 0) & (g.spread_line != 0)]
    fw = ((f.spread_line > 0) == (f.result > 0)).astype(int)
    band = lambda lo, hi, a, b: pct(int(fw[(f.spread_line.abs() >= lo) & (f.spread_line.abs() < hi) & f.season.between(a, b)].sum()),
                                    int(((f.spread_line.abs() >= lo) & (f.spread_line.abs() < hi) & f.season.between(a, b)).sum()))
    conf = dict(bands=[dict(lo=lo, hi=hi, label=lab, five=band(lo, hi, 2021, 2025), long=band(lo, hi, 2006, 2025)) for lo, hi, lab in BANDS],
                all3=dict(five=band(3, 99, 2021, 2025), long=band(3, 99, 2006, 2025), now=band(3, 99, SEASON, SEASON)), every=dict(five=band(0, 99, 2021, 2025), now=band(0, 99, SEASON, SEASON)))
    conf["span"] = "2021 to 2025"
    for b in conf["bands"]:
        b["hist"] = b["five"]
    conf["all3"]["hist"] = conf["all3"]["five"]
    return dict(hist=hist, live=live, week=dict(n=wk, picks=week), added=added, conf=conf, need=110 / 210,
                conf_nba=other_conf("nba/data/games.csv", "spread_home", "hs", "as", [(3, 6, "3 to 5.5"), (6, 9, "6 to 8.5"), (9, 12, "9 to 11.5"), (12, 99, "12 or more")], "since October 2025"),
                conf_cfb=other_conf("cfb/data/games.csv", "spread_line", "hs", "aw", [(3, 7, "3 to 6.5"), (7, 14, "7 to 13.5"), (14, 21, "14 to 20.5"), (21, 99, "21 or more")], "2025 and 2026"))


def other_conf(path, sp, hs, as_, bands, span):
    """Favorite's straight-up record by size of spread for another sport, same shape as the NFL table (positive spread = home favored)."""
    try:
        g = pd.read_csv(os.path.join(ROOT, path), low_memory=False)
        g = g[g[hs].notna() & g[sp].notna() & (g[sp] != 0) & (g[hs] != g[as_])]
        fw = ((g[sp] > 0) == (g[hs] > g[as_])).astype(int)
        ab = g[sp].abs()
        rec = lambda m: pct(int(fw[m].sum()), int(m.sum()))
        cur = g.season == g.season.max()
        return dict(span=span, bands=[dict(lo=lo, hi=hi, label=lab, hist=rec((ab >= lo) & (ab < hi))) for lo, hi, lab in bands],
                    all3=dict(hist=rec(ab >= 3), now=rec((ab >= 3) & cur)), every=dict(hist=rec(ab > 0)))
    except Exception as e:  # noqa: BLE001
        print("confident-winner table skipped for", path, repr(e))
        return None


if __name__ == "__main__":
    r = build()
    f = lambda t: "      –" if not t.get("n") else f"{t['pct'] * 100:5.1f}% of {t['n']:<5}" + (f" {t['units']:+6.1f}u" if "units" in t else "")
    print(f"{'rule':<52}{'2006-20':>24}{'2021-25':>24}{'2026 at the close':>24}")
    for h in r["hist"]:
        print(f"{h['label']:<52}{f(h['long']):>24}{f(h['five']):>24}{f(h['now']):>24}")
    print("\nlive log:", [(x["label"][:20], x.get("n", 0), x["open"]) for x in r["live"]], "| new picks logged:", r["added"])
    print("this week:", r["week"]["n"], [(p["team"], p["line"], p["price"], p["rules"]) for p in r["week"]["picks"]])
    c = r["conf"]
    print("\nfavorites of 3+:", f(c["all3"]["five"]), "(2021-25)", f(c["all3"]["long"]), "(2006-25)", f(c["all3"]["now"]), "(2026)")
    for b in c["bands"]:
        print(f"  {b['label']:<12} {f(b['five'])}  |  {f(b['long'])}")
