"""Does the model beat DraftKings' prices? The proof layer: profit at real prices, not hit rate.

  ../venv/bin/python edge_nba.py backtest   out-of-sample test on games the price blend never saw -> output/edge.json
  ../venv/bin/python edge_nba.py log [--date YYYY-MM-DD]
                                            record tonight's priced sides with the price on offer right now -> output/picks/<date>.csv.gz
                                            (the first price seen for a side is kept, so a later rerun cannot improve the record)
  ../venv/bin/python edge_nba.py grade      grade every logged night that is final, at the logged prices, and compare with the closing prices
                                            -> output/edge_live.json

Definitions
  price        American odds. pay = profit per 1 unit staked when the bet wins.
  break-even   the hit rate a price needs: 1 / (1 + pay).
  claimed edge our final probability for the side minus its break-even. EV = p * pay - (1 - p), per unit staked.
  ROI          units won per unit staked, flat 1-unit stakes.
  CLV          closing line value: did the market move toward our side after we logged it? Positive = we had the better number.
Ranges are 80% intervals from resampling whole games (bets on one game move together, so games are the independent unit).
"""
import argparse
import glob
import json
import os
import pickle
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import nba_model as M

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = f"{HERE}/output"
PICKS = f"{OUT}/picks"
SPLIT = "2026-10-01"          # same split as calibrate_nba.py: the blend is fit before it, tested after it
MKTS = ["pts", "reb", "ast", "fg3", "pr", "pa", "ra", "pra", "stl", "blk"]
EV_BUCKETS = [(-9, -.04, "below -4%"), (-.04, -.02, "-4% to -2%"), (-.02, 0, "-2% to 0%"), (0, .02, "0% to +2%"), (.02, .04, "+2% to +4%"), (.04, .07, "+4% to +7%"), (.07, 9, "+7% and up")]
EDGE_MIN = .03                # claimed EV per unit that makes a side an "edge pick"
EDGE_START = "2026-11-13"     # the edge-pick rule and the cleaned price blend start here; earlier nights are not back-filled under a rule nobody used
MAX_FAV = -900                # prices longer than this are ignored everywhere (no return)
KEY = ["game_id", "pid", "market", "kind", "line", "side"]


def american_p(a):
    a = np.asarray(a, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(a < 0, -a / (-a + 100), 100 / (a + 100))


def pay(a):
    a = np.asarray(a, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(a < 0, 100 / -a, a / 100)


def lg(p):
    p = np.clip(np.asarray(p, float), 1e-3, 1 - 1e-3)
    return np.log(p / (1 - p))


def blend(p_sim, dk, milestone, w):
    z = w["a"] + w["b"] * lg(p_sim) + w["c"] * lg(dk) + np.where(milestone, w["cm"] * lg(dk), 0)
    return 1 / (1 + np.exp(-z))


def sides(x):
    """One row per bettable side. x: game_id, pid, name, market, kind, line, p_over (final), price_over, price_under, dk_over.
    Main lines give an over and an under; milestones ("N+") only an over."""
    keep = [c for c in ("date", "game_id", "pid", "name", "team", "market", "kind", "line", "proj_min", "actual", "p_sim") if c in x]
    o = x[x.price_over.notna()].assign(side="over", price=lambda d: d.price_over, p=lambda d: d.p_over,
                                       fair=lambda d: np.where(d.kind == "line", d.dk_over, np.nan))
    u = x[(x.kind == "line") & x.price_under.notna()].assign(side="under", price=lambda d: d.price_under, p=lambda d: 1 - d.p_over, fair=lambda d: 1 - d.dk_over)
    s = pd.concat([o, u], ignore_index=True)[keep + ["side", "price", "p", "fair"]]
    s = s[(s.price >= MAX_FAV) & (s.price != 0)].copy()
    s["be"] = american_p(s.price)
    s["pay"] = pay(s.price)
    s["ev"] = s.p * s["pay"] - (1 - s.p)
    return s


def settle(s):
    """Add won / profit for sides that have `actual`. Pushes (actual == line) are dropped, as books refund them."""
    s = s[s.actual.notna() & (s.actual != s.line)].copy()
    over = s.actual > s.line
    s["won"] = np.where(s.side == "over", over, ~over).astype(int)
    s["profit"] = np.where(s.won == 1, s["pay"], -1.0)
    return s


def best_side(s):
    """One side per line: the one with the higher claimed EV (for a milestone that is its only side)."""
    return s.sort_values("ev", ascending=False).drop_duplicates(["game_id", "pid", "market", "kind", "line"])


def stat(s, B=400, seed=11):
    """Flat-stake results with 80% ranges from resampling games."""
    if not len(s):
        return dict(n=0)
    g = s.groupby("game_id").agg(n=("profit", "size"), profit=("profit", "sum"), won=("won", "sum"))
    out = dict(n=int(len(s)), games=int(len(g)), players=int(s.drop_duplicates(["game_id", "pid"]).shape[0]), hit=float(s.won.mean()), need=float(s.be.mean()),
               said=float(s.p.mean()), units=float(s.profit.sum()), roi=float(s.profit.mean()), ev=float(s.ev.mean()))
    if len(g) >= 8:
        rng = np.random.default_rng(seed)
        ix = rng.integers(0, len(g), (B, len(g)))
        n_, pr_ = g.n.values[ix].sum(1), g.profit.values[ix].sum(1)
        r = pr_ / np.maximum(n_, 1)
        out.update(roi_lo=float(np.percentile(r, 10)), roi_hi=float(np.percentile(r, 90)), p_pos=float((r > 0).mean()))
    return out


def ll(p, y):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def bucket(ev):
    for lo, hi, lab in EV_BUCKETS:
        if lo <= ev < hi:
            return lab
    return EV_BUCKETS[-1][2]


def role(m):
    return "Starter (28+ min)" if m >= 28 else ("Rotation (18-28)" if m >= 18 else "Bench (under 18)")


def price_band(a):
    return "Longer than -200" if a < -200 else ("-200 to -111" if a <= -111 else ("-110 to +110" if a <= 110 else "+111 and up"))


# ---------- out-of-sample backtest at real prices ----------
def priced_backtest():
    cal = json.load(open(f"{OUT}/calibration.json"))
    b = pd.read_csv(f"{OUT}/bt_rows.csv.gz", dtype={"pid": str, "game_id": str})
    b = b[(b.date >= SPLIT) & b.dk_p.notna() & b.kind.isin(["line", "milestone"])].copy()
    pr = pd.read_csv(f"{M.D}/props.csv.gz", dtype={"pid": str, "game_id": str})
    pr = pr[pr.market.isin(MKTS)].drop_duplicates(KEY)
    o = pr[pr.side == "over"][["game_id", "pid", "market", "kind", "line", "price"]].rename(columns={"price": "price_over"})
    u = pr[(pr.side == "under") & (pr.kind == "line")][["game_id", "pid", "market", "kind", "line", "price"]].rename(columns={"price": "price_under"})
    b = b.merge(o, on=["game_id", "pid", "market", "kind", "line"], how="left").merge(u, on=["game_id", "pid", "market", "kind", "line"], how="left")
    b["p_sim"] = b.p_over
    b["dk_over"] = b.dk_p
    b["p_over"] = blend(b.p_sim.values, b.dk_p.values, (b.kind == "milestone").values, cal["blend"])
    return b, cal


def backtest():
    b, cal = priced_backtest()
    s = settle(sides(b))
    s["role"] = s.proj_min.map(role)
    s["pband"] = s.price.map(price_band)
    s["bk"] = s.ev.map(bucket)
    main, mile = s[s.kind == "line"], s[s.kind == "milestone"]
    mb = best_side(main)
    res = dict(built=datetime.now(timezone.utc).isoformat(timespec="seconds"), split=SPLIT, from_=str(b.date.min()), to=str(b.date.max()), games=int(b.game_id.nunique()),
               edge_min=EDGE_MIN, max_fav=MAX_FAV)
    # 1. is the final probability sharper than DraftKings' own no-vig number? main lines only, where both sides are priced
    L = b[(b.kind == "line") & b.price_under.notna() & (b.actual != b.line)]
    y = (L.actual > L.line).astype(int).values
    res["logloss"] = dict(n=int(len(L)), blend=ll(L.p_over, y), dk=ll(L.dk_over, y), sim=ll(L.p_sim, y))
    # 2. baselines: what the book's cut costs, and what following the model's side does
    res["all_sides"] = stat(main)                      # every over and every under: the pure cost of the vig
    res["model_side"] = stat(mb)                       # one side per line, the one the model prefers
    res["edge_picks"] = stat(mb[mb.ev >= EDGE_MIN])    # the betting rule
    res["mile_all"] = stat(mile)
    res["mile_edge"] = stat(mile[mile.ev >= EDGE_MIN])
    # 3. when the model claimed an edge of X, what happened?
    res["by_claim"] = dict(line=[dict(bucket=lab, **stat(mb[mb.bk == lab])) for _, _, lab in EV_BUCKETS],
                           milestone=[dict(bucket=lab, **stat(mile[mile.bk == lab])) for _, _, lab in EV_BUCKETS])
    # 4. where the edge might live: every segment of the model's side on main lines, plus milestones by price
    seg = []
    for name, col in (("Stat", "market"), ("Side", "side"), ("Role", "role"), ("Price", "pband")):
        for k, x in mb.groupby(col):
            seg.append(dict(group=name, key=str(k), **stat(x)))
    for k, x in mile.groupby("pband"):
        seg.append(dict(group="N+ lines by price", key=str(k), **stat(x)))
    res["segments"] = seg
    json.dump(res, open(f"{OUT}/edge.json", "w"), indent=1)
    f = lambda t: (f"n {t['n']:>6}  games {t.get('games', 0):>3}  hit {t.get('hit', 0):.3f}  need {t.get('need', 0):.3f}  ROI {t.get('roi', 0):+.3f}"
                   + (f"  [{t['roi_lo']:+.3f}, {t['roi_hi']:+.3f}]  P(profit) {t['p_pos']:.2f}" if "roi_lo" in t else "")) if t["n"] else "n 0"
    print(f"Out of sample {res['from_']}..{res['to']}, {res['games']} games. Blend fit before {SPLIT}.")
    print(f"log loss on {res['logloss']['n']} main lines: blend {res['logloss']['blend']:.4f}  DraftKings no-vig {res['logloss']['dk']:.4f}  raw sim {res['logloss']['sim']:.4f}")
    for k, lab in (("all_sides", "every over and under"), ("model_side", "model's side, every line"), ("edge_picks", f"model's side, EV >= {EDGE_MIN:.0%}"), ("mile_all", "every N+ line"), ("mile_edge", f"N+ lines, EV >= {EDGE_MIN:.0%}")):
        print(f"{lab:>28}: {f(res[k])}")
    for kind in ("line", "milestone"):
        print(f"claimed EV -> result ({kind}):")
        for t in res["by_claim"][kind]:
            print(f"  {t['bucket']:>12}: {f(t)}")
    print("segments (model's side on main lines):")
    for t in seg:
        print(f"  {t['group']:>18} {t['key']:<18}: {f(t)}")
    return res


# ---------- live: log picks at the price on offer, grade them later ----------
def log(date):
    f = f"{OUT}/predictions_{date}.csv"
    if not os.path.exists(f):
        print("no forecasts for", date)
        return
    p = pd.read_csv(f, dtype={"pid": str, "game_id": str})
    p = p[p.kind.isin(["line", "milestone"]) & p.market.isin(MKTS)]
    s = sides(p)
    try:
        seen = pickle.load(open(f"{OUT}/nba_{date}.pkl", "rb")).get("built")
    except (OSError, pickle.UnpicklingError):
        seen = None
    s["seen"] = seen or datetime.now(timezone.utc).isoformat(timespec="seconds")
    os.makedirs(PICKS, exist_ok=True)
    out = f"{PICKS}/{date}.csv.gz"
    cols = ["game_id", "pid", "name", "team", "market", "kind", "line", "side", "price", "p", "fair", "be", "pay", "ev", "proj_min", "seen"]
    s = s[cols].round({"p": 4, "fair": 4, "be": 4, "pay": 4, "ev": 4, "proj_min": 1})
    if os.path.exists(out):
        old = pd.read_csv(out, dtype={"pid": str, "game_id": str})
        new = s.merge(old[KEY], on=KEY, how="left", indicator=True)
        s = pd.concat([old, new[new._merge == "left_only"].drop(columns="_merge")], ignore_index=True)
    s.to_csv(out, index=False)
    print(f"pick log {date}: {len(s)} priced sides ({int((s.ev >= EDGE_MIN).sum())} with claimed EV >= {EDGE_MIN:.0%})")


def closing():
    """Closing prices: the last DraftKings numbers ESPN holds for each finished game."""
    pr = pd.read_csv(f"{M.D}/props.csv.gz", dtype={"pid": str, "game_id": str})
    pr = pr[pr.market.isin(MKTS)].drop_duplicates(KEY)
    same = pr[KEY + ["price"]].rename(columns={"price": "close_price"})
    ln = pr[pr.kind == "line"]
    o = ln[ln.side == "over"][["game_id", "pid", "market", "line", "price"]].rename(columns={"line": "close_line", "price": "co"})
    u = ln[ln.side == "under"][["game_id", "pid", "market", "line", "price"]].rename(columns={"line": "close_line", "price": "cu"})
    main = o.merge(u, on=["game_id", "pid", "market", "close_line"]).drop_duplicates(["game_id", "pid", "market"])
    po, pu = american_p(main.co), american_p(main.cu)
    main["close_fair_over"] = po / (po + pu)
    return same, main[["game_id", "pid", "market", "close_line", "close_fair_over"]]


def grade_night(date, d, same, main):
    s = pd.read_csv(f"{PICKS}/{date}.csv.gz", dtype={"pid": str, "game_id": str})
    g = d["games"].set_index("game_id")
    gids = [i for i in s.game_id.unique() if i in g.index]
    if not gids or not all(g.loc[i, "completed"] for i in gids):
        return None
    box = d["players"][d["players"].game_id.isin(gids)]
    box = box[box["min"] > 0]
    act = {}
    for r in box.itertuples():
        for mk in MKTS:
            act[(r.game_id, r.pid, mk)] = float(sum(getattr(r, x) for x in M.COMBO.get(mk, (mk,))))
    s["actual"] = [act.get((a, b, c), np.nan) for a, b, c in zip(s.game_id, s.pid, s.market)]  # no box score = did not play = void
    s = settle(s)
    s = s.merge(same, on=KEY, how="left").merge(main, on=["game_id", "pid", "market"], how="left")
    s = s[~((s.kind == "milestone") & (s.market == "reb") & s.close_price.isna())]  # rebounds N+ prices the parser now rejects as another market
    # closing line value
    cfair = np.where(s.side == "over", s.close_fair_over, 1 - s.close_fair_over)
    is_line = s.kind == "line"
    moved = is_line & s.close_line.notna() & (s.close_line != s.line)
    s["clv"] = np.where(is_line & ~moved, cfair - s.fair, np.where(~is_line, american_p(s.close_price) - s.be, np.nan))  # points of probability
    better_num = np.where(s.side == "over", s.close_line > s.line, s.close_line < s.line)
    s["beat"] = np.where(moved, better_num, np.where(s["clv"].notna(), s["clv"] > 1e-9, np.nan)).astype(float)
    s["tie"] = (~moved & s["clv"].notna() & (s["clv"].abs() <= 1e-9)).astype(float)
    s["date"] = date
    return s


def portfolios(s):
    main = best_side(s[s.kind == "line"])
    edge = best_side(s)[lambda x: (x.ev >= EDGE_MIN) & (x.date >= EDGE_START)]
    likely = s[(s.p >= .7) & (s.proj_min >= 10)].sort_values("p", ascending=False).groupby("date").head(10)
    return dict(model_side=main, edge_picks=edge, most_likely=likely)


def clv_stat(s):
    x = s[s.beat.notna()]
    if not len(x):
        return dict(n=0)
    return dict(n=int(len(x)), beat=float((x.beat == 1).mean()), tie=float(x.tie.mean()), avg=float(s["clv"].mean(skipna=True)) if s["clv"].notna().any() else None)


def grade():
    d = M.load()
    same, main = closing()
    nights = []
    today = f"{datetime.now(timezone.utc) - timedelta(hours=5):%Y-%m-%d}"
    for f in sorted(glob.glob(f"{PICKS}/*.csv.gz")):
        date = os.path.basename(f)[:10]
        if date >= today:
            continue
        s = grade_night(date, d, same, main)
        if s is not None and len(s):
            nights.append(s)
    if not nights:
        print("no logged night is final yet")
        return
    a = pd.concat(nights, ignore_index=True)
    P = portfolios(a)
    res = dict(built=datetime.now(timezone.utc).isoformat(timespec="seconds"), nights=sorted(a.date.unique().tolist()), edge_min=EDGE_MIN,
               edge_start=EDGE_START, total={k: dict(**stat(v), clv=clv_stat(v)) for k, v in P.items()},
               by_night=[dict(date=dt, **{k: dict(**stat(v[v.date == dt]), clv=clv_stat(v[v.date == dt])) for k, v in P.items()}) for dt in sorted(a.date.unique())],
               picks=[dict(date=r.date, name=r.name, market=r.market, kind=r.kind, line=r.line, side=r.side, price=r.price, p=round(r.p, 4), ev=round(r.ev, 4), won=int(r.won),
                           close=None if pd.isna(r.close_price) else float(r.close_price)) for r in P["edge_picks"].sort_values(["date", "ev"], ascending=[False, False]).head(60).itertuples()])
    json.dump(res, open(f"{OUT}/edge_live.json", "w"), indent=1, default=float)
    for k, lab in (("model_side", "model's side, every main line"), ("edge_picks", f"edge picks (EV >= {EDGE_MIN:.0%})"), ("most_likely", "ten most likely each night")):
        t, c = res["total"][k], res["total"][k]["clv"]
        if not t["n"]:
            print(f"{lab}: none")
            continue
        print(f"{lab}: {t['n']} bets, {t['games']} games, hit {t['hit']:.3f} (need {t['need']:.3f}), {t['units']:+.1f} units, ROI {t['roi']:+.3f}"
              + (f" [{t['roi_lo']:+.3f}, {t['roi_hi']:+.3f}]" if "roi_lo" in t else "")
              + (f"; beat the close {c['beat']:.0%} of {c['n']}" + (f", avg move {c['avg'] * 100:+.2f} pts" if c.get('avg') is not None else "") if c["n"] else ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["backtest", "log", "grade"])
    ap.add_argument("--date", default=f"{datetime.now(timezone.utc) - timedelta(hours=5):%Y-%m-%d}")
    a = ap.parse_args()
    if a.cmd == "backtest":
        backtest()
    elif a.cmd == "log":
        log(a.date)
    else:
        grade()
