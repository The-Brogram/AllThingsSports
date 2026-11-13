"""Turn the downloaded ESPN files into three tables under data/:
  games.csv    one row per game: date, season, home/away, final score, DraftKings spread (home), total, moneylines
  players.csv  one row per player per game: minutes, points, rebounds, assists, threes, ... (DNPs kept with min=0)
  props.csv    one row per DraftKings prop price: market, line, price, open line/price, side (over/under or milestone)
Run: ../venv/bin/python parse_nba.py
"""
import glob
import json
import math
import os
import re
import numpy as np
import pandas as pd

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
# DraftKings prop markets kept, with the stat they settle on
MK = {"Total Points": "pts", "Total Rebounds": "reb", "Total Assists": "ast", "Total 3-Point Field Goals": "fg3", "Total Points and Rebounds": "pr",
      "Total Points and Assists": "pa", "Total Assists and Rebounds": "ra", "Total Points, Rebounds, and Assists": "pra", "Total Steals": "stl", "Total Blocks": "blk",
      "Points Milestones": "pts", "Rebounds Milestones": "reb", "Assists Milestones": "ast", "3-Point Field Goals Milestones": "fg3", "Points + Rebounds Milestones": "pr",
      "Points + Assists Milestones": "pa", "Rebounds + Assists Milestones": "ra", "Points + Assists + Rebounds Milestones": "pra", "Steals Milestones": "stl", "Blocks Milestones": "blk",
      "To Record a Double Double": "dd"}


def _pois_sf(k, mu):
    """P(X > k) for Poisson(mu), k a whole number."""
    t, c = math.exp(-mu), 0.0
    for n in range(int(k) + 1):
        c += t
        t *= mu / (n + 1)
    return max(0.0, 1.0 - c)


def _expected(ml, fair, line):
    """What a full-game "over `line`" should be worth, given the same player's main line `ml` at no-vig over probability `fair`."""
    lo, hi = .05, 150.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if _pois_sf(math.floor(ml), mid) < fair:
            lo = mid
        else:
            hi = mid
    return _pois_sf(math.floor(line), (lo + hi) / 2)


def drop_wrong_market(r):
    """ESPN files some other market (first-quarter rebounds, by the look of it) under the full-game "Rebounds Milestones" name: a starter
    whose rebound line is 6.5 shows "2+ rebounds" at -137. The main over/under line tells us what any full-game "N+" should be worth
    (Poisson through the main line), so an N+ priced 15+ points of probability away from that cannot be the same market and is dropped.
    A rebounds milestone with no main line to check against is dropped too."""
    if not len(r):
        return r
    imp = lambda a: np.where(np.asarray(a, float) < 0, -np.asarray(a, float) / (-np.asarray(a, float) + 100), 100 / (np.asarray(a, float) + 100))
    ln = r[r.kind == "line"]
    o = ln[ln.side == "over"][["game_id", "pid", "market", "line", "price"]].rename(columns={"line": "ml", "price": "po"})
    u = ln[ln.side == "under"][["game_id", "pid", "market", "line", "price"]].rename(columns={"line": "ml", "price": "pu"})
    m = o.merge(u, on=["game_id", "pid", "market", "ml"]).drop_duplicates(["game_id", "pid", "market"])
    m["fair"] = imp(m.po) / (imp(m.po) + imp(m.pu))
    x = r.merge(m[["game_id", "pid", "market", "ml", "fair"]], on=["game_id", "pid", "market"], how="left")
    ms = (x.kind == "milestone").values
    chk = ms & x.ml.notna().values & (x.line.values != x.ml.values) & x.price.notna().values
    exp = np.full(len(x), np.nan)
    cache = {}
    for i in np.nonzero(chk)[0]:
        k = (x.ml.values[i], round(float(x.fair.values[i]), 2), x.line.values[i])
        if k not in cache:
            cache[k] = _expected(*k)
        exp[i] = cache[k]
    bad = chk & (np.abs(imp(x.price) - exp) > .15) & ~((x.line.values > x.ml.values) & (imp(x.price) > exp))  # longshot rungs carry extra vig: not flagged
    bad |= ms & (x.market == "reb").values & x.ml.isna().values
    print(f"props: dropped {int(bad.sum())} milestone prices filed under the wrong market ({int((bad & (x.market == 'reb').values).sum())} rebounds)")
    return r[~bad].reset_index(drop=True)


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def made(x):
    m = re.match(r"(\d+)-(\d+)", str(x or ""))
    return (int(m.group(1)), int(m.group(2))) if m else (np.nan, np.nan)


def spread_home(details, home, away):
    """'NY -15.5' -> home favored by 15.5 (positive = home favored); 'EVEN' -> 0."""
    if not details:
        return np.nan
    if details.strip().upper() in ("EVEN", "PK", "PICK"):
        return 0.0
    m = re.match(r"([A-Z]+) (-?[\d.]+)", details)
    if not m:
        return np.nan
    v = abs(float(m.group(2)))
    return v if m.group(1) == home else (-v if m.group(1) == away else np.nan)


def main():
    G, P, R = [], [], []
    for f in glob.glob(f"{D}/games/*.json"):
        g = json.load(open(f))
        t = {x["ha"]: x for x in g["teams"]}
        if "home" not in t or "away" not in t:
            continue
        h, a = t["home"]["ab"], t["away"]["ab"]
        dk = next((p for p in g["pick"] if p.get("provider") == "DraftKings"), (g["pick"] or [{}])[0])
        G.append(dict(game_id=g["id"], date=g["date"], season=g["season"], stype=g["stype"], completed=g["completed"], neutral=g["neutral"], home=h, away=a,
                      hs=num(t["home"].get("score")) if g["completed"] else np.nan, as_=num(t["away"].get("score")) if g["completed"] else np.nan,
                      spread_home=spread_home(dk.get("details"), h, a), total=num(dk.get("ou")), ml_h=num(dk.get("ml_h")), ml_a=num(dk.get("ml_a"))))
        for b in g["box"]:
            s = b["stats"]
            fg, fg3, ft = made(s.get("FG")), made(s.get("3PT")), made(s.get("FT"))
            mn = num(s.get("MIN"))
            P.append(dict(game_id=g["id"], team=b["team"], opp=a if b["team"] == h else h, home=int(b["team"] == h), pid=b["id"], name=b["name"], pos=b["pos"], starter=int(b["starter"]),
                          dnp=int(b["dnp"] or not (mn > 0)), reason=b.get("reason"), min=mn if mn > 0 else 0.0, pts=num(s.get("PTS")), reb=num(s.get("REB")), ast=num(s.get("AST")),
                          oreb=num(s.get("OREB")), dreb=num(s.get("DREB")), stl=num(s.get("STL")), blk=num(s.get("BLK")), tov=num(s.get("TO")), pf=num(s.get("PF")), pm=num(s.get("+/-")),
                          fgm=fg[0], fga=fg[1], fg3=fg3[0], fg3a=fg3[1], ftm=ft[0], fta=ft[1]))
        pf = f"{D}/props/{g['id']}.json"
        if os.path.exists(pf):
            rows = json.load(open(pf))
            seen = {}
            for aid, typ, line, price, oline, oprice, k, *_ in rows:  # newer files also carry the market id and price timestamp
                if typ not in MK or not aid:
                    continue
                ms = "Milestones" in typ or typ.startswith("To Record")
                ln = num(str(line).rstrip("+")) if line is not None else np.nan
                if ms:
                    side, ln = "over", (ln - .5 if typ != "To Record a Double Double" else .5)  # "13+" settles like over 12.5
                else:
                    key = (aid, typ, line)
                    seen[key] = seen.get(key, 0) + 1
                    side = "over" if seen[key] == 1 else "under"  # ESPN lists the over first; parse checks this against milestones below
                R.append(dict(game_id=g["id"], pid=aid, market=MK[typ], kind="milestone" if ms else "line", line=ln, side=side, price=num(price),
                              open_line=num(str(oline).rstrip("+")) - (.5 if ms and oline else 0) if oline else np.nan, open_price=num(oprice)))
    g = pd.DataFrame(G).rename(columns={"as_": "as"}).sort_values("date")
    g["date_et"] = pd.to_datetime(g.date, utc=True).dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
    p = pd.DataFrame(P)
    r = drop_wrong_market(pd.DataFrame(R))
    g.to_csv(f"{D}/games.csv", index=False)
    p.to_csv(f"{D}/players.csv.gz", index=False)
    r.to_csv(f"{D}/props.csv.gz", index=False)
    # side check: an over line at L should price close to the "L+0.5" milestone; the under should not
    ln = r[r.kind == "line"]
    ms = r[r.kind == "milestone"][["game_id", "pid", "market", "line", "price"]].rename(columns={"price": "ms_price"})
    j = ln.merge(ms, on=["game_id", "pid", "market", "line"])
    if len(j):
        d = j.assign(gap=(j.price - j.ms_price).abs()).groupby("side").gap.median()
        print("side check (median |price - milestone price|; over should be much smaller):", d.round(1).to_dict(), "pairs:", len(j))
    print(f"{len(g)} games ({int(g.completed.sum())} final), {len(p)} player rows, {len(r)} prop prices, {r.game_id.nunique() if len(r) else 0} games with props")


if __name__ == "__main__":
    main()
