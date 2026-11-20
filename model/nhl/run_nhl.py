"""Today's NHL slate: simulate each game from the DraftKings price, price the player props, log every priced side.

Run: ../venv/bin/python run_nhl.py [--mode morning|pregame] [--date YYYY-MM-DD] [--n 20000]
- morning: every game today that has a line, with ESPN's Expected goalies (goalie saves props held).
- pregame: only games starting in the next 30-110 minutes, after starters are confirmed. A game re-run here replaces its morning
  numbers, and goalie saves props unlock for a Confirmed starter. GitHub runs this every 30 minutes through the evening.
A game that has started is never re-run: its last pre-game numbers stay.

Writes output/slate_<date>.json (what the hub shows) and adds to output/picks_<date>.json (the price-on-offer log for grading).
"""
import argparse
import json
import os
from datetime import datetime, timedelta
import numpy as np
import nhl_data as ND
import nhl_model as M

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
MKT = {"Anytime Goalscorer": "goal", "Total Assists": "ast", "Total Points": "pts", "Total Shots on Goal": "sog", "Total Saves": "saves"}


def side_info(c):
    t = c["team"]
    g = None
    for p in c.get("probables") or []:
        if p.get("name") == "probableStartingGoalie":
            g = dict(n=p["athlete"]["displayName"], st=p["status"]["name"], id=p["athlete"].get("id"))
    return dict(ab=ND.ab(t["abbreviation"]), id=t["id"], g=g)


def odds(c):
    o = (c.get("odds") or [None])[0]
    if not o or not o.get("moneyline"):
        return None
    m, ps, tt = o["moneyline"], o.get("pointSpread") or {}, o.get("total") or {}
    cur = lambda x, k="close": (x or {}).get(k) or {}
    r = dict(ml_a=cur(m.get("away")).get("odds"), ml_h=cur(m.get("home")).get("odds"), total=o.get("overUnder"),
             over=cur(tt.get("over")).get("odds"), under=cur(tt.get("under")).get("odds"),
             pl_a=cur(ps.get("away")).get("line"), pl_a_p=cur(ps.get("away")).get("odds"), pl_h=cur(ps.get("home")).get("line"),
             pl_h_p=cur(ps.get("home")).get("odds"))
    return r if r["ml_a"] and r["ml_h"] else None


def num(x):
    try:
        return float(str(x).replace("+", ""))
    except (TypeError, ValueError):
        return None


def goalie_status(espn_g, dfo):
    """Starter, its status and whether ESPN and DailyFaceoff agree."""
    if not espn_g and not dfo:
        return None
    g = dict(espn_g or dict(n=dfo["n"], st="Unconfirmed"))
    g["src"] = "ESPN" if espn_g else "DailyFaceoff"
    if dfo and dfo.get("n"):
        g["dfo"] = dict(n=dfo["n"], st=dfo["st"], rep=dfo.get("rep"), t=dfo.get("t"))
        g["agree"] = ND.name_key(dfo["n"]) == ND.name_key(g["n"])
        if dfo["st"] == "Confirmed" and g["agree"]:
            g["st"] = "Confirmed"
        elif dfo["st"] == "Confirmed" and not g["agree"] and g.get("st") != "Confirmed":
            g = dict(n=dfo["n"], st="Confirmed", src="DailyFaceoff", dfo=g["dfo"], agree=False, espn=espn_g and espn_g["n"])
    return g


def stat_index(st):
    by = {}   # (name, team) -> list: two players can share a name on one team (VAN has two Elias Petterssons)
    for r in st["sk"]:
        for t in str(r.get("teamAbbrevs") or "").split(","):
            by.setdefault((ND.name_key(r["skaterFullName"]), ND.ab(t.strip())), []).append(r)
    gl = {}
    for r in st["gl"]:
        for t in str(r.get("teamAbbrevs") or "").split(","):
            gl[(ND.name_key(r["goalieFullName"]), ND.ab(t.strip()))] = r
    tm = {ND.name_key(r["teamFullName"]): r for r in st["tm"]}
    return by, gl, tm


def sim_game(e, st_idx, dfo, n, seed):
    c = e["competitions"][0]
    H = next(x for x in c["competitors"] if x["homeAway"] == "home")
    A = next(x for x in c["competitors"] if x["homeAway"] == "away")
    h, a = side_info(H), side_info(A)
    o = odds(c)
    gh, ga = goalie_status(h["g"], dfo.get(h["ab"])), goalie_status(a["g"], dfo.get(a["ab"]))
    rec = dict(id=e["id"], home=h["ab"], away=a["ab"], t=e["date"], goalies=dict(h=gh, a=ga), odds=o)
    if not o or o["total"] is None:
        return rec
    playoff = e.get("season", {}).get("type") == 3
    ph = M.novig(o["ml_h"], o["ml_a"])
    po = M.novig(o["over"], o["under"]) if o["over"] and o["under"] else 0.5
    line = float(o["total"])
    d = M.Draws(n, seed)
    mh, ma = M.fit(ph, line, po, d, playoff)
    s = M.sim(mh, ma, d, playoff)
    q = M.summary(s, total=line)
    m = dict(ph=round(q["ph"], 4), pa=round(q["pa"], 4), rh=round(q["rh"], 4), ra=round(q["ra"], 4), pot=round(q["pot"], 4),
             pso=round(q["pso"], 4), xh=round(q["xh"], 2), xa=round(q["xa"], 2), mu=[round(mh, 3), round(ma, 3)])
    m["ml"] = dict(edge_h=round(q["ph"] - ph, 4), edge_a=round(q["pa"] - (1 - ph), 4), nv_h=round(ph, 4))
    # puck line: DraftKings posts the favourite at -1.5
    if o["pl_h"] is not None and o["pl_h_p"] and o["pl_a_p"]:
        hl = num(o["pl_h"])
        fav_home = hl is not None and hl < 0
        p_fav = q["home_m15"] if fav_home else q["away_m15"]
        nv_fav = M.novig(o["pl_h_p"], o["pl_a_p"]) if fav_home else M.novig(o["pl_a_p"], o["pl_h_p"])
        # Backtest (2024-25 on, 2,963 games): the simulation's puck line was slightly worse than the market's own puck-line price (mixed books: ESPN BET and DraftKings; DK-only +0.0021)
        # (log loss +0.0015, range +0.0005 to +0.0024), so the likelihood shown is the market's; the simulation's is kept beside it.
        p_sim, p_fav = p_fav, nv_fav
        m["pl"] = dict(fav=h["ab"] if fav_home else a["ab"], line=-1.5, p_fav=round(p_fav, 4), p_dog=round(1 - p_fav, 4), p_sim=round(p_sim, 4),
                       price_fav=o["pl_h_p"] if fav_home else o["pl_a_p"], price_dog=o["pl_a_p"] if fav_home else o["pl_h_p"],
                       nv_fav=round(nv_fav, 4), edge_fav=round(p_fav - nv_fav, 4))
    m["tot"] = dict(line=line, po=round(q["po"], 4), pu=round(1 - q["po"], 4), price_o=o["over"], price_u=o["under"], nv_o=round(po, 4))
    rec["m"] = m
    rec["props"] = price_props(e, s, h, a, gh, ga, q, st_idx, seed)
    return rec


def calib():
    """Blend weights and the anytime-goal margin factor, fit by backtest_props.py on the tuning period (output/backtest_props.json)."""
    try:
        b = json.load(open(os.path.join(OUT, "backtest_props.json")))["markets"]
    except (OSError, ValueError, KeyError):
        return {}
    return {k: dict(w=v.get("blend_weight", 0.0), c=v.get("margin_factor") or 1.0) for k, v in b.items() if "blend_weight" in v}


def price_props(e, s, h, a, gh, ga, q, H, seed):
    """Props v2 (nhl_props.py): pre-game player history by ESPN athlete id, ice time, position-specific rates, the same simulated games."""
    import nhl_props as NP
    rng = np.random.default_rng(seed + 7)
    raw = ND.props(e["id"])
    if not raw:
        return []
    day = f"{ND.to_et(e['date'][:16]):%Y-%m-%d}"
    names = {}
    for side in (h, a):
        for aid, (nm, pos) in ND.roster(side["id"]).items():
            names[aid] = (nm, pos, side["ab"])
    team = {x["ab"]: H.team(x["ab"], day) for x in (h, a)}
    cal = calib()
    pairs = {}
    for r in raw:
        mk = MKT.get(r["market"])
        if not mk or not r["aid"] or r["aid"] not in names or not r["price"]:
            continue
        pairs.setdefault((r["aid"], mk, r["line"]), []).append(r)
    # a two-way prop needs both prices, and they must add up like a real market (100-112% implied)
    for key in list(pairs):
        it = pairs[key]
        if key[1] == "goal":
            pairs[key] = it[:1]
        elif len(it) < 2 or not (1.0 <= M.implied(it[0]["price"]) + M.implied(it[1]["price"]) <= 1.12):
            del pairs[key]
    cache, rows = {}, []
    for (aid, mk, line), items in pairs.items():
        nm, pos, tab = names[aid]
        side = "h" if tab == h["ab"] else "a"
        T, O = team.get(tab), team.get(a["ab"] if side == "h" else h["ab"])
        if not T or not O:
            continue
        row = dict(aid=aid, n=nm, tm=tab, pos="G" if mk == "saves" else pos, mk=mk, line=line, side="yes" if mk == "goal" else "over",
                   price=items[0]["price"], under=items[1]["price"] if mk != "goal" else None)
        if mk == "saves":
            g = gh if side == "h" else ga
            if not g or ND.name_key(g["n"]) != ND.name_key(nm):
                continue  # not the projected starter: DraftKings voids it if they do not start
            if g.get("st") != "Confirmed":
                row["hold"] = True
                rows.append(row)
                continue
            v = cache.setdefault(aid, dict(saves=NP.saves_draws(rng, s, side, O["sf"], T["sa"])))["saves"]
        else:
            if aid not in cache:
                f = H.skater(aid, day)
                if not f:
                    continue
                cache[aid] = NP.skater_draws(rng, s, side, f, T["gf"])
            v = cache[aid][mk]
        L = 0.5 if mk == "goal" else float(line)
        pm = float((v > L).mean())
        c = cal.get(mk, dict(w=0.0, c=1.0))
        if mk == "goal":   # one-way price: the market chance is the implied chance less DraftKings' fitted margin
            nv = min(M.implied(row["price"]) / c["c"], 0.99)
            row["imp"] = round(M.implied(row["price"]), 4)
        else:
            nv = M.novig(row["price"], row["under"])
        z = c["w"] * M.lg(pm) + (1 - c["w"]) * M.lg(nv)
        p = float(1 / (1 + np.exp(-z)))
        row.update(p=round(p, 4), p_model=round(pm, 4), nv=round(nv, 4), edge=round(p - nv, 4), w=c["w"])
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="morning", choices=["morning", "pregame"])
    ap.add_argument("--date")
    ap.add_argument("--n", type=int, default=20000)
    a = ap.parse_args()
    now = ND.et_now()
    day = a.date or f"{now:%Y-%m-%d}"
    os.makedirs(OUT, exist_ok=True)
    f = os.path.join(OUT, f"slate_{day}.json")
    old = json.load(open(f)) if os.path.exists(f) else dict(date=day, games={})
    evs = [e for e in ND.scoreboard(day) if e.get("season", {}).get("type") in (2, 3)
           and all(ND.ab(c["team"]["abbreviation"]) in ND.CLUBS for c in e["competitions"][0]["competitors"])]   # no All-Star or 4 Nations games
    todo = []
    for e in evs:
        state = e["status"]["type"]["state"]
        prev = old["games"].get(e["id"])
        if state != "pre":
            continue
        start = ND.to_et(e["date"]).replace(tzinfo=None)
        mins = (start - now.replace(tzinfo=None)).total_seconds() / 60
        if a.mode == "pregame" and not (30 <= mins <= 110 and not (prev and prev.get("m", {}).get("run") == "pregame")):
            continue
        todo.append(e)
    if not todo:
        print(f"{day} {a.mode}: nothing to run ({len(evs)} games on the slate)")
        return
    import nhl_props as NP
    st_idx = NP.History()   # pre-game player and team history (players.csv.gz)
    dfo = ND.dailyfaceoff(day)
    for e in todo:
        rec = sim_game(e, st_idx, dfo, a.n, int(e["id"]) % 100000)
        if "m" in rec:
            rec["m"].update(run=a.mode, at=f"{ND.et_now():%-I:%M %p} ET",
                            goalies="confirmed" if all((rec["goalies"].get(k) or {}).get("st") == "Confirmed" for k in "ha") else "expected")
            print(f"{rec['away']} @ {rec['home']}: home {rec['m']['ph']:.0%}, OT {rec['m']['pot']:.0%}, {len(rec['props'])} props")
        else:
            print(f"{rec['away']} @ {rec['home']}: no line yet")
        old["games"][e["id"]] = rec
    old["built"] = datetime.now().isoformat(timespec="seconds")
    json.dump(old, open(f, "w"), separators=(",", ":"))
    log_picks(day, old)


def log_picks(day, slate):
    """Every priced side at the price on offer, first sighting per run type kept (morning and pregame both logged)."""
    f = os.path.join(OUT, f"picks_{day}.json")
    log = json.load(open(f)) if os.path.exists(f) else []
    seen = {(x["id"], x["mk"], x.get("who"), x.get("line"), x.get("side"), x["run"]) for x in log}
    for gid, g in slate["games"].items():
        m = g.get("m")
        if not m:
            continue
        o = g["odds"]
        rows = [dict(mk="ml", who=g["home"], p=m["ph"], nv=m["ml"]["nv_h"], price=o["ml_h"]),
                dict(mk="ml", who=g["away"], p=m["pa"], nv=round(1 - m["ml"]["nv_h"], 4), price=o["ml_a"]),
                dict(mk="tot_o", who="over", line=m["tot"]["line"], p=m["tot"]["po"], nv=m["tot"]["nv_o"], price=o["over"]),
                dict(mk="tot_u", who="under", line=m["tot"]["line"], p=m["tot"]["pu"], nv=round(1 - m["tot"]["nv_o"], 4), price=o["under"])]
        if m.get("pl"):
            pl = m["pl"]
            dog = g["away"] if pl["fav"] == g["home"] else g["home"]
            rows += [dict(mk="pl", who=pl["fav"], line=-1.5, p=pl["p_fav"], nv=pl["nv_fav"], price=pl["price_fav"]),
                     dict(mk="pl", who=dog, line=1.5, p=pl["p_dog"], nv=round(1 - pl["nv_fav"], 4), price=pl["price_dog"])]
        for pr in g.get("props") or []:
            if pr.get("hold"):
                continue
            rows.append(dict(mk=pr["mk"], who=pr["n"], aid=pr.get("aid"), tm=pr["tm"], line=pr["line"], side="over", p=pr["p"], nv=pr.get("nv"), price=pr["price"]))
            if pr.get("under"):   # both sides of a two-way prop are logged, so the model's "under" views get graded too
                rows.append(dict(mk=pr["mk"], who=pr["n"], aid=pr.get("aid"), tm=pr["tm"], line=pr["line"], side="under", p=round(1 - pr["p"], 4),
                                 nv=round(1 - pr["nv"], 4) if pr.get("nv") is not None else None, price=pr["under"]))
        for r in rows:
            k = (gid, r["mk"], r.get("who"), r.get("line"), r.get("side"), m["run"])
            if k in seen or not r.get("price"):
                continue
            seen.add(k)
            log.append(dict(r, id=gid, run=m["run"], at=m["at"]))
    json.dump(log, open(f, "w"), separators=(",", ":"))


if __name__ == "__main__":
    main()
