"""Download the history the props model is fit and tested on (run once, then it only adds new finished games).

- Player box scores for every NHL game from 2025-26 on (ESPN summary): goals, assists, shots on goal, time on ice, power-play time,
  goalie saves. -> data/players.csv.gz
- DraftKings player props for every game ESPN kept them for (from May 2026: the 2026 playoffs and 2026-27). -> data/props_hist.csv.gz

Run: ../venv/bin/python fetch_history.py   (raw downloads are kept in data/raw/, which is not published)
"""
import csv
import gzip
import json
import os
from concurrent.futures import ThreadPoolExecutor
import nhl_data as ND

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data", "raw")
FIRST = "2025-10"   # last season onward


def months():
    """Every month from FIRST to this one (the off-season months simply have no games)."""
    y, m = map(int, FIRST.split("-"))
    now = ND.et_now()
    out = []
    while (y, m) <= (now.year, now.month):
        out.append(f"{y}{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out
PROPS_FROM = "2026-05-01"
MKT = {"Anytime Goalscorer": "goal", "Total Assists": "ast", "Total Points": "pts", "Total Shots on Goal": "sog", "Total Saves": "saves"}


def secs(t):
    try:
        m, s = str(t).split(":")
        return int(m) * 60 + int(s)
    except ValueError:
        return 0


def games():
    out = {}
    for m in months():
        d = ND.get(f"{ND.SITE}/scoreboard?dates={m}&limit=1000") or {}
        for e in d.get("events", []):
            c = e["competitions"][0]
            teams = [ND.ab(x["team"]["abbreviation"]) for x in c["competitors"]]
            if e["season"]["type"] not in (2, 3) or not e["status"]["type"].get("completed") or not all(t in ND.CLUBS for t in teams):
                continue
            h = next(x for x in c["competitors"] if x["homeAway"] == "home")
            a = next(x for x in c["competitors"] if x["homeAway"] == "away")
            out[e["id"]] = dict(id=e["id"], date=f"{ND.to_et(e['date'][:16]):%Y-%m-%d}", t=e["date"], stype=e["season"]["type"],
                                home=ND.ab(h["team"]["abbreviation"]), away=ND.ab(a["team"]["abbreviation"]), hs=int(h["score"]), as_=int(a["score"]),
                                det=e["status"]["type"].get("shortDetail", ""))
    return out


def box(g):
    f = os.path.join(RAW, "box", f"{g['id']}.json")
    if os.path.exists(f):
        return json.load(open(f))
    s = ND.get(f"{ND.SITE}/summary?event={g['id']}")
    if not s or not (s.get("boxscore") or {}).get("players"):
        return None
    rows = []
    for t in s["boxscore"]["players"]:
        tm = ND.ab(t["team"]["abbreviation"])
        for grp in t["statistics"]:
            nm = grp.get("name")
            for i, a in enumerate(grp.get("athletes", [])):
                r = dict(zip(grp["labels"], a["stats"]))
                if nm in ("forwards", "defenses"):
                    rows.append([a["athlete"]["id"], a["athlete"]["displayName"], tm, "D" if nm == "defenses" else "F", int(r.get("G") or 0),
                                 int(r.get("A") or 0), int(r.get("S") or 0), secs(r.get("TOI")), secs(r.get("PPTOI")), "", ""])
                elif nm == "goalies":
                    rows.append([a["athlete"]["id"], a["athlete"]["displayName"], tm, "G" if i == 0 else "G2", 0, 0, 0, secs(r.get("TOI")), 0,
                                 int(r.get("SV") or 0), int(r.get("SA") or 0)])
    os.makedirs(os.path.dirname(f), exist_ok=True)
    json.dump(rows, open(f, "w"), separators=(",", ":"))
    return rows


def props(g):
    f = os.path.join(RAW, "props", f"{g['id']}.json")
    if os.path.exists(f):
        return json.load(open(f))
    rows = [[r["aid"], MKT[r["market"]], r["line"], r["price"], r["open"]] for r in ND.props(g["id"]) if r["market"] in MKT and r["aid"] and r["price"]]
    os.makedirs(os.path.dirname(f), exist_ok=True)
    json.dump(rows, open(f, "w"), separators=(",", ":"))
    return rows


def kept(path, key="gid"):
    """Rows already in a compiled file, by game: games on file are never fetched again (GitHub keeps no raw cache)."""
    out = {}
    if os.path.exists(path):
        for r in csv.DictReader(gzip.open(path, "rt")):
            out.setdefault(r[key], []).append(r)
    return out


def main():
    gs = games()
    print(len(gs), "finished games")
    P, Q = os.path.join(HERE, "data", "players.csv.gz"), os.path.join(HERE, "data", "props_hist.csv.gz")
    old_b, old_p = kept(P), kept(Q)
    for gid, rows in old_b.items():   # seed the raw cache from the compiled file
        f = os.path.join(RAW, "box", f"{gid}.json")
        if not os.path.exists(f):
            os.makedirs(os.path.dirname(f), exist_ok=True)
            json.dump([[r["aid"], r["name"], r["tm"], r["pos"], int(r["g"]), int(r["a"]), int(r["sog"]), int(r["toi"]), int(r["pptoi"]),
                        int(r["sv"]) if r["sv"] else "", int(r["sa"]) if r["sa"] else ""] for r in rows], open(f, "w"))
    for gid, rows in old_p.items():
        f = os.path.join(RAW, "props", f"{gid}.json")
        if not os.path.exists(f):
            os.makedirs(os.path.dirname(f), exist_ok=True)
            json.dump([[r["aid"], r["mk"], r["line"], r["price"], r["open"]] for r in sorted(rows, key=lambda x: int(x["seq"]))], open(f, "w"))
    with ThreadPoolExecutor(6) as ex:
        boxes = dict(zip(gs, ex.map(box, gs.values())))
        pg = [g for g in gs.values() if g["date"] >= PROPS_FROM]
        prs = dict(zip([g["id"] for g in pg], ex.map(props, pg)))
    with gzip.open(os.path.join(HERE, "data", "players.csv.gz"), "wt") as fh:
        w = csv.writer(fh)
        w.writerow(["gid", "date", "stype", "home", "away", "hs", "as", "det", "aid", "name", "tm", "pos", "g", "a", "sog", "toi", "pptoi", "sv", "sa"])
        for gid, rows in boxes.items():
            g = gs[gid]
            for r in rows or []:
                w.writerow([gid, g["date"], g["stype"], g["home"], g["away"], g["hs"], g["as_"], g["det"]] + r)
    with gzip.open(os.path.join(HERE, "data", "props_hist.csv.gz"), "wt") as fh:
        w = csv.writer(fh)
        w.writerow(["gid", "date", "seq", "aid", "mk", "line", "price", "open"])
        for gid, rows in prs.items():
            for i, r in enumerate(rows):
                w.writerow([gid, gs[gid]["date"], i] + r)
    print(sum(1 for b in boxes.values() if b), "box scores;", sum(1 for p in prs.values() if p), "games with props;",
          sum(len(p) for p in prs.values()), "prop prices")


if __name__ == "__main__":
    main()
