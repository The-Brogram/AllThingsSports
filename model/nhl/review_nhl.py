"""Grade logged NHL picks once every game of a night is final. DraftKings settlement:
- a shootout adds one goal to the winner's score and to the total;
- player props count overtime, never the shootout;
- a goalie saves prop is void if that goalie did not start.

Run: ../venv/bin/python review_nhl.py   (grades every finished night that has a picks file and no review yet)
Writes output/review_<date>.json: one row per logged side with hit (1, 0, or null for a push or void) and profit at the logged price.
"""
import glob
import json
import os
import nhl_data as ND

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")


def profit(price, hit):
    p = float(str(price).replace("+", ""))
    if hit is None:
        return 0.0
    if not hit:
        return -1.0
    return p / 100 if p > 0 else 100 / -p


def box(eid):
    s = ND.summary(eid, final=True)
    if not s:
        return None
    comp = s["header"]["competitions"][0]
    st = comp["status"]["type"]
    if st.get("state") == "post" and not st.get("completed", True) or any(w in (st.get("name") or "") for w in ("POSTPONED", "CANCELED", "SUSPENDED")):
        return dict(void=True)   # postponed or cancelled: every side on it is void, and the rest of the night still grades
    if st["state"] != "post":
        return None
    det = comp["status"]["type"].get("shortDetail") or comp["status"]["type"].get("detail") or ""
    sc = {ND.ab(c["team"]["abbreviation"]): int(c["score"]) for c in comp["competitors"]}
    pl = {}
    for t in s["boxscore"].get("players", []):
        for grp in t["statistics"]:
            for i, a in enumerate(grp.get("athletes", [])):   # ESPN lists the starting goalie first (no starter flag in the feed)
                r = dict(zip(grp["labels"], a["stats"]))
                pl[a["athlete"]["id"]] = pl[ND.name_key(a["athlete"]["displayName"])] = dict(
                    goal=int(r.get("G", 0) or 0), ast=int(r.get("A", 0) or 0), sog=int(r.get("S", 0) or 0),
                    saves=int(r["SV"]) if grp.get("name") == "goalies" and r.get("SV") not in (None, "") else None,
                    started=grp.get("name") == "goalies" and i == 0)
    return dict(score=sc, so="SO" in det, ot="OT" in det or "SO" in det, players=pl)


def grade(p, b, home, away):
    if b.get("void"):
        return None
    sc = b["score"]
    tot = sc[home] + sc[away]
    mk = p["mk"]
    if mk == "ml":
        return int(sc[p["who"]] > sc[away if p["who"] == home else home])
    if mk == "pl":
        other = away if p["who"] == home else home
        m = sc[p["who"]] - sc[other] + p["line"]
        return None if m == 0 else int(m > 0)
    if mk in ("tot_o", "tot_u"):
        if tot == p["line"]:
            return None
        return int((tot > p["line"]) == (mk == "tot_o"))
    r = b["players"].get(p.get("aid")) or b["players"].get(ND.name_key(p["who"]))   # ESPN athlete id first (names can repeat)
    if r is None:
        return None  # did not play: DraftKings voids
    if mk == "goal":
        return int(r["goal"] >= 1)
    v = r.get(mk)
    if v is None:
        return None
    if mk == "saves" and not r.get("started"):
        return None
    if v == p["line"]:
        return None
    return int((v > p["line"]) == (p.get("side", "over") == "over"))


def main():
    for f in sorted(glob.glob(os.path.join(OUT, "picks_*.json"))):
        day = os.path.basename(f)[6:16]
        out = os.path.join(OUT, f"review_{day}.json")
        if os.path.exists(out):
            continue
        slate = json.load(open(os.path.join(OUT, f"slate_{day}.json")))
        picks = json.load(open(f))
        boxes, done = {}, True
        for gid in {p["id"] for p in picks}:
            b = box(gid)
            if b is None:
                done = False
                break
            boxes[gid] = b
        if not done:
            print(day, "not all final yet")
            continue
        rows = []
        for p in picks:
            g = slate["games"][p["id"]]
            hit = grade(p, boxes[p["id"]], g["home"], g["away"])
            rows.append(dict(p, hit=hit, profit=round(profit(p["price"], hit), 4)))
        games = {gid: (dict(void=True) if b.get("void") else dict(score=b["score"], ot=b["ot"], so=b["so"])) for gid, b in boxes.items()}
        json.dump(dict(date=day, games=games, rows=rows), open(out, "w"), separators=(",", ":"))
        print(day, f"graded {len(rows)} sides over {len(games)} games")


if __name__ == "__main__":
    main()
