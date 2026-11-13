"""Live scores and finals for the hub, light enough to run every few minutes on GitHub's servers (no installs, no project data).

Run: python3 cloud_scores.py [output folder, default docs]   -> <folder>/scores.json
Reads ESPN's public scoreboards for the NFL, college football (FBS) and the NBA (yesterday and today, Eastern time).
The hub page loads scores.json when it opens and overlays live scores and finals on games its own data does not have yet.
"""
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone

API = "https://site.api.espn.com/apis/site/v2/sports"
NFL_FIX = {"WSH": "WAS", "LAR": "LA"}  # ESPN abbreviations -> the hub's (nflverse)


def get(url):
    """curl, like the rest of the project: it is on every Mac and on GitHub's servers, and avoids Python certificate setup."""
    try:
        r = subprocess.run(["curl", "-sf", "--max-time", "30", "-A", "scores-refresh", url], capture_output=True, text=True)
        return json.loads(r.stdout) if r.returncode == 0 else {}
    except Exception as e:  # noqa: BLE001  one feed failing must not stop the others
        print("failed:", url, repr(e))
        return {}


def games(d, fix=None):
    out = []
    for e in d.get("events", []):
        c = e["competitions"][0]
        t = {x["homeAway"]: x for x in c["competitors"]}
        if "home" not in t or "away" not in t:
            continue
        st = e["status"]["type"]
        ab = lambda x: (fix or {}).get(x["team"]["abbreviation"], x["team"]["abbreviation"])
        sc = lambda x: int(x["score"]) if str(x.get("score", "")).isdigit() else None
        out.append(dict(id=e["id"], date=e["date"], home=ab(t["home"]), away=ab(t["away"]), hs=sc(t["home"]), as_=sc(t["away"]),
                        state=st.get("state"), final=bool(st.get("completed")), detail=st.get("shortDetail")))
    return out


PATH = {"nfl": "football/nfl", "cfb": "football/college-football", "nba": "basketball/nba"}


def review(sport, g, fix=None):
    """Post-game review of one final from ESPN's game summary: winner and price, spread and total result, each team's leaders."""
    d = get(f"{API}/{PATH[sport]}/summary?event={g['id']}")
    if not d or g["hs"] is None or g["as_"] is None:
        return None
    ab = lambda x: (fix or {}).get(x, x)
    p = (d.get("pickcenter") or [{}])[0]
    sp, ou = p.get("spread"), p.get("overUnder")          # spread is the home team's line: negative = home favored
    mlh, mla = (p.get("homeTeamOdds") or {}).get("moneyLine"), (p.get("awayTeamOdds") or {}).get("moneyLine")
    hm = g["hs"] - g["as_"]
    r = dict(sport=sport, id=g["id"], date=g["date"], home=g["home"], away=g["away"], hs=g["hs"], as_=g["as_"], detail=g.get("detail"),
             winner=None if hm == 0 else (g["home"] if hm > 0 else g["away"]), ml_h=mlh, ml_a=mla, spread_home=sp, total=ou, line=p.get("details"), book=(p.get("provider") or {}).get("name"))
    if sp is not None:
        ats = hm + sp                                      # home team's margin against its line
        r["cover"] = None if ats == 0 else (g["home"] if ats > 0 else g["away"])
        r["fav"] = None if sp == 0 else (g["home"] if sp < 0 else g["away"])
    if ou is not None:
        t = g["hs"] + g["as_"]
        r["ou"] = "push" if t == ou else ("over" if t > ou else "under")
    lead = []
    for t in d.get("leaders") or []:
        rows = []
        for c in t.get("leaders") or []:
            for x in (c.get("leaders") or [])[:1]:
                nm = (x.get("athlete") or {}).get("displayName")
                if nm and x.get("displayValue"):
                    rows.append(dict(cat=c.get("displayName") or c.get("name"), name=nm, line=x["displayValue"]))
        lead.append(dict(team=ab((t.get("team") or {}).get("abbreviation")), rows=rows[:5]))
    r["leaders"] = lead
    return r


def reviews(out, res):
    """Write a review for every final that does not have one yet (at most 25 per run), and drop reviews older than three weeks."""
    f = os.path.join(out, "reviews.json")
    try:
        R = json.load(open(f))
    except (OSError, ValueError):
        R = {}
    n, cut = 0, (datetime.now(timezone.utc) - timedelta(days=21)).isoformat()
    for sport, fix in (("nfl", NFL_FIX), ("cfb", None), ("nba", None)):
        for g in res[sport]:
            k = f"{sport}:{g['id']}"
            if not g["final"] or k in R or n >= 25:
                continue
            r = review(sport, g, fix)
            if r:
                R[k] = r
                if sport == "nfl":
                    R[f"nfl:{g['away']}@{g['home']}"] = k   # the hub finds NFL games by teams
                n += 1
    keep = {k: v for k, v in R.items() if isinstance(v, str) or v.get("date", "9") >= cut}
    keep = {k: v for k, v in keep.items() if not isinstance(v, str) or v in keep}
    if n or len(keep) != len(R):
        json.dump(keep, open(f, "w"), separators=(",", ":"))
    print(f"reviews: {n} new, {sum(not isinstance(v, str) for v in keep.values())} on file")


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "docs"
    et = datetime.now(timezone.utc) - timedelta(hours=5)
    nba = []
    for d in (et - timedelta(days=1), et):
        nba += games(get(f"{API}/basketball/nba/scoreboard?dates={d:%Y%m%d}"))
    res = dict(updated=datetime.now(timezone.utc).isoformat(timespec="seconds"), nfl=games(get(f"{API}/football/nfl/scoreboard"), NFL_FIX),
               cfb=games(get(f"{API}/football/college-football/scoreboard?groups=80&limit=300")), nba=nba)
    os.makedirs(out, exist_ok=True)
    reviews(out, res)
    f = os.path.join(out, "scores.json")
    try:  # leave the file alone when nothing but the timestamp would change, so GitHub only records real changes
        old = json.load(open(f))
        if all(old.get(k) == res[k] for k in ("nfl", "cfb", "nba")):
            print("scores: no change")
            return
    except (OSError, ValueError):
        pass
    json.dump(res, open(f, "w"), separators=(",", ":"))
    print("scores:", {k: f"{sum(g['final'] for g in v)}/{len(v)} final" for k, v in res.items() if k != "updated"})


if __name__ == "__main__":
    main()
