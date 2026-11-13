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


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "docs"
    et = datetime.now(timezone.utc) - timedelta(hours=5)
    nba = []
    for d in (et - timedelta(days=1), et):
        nba += games(get(f"{API}/basketball/nba/scoreboard?dates={d:%Y%m%d}"))
    res = dict(updated=datetime.now(timezone.utc).isoformat(timespec="seconds"), nfl=games(get(f"{API}/football/nfl/scoreboard"), NFL_FIX),
               cfb=games(get(f"{API}/football/college-football/scoreboard?groups=80&limit=300")), nba=nba)
    os.makedirs(out, exist_ok=True)
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
