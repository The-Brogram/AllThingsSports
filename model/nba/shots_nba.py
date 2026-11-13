"""Shot locations from ESPN play-by-play (this season's completed games), for the scoring-zone sections of the hub.

Run: ../venv/bin/python shots_nba.py   (incremental: only games not yet saved)
Saves data/shots/<game_id>.json as a list of [shooter_id, team, x, y, points_attempted, made, free_throw, period, text_type]
and data/shots.csv.gz with every shot and its zone.

ESPN coordinates are in feet on a half court: x 0-50 across, y up from the baseline, hoop at about (25, 0) (baseline ~5 ft behind).
Zones follow the NBA's shot-zone split: Restricted Area (within 4 ft of the hoop), In The Paint (non-RA), Mid-Range,
Left / Right Corner 3, Above the Break 3. Free throws are kept as their own "zone".
"""
import json
import math
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
D = f"{HERE}/data"
OUT = f"{D}/shots"
SITE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
HOOP = (25.0, 0.0)  # ESPN y is measured from the hoop line (layups sit at y 0-3)


def get(gid):
    r = subprocess.run(["curl", "-s", "--max-time", "30", f"{SITE}/summary?event={gid}"], capture_output=True, text=True)
    try:
        return json.loads(r.stdout)
    except Exception:
        return None


def fetch(gid):
    s = get(gid)
    if not s or not s.get("plays"):
        return gid, False
    teams = {c["team"]["id"]: c["team"]["abbreviation"] for c in s["header"]["competitions"][0]["competitors"]}
    rows = []
    for p in s["plays"]:
        if not p.get("shootingPlay") or not p.get("participants"):
            continue
        c = p.get("coordinate") or {}
        ft = "Free Throw" in (p.get("type") or {}).get("text", "")
        x, y = c.get("x"), c.get("y")
        if not ft and (x is None or y is None or abs(x) > 1000 or abs(y) > 1000):
            continue
        rows.append([p["participants"][0]["athlete"]["id"], teams.get((p.get("team") or {}).get("id")), None if ft else x, None if ft else y,
                     p.get("pointsAttempted") or (1 if ft else 2), int(bool(p.get("scoringPlay"))), int(ft), (p.get("period") or {}).get("number"),
                     (p.get("type") or {}).get("text")])
    json.dump(rows, open(f"{OUT}/{gid}.json", "w"), separators=(",", ":"))
    return gid, True


def zone(x, y, pa, ft):
    if ft:
        return "FT"
    d = math.hypot(x - HOOP[0], y - HOOP[1])
    if pa == 3:
        if y <= 9:
            return "LC3" if x < 25 else "RC3"
        return "AB3"
    if d <= 4:
        return "RA"
    if abs(x - 25) <= 8 and y <= 14:
        return "PNT"
    return "MID"


def main():
    os.makedirs(OUT, exist_ok=True)
    g = pd.read_csv(f"{D}/games.csv", dtype={"game_id": str})
    g = g[g.completed & (g.season == g.season.max())]
    todo = [i for i in g.game_id if not os.path.exists(f"{OUT}/{i}.json")]
    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(fetch, todo))
    bad = [i for i, ok in res if not ok]
    print(f"shots: fetched {len(todo) - len(bad)} new games" + (f", {len(bad)} without play-by-play" if bad else ""))
    rows = []
    for gid in g.game_id:
        f = f"{OUT}/{gid}.json"
        if os.path.exists(f):
            for r in json.load(open(f)):
                rows.append([gid] + r)
    s = pd.DataFrame(rows, columns=["game_id", "pid", "team", "x", "y", "pa", "made", "ft", "period", "kind"])
    s["zone"] = [zone(x, y, pa, ft) for x, y, pa, ft in zip(s.x, s.y, s.pa, s.ft)]
    s["pts"] = s.made * s.pa.where(s.ft == 0, 1)
    s.to_csv(f"{D}/shots.csv.gz", index=False)
    print(f"shots: {len(s):,} shots from {s.game_id.nunique()} games -> data/shots.csv.gz")


if __name__ == "__main__":
    main()
