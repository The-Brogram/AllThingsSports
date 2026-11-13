"""Download ESPN college scoreboards + per-game summaries.
Run: python3 fetch_cfb.py [--week N] [--today-only]
The slate (data/sb.json) is ESPN's whole week N (Tue-Sat), default the current week."""
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
API = "https://site.api.espn.com/apis/site/v2/sports/football/college-football"


def week_arg():
    a = sys.argv
    if "--week" in a:
        return int(a[a.index("--week") + 1])
    return None


def get(url, out):
    subprocess.run(["curl", "-s", "-m", "40", "--retry", "2", "-o", out, url], check=False)


def load(f):
    try:
        return json.load(open(f))
    except (OSError, ValueError):
        return {}


def main():
    os.makedirs(f"{D}/games", exist_ok=True)
    os.makedirs(f"{D}/sb", exist_ok=True)
    wk = week_arg()
    get(f"{API}/scoreboard?groups=80&limit=400" + (f"&seasontype=2&week={wk}" if wk else ""), f"{D}/sb.json")
    get(f"{API}/rankings", f"{D}/rank.json")
    subprocess.run(["curl", "-s", "-m", "40", "-A", "Mozilla/5.0", "-o", f"{D}/covers_inj.html", "https://www.covers.com/sport/football/ncaaf/injuries"], check=False)
    ids = [e["id"] for e in load(f"{D}/sb.json").get("events", [])]
    if "--today-only" not in sys.argv:
        jobs = [(y, w, t) for y in (2025, 2026) for w, t in [(w, 2) for w in range(1, 17)] + [(1, 3)]]
        for y, w, t in jobs:
            f = f"{D}/sb/{y}_{t}_{w}.json"
            get(f"{API}/scoreboard?dates={y}&seasontype={t}&week={w}&groups=80&limit=400", f)
            for e in load(f).get("events", []):
                if e["status"]["type"].get("completed"):
                    ids.append(e["id"])
    ids = sorted(set(ids))
    def final(i):  # a summary saved before kickoff must be fetched again once the game is over
        c = (load(f"{D}/games/{i}.json").get("header", {}).get("competitions") or [{}])[0]
        return c.get("status", {}).get("type", {}).get("completed", False)
    todo = [i for i in ids if "--today-only" in sys.argv or not final(i)]
    with ThreadPoolExecutor(10) as ex:
        list(ex.map(lambda i: get(f"{API}/summary?event={i}", f"{D}/games/{i}.json"), todo))
    print(len(ids), "games on file;", len(todo), "fetched")


if __name__ == "__main__":
    main()
