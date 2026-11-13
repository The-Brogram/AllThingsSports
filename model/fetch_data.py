"""Refresh all inputs. Run before each slate and again after inactives: python fetch_data.py
python fetch_data.py results refreshes only the nflverse files, which is all a postgame review needs."""
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from config import COORD, DATE, ESPN

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
B = "https://github.com/nflverse/nflverse-data/releases/download"
CORE = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/events"
FILES = ["stats_player/stats_player_week_2026.csv", "pbp/play_by_play_2026.csv.gz", "snap_counts/snap_counts_2026.csv", "injuries/injuries_2026.csv", "ftn_charting/ftn_charting_2026.csv",
         "nextgen_stats/ngs_receiving.csv.gz", "nextgen_stats/ngs_passing.csv.gz", "nextgen_stats/ngs_rushing.csv.gz", "depth_charts/depth_charts_2026.csv", "rosters/roster_2026.csv",
         "pfr_advstats/advstats_week_pass_2026.csv", "pfr_advstats/advstats_week_rec_2026.csv", "pfr_advstats/advstats_week_rush_2026.csv", "pfr_advstats/advstats_week_def_2026.csv", "weekly_rosters/roster_weekly_2026.csv"]


def curl(url, out, *extra):
    """Download to a temp file and replace `out` only on success, so a failed or partial download never overwrites good data."""
    tmp = out + ".part"
    r = subprocess.run(["curl", "-sfL", "-m", "300", *extra, "-o", tmp, url], check=False)
    if r.returncode == 0 and os.path.exists(tmp) and os.path.getsize(tmp) > 0:
        os.replace(tmp, out)
    else:
        if os.path.exists(tmp):
            os.remove(tmp)
        print(f"download failed, kept the old copy: {os.path.basename(out)} (curl exit {r.returncode})")


def page(a):
    e, i = a
    curl(f"{CORE}/{e}/competitions/{e}/odds/100/propBets?limit=100&page={i}", f"{D}/espn/pp/{e}_{i}.json")


def fetch_nflverse():
    with ThreadPoolExecutor(6) as ex:
        list(ex.map(lambda f: curl(f"{B}/{f}", f"{D}/{os.path.basename(f)}"), FILES))
    curl("https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv", f"{D}/games.csv")


def fetch_espn():
    """Schedule, injuries, game odds and DraftKings prop lines for the slate date. Returns the events."""
    os.makedirs(f"{D}/espn/pp", exist_ok=True)
    curl(f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?dates={DATE}", f"{D}/{ESPN}")
    events = json.load(open(f"{D}/{ESPN}"))["events"]
    for ev in events:
        e = ev["id"]
        curl(f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary?event={e}", f"{D}/espn/sum_{e}.json")
        page((e, 1))
        try:
            n = json.load(open(f"{D}/espn/pp/{e}_1.json")).get("pageCount", 1)
        except (OSError, ValueError):
            n = 1  # no props posted for this game yet
        with ThreadPoolExecutor(6) as ex:
            list(ex.map(page, [(e, i) for i in range(2, n + 1)]))
    return events


def fetch_weather(events):
    os.makedirs(f"{D}/weather", exist_ok=True)
    ua = ("-H", "User-Agent: nfl_model personal research")
    homes = {c["team"]["abbreviation"] for ev in events for c in ev["competitions"][0]["competitors"] if c["homeAway"] == "home"}
    for t in sorted({"WSH": "WAS", "LAR": "LA"}.get(h, h) for h in homes):  # nflverse names, as report.py reads them
        la, lo = COORD[t]
        tmp = f"{D}/weather/_{t}.tmp"
        try:  # one forecast failure must not stop the refresh; the last good file stays in place
            curl(f"https://api.weather.gov/points/{la},{lo}", tmp, *ua)
            curl(json.load(open(tmp))["properties"]["forecastHourly"], tmp, *ua)
            json.load(open(tmp))["properties"]["periods"]
            os.replace(tmp, f"{D}/weather/{t}.json")
        except (OSError, ValueError, KeyError) as x:
            print(f"WARNING: weather for {t} not refreshed ({x}); using the last saved forecast")


def main():
    import odds
    fetch_nflverse()
    events = fetch_espn()
    fetch_weather(events)
    odds.snapshot()
    print("data refreshed")


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["results"]:
        fetch_nflverse()
    else:
        main()
