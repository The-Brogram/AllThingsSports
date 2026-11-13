"""Download NBA data from ESPN: daily scoreboards, per-game box scores and DraftKings lines, player props, and the injury feed.

Run: ../venv/bin/python fetch_nba.py            every date from 2025-10-21 through tomorrow (only what is missing or not final)
     ../venv/bin/python fetch_nba.py --today    today's slate only: scoreboard, summaries, props, injuries (fast; for the daily runs)
Saved trimmed, under data/:
  sb/<YYYYMMDD>.json      events on that date
  games/<id>.json         header, box score (minutes, points, rebounds, ...), DraftKings spread/total/moneyline, injuries listed for the game
  props/<id>.json         DraftKings player props: [athlete id, market, line, american price, open line, open price, order, market id, last updated]
  injuries.json           ESPN's league injury feed (status, date, comment) at fetch time
NBA.com blocks scripted requests (403), so ESPN is the only source.
"""
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SITE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
CORE = "https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba"
START = date(2025, 10, 21)  # 2025-26 opening night


def get(url):
    r = subprocess.run(["curl", "-s", "-m", "40", "--retry", "2", url], capture_output=True, text=True)
    try:
        return json.loads(r.stdout)
    except ValueError:
        return {}


def load(f):
    try:
        return json.load(open(f))
    except (OSError, ValueError):
        return None


def save(f, x):
    os.makedirs(os.path.dirname(f), exist_ok=True)
    with open(f, "w") as h:
        json.dump(x, h, separators=(",", ":"))


def scoreboard(d):
    s = get(f"{SITE}/scoreboard?dates={d:%Y%m%d}&limit=50")
    ev = []
    for e in s.get("events", []):
        c = e["competitions"][0]
        st = e["status"]["type"]
        ev.append(dict(id=e["id"], date=e["date"], name=e.get("shortName"), state=st.get("state"), status=st.get("name"), completed=bool(st.get("completed")),
                       season=e.get("season", {}).get("year"), stype=e.get("season", {}).get("type"), neutral=bool(c.get("neutralSite")),
                       venue=(c.get("venue") or {}).get("fullName"), city=((c.get("venue") or {}).get("address") or {}).get("city"),
                       teams=[dict(id=x["team"]["id"], ab=x["team"]["abbreviation"], name=x["team"].get("displayName"), short=x["team"].get("shortDisplayName"), color=x["team"].get("color"),
                                   ha=x["homeAway"], score=x.get("score"), rec=(x.get("records") or [{}])[0].get("summary")) for x in c["competitors"]],
                       tv=next((b.get("names", [None])[0] for b in c.get("broadcasts", []) if b.get("names")), None),
                       odds=[dict(details=o.get("details"), ou=o.get("overUnder")) for o in c.get("odds", [])]))
    return ev


def summary(gid):
    s = get(f"{SITE}/summary?event={gid}")
    if not s.get("header"):
        return None
    h = s["header"]
    c = h["competitions"][0]
    box = []
    for t in s.get("boxscore", {}).get("players", []):
        st = (t.get("statistics") or [{}])[0]
        lab = st.get("labels", [])
        for a in st.get("athletes", []):
            if not a.get("athlete", {}).get("id"):
                continue
            box.append(dict(team=t["team"]["abbreviation"], id=a["athlete"].get("id"), name=a["athlete"].get("displayName"), pos=(a["athlete"].get("position") or {}).get("abbreviation"),
                            starter=bool(a.get("starter")), dnp=bool(a.get("didNotPlay")), reason=a.get("reason"), stats=dict(zip(lab, a.get("stats", [])))))
    pick = [dict(provider=p.get("provider", {}).get("name"), details=p.get("details"), ou=p.get("overUnder"), spread=p.get("spread"),
                 ml_h=(p.get("homeTeamOdds") or {}).get("moneyLine"), ml_a=(p.get("awayTeamOdds") or {}).get("moneyLine"),
                 sp_h=(p.get("homeTeamOdds") or {}).get("spreadOdds"), sp_a=(p.get("awayTeamOdds") or {}).get("spreadOdds"),
                 fav_h=(p.get("homeTeamOdds") or {}).get("favorite"), over=p.get("overOdds"), under=p.get("underOdds")) for p in s.get("pickcenter", [])]
    inj = [dict(team=t.get("team", {}).get("abbreviation"), rows=[dict(id=(i.get("athlete") or {}).get("id"), name=(i.get("athlete") or {}).get("displayName"), status=i.get("status"),
                                                                     date=i.get("date"), detail=(i.get("details") or {}).get("type")) for i in t.get("injuries", [])]) for t in s.get("injuries", [])]
    return dict(id=gid, date=c.get("date"), completed=bool(c.get("status", {}).get("type", {}).get("completed")), neutral=bool(c.get("neutralSite")),
                season=h.get("season", {}).get("year"), stype=h.get("season", {}).get("type"),
                teams=[dict(id=x["team"]["id"], ab=x["team"]["abbreviation"], ha=x["homeAway"], score=x.get("score"), winner=x.get("winner")) for x in c["competitors"]],
                box=box, pick=pick, inj=inj, fetched=datetime.now(timezone.utc).isoformat(timespec="seconds"))


def props(gid):
    # ESPN ignores limit and serves 25 items per page
    url = f"{CORE}/events/{gid}/competitions/{gid}/odds/100/propBets"
    s = get(f"{url}?page=1")
    items = s.get("items", [])
    with ThreadPoolExecutor(6) as ex:
        for x in ex.map(lambda p: get(f"{url}?page={p}"), range(2, int(s.get("pageCount") or 1) + 1)):
            items += x.get("items", [])
    out = []
    for k, i in enumerate(items):
        o = i.get("odds", {})
        out.append([i.get("athlete", {}).get("$ref", "").split("/athletes/")[-1].split("?")[0] or None, i.get("type", {}).get("name"),
                    (o.get("total") or {}).get("value"), (o.get("american") or {}).get("value"), (o.get("total") or {}).get("open"), (o.get("american") or {}).get("open"), k,
                    i.get("type", {}).get("id"), i.get("lastUpdated")])  # market id and price timestamp, kept since Nov 12, 2026
    return out


def fetch_game(e, force):
    gid = e["id"]
    g = load(f"{D}/games/{gid}.json")
    if force or not g or not g.get("completed"):
        g2 = summary(gid)
        if g2:
            save(f"{D}/games/{gid}.json", g2)
            g = g2
    p = f"{D}/props/{gid}.json"
    if force or not os.path.exists(p) or not (g or {}).get("completed"):
        save(p, props(gid))


def main():
    today = "--today" in sys.argv
    now = datetime.now(timezone.utc) - timedelta(hours=5)
    days = [now.date()] if today else [START + timedelta(n) for n in range((now.date() - START).days + 2)]
    todo = []
    for d in days:
        f = f"{D}/sb/{d:%Y%m%d}.json"
        ev = load(f)
        if ev is None or today or d >= now.date() - timedelta(days=2) or any(not x["completed"] for x in ev):
            ev = scoreboard(d)
            save(f, ev)
        todo += [x for x in ev if x.get("stype") in (2, 3, 5)]  # regular season, playoffs, play-in
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda e: fetch_game(e, today), todo))
    inj = get(f"{SITE}/injuries")
    rows = [dict(team=(i.get("athlete", {}).get("team") or {}).get("abbreviation"), id=(i.get("athlete", {}).get("links") or [{}])[0].get("href", "").split("/id/")[-1].split("/")[0],
                 name=i.get("athlete", {}).get("displayName"), status=i.get("status"), date=i.get("date"), short=i.get("shortComment"), detail=(i.get("details") or {}).get("type"),
                 ret=(i.get("details") or {}).get("returnDate")) for t in inj.get("injuries", []) for i in t.get("injuries", [])]
    save(f"{D}/injuries.json", dict(fetched=datetime.now(timezone.utc).isoformat(timespec="seconds"), rows=rows))
    print(f"{len(days)} dates, {len(todo)} games checked, {len(rows)} injury listings")


if __name__ == "__main__":
    main()
