"""NHL downloads: ESPN (games, DraftKings lines and props, probable goalies, injuries, standings), the NHL stats API (player and
team season stats) and DailyFaceoff (starting goalies). Sources and traps: briefs/nhl/research.md and sources.json.

Everything is fetched with curl (Python's own SSL has failed on this machine). Finished games are cached once in data/final/;
everything else is re-fetched each run.
"""
import json
import os
import re
import subprocess
import time
import unicodedata
from datetime import datetime, timedelta, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data")
SITE = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl"
CORE = "https://sports.core.api.espn.com/v2/sports/hockey/leagues/nhl"
STATS = "https://api.nhle.com/stats/rest/en"
SEASON = "20262027"
# ESPN abbreviation <- other sources' abbreviations
TO_ESPN = {"LAK": "LA", "TBL": "TB", "NJD": "NJ", "SJS": "SJ", "UTAH": "UTA"}
CLUBS = {"ANA", "BOS", "BUF", "CAR", "CBJ", "CGY", "CHI", "COL", "DAL", "DET", "EDM", "FLA", "LA", "MIN", "MTL", "NJ", "NSH", "NYI",
         "NYR", "OTT", "PHI", "PIT", "SEA", "SJ", "STL", "TB", "TOR", "UTA", "VAN", "VGK", "WPG", "WSH"}


def ab(x):
    x = (x or "").upper()
    return TO_ESPN.get(x, x)


def et_now():
    """Eastern time (fixed -5 in winter; -4 from the second Sunday of March to the first Sunday of November)."""
    u = datetime.now(timezone.utc)
    y = u.year
    mar = datetime(y, 3, 8, 7, tzinfo=timezone.utc)
    mar += timedelta(days=(6 - mar.weekday()) % 7)
    nov = datetime(y, 11, 1, 6, tzinfo=timezone.utc)
    nov += timedelta(days=(6 - nov.weekday()) % 7)
    return u + timedelta(hours=-4 if mar <= u < nov else -5)


def to_et(iso):
    u = datetime.strptime(iso.replace("Z", ""), "%Y-%m-%dT%H:%M").replace(tzinfo=timezone.utc)
    off = et_now() - datetime.now(timezone.utc)
    return u + timedelta(seconds=round(off.total_seconds() / 3600) * 3600)


def get(url, tries=3, ua=None):
    for i in range(tries):
        cmd = ["curl", "-s", "-L", "--compressed", "--max-time", "30", url]
        if ua:
            cmd[1:1] = ["-A", ua]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip():
            if ua:
                return r.stdout
            try:
                return json.loads(r.stdout)
            except ValueError:
                pass
        time.sleep(1 + i)
    return None


def name_key(n):
    n = unicodedata.normalize("NFKD", n or "").encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z]", "", n)


# ---------- ESPN ----------
def scoreboard(day):
    """day: YYYY-MM-DD (ET). ESPN's dates= is the ET game date."""
    return (get(f"{SITE}/scoreboard?dates={day.replace('-', '')}&limit=100") or {}).get("events", [])


def summary(eid, final=False):
    f = os.path.join(D, "final", f"{eid}.json")
    if final and os.path.exists(f):
        return json.load(open(f))
    s = get(f"{SITE}/summary?event={eid}")
    done = bool(s) and ((((s.get("header") or {}).get("competitions") or [{}])[0].get("status") or {}).get("type") or {}).get("state") == "post"
    if s and final and done:   # only a finished game is cached for good (a pre-game copy would block grading forever)
        os.makedirs(os.path.dirname(f), exist_ok=True)
        keep = {k: s.get(k) for k in ("boxscore", "plays", "header", "gameInfo", "seasonseries", "leaders", "lastFiveGames", "injuries",
                                       "pickcenter", "goalies")}
        keep["plays"] = [p for p in (s.get("plays") or []) if p.get("scoringPlay")]
        json.dump(keep, open(f, "w"), separators=(",", ":"))
        return keep
    return s


def standings():
    return get("https://site.api.espn.com/apis/v2/sports/hockey/nhl/standings?level=3")


def props(eid):
    """DraftKings player props for one game: [{aid, market, line, price, open, upd}], in feed order (over first in each pair)."""
    out, page, pages = [], 1, 1
    while page <= pages:
        d = get(f"{CORE}/events/{eid}/competitions/{eid}/odds/100/propBets?limit=100&page={page}")
        if not d:
            break
        pages = d.get("pageCount", 1)
        for it in d.get("items", []):
            aid = re.search(r"athletes/(\d+)", (it.get("athlete") or {}).get("$ref", ""))
            am = (it.get("odds") or {}).get("american") or {}
            tgt = ((it.get("current") or {}).get("target") or {}).get("value")
            out.append(dict(aid=aid.group(1) if aid else None, market=it["type"]["name"], line=tgt, price=am.get("value"), open=am.get("open"),
                            upd=it.get("lastUpdated")))
        page += 1
    return out


def roster(team_id):
    """ESPN roster: {athlete id: (name, position)}."""
    r = get(f"{SITE}/teams/{team_id}/roster") or {}
    out = {}
    for grp in r.get("athletes", []):
        for a in grp.get("items", []):
            out[a["id"]] = (a.get("displayName"), (a.get("position") or {}).get("abbreviation"))
    return out


# ---------- DailyFaceoff ----------
def dailyfaceoff(day):
    """Starting goalies by ESPN team abbreviation: {ab: {n, st, rep, t}}. st is DailyFaceoff's status (Confirmed, Likely, Unconfirmed...)."""
    html = get(f"https://www.dailyfaceoff.com/starting-goalies/{day}", ua="Mozilla/5.0 (Macintosh) AllThingsSports-hub")
    if not html:
        return {}
    m = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.S)
    if not m:
        return {}
    try:
        rows = json.loads(m.group(1))["props"]["pageProps"]["data"]
    except (ValueError, KeyError):
        return {}
    out = {}
    for r in rows:
        for side in ("home", "away"):
            out[ab(r.get(f"{side}TeamAbbreviation"))] = dict(n=r.get(f"{side}GoalieName"), st=r.get(f"{side}NewsStrengthName") or "Unconfirmed",
                                                             rep=r.get(f"{side}NewsSourceName"), t=r.get(f"{side}NewsCreatedAt"))
    return out


# ---------- NHL stats API ----------
def nhl_rows(kind, report, sort="points"):
    rows, start = [], 0
    while True:
        q = (f"{STATS}/{kind}/{report}?limit=100&start={start}&sort=%5B%7B%22property%22:%22{sort}%22,%22direction%22:%22DESC%22%7D%5D"
             f"&cayenneExp=seasonId={SEASON}%20and%20gameTypeId=2")
        d = get(q) or {}
        part = d.get("data", [])
        rows += part
        start += 100
        if len(part) < 100 or start >= d.get("total", 0):
            break
    return rows


def season_stats():
    """Skaters, goalies and teams for this season, keyed for the hub and the props model."""
    sk = {r["playerId"]: r for r in nhl_rows("skater", "summary")}
    for r in nhl_rows("skater", "realtime", "hits"):
        if r["playerId"] in sk:
            sk[r["playerId"]].update(hits=r.get("hits"), blockedShots=r.get("blockedShots"))
    gl = nhl_rows("goalie", "summary", "wins")
    tm = {r["teamId"]: r for r in nhl_rows("team", "summary")}
    for r in nhl_rows("team", "penalties", "penaltyMinutes"):
        if r["teamId"] in tm:
            tm[r["teamId"]].update(penaltyMinutes=r.get("penaltyMinutes"), penalties=r.get("penalties"), drawn=r.get("totalPenaltiesDrawn"))
    for r in nhl_rows("team", "realtime", "hits"):
        if r["teamId"] in tm:
            tm[r["teamId"]].update(hits=r.get("hits"), blockedShots=r.get("blockedShots"))
    return dict(sk=list(sk.values()), gl=gl, tm=list(tm.values()))
