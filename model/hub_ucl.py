"""Champions League pages for the hub: ucl.json (season) and uclh.json (head-to-head history), loaded on demand.

Two sources:
- ESPN, fetched wherever this runs (this computer or GitHub): every 2026-27 fixture and result, the league-phase
  table, box scores and player stats for finished matches, each club's results in all competitions (home/away
  splits), and every Champions League result since 2003 for head-to-head history. Kept in ucl/cache so finished
  matches and past seasons are fetched once.
- The Champions League model (TheBrogram/soccer, this computer only): each analysed matchday's export
  (soccer/web/matchups_<date>.json), kept in ucl/model.json so GitHub's runs still publish it.
"""
import glob
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.abspath(__file__))
SOCCER = os.path.join(os.path.dirname(ROOT), "soccer", "web")
UCL = os.path.join(ROOT, "ucl")
CACHE = os.path.join(UCL, "cache")
MODEL = os.path.join(UCL, "model.json")
SITE = "https://site.api.espn.com/apis/site/v2/sports/soccer/"
SEASON, START, END = 2026, "2026-08-27", "2027-07-01"
# league-phase matchdays by date (ESPN lists the dates; UEFA numbers them)
MD = {"2026-09-08": 1, "2026-09-09": 1, "2026-09-10": 1, "2026-10-13": 2, "2026-10-14": 2, "2026-10-20": 3, "2026-10-21": 3,
      "2026-11-03": 4, "2026-11-04": 4, "2026-11-24": 5, "2026-11-25": 5, "2026-12-08": 6, "2026-12-09": 6,
      "2027-01-19": 7, "2027-01-20": 7, "2027-01-27": 8}
STAGES = [("2027-02-15", "Knockout playoffs"), ("2027-03-08", "Round of 16"), ("2027-04-05", "Quarterfinals"),
          ("2027-04-26", "Semifinals"), ("2027-06-01", "Final")]


def get(url):
    try:
        raw = subprocess.run(["curl", "-s", "--compressed", "--max-time", "25", url], capture_output=True, text=True, timeout=40).stdout
        return json.loads(raw)
    except Exception as ex:  # noqa: BLE001
        print("UCL fetch failed:", url, repr(ex))
        return None


def kept(name, max_age, make):
    """A compact result kept in ucl/cache. max_age None = keep for good. If a refresh fails, the kept copy is used."""
    path = os.path.join(CACHE, name)
    have = os.path.exists(path)
    if have and (max_age is None or time.time() - os.path.getmtime(path) < max_age):
        return json.load(open(path))
    try:
        data = make()
    except Exception as ex:  # noqa: BLE001
        print("UCL refresh failed:", name, repr(ex))
        data = None
    if data is None:
        return json.load(open(path)) if have else None
    os.makedirs(os.path.dirname(path), exist_ok=True)
    txt = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
    if not have or open(path).read() != txt:
        open(path, "w").write(txt)
    else:
        os.utime(path)
    return data


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def local_day(iso):
    """ESPN times are UTC; matchdays are European evenings, so the UTC date is the match date."""
    return iso[:10]


def stage(day):
    if day in MD:
        return MD[day], f"Matchday {MD[day]}"
    lab = "League phase"
    for d, name in STAGES:
        if day >= d:
            lab = name
    return None, lab


def fixtures():
    return kept("fixtures.json", 1800, _fixtures) or []


def _fixtures():
    out = []
    for y in (2026, 2027):
        d = get(SITE + f"uefa.champions/scoreboard?dates={y}&limit=1000")
        if not d or "events" not in d:
            return None
        for e in d["events"]:
            if not (START <= e["date"][:10] < END):
                continue
            c = e["competitions"][0]
            t = {x["homeAway"]: x for x in c["competitors"]}
            st = e["status"]["type"]
            day = local_day(e["date"])
            n, lab = stage(day)
            f = {"id": e["id"], "date": e["date"], "day": day, "md": n, "stage": lab,
                 "h": {"id": t["home"]["team"]["id"], "name": t["home"]["team"]["displayName"], "ab": t["home"]["team"].get("abbreviation", "")},
                 "a": {"id": t["away"]["team"]["id"], "name": t["away"]["team"]["displayName"], "ab": t["away"]["team"].get("abbreviation", "")},
                 "st": st.get("state"), "fin": bool(st.get("completed")), "detail": st.get("shortDetail", ""),
                 "venue": (c.get("venue") or {}).get("fullName", "")}
            if st.get("state") != "pre":
                f["hs"], f["as"] = int(num(t["home"].get("score")) or 0), int(num(t["away"].get("score")) or 0)
            o = (c.get("odds") or [None])[0]
            if o and st.get("state") == "pre":
                M = o.get("moneyline") or {}
                cl = lambda k: ((M.get(k) or {}).get("close") or {}).get("odds")
                tot = o.get("total") or {}
                ou = lambda k: ((tot.get(k) or {}).get("close") or {}).get("odds")
                f["odds"] = {"ml": [cl("home"), cl("draw"), cl("away")], "ou": o.get("overUnder"), "o": ou("over"), "u": ou("under"),
                             "by": (o.get("provider") or {}).get("name", "")}
            out.append(f)
    out.sort(key=lambda f: (f["date"], f["h"]["name"]))
    return out


def table():
    return kept("table.json", 1800, _table) or []


def _table():
    d = get(SITE.replace("/site/v2/", "/v2/") + "uefa.champions/standings")
    if not d or "children" not in d:
        return None
    rows = []
    for ch in d["children"]:
        for e in ch["standings"]["entries"]:
            s = {x["name"]: x.get("value") for x in e["stats"]}
            rows.append({"rank": int(s.get("rank") or 0), "id": e["team"]["id"], "name": e["team"]["displayName"],
                         "ab": e["team"].get("abbreviation", ""), "p": int(s.get("gamesPlayed") or 0), "w": int(s.get("wins") or 0),
                         "d": int(s.get("ties") or 0), "l": int(s.get("losses") or 0), "gf": int(s.get("pointsFor") or 0),
                         "ga": int(s.get("pointsAgainst") or 0), "pts": int(s.get("points") or 0),
                         "note": (e.get("note") or {}).get("description", "")})
    return sorted(rows, key=lambda r: r["rank"])


TEAM_KEYS = {"possessionPct": "poss", "totalShots": "sh", "shotsOnTarget": "sot", "wonCorners": "cor", "foulsCommitted": "fouls",
             "yellowCards": "yc", "redCards": "rc", "saves": "sv", "passPct": "pass", "accuratePasses": "apass", "totalPasses": "tpass",
             "effectiveTackles": "tkl", "interceptions": "int", "offsides": "off"}
PLAYER_KEYS = {"totalGoals": "g", "goalAssists": "a", "totalShots": "sh", "shotsOnTarget": "sot", "yellowCards": "yc",
               "redCards": "rc", "saves": "sv", "foulsCommitted": "fc", "foulsSuffered": "fs"}


def box(fx):
    """Team and player stats for one finished match (ESPN summary, cached for good once the match is final)."""
    path = os.path.join(CACHE, "box", fx["id"] + ".json")
    if os.path.exists(path):
        return json.load(open(path))
    d = get(SITE + f"uefa.champions/summary?event={fx['id']}")
    if not d or "boxscore" not in d:
        return None
    out = {"t": {}, "p": {}}
    for t in d["boxscore"].get("teams", []):
        out["t"][t["team"]["id"]] = {TEAM_KEYS[s["name"]]: num(s.get("displayValue")) for s in t.get("statistics", []) if s["name"] in TEAM_KEYS}
    for r in d.get("rosters", []):
        tid, L = r["team"]["id"], []
        for p in r.get("roster", []):
            st = {x["name"]: x.get("value") for x in p.get("stats", [])}
            if not st.get("appearances") and not p.get("starter"):
                continue
            row = {"n": p["athlete"]["displayName"], "id": p["athlete"].get("id"), "pos": (p.get("position") or {}).get("abbreviation", ""),
                   "st": 1 if p.get("starter") else 0}
            row.update({v: int(st.get(k) or 0) for k, v in PLAYER_KEYS.items() if st.get(k)})
            L.append(row)
        out["p"][tid] = L
    if fx["fin"]:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        json.dump(out, open(path, "w"), separators=(",", ":"))
    return out


def split_ucl(fx_list, boxes):
    """Champions League totals per club, overall and at home / away, plus season player totals."""
    T = defaultdict(lambda: {k: defaultdict(float) for k in ("all", "home", "away")})
    P = defaultdict(dict)
    for f in fx_list:
        if not f["fin"] or f["id"] not in boxes:
            continue
        b = boxes[f["id"]]
        for side, opp, ha in (("h", "a", "home"), ("a", "h", "away")):
            tid, oid = f[side]["id"], f[opp]["id"]
            gf, ga = f["hs" if side == "h" else "as"], f["as" if side == "h" else "hs"]
            mine, theirs = b["t"].get(tid, {}), b["t"].get(oid, {})
            for k in ("all", ha):
                a = T[tid][k]
                a["gp"] += 1; a["gf"] += gf; a["ga"] += ga
                a["w"] += gf > ga; a["d"] += gf == ga; a["l"] += gf < ga
                a["cs"] += ga == 0; a["fts"] += gf == 0; a["o25"] += gf + ga > 2.5; a["btts"] += gf > 0 and ga > 0
                for s, v in mine.items():
                    if v is not None:
                        a[s] += v; a["n_" + s] += 1
                for s in ("sh", "sot", "cor"):
                    if theirs.get(s) is not None:
                        a[s + "a"] += theirs[s]; a["n_" + s + "a"] += 1
            for p in b["p"].get(tid, []):
                key = p.get("id") or p["n"]
                q = P[tid].setdefault(key, {"n": p["n"], "pos": p["pos"], "app": 0, "st": 0})
                q["app"] += 1; q["st"] += p["st"]
                if p["pos"] and p["pos"] != "SUB":
                    q["pos"] = p["pos"]
                for k in PLAYER_KEYS.values():
                    if p.get(k):
                        q[k] = q.get(k, 0) + p[k]
    teams = {}
    for tid, parts in T.items():
        teams[tid] = {}
        for k, a in parts.items():
            if not a.get("gp"):
                continue
            o = {x: int(a[x]) for x in ("gp", "w", "d", "l", "gf", "ga", "cs", "fts", "o25", "btts")}
            for s in list(TEAM_KEYS.values()) + ["sha", "sota", "cora"]:
                if a.get("n_" + s):
                    o[s] = round(a[s] / a["n_" + s], 2)
            teams[tid][k] = o
    players = {tid: sorted(v.values(), key=lambda q: (-q.get("g", 0), -q.get("a", 0), -q.get("sot", 0))) for tid, v in P.items()}
    return teams, players


def all_comps(team_ids, names):
    """Each club's results in every competition this season, for home / away splits (ESPN team schedules, refreshed every 6 hours)."""
    return {tid: kept(f"sched/{tid}.json", 6 * 3600, lambda tid=tid: _sched(tid)) or [] for tid in team_ids}


def _sched(tid):
    d = get(SITE + f"all/teams/{tid}/schedule?season={SEASON}")
    if not d or "events" not in d:
        return None
    if True:
        games = []
        for e in d["events"]:
            c = e["competitions"][0]
            if not c["status"]["type"].get("completed"):
                continue
            t = {x["homeAway"]: x for x in c["competitors"]}
            if "home" not in t or "away" not in t:
                continue
            me = "home" if t["home"]["team"]["id"] == tid else "away"
            op = "away" if me == "home" else "home"
            sc = lambda x: int(num((x.get("score") or {}).get("value") if isinstance(x.get("score"), dict) else x.get("score")) or 0)
            games.append({"date": e["date"][:10], "ha": me, "opp": t[op]["team"]["displayName"], "gf": sc(t[me]), "ga": sc(t[op]),
                          "comp": (e.get("league") or {}).get("abbreviation") or (e.get("league") or {}).get("name", "")})
        games.sort(key=lambda g: g["date"])
        return games


def history():
    """Every Champions League result ESPN has (2003 on), compact, for head-to-head lookups. Past years are fetched once."""
    rows, this = [], datetime.now().year
    for y in range(2003, this + 1):
        rows += kept(f"hist/{y}.json", None if y < this - 1 else 6 * 3600, lambda y=y: _hist(y)) or []
    return sorted({r[0] + r[1] + r[2]: r for r in rows}.values())


def _hist(y):
    rows = []
    d = get(SITE + f"uefa.champions/scoreboard?dates={y}&limit=1000")
    if not d or "events" not in d:
        return None
    if True:
        for e in d["events"]:
            c = e["competitions"][0]
            if not c["status"]["type"].get("completed"):
                continue
            t = {x["homeAway"]: x for x in c["competitors"]}
            if "home" not in t or "away" not in t:
                continue
            s = e.get("season") or {}
            slug = (s.get("slug") or "").replace("-", " ")
            yr = s.get("year")
            rows.append([e["date"][:10], t["home"]["team"]["id"], t["away"]["team"]["id"], int(num(t["home"].get("score")) or 0),
                         int(num(t["away"].get("score")) or 0), t["home"]["team"]["displayName"], t["away"]["team"]["displayName"],
                         f"{yr}-{str(yr + 1)[2:]} {slug}".strip() if yr else slug])
    return rows


def model_pages():
    """The soccer model's analysed matchdays (this computer only); otherwise the kept copy."""
    old = json.load(open(MODEL)) if os.path.exists(MODEL) else {"days": []}
    src = sorted(glob.glob(os.path.join(SOCCER, "matchups_*.json")))
    if not src:
        return old
    days = []
    for f in src:
        m = re.search(r"matchups_(\d{4}-\d{2}-\d{2})\.json$", f)
        if not m:
            continue
        page = json.load(open(f))
        if isinstance(page, list):  # the first export was a bare list of matches, with no label or pick
            page = {"date": datetime.strptime(m.group(1), "%Y-%m-%d").strftime("%A %-d %B %Y"), "pick": None, "matches": page}
        days.append({"day": m.group(1), "label": page["date"], "pick": page.get("pick"), "matches": page["matches"]})
    data = {"days": days}
    os.makedirs(UCL, exist_ok=True)
    json.dump(data, open(MODEL, "w"), separators=(",", ":"))
    return data


def build(hub):
    os.makedirs(CACHE, exist_ok=True)
    fx = fixtures()
    if not fx:
        print("UCL: no fixtures from ESPN; keeping the last pages")
        return {k: os.path.join(hub, k) for k in ("ucl.json", "uclh.json") if os.path.exists(os.path.join(hub, k))}
    boxes = {}
    for f in fx:
        if f["st"] != "pre":
            b = box(f)
            if b:
                for t in b["t"].values():  # ESPN rounds pass accuracy to one decimal; use completed / attempted instead
                    if t.get("apass") and t.get("tpass"):
                        t["pass"] = round(t["apass"] / t["tpass"], 3)
                boxes[f["id"]] = b
    for f in fx:  # this match's own team stats, for the match page
        b = boxes.get(f["id"])
        if b and f["fin"]:
            f["bx"] = {"h": b["t"].get(f["h"]["id"], {}), "a": b["t"].get(f["a"]["id"], {})}
    teams, players = split_ucl(fx, boxes)
    tab = table()
    ids = sorted({f[s]["id"] for f in fx for s in "ha"})
    names = {f[s]["id"]: f[s]["name"] for f in fx for s in "ha"}
    sched = all_comps(ids, names)
    mod = model_pages()
    picks = {}
    for d in mod["days"]:
        for m in d["matches"]:
            picks[m["id"]] = d["day"]
    data = {"built": datetime.now().strftime("%Y-%m-%d %H:%M"), "season": "2026-27", "table": tab, "fx": fx,
            "teams": {tid: {"name": names.get(tid, tid), "ucl": teams.get(tid, {}), "players": players.get(tid, [])[:30],
                            "games": sched.get(tid, [])} for tid in ids},
            "model": mod["days"]}
    out = os.path.join(hub, "ucl.json")
    json.dump(data, open(out, "w"), separators=(",", ":"), ensure_ascii=False)
    hist = os.path.join(hub, "uclh.json")
    json.dump(history(), open(hist, "w"), separators=(",", ":"), ensure_ascii=False)
    fin = sum(f["fin"] for f in fx)
    print(f"UCL: {len(fx)} fixtures ({fin} final, {len(boxes)} box scores), {len(tab)} in the table, {len(mod['days'])} analysed matchdays")
    return {"ucl.json": out, "uclh.json": hist}


if __name__ == "__main__":
    print(build(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "output", "hub")))
