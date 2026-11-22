"""Home page board: every sport's games over the next day and a half, with the chance each team wins.

Writes output/hub/tonight.json from data the hub build already made (NFL games from season.py, plus
cfb.json, nba.json and nhl.json). The home page picks the upcoming games it shows from this small file,
so it does not have to download every sport's full data. Chances are the model's where it has one,
otherwise the market's no-vig price (NHL games before the model runs); each row says which.
Champions League matches are left out: they have three outcomes, so one win chance would mislead.
"""
import json
import os
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")


def _load(hub, name):
    p = os.path.join(hub, name)
    return json.load(open(p)) if os.path.exists(p) else None


def _iso(dt):
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")


def _p(v):
    try:
        v = float(v)
        return round(v, 4) if 0 < v < 1 else None
    except (TypeError, ValueError):
        return None


def build(hub, nfl):
    now = datetime.now(timezone.utc)
    lo, hi = now - timedelta(hours=4), now + timedelta(hours=40)
    rows = []

    def add(sport, a, h, t, ph, href, src="model", ra=None, rh=None):
        ph = _p(ph)
        if ph is None or not t:
            return
        dt = datetime.fromisoformat(t.replace("Z", "+00:00"))
        if lo <= dt <= hi:
            rows.append(dict(s=sport, a=a, h=h, t=_iso(dt), ph=ph, href=href, src=src, ra=ra, rh=rh))

    for g in (nfl or {}).get("games", []):
        if g.get("status") == "final" or not g.get("date") or not g.get("time"):
            continue
        try:
            dt = datetime.strptime(g["date"] + g["time"], "%Y%m%d%H:%M").replace(tzinfo=ET)
        except ValueError:
            continue
        add("nfl", g["away"], g["home"], _iso(dt), g.get("ph"), "#game-" + str(g["id"]).replace("_", "-").lower())

    cfb = _load(hub, "cfb.json") or {}
    for g in cfb.get("games", []):
        if g.get("done") or g.get("live") or g.get("pw") is None:
            continue
        pw = float(g["pw"])
        add("cfb", g["a"], g["h"], g.get("kick"), pw if g.get("pick") == g["h"] else 1 - pw, "#cfb-game-" + str(g["id"]), ra=g.get("ra"), rh=g.get("rh"))

    nba = (_load(hub, "nba.json") or {}).get("tonight") or {}
    for g in nba.get("games", []):
        if g.get("state") == "pre":
            add("nba", g["away"], g["home"], g.get("t"), g.get("ph"), "#nba-game-" + str(g["id"]))

    nhl = _load(hub, "nhl.json") or {}
    for day in (nhl.get("days") or {}).values():
        for g in day or []:
            if g.get("st") != "pre":
                continue
            m, mk = g.get("m") or {}, g.get("mk") or {}
            ph, src = (m.get("ph"), "model") if m.get("ph") is not None else (mk.get("h"), "market")
            add("nhl", (g.get("a") or {}).get("ab"), (g.get("h") or {}).get("ab"), g.get("t"), ph, "#nhl-game-" + str(g["id"]), src)

    rows.sort(key=lambda r: r["t"])
    out = dict(built=_iso(now), games=rows)
    path = os.path.join(hub, "tonight.json")
    json.dump(out, open(path, "w"), separators=(",", ":"))
    return {"tonight.json": path}


if __name__ == "__main__":
    import sys
    hub = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "hub")
    print(build(hub, None))
