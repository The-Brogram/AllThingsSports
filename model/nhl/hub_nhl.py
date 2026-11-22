"""Build the hub's NHL file, nhl.json (shape: briefs/nhl/hub_schema.md).

Contents: standings, the games for yesterday, today and the next two days with games, a game page for each, the model's numbers
(from output/slate_<date>.json), player and team stat leaders, and the track record (from output/review_*.json).

Called by season.py (build(hub_folder)) and by cloud_nhl.py on GitHub. Run by hand: ../venv/bin/python hub_nhl.py [folder]
"""
import glob
import json
import os
import sys
from datetime import datetime, timedelta
import nhl_data as ND

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
BASE = dict(home_win=.536, ot=.222, so=.073, fav_win=.610, fav_pl=.382, games=6918, span="2021-22 to Nov 19, 2026")
LONGEST = -400   # "most likely" lists hide prices longer than this (Josh, 2026-11-20)


def implied(o):
    o = float(str(o).replace("+", ""))
    return 100 / (o + 100) if o > 0 else -o / (-o + 100)


def novig(a, h):
    pa, ph = implied(a), implied(h)
    return round(pa / (pa + ph), 4), round(ph / (pa + ph), 4)


def standings():
    sd = ND.standings() or {}
    teams, stand = {}, []
    for conf in sd.get("children", []):
        C = {"conf": conf["name"].replace(" Conference", ""), "divs": []}
        for div in conf.get("children", []):
            rows = []
            for e in div["standings"]["entries"]:
                s = {}
                for x in e["stats"]:
                    s.setdefault(x.get("abbreviation", x["name"]), x.get("displayValue"))
                t = e["team"]
                a = ND.ab(t["abbreviation"])
                teams[a] = {"name": t["displayName"], "nick": t.get("shortDisplayName") or t["name"], "loc": t["location"],
                            "conf": C["conf"], "div": div["name"].replace(" Division", ""), "id": t["id"]}
                gi = lambda k: int(float(s.get(k) or 0))
                rows.append({"ab": a, "gp": gi("GP"), "w": gi("W"), "l": gi("L"), "otl": gi("OTL"), "pts": gi("PTS"), "rw": gi("RW"),
                             "row": gi("ROW"), "gf": gi("GF"), "ga": gi("GA"), "diff": gi("GF") - gi("GA"), "strk": s.get("STRK"),
                             "l10": (s.get("L10") or "").split(",")[0], "home": s.get("HOME"), "away": s.get("AWAY"), "road": s.get("AWAY"),
                             "ppct": round(gi("PTS") / (2 * gi("GP")), 3) if gi("GP") else None})
            C["divs"].append({"name": div["name"].replace(" Division", ""), "rows": rows})
        stand.append(C)
    return teams, stand


def side(c):
    t = c["team"]
    recs = {r.get("type", r.get("name")): r.get("summary") for r in c.get("records", [])}
    g = None
    for p in c.get("probables") or []:
        if p.get("name") == "probableStartingGoalie":
            g = {"n": p["athlete"]["displayName"], "st": p["status"]["name"], "src": "ESPN"}
    return {"ab": ND.ab(t["abbreviation"]), "score": c.get("score"), "rec": recs.get("total") or recs.get("ytd"),
            "ha": recs.get("home") if c["homeAway"] == "home" else recs.get("road"), "g": g,
            "ls": [l.get("displayValue", l.get("value")) for l in c.get("linescores", [])]}


def game_row(e, sm=None):
    c = e["competitions"][0]
    H = next(x for x in c["competitors"] if x["homeAway"] == "home")
    A = next(x for x in c["competitors"] if x["homeAway"] == "away")
    g = {"id": e["id"], "t": e["date"], "st": e["status"]["type"]["state"], "det": e["status"]["type"]["shortDetail"],
         "venue": (c.get("venue") or {}).get("fullName"), "h": side(H), "a": side(A),
         "tv": [n for b in c.get("broadcasts", []) if b.get("market") in (None, "national") for n in b.get("names", [])][:2]}
    o = (c.get("odds") or [None])[0]
    if o and o.get("moneyline"):
        m, ps, tt = o["moneyline"], o.get("pointSpread") or {}, o.get("total") or {}
        cl = lambda x: (x or {}).get("close") or {}
        op = lambda x: (x or {}).get("open") or {}
        g["odds"] = {"src": "DraftKings", "ml": [cl(m.get("away")).get("odds"), cl(m.get("home")).get("odds")],
                     "mlo": [op(m.get("away")).get("odds"), op(m.get("home")).get("odds")],
                     "pl": [[cl(ps.get("away")).get("line"), cl(ps.get("away")).get("odds")], [cl(ps.get("home")).get("line"), cl(ps.get("home")).get("odds")]],
                     "tot": [o.get("overUnder"), cl(tt.get("over")).get("odds"), cl(tt.get("under")).get("odds")]}
    if g["st"] == "post" and sm:
        pc = (sm.get("pickcenter") or [None])[0]
        if pc and (pc.get("awayTeamOdds") or {}).get("moneyLine") is not None:
            sg = lambda v: ("+" if v > 0 else "") + str(v)
            g["odds"] = {"src": "DraftKings (last ESPN line before puck drop)",
                         "ml": [sg(int(pc["awayTeamOdds"]["moneyLine"])), sg(int(pc["homeTeamOdds"]["moneyLine"]))],
                         "pl": [[sg(-pc["spread"]), sg(int(pc["awayTeamOdds"].get("spreadOdds") or 0))], [sg(pc["spread"]), sg(int(pc["homeTeamOdds"].get("spreadOdds") or 0))]],
                         "tot": [pc.get("overUnder"), sg(int(pc["overOdds"])) if pc.get("overOdds") else None, sg(int(pc["underOdds"])) if pc.get("underOdds") else None]}
        for t in sm["boxscore"]["teams"]:
            st = {x.get("label", x["name"]): x["displayValue"] for x in t["statistics"]}
            k = "h" if ND.ab(t["team"]["abbreviation"]) == g["h"]["ab"] else "a"
            g[k]["sog"] = st.get("Shots")
        for p in sm["boxscore"].get("players", []):
            k = "h" if ND.ab(p["team"]["abbreviation"]) == g["h"]["ab"] else "a"
            for grp in p["statistics"]:
                if grp.get("name") == "goalies":
                    gl = [dict(zip(grp["labels"], a["stats"])) | {"n": a["athlete"]["displayName"]} for a in grp["athletes"]]
                    g[k]["gp"] = [{"n": x["n"], "sv": x.get("SV"), "sa": x.get("SA"), "pct": x.get("SV%"), "toi": x.get("TOI")} for x in gl]
        hs, as_ = int(g["h"]["score"]), int(g["a"]["score"])
        extra = "OT" in g["det"] or "SO" in g["det"]
        g["res"] = {"h": "W" if hs > as_ else ("OTL" if extra else "L"), "a": "W" if as_ > hs else ("OTL" if extra else "L")}
    if g.get("odds") and g["odds"]["ml"][0] and g["odds"]["ml"][1]:
        pa, ph = novig(g["odds"]["ml"][0], g["odds"]["ml"][1])
        g["mk"] = {"a": pa, "h": ph}
    return g


def detail(eid, final):
    s = ND.summary(eid, final=final)
    if not s:
        return None
    out = {"team": {}, "goalies": {}, "leaders": {}, "last5": {}, "inj": {}}
    for t in (s.get("boxscore") or {}).get("teams", []):
        out["team"][ND.ab(t["team"]["abbreviation"])] = {x.get("label", x["name"]): x["displayValue"] for x in t.get("statistics", [])}
    gg = s.get("goalies") or {}
    for k in ("awayTeam", "homeTeam"):
        if k not in gg:
            continue
        rows = []
        for a in gg[k].get("athletes", []):
            st = {}
            for x in a.get("statistics", []):
                st.setdefault(x["abbreviation"], x["displayValue"])
            rows.append({"n": a["displayName"], "no": a.get("jersey"), "gp": st.get("GP"), "w": st.get("WINS"), "l": (st.get("L") or "").split(".")[0],
                         "otl": st.get("OTL"), "gaa": st.get("GAA"), "sv": st.get("SV%"), "so": st.get("SO")})
        out["goalies"][k[:-4]] = rows
    for L in s.get("leaders") or []:
        try:
            out["leaders"][ND.ab(L["team"]["abbreviation"])] = [[c["displayName"], c["leaders"][0]["athlete"]["displayName"], c["leaders"][0]["displayValue"]]
                                                                for c in L["leaders"] if c.get("leaders")]
        except (KeyError, IndexError):
            pass
    for L in s.get("lastFiveGames") or []:
        out["last5"][ND.ab(L["team"]["abbreviation"])] = [{"r": e.get("gameResult"), "s": e.get("score"), "v": e.get("atVs"),
                                                          "o": ND.ab(e["opponent"]["abbreviation"]), "d": et_day(e.get("gameDate"))} for e in L.get("events", [])]
    for I in s.get("injuries") or []:
        out["inj"][ND.ab(I["team"]["abbreviation"])] = [{"n": x["athlete"]["displayName"], "pos": (x["athlete"].get("position") or {}).get("abbreviation"),
                                                         "st": x.get("status"), "t": (x.get("details") or {}).get("type"),
                                                         "ret": (x.get("details") or {}).get("returnDate")} for x in I.get("injuries", [])]
    ss = (s.get("seasonseries") or [None])[0]
    if ss:
        out["series"] = {"sum": ss.get("summary"), "games": [{"d": e["date"][:10], "det": e["statusType"]["shortDetail"],
                         "c": [[ND.ab(c["team"]["abbreviation"]), c["homeAway"], c.get("score"), c.get("winner")] for c in e["competitors"]]}
                         for e in ss.get("events", [])]}
    if final:
        ids = {}
        for t in (s.get("boxscore") or {}).get("teams", []):
            ids[t["team"]["id"]] = ND.ab(t["team"]["abbreviation"])
        out["plays"] = [{"p": x["period"]["number"], "c": x["clock"]["displayValue"], "tx": x["text"], "as": x.get("awayScore"), "hs": x.get("homeScore"),
                         "tm": ids.get((x.get("team") or {}).get("id"), "")} for x in s.get("plays") or [] if x.get("scoringPlay")]
        sk = []
        for p in s["boxscore"].get("players", []):
            a_ = ND.ab(p["team"]["abbreviation"])
            for grp in p["statistics"]:
                if grp.get("name") in ("forwards", "defenses"):
                    for a in grp["athletes"]:
                        r = dict(zip(grp["labels"], a["stats"]))
                        num = lambda k: int(r.get(k) or 0)
                        sk.append({"n": a["athlete"]["displayName"], "tm": a_, "pos": "D" if grp["name"] == "defenses" else "F", "g": num("G"),
                                   "a": num("A"), "sog": num("S"), "toi": r.get("TOI"), "pm": r.get("+/-"), "ht": num("HT"), "bs": num("BS"),
                                   "pim": num("PIM")})
        out["sk"] = sk
        out["att"] = (s.get("gameInfo") or {}).get("attendance")
    return out


def et_day(iso):
    """ESPN game times are UTC: a 7 PM ET game is dated the next day there."""
    if not iso:
        return ""
    try:
        return f"{ND.to_et(iso[:16]):%Y-%m-%d}"
    except ValueError:
        return iso[:10]


def mmss(sec):
    sec = int(round(sec or 0))
    return f"{sec // 60}:{sec % 60:02d}"


def leaders(st, teams):
    by_name = {ND.name_key(v["name"]): k for k, v in teams.items()}
    sk = []
    for r in st["sk"]:
        if not r.get("gamesPlayed"):
            continue
        tm = ND.ab(str(r.get("teamAbbrevs") or "").split(",")[-1].strip())
        sk.append({"id": r["playerId"], "n": r["skaterFullName"], "tm": tm, "pos": r.get("positionCode"), "gp": r["gamesPlayed"], "g": r["goals"], "a": r["assists"],
                   "pts": r["points"], "pm": r.get("plusMinus"), "pim": r.get("penaltyMinutes"), "ppg": r.get("ppGoals"), "ppp": r.get("ppPoints"),
                   "shg": r.get("shGoals"), "gwg": r.get("gameWinningGoals"), "sog": r.get("shots"),
                   "shp": round(100 * r["shootingPct"], 1) if r.get("shootingPct") is not None else None,
                   "toi": mmss(r.get("timeOnIcePerGame")), "toi_s": round(r.get("timeOnIcePerGame") or 0),
                   "fo": round(100 * r["faceoffWinPct"], 1) if r.get("faceoffWinPct") else None, "hit": r.get("hits"), "blk": r.get("blockedShots")})
    gl = []
    for r in st["gl"]:
        if not r.get("gamesPlayed"):
            continue
        gl.append({"n": r["goalieFullName"], "tm": ND.ab(str(r.get("teamAbbrevs") or "").split(",")[-1].strip()), "gp": r["gamesPlayed"],
                   "gs": r.get("gamesStarted"), "w": r.get("wins"), "l": r.get("losses"), "otl": r.get("otLosses"),
                   "gaa": f"{r['goalsAgainstAverage']:.2f}" if r.get("goalsAgainstAverage") is not None else None,
                   "sv": f"{r['savePct']:.3f}".lstrip("0") if r.get("savePct") is not None else None, "so": r.get("shutouts"),
                   "sa": r.get("shotsAgainst"), "ga": r.get("goalsAgainst"),
                   "svs": (r["shotsAgainst"] - r["goalsAgainst"]) if r.get("shotsAgainst") is not None and r.get("goalsAgainst") is not None else r.get("saves")})
    tl = []
    for r in st["tm"]:
        a = by_name.get(ND.name_key(r["teamFullName"]))
        if not a:
            continue
        gp = r["gamesPlayed"] or 1
        tl.append({"ab": a, "gp": r["gamesPlayed"], "w": r["wins"], "l": r["losses"], "otl": r["otLosses"], "pts": r["points"],
                   "ppct": round(r["pointPct"], 3), "gf": r["goalsFor"], "ga": r["goalsAgainst"], "diff": r["goalsFor"] - r["goalsAgainst"],
                   "gfpg": round(r["goalsForPerGame"], 2), "gapg": round(r["goalsAgainstPerGame"], 2),
                   "pp": round(100 * r["powerPlayPct"], 1) if r.get("powerPlayPct") is not None else None,
                   "pk": round(100 * r["penaltyKillPct"], 1) if r.get("penaltyKillPct") is not None else None,
                   "sfpg": round(r["shotsForPerGame"], 1), "sapg": round(r["shotsAgainstPerGame"], 1),
                   "fo": round(100 * r["faceoffWinPct"], 1) if r.get("faceoffWinPct") else None, "pim": r.get("penaltyMinutes"),
                   "pimpg": round((r.get("penaltyMinutes") or 0) / gp, 1), "pen": r.get("penalties"), "drawn": r.get("drawn"),
                   "rw": r.get("winsInRegulation"), "sho": r.get("teamShutouts"), "hit": r.get("hits"), "blk": r.get("blockedShots")})
    return sk, gl, tl


def record():
    rows, nights = [], 0
    for f in sorted(glob.glob(os.path.join(OUT, "review_*.json"))):
        nights += 1
        rows += json.load(open(f))["rows"]
    if not rows:
        return None
    since = os.path.basename(sorted(glob.glob(os.path.join(OUT, "review_*.json")))[0])[7:17]
    # one row per side: the latest run before puck drop (pregame if there was one)
    best = {}
    for r in rows:
        k = (r["id"], r["mk"], r.get("who"), r.get("line"))
        if k not in best or r["run"] == "pregame":
            best[k] = r
    rr = list(best.values())
    ml = [r for r in rr if r["mk"] == "ml" and r["p"] > 0.5 and r["hit"] is not None]
    bands = []
    for lo, hi in ((.5, .55), (.55, .6), (.6, .65), (.65, .7), (.7, 1.01)):
        b = [r for r in ml if lo <= r["p"] < hi]
        if b:
            bands.append(dict(lo=lo, hi=min(hi, 1), n=len(b), hit=round(sum(r["hit"] for r in b) / len(b), 3), avg_p=round(sum(r["p"] for r in b) / len(b), 3)))
    bets = {}
    for name, keys in (("ml", ("ml",)), ("pl", ("pl",)), ("tot", ("tot_o", "tot_u")), ("props", ("goal", "ast", "pts", "sog", "saves"))):
        b = [r for r in rr if r["mk"] in keys and r["hit"] is not None and r.get("nv") is not None and r["p"] - r["nv"] >= 0.03]
        if b:
            u = sum(r["profit"] for r in b)
            bets[name] = dict(n=len(b), units=round(u, 2), roi=round(u / len(b), 4), hit=round(sum(r["hit"] for r in b) / len(b), 3))
    return dict(since=since, nights=nights, ml=dict(n=len(ml), hit=round(sum(r["hit"] for r in ml) / len(ml), 3) if ml else None,
                                                   avg_p=round(sum(r["p"] for r in ml) / len(ml), 3) if ml else None),
                bands=bands, bets=bets, rule="a side counts as a bet when the model's chance is at least 3 points above the no-vig price")


def shown_model(m):
    """What the page shows (skeptic review, 2026-11-20). The win and over chances are DraftKings' no-vig price exactly (the
    simulation only matches them to within noise), so no moneyline or total edge is shown. The overtime/shootout split was refit on 2021-24
    and checked on 2024-26 (shootouts 7.1% vs 7.5% actual). The puck line shown is the market's."""
    m = json.loads(json.dumps(m))
    nv = m.get("ml", {}).get("nv_h")
    if nv is not None:
        m["ph"], m["pa"] = round(nv, 4), round(1 - nv, 4)
    m["ml"] = {"nv_h": nv}
    t = m.get("tot")
    if t and t.get("nv_o") is not None:
        t["po"], t["pu"] = t["nv_o"], round(1 - t["nv_o"], 4)
        t.pop("nv_o")
    if m.get("pl"):
        m["pl"].pop("edge_fav", None)
        m["pl"].pop("nv_fav", None)
    return m


def shown_props(rows):
    """Props v2 (backtested at real DraftKings prices, backtest_props.py): the chance is the fitted blend, the edge sits beside it.
    Rows from a v1 run (no blend weight on file) show DraftKings' chance only."""
    out = []
    for r in rows:
        if "w" not in r:
            r = {k: v for k, v in r.items() if k not in ("p", "p_model", "edge")}
        out.append(r)
    return out


def build(hub):
    now = ND.et_now()
    today = now.date()
    teams, stand = standings()
    days, det, sched = {}, {}, {}
    shown = [today - timedelta(days=1), today]
    d = today + timedelta(days=1)
    while len(shown) < 4 and d <= today + timedelta(days=10):
        shown.append(d)
        d += timedelta(days=1)
    evs = {}
    for dd in [shown[0] - timedelta(days=1)] + shown:
        k = f"{dd:%Y-%m-%d}"
        evs[k] = [e for e in ND.scoreboard(k) if e.get("season", {}).get("type") in (2, 3)
                  and all(ND.ab(c["team"]["abbreviation"]) in ND.CLUBS for c in e["competitions"][0]["competitors"])]
        sched[k] = {ND.ab(c["team"]["abbreviation"]) for e in evs[k] for c in e["competitions"][0]["competitors"]}
    # keep yesterday, today, then up to two more days that have games
    keep = [shown[0], shown[1]] + [x for x in shown[2:] if evs[f"{x:%Y-%m-%d}"]][:2]
    nxt = next((f"{x:%Y-%m-%d}" for x in shown[1:] if evs[f"{x:%Y-%m-%d}"]), None)
    for dd in keep:
        k = f"{dd:%Y-%m-%d}"
        prev = f"{dd - timedelta(days=1):%Y-%m-%d}"
        slate = {}
        sf = os.path.join(OUT, f"slate_{k}.json")
        if os.path.exists(sf):
            slate = json.load(open(sf)).get("games", {})
        L = []
        for e in evs[k]:
            final = e["status"]["type"]["state"] == "post"
            dt = detail(e["id"], final)
            g = game_row(e, dt and final and ND.summary(e["id"], final=True))
            if dt:
                det[e["id"]] = dt
            if prev in sched:
                g["b2b"] = {"h": g["h"]["ab"] in sched[prev], "a": g["a"]["ab"] in sched[prev]}
            sl = slate.get(e["id"])
            if sl:
                for s_ in ("h", "a"):
                    if (sl.get("goalies") or {}).get(s_) and g["st"] == "pre":
                        g[s_]["g"] = sl["goalies"][s_]
                o = sl.get("odds") or {}
                if not g.get("odds") and o.get("ml_a") and o.get("ml_h"):
                    # ESPN drops the line once a game starts: use the DraftKings line saved at the model's pre-game run
                    g["odds"] = {"src": "DraftKings (line at the model's pre-game run)", "ml": [o["ml_a"], o["ml_h"]],
                                 "pl": [[o.get("pl_a"), o.get("pl_a_p")], [o.get("pl_h"), o.get("pl_h_p")]],
                                 "tot": [o.get("total"), o.get("over"), o.get("under")]}
                    pa, ph = novig(o["ml_a"], o["ml_h"])
                    g["mk"] = {"a": pa, "h": ph}
                if sl.get("m"):
                    g["m"] = shown_model(sl["m"])
                if sl.get("props"):
                    g["props"] = shown_props(sl["props"])
            L.append(g)
        days[k] = sorted(L, key=lambda x: x["t"])
    sk, gl, tl = leaders(ND.season_stats(), teams)
    # one set of goal totals on every page: NHL.com's, which leave out the shootout winner's extra goal (ESPN's standings count it)
    by = {t["ab"]: t for t in tl}
    for C in stand:
        for dv in C["divs"]:
            for r in dv["rows"]:
                t = by.get(r["ab"])
                if t:
                    r.update(gf=t["gf"], ga=t["ga"], diff=t["gf"] - t["ga"])
    out = {"built": f"{now:%b %-d, %Y %-I:%M %p} ET", "season": "2026-27", "teams": teams, "stand": stand, "days": days, "next": nxt,
           "det": det, "sk": sk, "gl": gl, "tl": tl, "base": BASE, "longest": LONGEST, "props_model": True,
           "credit": "Data: ESPN, NHL.com, DailyFaceoff. Lines: DraftKings via ESPN. Goals for/against exclude shootout goals (NHL.com)."}
    rec = record()
    if rec:
        out["record"] = rec
    os.makedirs(hub, exist_ok=True)
    json.dump(out, open(os.path.join(hub, "nhl.json"), "w"), separators=(",", ":"))
    return {"nhl.json": os.path.abspath(os.path.join(hub, "nhl.json"))}


if __name__ == "__main__":
    print(build(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "output", "hub")))
