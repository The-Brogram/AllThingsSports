"""Turn ESPN game JSON into games.csv, plays.csv, players.csv."""
import glob
import json
import os
import re
import pandas as pd

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
# two ESPN text formats: "#10 D.Mensah pass complete short right to #1 M.Toney" and "Miller Moss pass complete to Chris Bell"
PASS = re.compile(r"pass (complete|incomplete)(?: (short|deep))?(?: (left|middle|right))?(?: to (?:#(\d+) )?([A-Z][\w.'\- ]*?)(?= caught| thrown| for|,|$))?")
RUSH = re.compile(r"(?:#(\d+) )?([A-Z][\w.'\- ]*?) (?:rush|run|scramble|kneel)")
PLAYS = {"Rush", "Rushing Touchdown", "Pass Reception", "Pass Incompletion", "Passing Touchdown", "Sack", "Pass Interception Return", "Interception Return Touchdown", "Fumble Recovery (Opponent)", "Fumble Return Touchdown"}


def abbr(name):
    p = name.replace(".", " ").split()
    return (p[0][0] + "." + p[1]).lower() if len(p) > 1 else name.lower()


def parse(f):
    d = json.load(open(f))
    h = d.get("header", {})
    c = (h.get("competitions") or [{}])[0]
    if not c.get("status", {}).get("type", {}).get("completed"):
        return None
    cs = c.get("competitors", [])
    if len(cs) != 2:
        return None
    home = [x for x in cs if x["homeAway"] == "home"][0]
    away = [x for x in cs if x["homeAway"] == "away"][0]
    pc = (d.get("pickcenter") or [{}])
    pc = next((p for p in pc if p.get("provider", {}).get("id") == "100"), pc[0])
    spread = None
    m = re.match(r"([A-Z&]+) -?([\d.]+)", pc.get("details") or "")
    ha, aa = home["team"].get("abbreviation"), away["team"].get("abbreviation")
    if m:
        spread = float(m.group(2)) * (1 if m.group(1) == ha else -1)
    elif (pc.get("details") or "").upper() == "EVEN":
        spread = 0.0
    gid = int(h["id"])
    g = dict(game_id=gid, season=h["season"]["year"], stype=h["season"]["type"], week=h.get("week"), date=c.get("date"), neutral=bool(c.get("neutralSite")), home=ha, away=aa, home_id=home["team"]["id"],
             away_id=away["team"]["id"], hs=float(home.get("score") or 0), aw=float(away.get("score") or 0), spread_line=spread, total_line=pc.get("overUnder"),
             venue=d.get("gameInfo", {}).get("venue", {}).get("fullName"), attendance=d.get("gameInfo", {}).get("attendance"))
    id2ab = {str(home["team"]["id"]): ha, str(away["team"]["id"]): aa}
    players, keymap = {}, {}

    def P(t, a):
        i = a["id"]
        k = (t, i)
        if k not in players:
            players[k] = dict(game_id=gid, team=t, pid=int(i), name=a.get("displayName"), jersey=a.get("jersey"))
            # first in wins: passing/rushing/receiving tables come before defense, so a defender with the same initials cannot take a receiver's targets
            keymap.setdefault((t, abbr(a.get("displayName", ""))), i)
            keymap.setdefault((t, str(a.get("jersey"))), i)
        return players[k]
    for tb in d.get("boxscore", {}).get("players", []):
        t = id2ab.get(str(tb["team"]["id"]), tb["team"].get("abbreviation"))
        for st in tb.get("statistics", []):
            keys = st.get("keys", [])
            for x in st.get("athletes", []):
                row, v = P(t, x["athlete"]), dict(zip(keys, x.get("stats", [])))
                num = lambda k: float(v.get(k, 0) or 0) if str(v.get(k, "0")).replace(".", "").replace("-", "").isdigit() else 0.0
                if st["name"] == "passing":
                    a, b = (v.get("completions/passingAttempts", "0/0").split("/") + ["0"])[:2]
                    row.update(comp=float(a), att=float(b), pass_y=num("passingYards"), pass_td=num("passingTouchdowns"), ints=num("interceptions"))
                elif st["name"] == "rushing":
                    row.update(car=num("rushingAttempts"), rush_y=num("rushingYards"), rush_td=num("rushingTouchdowns"))
                elif st["name"] == "receiving":
                    row.update(rec=num("receptions"), rec_y=num("receivingYards"), rec_td=num("receivingTouchdowns"))
                elif st["name"] == "defensive":
                    row.update(tackles=num("totalTackles"), sacks=num("sacks"), tfl=num("tacklesForLoss"), pd=num("passesDefended"), hur=num("hurries"))
    plays = []
    for dr in d.get("drives", {}).get("previous", []):
        dres = dr.get("result") or dr.get("displayResult")
        did = dr.get("id")
        for p in dr.get("plays", []):
            ty = p.get("type", {}).get("text", "")
            if ty not in PLAYS:
                continue
            part = p.get("teamParticipants") or []
            off = next((id2ab.get(str(x.get("id"))) for x in part if x.get("type") == "offense"), None)
            if off is None:
                off = id2ab.get(str(dr.get("team", {}).get("id"))) or dr.get("team", {}).get("abbreviation")
            de = aa if off == ha else ha
            s = p.get("start", {})
            txt = p.get("text", "")
            isp = bool(re.search(r"pass |sacked", txt)) and "rush" not in ty.lower()
            hsc, asc = p.get("homeScore", 0), p.get("awayScore", 0)
            r = dict(game_id=gid, drive=did, off=off, de=de, ty=ty, down=s.get("down"), dist=s.get("distance"), yl=s.get("yardsToEndzone"), yds=p.get("statYardage", 0), qtr=p.get("period", {}).get("number"),
                     sd=(hsc - asc) if off == ha else (asc - hsc), td=int(p.get("scoringPlay", False) and "ouchdown" in ty), sack=int(ty == "Sack" or "sacked" in txt), pass_play=int(isp), dres=dres, scramble=0)
            mm = PASS.search(txt)
            if mm:
                r.update(comp=int(mm.group(1) == "complete"), depth=mm.group(2), side=mm.group(3))
                if mm.group(5):
                    k = keymap.get((off, mm.group(4))) or keymap.get((off, abbr(mm.group(5))))  # jersey first: it is unique on a team
                    r["tgt"] = k
                    r["tgt_name"] = mm.group(5)
            if not isp:
                mr = RUSH.search(txt)
                sd_ = re.search(r"(?:rush|run) (left|middle|right)", txt)
                if sd_:
                    r["rside"] = sd_.group(1)
                if mr:
                    r["rusher"] = keymap.get((off, mr.group(1))) or keymap.get((off, abbr(mr.group(2))))
            if "scramble" in txt:
                r["scramble"] = 1
            plays.append(r)
    return g, list(players.values()), plays


def main():
    G, P, L = [], [], []
    for f in glob.glob(f"{D}/games/*.json"):
        try:
            x = parse(f)
        except (ValueError, KeyError, IndexError, TypeError) as e:
            print("skip", f, repr(e)[:80])
            continue
        if x:
            G.append(x[0]); P += x[1]; L += x[2]
    g, p, l = pd.DataFrame(G), pd.DataFrame(P), pd.DataFrame(L)
    g = g[g.season.isin([2025, 2026])]
    g.to_csv(f"{D}/games.csv", index=False)
    l.to_csv(f"{D}/plays.csv.gz", index=False)
    tg = l[l.tgt.notna()].groupby(["game_id", "off", "tgt"]).agg(targets=("yds", "size"), rz_tgt=("yl", lambda s: int((s <= 20).sum())), deep_t=("depth", lambda s: int((s == "deep").sum()))).reset_index().rename(columns={"off": "team", "tgt": "pid"})
    tg["pid"] = tg.pid.astype(float).astype(int)
    ru = l[l.rusher.notna()].groupby(["game_id", "off", "rusher"]).agg(i10_car=("yl", lambda s: int((s <= 10).sum())), i5_car=("yl", lambda s: int((s <= 5).sum()))).reset_index().rename(columns={"off": "team", "rusher": "pid"})
    ru["pid"] = ru.pid.astype(float).astype(int)
    p = p.merge(tg, how="outer", on=["game_id", "team", "pid"]).merge(ru, how="left", on=["game_id", "team", "pid"])
    num = ["comp", "att", "pass_y", "pass_td", "ints", "car", "rush_y", "rush_td", "rec", "rec_y", "rec_td", "tackles", "sacks", "tfl", "pd", "hur", "targets", "rz_tgt", "deep_t", "i10_car", "i5_car"]
    p[num] = p[num].fillna(0)
    p.to_csv(f"{D}/players.csv.gz", index=False)
    print(len(g), "games", g.groupby("season").size().to_dict(), g.spread_line.notna().mean(), len(l), "plays", l.tgt.notna().sum(), "targets", l.pass_play.sum(), "passes", l.rusher.notna().sum(), "rushes")
    print("target parse rate", l[l.ty.isin(["Pass Reception", "Pass Incompletion", "Passing Touchdown"])].tgt.notna().mean(), "rec tie", p.rec.sum() / max(p.targets.sum(), 1))


if __name__ == "__main__":
    main()
