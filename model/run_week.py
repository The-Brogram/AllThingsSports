"""Run the live slate: python run_week.py  -> output/slate.pkl, output/predictions_<date>.csv"""
import glob
import json
import os
import pickle
import re
import numpy as np
import pandas as pd
import engine as E
from backtest import apply_cal, cal_key

from config import CACHE, DATE, ESPN, OUT_IDS, PLAY_IDS, PRED, QB_START, SEASON, WEEK
ESPN2NFLV = {"WSH": "WAS", "LAR": "LA"}
N = 20000
YD = {"rec": [20, 30, 40, 50, 60, 70, 80, 90, 100, 120], "rush": [20, 30, 40, 50, 60, 70, 80, 90, 100, 120],
      "pass": [150, 175, 200, 225, 250, 275, 300, 325]}
PLAY_P = {"Full Participation in Practice": .95, "Limited Participation in Practice": .8, "Did Not Participate In Practice": .5}


def tm(a):
    return ESPN2NFLV.get(a, a)


def slate():
    ev = json.load(open(f"{E.DATA}/{ESPN}"))["events"]
    out = []
    for e in ev:
        c = e["competitions"][0]
        s = json.load(open(f"{E.DATA}/espn/sum_{e['id']}.json"))
        h = [x for x in c["competitors"] if x["homeAway"] == "home"][0]
        a = [x for x in c["competitors"] if x["homeAway"] == "away"][0]
        pc = [p for p in s.get("pickcenter", []) if p.get("provider", {}).get("id") == "100"]
        pc = pc[0] if pc else s["pickcenter"][0]
        m = re.match(r"([A-Z]+) -?([\d.]+)", pc["details"])
        spread_home = float(m.group(2)) * (1 if tm(m.group(1)) == tm(h["team"]["abbreviation"]) else -1)
        out.append(dict(id=e["id"], home=tm(h["team"]["abbreviation"]), away=tm(a["team"]["abbreviation"]),
                        home_name=h["team"]["displayName"], away_name=a["team"]["displayName"],
                        kick=e["date"], venue=c["venue"]["fullName"], city=c["venue"].get("address", {}).get("city", ""),
                        indoor=c["venue"].get("indoor"), rec_h=h["records"][0]["summary"], rec_a=a["records"][0]["summary"], weather=c.get("weather") or e.get("weather"),
                        spread_home=spread_home, total=pc["overUnder"], over_odds=pc.get("overOdds"), under_odds=pc.get("underOdds"),
                        ml_h=pc["homeTeamOdds"].get("moneyLine"), ml_a=pc["awayTeamOdds"].get("moneyLine"), sp_h=pc["homeTeamOdds"].get("spreadOdds"), sp_a=pc["awayTeamOdds"].get("spreadOdds"),
                        details=pc["details"], status=e["status"]["type"]["description"], injuries=s.get("injuries", [])))
    return out


def availability(games, d):
    r = d["roster"]
    e2g = dict(zip(r.espn_id.dropna().astype(int), r[r.espn_id.notna()].gsis_id))
    inj = pd.read_csv(f"{E.DATA}/injuries_2026.csv")
    w = inj[inj.week == WEEK]
    prac = dict(zip(w.gsis_id, w.practice_status))
    av, stat_by_espn, lists = {}, {}, {}
    for g in games:
        for t in g["injuries"]:
            ab = tm(t["team"]["abbreviation"])
            for x in t.get("injuries", []):
                aid = int(x["athlete"]["id"])
                st = x.get("status", "")
                pid = e2g.get(aid)
                stat_by_espn[aid] = st
                if st in ("Out", "Injured Reserve", "Suspension", "Physically Unable to Perform", "Non-Football Illness", "Non-Football Injury"):
                    p = 0.0
                elif st == "Doubtful":
                    p = .12
                elif st == "Questionable":
                    p = PLAY_P.get(prac.get(pid), .75)
                else:
                    p = .9
                if pid:
                    av[pid] = p
                d_ = x.get("details", {}) or {}
                lists.setdefault(ab, []).append(dict(name=x["athlete"]["displayName"], pos=x["athlete"].get("position", {}).get("abbreviation", ""), status=st,
                                                     injury=d_.get("type", "") or "", practice=prac.get(pid, ""), play_p=p, updated=x.get("date", "")[:10]))
    # players who left the roster should not keep a share
    cur = r[r.status == "ACT"].groupby("team").gsis_id.apply(set).to_dict()
    st = d["stats"]
    for team, ids in cur.items():
        for pid in st[st.team == team].player_id.unique():
            if pid not in ids:
                av[(team, pid)] = 0.0  # keyed by team: he left this team, not the league
    dc = pd.read_csv(f"{E.DATA}/depth_charts_2026.csv", low_memory=False)
    dc = dc[(dc.dt == dc.groupby("team").dt.transform("max")) & (dc.pos_abb == "QB") & (dc.pos_rank == 1)]
    for r_ in dc.drop_duplicates("team").itertuples():
        av[("qb", r_.team)] = r_.gsis_id
    # slate.json overrides: confirmed news the feeds have not caught yet
    nm = dict(zip(r.gsis_id, r.full_name))
    for team, ids in OUT_IDS.items():
        for pid in ids:
            av[pid] = 0.0
            if not any(x["name"] == nm.get(pid) for x in lists.get(team, [])):
                lists.setdefault(team, []).append(dict(name=nm.get(pid, pid), pos="", status="Out (news)", injury="", practice="", play_p=0.0, updated=""))
    for team, ps in PLAY_IDS.items():
        for pid, p in ps.items():
            av[pid] = float(p)
            for x in lists.get(team, []):
                if x["name"] == nm.get(pid):
                    x.update(status=x["status"] + " (feed) - expected to play per news", play_p=float(p))
    for team, pid in QB_START.items():
        av[("qb", team)] = pid
    return av, stat_by_espn, lists, e2g


def props():
    rows = {}
    for f in glob.glob(f"{E.DATA}/espn/pp/*.json"):
        eid = f.split("/")[-1].split("_")[0]
        for r in json.load(open(f)).get("items", []):
            n = r["type"]["name"]
            if "athlete" not in r:
                continue
            aid = int(re.search(r"athletes/(\d+)", r["athlete"]["$ref"]).group(1))
            tgt = r.get("current", {}).get("target", {}).get("value")
            rows.setdefault((eid, aid, n), set()).add(tgt)
    return rows


def main():
    d = E.load()
    games = slate()
    av, stat_espn, inj_lists, e2g = availability(games, d)
    g2e = {v: k for k, v in e2g.items()}
    pr = props()
    fit = E.fit_script(E.script_rows(d, E.tidx(2025, 4), E.tidx(SEASON, WEEK - 1)))
    resid = E.residual_pairs(d["games"])
    t = E.tidx(SEASON, WEEK)
    r = E.ratings(d["team_games"], t)
    bt = json.load(open("output/backtest.json"))
    cal = {k: tuple(v) for k, v in bt["cal"].items()}
    br = pd.read_pickle("output/bt_rows.pkl")
    qb = br[(br.mkt == "atd") & (br.pos == "QB")]
    off = float(np.log(qb.act.gt(0).mean() / (1 - qb.act.gt(0).mean())) - np.log(apply_cal(qb.proj.values, cal["atd"]).mean() / (1 - apply_cal(qb.proj.values, cal["atd"]).mean())))
    rng = np.random.default_rng(int(DATE))
    res, preds = [], []
    for g in games:
        s = E.simulate_game(rng, d, g["home"], g["away"], g["spread_home"], g["total"], t, fit, resid, n=N, avail=av, ratings_cache=r)
        s["game"] = g
        s["inj"] = {g["home"]: inj_lists.get(g["home"], []), g["away"]: inj_lists.get(g["away"], [])}
        for side, df, sim, team, opp in (("h", s["dfh"], s["sh"], g["home"], g["away"]), ("a", s["dfa"], s["sa"], g["away"], g["home"])):
            P = []
            for j, p in df.iterrows():
                atd = (sim["ptds"][:, j] + sim["rtds"][:, j]) > 0
                c = (cal["atd"][0] + (off if p.pos == "QB" else 0), cal["atd"][1])
                x = dict(p, team=team, opp=opp, j=j, espn=g2e.get(p.player_id), tg=sim["targets"][:, j].mean(), rec=sim["rec"][:, j].mean(), car=sim["cars"][:, j].mean(),
                         rec_y=sim["rec_y"][:, j], rush_y=sim["rush_y"][:, j], atd_raw=atd.mean(), atd=float(apply_cal(np.array([atd.mean()]), c)[0]), rec_td=sim["ptds"][:, j].mean(), rush_td=sim["rtds"][:, j].mean())
                if p.pos == "QB":
                    x["pass_y"] = sim["pass_y"]
                P.append(x)
            s["pl_" + side] = P
            for x in P:
                for mk, arr, ths in (("rec", x["rec_y"], YD["rec"]), ("rush", x["rush_y"], YD["rush"])):
                    if arr.mean() >= 6:
                        for k in ths:
                            preds.append(dict(game=f"{g['away']}@{g['home']}", player=x["name"], pid=x["player_id"], team=team, market=mk, threshold=k, raw_p=float((arr >= k).mean()), cal_p=float(apply_cal(np.array([(arr >= k).mean()]), cal[cal_key(cal, mk, x["tshare"])])[0]), proj=float(arr.mean())))
                if "pass_y" in x:
                    for k in YD["pass"]:
                        preds.append(dict(game=f"{g['away']}@{g['home']}", player=x["name"], pid=x["player_id"], team=team, market="pass", threshold=k, raw_p=float((x["pass_y"] >= k).mean()), cal_p=float(apply_cal(np.array([(x["pass_y"] >= k).mean()]), cal["pass"])[0]), proj=float(x["pass_y"].mean())))
                preds.append(dict(game=f"{g['away']}@{g['home']}", player=x["name"], pid=x["player_id"], team=team, market="atd", threshold=1, raw_p=x["atd_raw"], cal_p=x["atd"], proj=x["atd"]))
        s["props"] = {k: v for k, v in pr.items() if k[0] == g["id"]}
        res.append(s)
        print(g["away"], "@", g["home"], "line", g["details"], g["total"], "| model", round(s["model_a"], 1), round(s["model_h"], 1), "| QB", s["dfa"][s["dfa"].pos == "QB"].name.tolist(), s["dfh"][s["dfh"].pos == "QB"].name.tolist(), flush=True)
    pd.DataFrame(preds).to_csv(PRED, index=False)  # one file per slate, so a later run never overwrites forecasts awaiting review
    pickle.dump(dict(res=res, inj_status=stat_espn, ratings=r, cal=cal, bt=bt, fit=fit, stat_espn=stat_espn, qb_off=off), open(CACHE, "wb"))
    for s in res:
        print(s["game"]["away"], "@", s["game"]["home"])
        for key in ("pl_a", "pl_h"):
            print("  ", " | ".join(f"{p['name']} {p['pos']} {p['tg']:.1f}t {p['rec_y'].mean():.0f}y {p['car']:.1f}c {p['rush_y'].mean():.0f}r {p['atd']:.0%} ({p['play_p']:.2f})" for p in sorted(s[key], key=lambda z: -(z['rec_y'].mean() + z['rush_y'].mean()))[:8]))


if __name__ == "__main__":
    main()
