"""Simulate this week's games: every game with an AP Top 25 team or an SEC, ACC, Big Ten or Big 12 team.
Run: python fetch_cfb.py && python parse_cfb.py && python weather_cfb.py && python run_cfb.py && python report_cfb.py
The week is whatever fetch_cfb.py saved in data/sb.json. Starters and notes checked by hand go in qb.json under the week number."""
import json
import os
import pickle
import re
import numpy as np
import pandas as pd
import cfb_model as C
import engine as E
from backtest import apply_cal

N = 20000
HERE = os.path.dirname(os.path.abspath(__file__))
TOP = 25
CONF = {"8": "SEC", "1": "ACC", "5": "Big Ten", "4": "Big 12"}  # ESPN conference ids


def week():
    return int(json.load(open(f"{C.D}/sb.json"))["week"]["number"])


def cache(wk):
    return os.environ.get("CFB_CACHE", f"{C.D}/cfb_slate_w{wk}.pkl")


def hand(wk):
    """Starters and notes confirmed by hand for week wk: {"pick": {team: name}, "note": {team: text}}."""
    f = os.path.join(HERE, "qb.json")
    q = json.load(open(f)) if os.path.exists(f) else {}
    return q.get(str(wk), {"pick": {}, "note": {}})
THR = {"rec": [20, 30, 40, 50, 60, 70, 80, 90, 100, 120], "rush": [20, 30, 40, 50, 60, 70, 80, 90, 100, 120], "pass": [150, 175, 200, 225, 250, 275, 300, 325]}
QB_DISCOUNT = .88  # backup/new starter: yards per attempt versus the team's sample


def slate():
    sb = json.load(open(f"{C.D}/sb.json"))
    rank = {}
    for t in json.load(open(f"{C.D}/rank.json"))["rankings"][0]["ranks"]:
        rank[t["team"]["abbreviation"]] = t["current"]
    out = []
    for e in sb["events"]:
        c = e["competitions"][0]
        h = [x for x in c["competitors"] if x["homeAway"] == "home"][0]
        a = [x for x in c["competitors"] if x["homeAway"] == "away"][0]
        ha, aa = h["team"]["abbreviation"], a["team"]["abbreviation"]
        confs = {x["team"].get("conferenceId") for x in (h, a)} & set(CONF)
        if min(rank.get(ha, 99), rank.get(aa, 99)) > TOP and not confs:
            continue
        if not os.path.exists(f"{C.D}/games/{e['id']}.json"):
            print("no summary", aa, "@", ha)
            continue
        s = json.load(open(f"{C.D}/games/{e['id']}.json"))
        pc = next((p for p in s.get("pickcenter", []) if p.get("provider", {}).get("id") == "100"), (s.get("pickcenter") or [{}])[0])
        m = re.match(r"([A-Z&]+) -?([\d.]+)", pc.get("details") or "")
        if not m and "EVEN" in (pc.get("details") or "").upper():
            m = re.match(r"()(0)", "0")
        if not m or pc.get("overUnder") is None:
            print("no line", aa, "@", ha)
            continue
        sp = float(m.group(2)) * (1 if m.group(1) == ha else -1)
        out.append(dict(id=e["id"], home=ha, away=aa, home_name=h["team"]["displayName"], away_name=a["team"]["displayName"], hloc=h["team"]["location"], aloc=a["team"]["location"], rank_h=rank.get(ha), rank_a=rank.get(aa),
                        kick=e["date"], venue=c["venue"]["fullName"], city=c["venue"].get("address", {}).get("city", ""), state=c["venue"].get("address", {}).get("state", ""), indoor=c["venue"].get("indoor"), neutral=bool(c.get("neutralSite")), conf=bool(c.get("conferenceCompetition")),
                        rec_h=h["records"][0]["summary"], rec_a=a["records"][0]["summary"], conf_h=CONF.get(h["team"].get("conferenceId")), conf_a=CONF.get(a["team"].get("conferenceId")), espn_status=e["status"]["type"]["name"], weather=c.get("weather") or e.get("weather"), status=e["status"]["type"]["description"], spread_home=sp, total=pc["overUnder"], over_odds=pc.get("overOdds"), under_odds=pc.get("underOdds"),
                        ml_h=pc["homeTeamOdds"].get("moneyLine"), ml_a=pc["awayTeamOdds"].get("moneyLine"), sp_h=pc["homeTeamOdds"].get("spreadOdds"), sp_a=pc["awayTeamOdds"].get("spreadOdds"), details=pc["details"]))
    return sorted(out, key=lambda x: x["kick"]), rank


def stamp(s):
    return pd.Timestamp(s.split(", ")[-1] + " 2026 23:59")


def availability(games, d, inj):
    av, lists = {}, {}
    pl = d["players"]
    dates = pd.to_datetime(d["games"].set_index("game_id").date, utc=True).dt.tz_localize(None)
    x = pl[(pl.season == 2026) & ((pl.att + pl.car + pl.targets) > 0)]
    seen = x.assign(dt=x.game_id.map(dates)).groupby(["team", "name"]).dt.max().to_dict()
    for g in games:
        for loc, ab in ((g["hloc"], g["home"]), (g["aloc"], g["away"])):
            rows = inj.get(loc, [])
            if not rows:
                alt = {"Miami": "Miami (FL)", "Ole Miss": "Mississippi", "UCLA": "UCLA", "USC": "USC", "BYU": "BYU", "SMU": "SMU"}.get(loc)
                rows = inj.get(alt, []) if alt else []
            names = pl[(pl.team == ab) & (pl.season == 2026)].name.dropna().unique()
            for r in rows:
                tok = re.sub(r"\b(Jr|Sr|II|III|IV)\b\.?", "", r["name"]).replace(".", " ").split()
                if len(tok) < 2:
                    continue
                last, ini = tok[-1].lower(), tok[0][0].lower()
                hit = [n for n in names if n.split()[0][0].lower() == ini and last in n.lower().replace("jr.", "").split()]
                r2 = dict(r, play_p=C.status_p(r["status"]), team=ab, matched=hit[0] if hit else None)
                lists.setdefault(ab, []).append(r2)
                for n in hit:
                    last = seen.get((ab, n))
                    if last is not None and r.get("date") and last > stamp(r["date"]):  # listed, then played: the note is stale
                        r2.update(play_p=1.0, stale=True)
                    if (ab, n) not in {(k, v) for k, v in C.QB_PICK.items()}:
                        av[(ab, n)] = r2["play_p"]
    return av, lists


def qb_flag(d, team, df, t):
    pl = d["players"]
    q = df[df.pos == "QB"]
    if q.empty:
        return None
    pid = q.player_id.iat[0]
    x = pl[(pl.team == team) & (pl.season == 2026) & (pl.t < t) & (pl.att > 0)]
    if x.empty:
        return None
    share = x[x.pid == pid].att.sum() / max(x.att.sum(), 1)
    return share if share < .4 else None


def role_changes(d, team, t):
    """Touch share swing in the latest game vs earlier 2026 games; college has no snap counts."""
    pl = d["players"]
    x = pl[(pl.team == team) & (pl.season == 2026) & (pl.t < t)].copy()
    x["touch"] = x.targets + x.car
    tot = x.groupby("t").touch.transform("sum")
    x["s"] = x.touch / tot.clip(1)
    last = x.t.max()
    out = []
    for pid, y in x.groupby("pid"):
        a, b = y[y.t == last].s.sum(), y[y.t < last].s.sum() / max(x[x.t < last].t.nunique(), 1)
        if abs(a - b) >= .10 and max(a, b) >= .14:
            out.append((team, y.name.dropna().iloc[-1], float(b), float(a)))
    return out


def main():
    wk = week()
    hd = hand(wk)
    C.QB_PICK.clear()
    C.QB_PICK.update(hd.get("pick", {}))
    d = C.load()
    games, rank = slate()
    old = {}
    if os.path.exists(cache(wk)):  # games already under way keep the forecast made before kickoff
        old = {s["game"]["id"]: s for s in pickle.load(open(cache(wk), "rb"))["res"]}
    inj = C.covers_injuries()
    av, lists = availability(games, d, inj)
    bt = json.load(open("output/backtest.json"))
    cal = {k: tuple(v) for k, v in bt["cal"].items()}
    g = d["games"]
    res = C.residuals(g[g.season == 2026])
    t = C.t_of(2026, wk)
    fit = E.fit_script(C.script_rows(d, C.t_of(2026, 4), C.t_of(2026, wk - 1)))
    r = C.ratings(d, t)
    sim = C.Sim(d, fit, res, 0.0)  # backtest: our line adds nothing to the close, so game script uses the market
    rng = np.random.default_rng(2026_00 + wk)
    out, preds = [], []
    tg = d["team_games"]
    orig = E.team_prior
    for gm in games:
        if gm["espn_status"] != "STATUS_SCHEDULED" and gm["id"] in old:
            out.append(dict(old[gm["id"]], kept=True))
            print("kept pre-kick forecast:", gm["away"], "@", gm["home"], gm["status"])
            continue
        flags = {}
        for tm, op in ((gm["home"], gm["away"]), (gm["away"], gm["home"])):
            flags[tm] = qb_flag(d, tm, C.table(d, tm, t, op, av), t)

        def tp(tgx, team, tt):
            x = orig(tgx, team, tt)
            if flags.get(team) is not None:
                x = dict(x, ypa=x["ypa"] * QB_DISCOUNT)
            return x
        E.team_prior = tp
        s = sim.game(rng, gm["home"], gm["away"], gm["spread_home"], gm["total"], gm["neutral"], t, r, N, av)
        E.team_prior = orig
        s["game"] = gm
        s["inj"] = {gm["home"]: lists.get(gm["home"], []), gm["away"]: lists.get(gm["away"], [])}
        s["qb_flag"] = flags
        s["roles"] = role_changes(d, gm["home"], t) + role_changes(d, gm["away"], t)
        for side, df, o, team, opp in (("h", s["dfh"], s["sh"], gm["home"], gm["away"]), ("a", s["dfa"], s["sa"], gm["away"], gm["home"])):
            P = []
            for j, p in df.iterrows():
                atd = (o["ptds"][:, j] + o["rtds"][:, j]) > 0
                x = dict(p, team=team, opp=opp, j=j, espn=None, tg=o["targets"][:, j].mean(), rec=o["rec"][:, j].mean(), car=o["cars"][:, j].mean(), rec_y=o["rec_y"][:, j], rush_y=o["rush_y"][:, j], atd_raw=float(atd.mean()), atd=float(apply_cal(np.array([atd.mean()]), cal["atd"])[0]), rec_td=o["ptds"][:, j].mean(), rush_td=o["rtds"][:, j].mean())
                if p.pos == "QB":
                    x["pass_y"] = o["pass_y"]
                P.append(x)
                lab = f"{gm['away']}@{gm['home']}"
                for mk, arr in (("rec", x["rec_y"]), ("rush", x["rush_y"])):
                    if arr.mean() >= 6:
                        preds += [dict(game=lab, player=x["name"], pid=x["player_id"], team=team, market=mk, threshold=k, raw_p=float((arr >= k).mean()), cal_p=float(apply_cal(np.array([(arr >= k).mean()]), cal[mk])[0]), proj=float(arr.mean())) for k in THR[mk]]
                if "pass_y" in x:
                    preds += [dict(game=lab, player=x["name"], pid=x["player_id"], team=team, market="pass", threshold=k, raw_p=float((x["pass_y"] >= k).mean()), cal_p=float(apply_cal(np.array([(x["pass_y"] >= k).mean()]), cal["pass"])[0]), proj=float(x["pass_y"].mean())) for k in THR["pass"]]
                preds.append(dict(game=lab, player=x["name"], pid=x["player_id"], team=team, market="atd", threshold=1, raw_p=x["atd_raw"], cal_p=x["atd"], proj=x["atd"]))
            s["pl_" + side] = P
        s["props"] = {}
        out.append(s)
        print(gm["away"], "@", gm["home"], gm["details"], gm["total"], "| ours", round(s["model_a"], 1), round(s["model_h"], 1), "| sim", round(s["a_pts"].mean(), 1), round(s["h_pts"].mean(), 1), "| QB", s["dfa"][s["dfa"].pos == "QB"].name.tolist(), s["dfh"][s["dfh"].pos == "QB"].name.tolist(), {k: round(v, 2) for k, v in flags.items() if v is not None}, flush=True)
        for key in ("pl_a", "pl_h"):
            print("   ", " | ".join(f"{p['name']} {p['pos']} {p['tg']:.1f}t {p['rec_y'].mean():.0f}y {p['car']:.1f}c {p['rush_y'].mean():.0f}r {p['atd']:.0%}" + (f" Q{p['play_p']:.1f}" if p['play_p'] < 1 else "") for p in sorted(s[key], key=lambda z: -(z['rec_y'].mean() + z['rush_y'].mean()))[:6]))
    if old and preds:  # keep forecasts of games carried over from the earlier run
        f0 = f"output/predictions_w{wk}.csv"
        if os.path.exists(f0):
            p0 = pd.read_csv(f0)
            kept = {f"{s['game']['away']}@{s['game']['home']}" for s in out if s.get("kept")}
            preds = p0[p0.game.isin(kept)].to_dict("records") + preds
    pd.DataFrame(preds).to_csv(f"output/predictions_w{wk}.csv", index=False)
    pickle.dump(dict(res=out, ratings=r, cal=cal, bt=bt, rank=rank, week=wk, built=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"), hand=hd), open(cache(wk), "wb"))
    print(f"Week {wk}: {len(out)} games -> {cache(wk)}")


if __name__ == "__main__":
    main()
