"""College football data for the season hub: output/hub/cfb.json plus the week Game Books in output/hub/reports/.

Scope: every game with an AP Top 25 team (rank at kickoff) or an SEC, ACC, Big Ten or Big 12 team.
  weeks      - every 2026 regular-season week, with each in-scope game's line, score, and our pick when one was made
  record     - game picks graded from the week summaries (output/summary_w<N>.json), player forecasts from review_cfb.py
  teampos    - per-game stats each defense has allowed to QBs, RBs and WRs (college box scores carry no TE label)
Called by season.py; run alone with: venv/bin/python cfb/hub_cfb.py
"""
import glob
import json
import os
import re
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cfb_model as C  # noqa: E402

D = C.D
OUT = os.path.join(HERE, "output")
CONF = {"8": "SEC", "1": "ACC", "5": "Big Ten", "4": "Big 12"}
SEASON = 2026


def load(f):
    try:
        return json.load(open(f))
    except (OSError, ValueError):
        return {}


def ap_ranks():
    r = load(f"{D}/rank.json").get("rankings", [])
    ap = next((x for x in r if x.get("type") == "ap"), r[0] if r else {"ranks": []})
    return {t["team"]["abbreviation"]: t["current"] for t in ap["ranks"]}, ap.get("headline") or ap.get("name", "AP Top 25")


def odds(c):
    o = (c.get("odds") or [{}])[0]
    return o.get("details"), o.get("overUnder")


def scoreboard_games(ap):
    """Every in-scope game of the season from ESPN's weekly scoreboards, plus team info."""
    games, teams, weeks = [], {}, []
    files = sorted(glob.glob(f"{D}/sb/{SEASON}_2_*.json"), key=lambda f: int(re.search(r"_(\d+)\.json$", f).group(1)))
    cur = load(f"{D}/sb.json")
    cw = int(cur.get("week", {}).get("number", 0)) if cur else 0
    for f in files:
        wk = int(re.search(r"_(\d+)\.json$", f).group(1))
        ev = load(f).get("events", [])
        if wk == cw and cur.get("events"):
            ev = cur["events"]  # the slate file is fresher
        if not ev:
            continue
        kept = []
        for e in ev:
            c = e["competitions"][0]
            cs = c["competitors"]
            h = next(x for x in cs if x["homeAway"] == "home")
            a = next(x for x in cs if x["homeAway"] == "away")
            if wk == cw:  # the current slate uses the AP poll, as run_cfb.py does; past weeks keep ESPN's rank at kickoff
                rk = lambda x: ap.get(x["team"]["abbreviation"], 99)
            else:
                rk = lambda x: (x.get("curatedRank") or {}).get("current", 99)
            cf = lambda x: CONF.get(str(x["team"].get("conferenceId")))
            if min(rk(h), rk(a)) > 25 and not (cf(h) or cf(a)):
                continue
            for x in (h, a):
                t = x["team"]
                ab = t["abbreviation"]
                rec = (x.get("records") or [{}])[0].get("summary")
                old = teams.get(ab, {})
                teams[ab] = dict(name=t.get("displayName"), short=t.get("location"), nick=t.get("name"), conf=cf(x) or old.get("conf"),
                                 color="#" + (t.get("color") or "888888"), rec=rec or old.get("rec"), wk=max(wk, old.get("wk", 0)))
            st = e["status"]["type"]
            det, ou = odds(c)
            kept.append(dict(id=e["id"], wk=wk, kick=e["date"], a=a["team"]["abbreviation"], h=h["team"]["abbreviation"], ra=rk(a) if rk(a) <= 25 else None, rh=rk(h) if rk(h) <= 25 else None,
                             ca=cf(a), ch=cf(h), neutral=bool(c.get("neutralSite")), confg=bool(c.get("conferenceCompetition")), venue=(c.get("venue") or {}).get("fullName"),
                             done=bool(st.get("completed")), live=st.get("state") == "in", sa=int(a["score"]) if st.get("completed") else None, sh=int(h["score"]) if st.get("completed") else None,
                             line=det, total=ou, tv=next((b.get("names", [None])[0] for b in c.get("broadcasts", []) if b.get("names")), None)))
        if kept:
            ds = sorted(x["kick"] for x in kept)
            weeks.append(dict(n=wk, start=ds[0][:10], end=ds[-1][:10], games=len(kept)))
            games += kept
    return games, teams, weeks, cw


def closing_lines():
    g = pd.read_csv(f"{D}/games.csv")
    g = g[g.season == SEASON]
    return {str(r.game_id): (r.spread_line, r.total_line) for r in g.itertuples()}


def attach_picks(games, copy_report, hub):
    """Join each week's summary (our pick, projected score, Game Book link) onto the scoreboard games."""
    files = {}
    by_id = {g["id"]: g for g in games}
    by_lab = {(g["wk"], g["a"], g["h"]): g for g in games}
    for f in sorted(glob.glob(f"{OUT}/summary_w*.json")):
        s = load(f)
        wk = s["week"]
        rep = f"{OUT}/cfb_w{wk}_report.html" if os.path.exists(f"{OUT}/cfb_w{wk}_report.html") else (f"{OUT}/cfb_week10_report.html" if wk == 9 else None)
        rel = f"reports/cfb_w{wk}.html"
        if rep and os.path.exists(rep):
            copy_report(rep, os.path.join(hub, rel))
            files[rel] = os.path.join(hub, rel)
        for x in s["games"]:
            g = by_id.get(str(x.get("espn"))) or by_lab.get((wk, x["a"], x["h"]))
            if not g:
                continue
            g.update(pick=x["win"], pw=round(float(x["pw"]), 3), pa=round(float(x["pa"]), 1), ph=round(float(x["ph"]), 1), built=s.get("built"),
                     report=f"{rel}#{x['id']}" if rep else None, notes=x.get("notes") or [])
            if x.get("spread_home") is not None:
                g["sp"] = x["spread_home"]
            if x.get("line") and not g.get("line"):
                g["line"] = x["line"]
            if x.get("total") and not g.get("total"):
                g["total"] = x["total"]
    cl = closing_lines()
    for g in games:
        if g["id"] in cl:
            sp, tot = cl[g["id"]]
            if pd.notna(sp):
                g["close_sp"] = float(sp)  # home favored by this many
            if pd.notna(tot):
                g["close_tot"] = float(tot)
        if g["done"] and g.get("pick"):
            won = g["h"] if g["sh"] > g["sa"] else g["a"]
            g["right"] = g["pick"] == won
    return files


def record(games):
    """Game picks by week and by confidence, plus the player-forecast reviews."""
    gr = [g for g in games if "right" in g]
    by_wk = []
    for wk in sorted({g["wk"] for g in gr}):
        x = [g for g in gr if g["wk"] == wk]
        by_wk.append(dict(wk=wk, n=len(x), right=sum(g["right"] for g in x), exp=round(sum(g["pw"] for g in x), 1)))
    bands = []
    for lo, hi, lab in ((.5, .6, "50-59%"), (.6, .7, "60-69%"), (.7, .8, "70-79%"), (.8, .9, "80-89%"), (.9, 1.01, "90%+")):
        x = [g for g in gr if lo <= g["pw"] < hi]
        if x:
            bands.append(dict(band=lab, n=len(x), right=sum(g["right"] for g in x), exp=round(float(np.mean([g["pw"] for g in x])), 3)))
    rev = [load(f) for f in sorted(glob.glob(f"{OUT}/review_w*.json"), key=lambda f: int(re.search(r"_w(\d+)", f).group(1)))]
    return dict(weeks=by_wk, bands=bands, players=[r for r in rev if r])


def teampos(d, teams, ranks):
    """Per-game averages each defense allowed to QBs, RBs and WRs, for every in-scope team."""
    pl = d["players"]
    pl = pl[(pl.season == SEASON)].copy()
    g = d["games"]
    reg = set(g[(g.season == SEASON) & (g.stype == 2)].game_id)
    pl = pl[pl.game_id.isin(reg)]
    pl["pos"] = pl.pid.map(d["pos"]).fillna("WR")
    num = ["comp", "att", "pass_y", "pass_td", "ints", "car", "rush_y", "rush_td", "targets", "rec", "rec_y", "rec_td"]
    pl[num] = pl[num].fillna(0)
    per = pl.groupby(["game_id", "opp", "pos"])[num].sum().reset_index()
    gdates = g.set_index("game_id").t
    per["t"] = per.game_id.map(gdates)
    keep = {t for t, v in teams.items() if v.get("conf")} | set(ranks)
    through = int(g[(g.season == SEASON) & (g.stype == 2)].week.max())
    tfs = []
    for key, label, last in (("season", f"{SEASON} season", None), ("last3", "Last 3 games", 3)):
        out = {}
        for pos in ("QB", "RB", "WR"):
            rows = []
            for team in sorted(keep):
                gids = sorted(set(pl[pl.opp == team].game_id), key=lambda x: gdates.get(x, 0))
                if last:
                    gids = gids[-last:]
                if not gids:
                    continue
                x = per[(per.opp == team) & (per.pos == pos) & per.game_id.isin(gids)]
                s = x[num].sum() / len(gids)
                tot = x[num].sum()
                att = tot.att
                rate = None
                if pos == "QB" and att >= 20:  # NCAA passer rating from totals
                    rate = round(float((8.4 * tot.pass_y + 330 * tot.pass_td + 100 * tot.comp - 200 * tot.ints) / att), 1)
                rows.append(dict(team=team, g=len(gids), att=round(s.att, 2), cmp=round(s.comp, 2), py=round(s.pass_y, 2), ptd=round(s.pass_td, 2), int=round(s.ints, 2), rate=rate,
                                 car=round(s.car, 2), ry=round(s.rush_y, 2), rtd=round(s.rush_td, 2), tgt=round(s.targets, 2), rec=round(s.rec, 2), ryd=round(s.rec_y, 2), rctd=round(s.rec_td, 2)))
            out[pos] = rows
        tfs.append(dict(key=key, label=label, through=through, pos=out))
    return tfs


LKEYS = ["att", "cmp", "py", "ptd", "int", "car", "ry", "rtd", "tgt", "rec", "ryd", "rctd"]
LSRC = dict(att="att", cmp="comp", py="pass_y", ptd="pass_td", int="ints", car="car", ry="rush_y", rtd="rush_td", tgt="targets", rec="rec", ryd="rec_y", rctd="rec_td")
LSTAT = {"QB": ["py", "ptd", "cmp", "ry"], "RB": ["ry", "rec", "car", "rtd"], "WR": ["ryd", "rec", "rctd", "tgt"]}
DVP = {"QB": ["py", "ptd", "cmp", "att", "int", "ry"], "RB": ["ry", "rtd", "car", "rec", "ryd"], "WR": ["ryd", "rec", "rctd", "tgt"]}
DFLIP = {"int"}  # more allowed is worse for the player


def leaders(d, hub):
    """Season leaders (top 20 per stat for QB / RB / WR), each leader's game log and remaining schedule,
    and what every FBS defense allows per game to each position, with national ranks (1 = allows the most, the best matchup)."""
    g = d["games"]
    reg = g[(g.season == SEASON) & (g.stype == 2)]
    fbs = d["fbs"]
    pl = d["players"]
    pl = pl[pl.game_id.isin(set(reg[reg.hs.notna()].game_id))].copy()
    pl["pos"] = pl.pid.map(d["pos"])
    src = list(LSRC.values())
    pl[src] = pl[src].fillna(0)
    pl = pl.rename(columns={v: k for k, v in LSRC.items()})
    off = pl[(pl.pid > 0) & pl.pos.notna() & pl.team.isin(fbs)]
    off = off[off[LKEYS].abs().sum(axis=1) > 0]
    tot = off.groupby("pid").agg(n=("name", "last"), t=("team", "last"), pos=("pos", "first"), g=("game_id", "nunique"), **{k: (k, "sum") for k in LKEYS})
    top, keep = {}, set()
    for pos, ks in LSTAT.items():
        x = tot[tot.pos == pos]
        top[pos] = {}
        for k in ks:
            ids = x.sort_values([k, "g"], ascending=[False, True]).head(20).index.tolist()
            top[pos][k] = [str(i) for i in ids]
            keep |= set(ids)
    players = {str(i): [r.n, r.t, r.pos, int(r.g)] + [int(r[k]) for k in LKEYS] for i, r in tot.loc[sorted(keep)].iterrows()}
    log = {}
    gw = reg.set_index("game_id")
    for pid, x in off[off.pid.isin(keep)].groupby("pid"):
        rows = []
        for r in x.sort_values("t").itertuples():
            gm = gw.loc[r.game_id]
            home = gm.home == r.team
            us, th = (gm.hs, gm.aw) if home else (gm.aw, gm.hs)
            rows.append([int(gm.week), r.opp, 1 if home else (2 if gm.neutral else 0), f"{'W' if us > th else 'L' if us < th else 'T'} {int(us)}-{int(th)}"] + [int(getattr(r, k)) for k in LKEYS])
        log[str(pid)] = rows
    # defense vs position: per game, all FBS defenses
    per = pl[pl.pos.notna()].groupby(["game_id", "opp", "pos"])[LKEYS].sum().reset_index()
    gp = pl.groupby("opp").game_id.nunique()
    dvp = {}
    for pos, ks in DVP.items():
        x = per[per.pos == pos].groupby("opp")[ks].sum()
        x = x[x.index.isin(fbs)].div(gp.reindex(x[x.index.isin(fbs)].index), axis=0)
        rk = pd.DataFrame({k: x[k].rank(ascending=k in DFLIP, method="min") for k in ks})
        dvp[pos] = {t: {k: [round(float(x.at[t, k]), 1), int(rk.at[t, k])] for k in ks} for t in x.index}
    # remaining schedule + team names for every FBS team, from the weekly scoreboards
    sched, names = {}, {}
    for f in sorted(glob.glob(f"{D}/sb/{SEASON}_2_*.json"), key=lambda f: int(re.search(r"_(\d+)\.json$", f).group(1))):
        wk = int(re.search(r"_(\d+)\.json$", f).group(1))
        for e in load(f).get("events", []):
            c = e["competitions"][0]
            cs = {x["homeAway"]: x for x in c["competitors"]}
            if "home" not in cs or "away" not in cs:
                continue
            for side, o in (("home", "away"), ("away", "home")):
                t = cs[side]["team"]
                names[t["abbreviation"]] = [t.get("location") or t.get("displayName"), (cs[side].get("records") or [{}])[0].get("summary") or names.get(t["abbreviation"], [None, None])[1], "#" + (t.get("color") or "888888")]
                if e["status"]["type"].get("completed"):
                    continue
                sched.setdefault(t["abbreviation"], []).append([wk, e["date"], cs[o]["team"]["abbreviation"], 2 if c.get("neutralSite") else (1 if side == "home" else 0)])
    tms = {p[1] for p in players.values()}
    for t in list(sched):
        sched[t] = sorted({tuple(x) for x in sched[t]}, key=lambda z: z[1])
    sched = {t: [list(x) for x in v] for t, v in sched.items() if t in tms}
    need = tms | {x[2] for v in sched.values() for x in v} | {r[1] for v in log.values() for r in v}
    out = dict(season=SEASON, through=int(reg[reg.hs.notna()].week.max()), keys=["n", "t", "pos", "g"] + LKEYS, stats=LSTAT, dvpk=DVP, nfbs=len(dvp["WR"]),
               players=players, top=top, log=log, sched=sched, dvp=dvp, names={t: v for t, v in names.items() if t in need})
    f = os.path.join(hub, "cfbl.json")
    json.dump(out, open(f, "w"), separators=(",", ":"))
    print(f"CFB leaders: {len(players)} players, {len(dvp['WR'])} FBS defenses, through week {out['through']} -> {f} {os.path.getsize(f) / 1e3:.0f} KB")
    return f


def build(hub, copy_report):
    ranks, poll = ap_ranks()
    games, teams, weeks, cw = scoreboard_games(ranks)
    for t, r in ranks.items():
        if t in teams:
            teams[t]["rank"] = r
    files = attach_picks(games, copy_report, hub)
    d = C.load()
    hand = load(os.path.join(HERE, "qb.json")).get(str(cw), {})
    data = dict(season=SEASON, built=pd.Timestamp.now().strftime("%a %b %-d, %-I:%M %p"), week=cw, poll=poll, weeks=weeks, teams=teams, games=games,
                record=record(games), teampos=teampos(d, teams, ranks), qb_notes=hand.get("note", {}), qb_checked=hand.get("checked"),
                bt=load(f"{OUT}/backtest.json").get("mae_margin"))
    out = os.path.join(hub, "cfb.json")
    with open(out, "w") as f:
        json.dump(data, f, separators=(",", ":"), default=lambda o: o.item() if hasattr(o, "item") else str(o))
    files["cfb.json"] = out
    files["cfbl.json"] = leaders(d, hub)
    n = sum(1 for g in games if g.get("pick"))
    print(f"CFB: week {cw} current, {len(games)} in-scope games across {len(weeks)} weeks, {n} with our pick, {len(teams)} teams, {os.path.getsize(out) / 1e3:.0f} KB -> {out}")
    return files


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(HERE))
    import season
    build(os.path.join(os.path.dirname(HERE), "output", "hub"), season.copy_report)
