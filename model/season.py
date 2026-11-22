"""Season hub: python season.py (or slate.py hub) -> output/hub/index.html plus output/hub/reports/.

One page for every game of the season, rebuilt from whatever data exists right now:
  Final        the score, and whether our pre-game pick was right
  Game Book    a full slate report exists (slate.py run): its score, win chance and Most likely tier
  Lines posted the market line is out (nflverse look-ahead lines, usually 1-2 weeks early): 90% market / 10% model
  Early        no line yet: our ratings-only line, which the backtest found less accurate than the market

Each game has a side-by-side panel (each offense against the defense it faces), any two teams can be compared,
and the Track record tab scores every reviewed slate. Publish output/hub/index.html with the files map in
output/hub/files.json to the URL in output/artifacts.json under "hub", so the link never changes.
"""
import glob
import json
import shutil
import sys
import math
import os
import re
from datetime import datetime
import numpy as np
import pandas as pd
import engine as E
import scheme as S
from config import ROOT
from report import CSS as REPORT_CSS, FONTS as REPORT_FONTS, TEAM_M, build_ranks

SEASON = 2026
OUT = os.path.join(ROOT, "output")
HUB = os.path.join(OUT, "hub")
SD = 12.4  # SD of (result - closing spread), 2024-26 finals: turns a projected margin into a win chance
W_MKT = 0.9
ROUND = {19: "Wild Card", 20: "Divisional", 21: "Conference", 22: "Super Bowl"}
NAMES = {"ARI": "Cardinals", "ATL": "Falcons", "BAL": "Ravens", "BUF": "Bills", "CAR": "Panthers", "CHI": "Bears", "CIN": "Bengals", "CLE": "Browns",
         "DAL": "Cowboys", "DEN": "Broncos", "DET": "Lions", "GB": "Packers", "HOU": "Texans", "IND": "Colts", "JAX": "Jaguars", "KC": "Chiefs",
         "LA": "Rams", "LAC": "Chargers", "LV": "Raiders", "MIA": "Dolphins", "MIN": "Vikings", "NE": "Patriots", "NO": "Saints", "NYG": "Giants",
         "NYJ": "Jets", "PHI": "Eagles", "PIT": "Steelers", "SEA": "Seahawks", "SF": "49ers", "TB": "Buccaneers", "TEN": "Titans", "WAS": "Commanders"}
LEGACY = {"nfl_week8_report.html": "20261101"}  # reports saved before the date naming


def phi(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def num(v):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)


def reports():
    """Slate date -> report file name inside the hub."""
    out = {}
    for f in glob.glob(os.path.join(OUT, "nfl_*_report.html")):
        b = os.path.basename(f)
        m = re.match(r"nfl_(\d{8})_report\.html$", b)
        d = m.group(1) if m else LEGACY.get(b)
        if d:
            out[d] = (f, f"reports/nfl_{d}.html")
    return out


def copy_report(src, dst):
    """Reports are written as fragments; a published supporting page needs its own doctype and viewport."""
    h = open(src).read()
    # older reports carry older styling; give every report the current look
    h = re.sub(r"<style>.*?</style>", lambda m: f"<style>{REPORT_CSS}</style>", h, count=1, flags=re.S)
    h = re.sub(r'<link rel="stylesheet" href="https://fonts\.googleapis\.com/css2\?[^"]*">', lambda m: REPORT_FONTS, h, count=1)
    if not h.lstrip().lower().startswith("<!doctype"):
        h = '<!doctype html><meta name="viewport" content="width=device-width,initial-scale=1">' + h
    back = '<a href="../index.html" style="color:inherit;opacity:.8;font-size:14px;text-decoration:none">&larr; Season hub</a>'
    h = h.replace('<div class="top">', f'<div class="top">{back}', 1)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    open(dst, "w").write(h)


def summaries():
    out = {}
    for f in glob.glob(os.path.join(OUT, "summary_????????.json")):
        s = json.load(open(f))
        for g in s["games"]:
            out[(s["date"], g["a"], g["h"])] = dict(g, built=s["built"])
    return out


def games(d, rep, summ):
    g = pd.read_csv(os.path.join(ROOT, "data", "games.csv"))
    g = g[g.season == SEASON].copy()
    g["date"] = g.gameday.str.replace("-", "")
    tg = d["team_games"]
    rcache = {}
    rows = []
    for r in g.sort_values(["week", "gameday", "gametime", "game_id"]).itertuples():
        if r.week not in rcache:
            rcache[r.week] = E.ratings(tg, E.tidx(SEASON, r.week))  # only games before this week, so past picks are honest
        mh, ma = E.model_line(rcache[r.week], r.home_team, r.away_team, neutral=r.location == "Neutral")
        sp, tot = num(r.spread_line), num(r.total_line)
        if sp is not None and tot is not None:  # nflverse spread_line is home margin
            kh, ka = (tot + sp) / 2, (tot - sp) / 2
            bh, ba = W_MKT * kh + (1 - W_MKT) * mh, W_MKT * ka + (1 - W_MKT) * ma
        else:
            bh, ba = mh, ma
        x = dict(id=r.game_id, week=int(r.week), date=r.date, day=r.weekday[:3], time=r.gametime, away=r.away_team, home=r.home_team,
                 stadium=r.stadium, roof=r.roof if isinstance(r.roof, str) else "", spread=sp, total=tot,
                 ml_a=num(r.away_moneyline), ml_h=num(r.home_moneyline), mh=round(mh, 1), ma=round(ma, 1), bh=round(bh, 1), ba=round(ba, 1),
                 ph=round(phi((bh - ba) / SD), 3), status="early" if sp is None else "lines", qb_a=r.away_qb_name if isinstance(r.away_qb_name, str) else None,
                 qb_h=r.home_qb_name if isinstance(r.home_qb_name, str) else None)
        s = summ.get((r.date, r.away_team, r.home_team))
        if s:  # the Game Book's sim is the better number for its own game
            x.update(status="book", bh=round(s["ph"], 1), ba=round(s["pa"], 1), ph=round(s["pw"] if s["win"] == r.home_team else 1 - s["pw"], 3),
                     likely=s["likely"][:5], td=s["td"][:3], built=s["built"])
        if r.date in rep:
            x["report"] = rep[r.date][1] + "#" + f"{r.away_team}-{r.home_team}".lower()
        if pd.notna(r.result):
            x.update(status="final", sa=int(r.away_score), sh=int(r.home_score))
            pick_home = x["ph"] >= .5
            x["right"] = None if r.result == 0 else bool(pick_home == (r.result > 0))
        rows.append(x)
    return rows, rcache


def teams(c, R, rt, rows):
    rec = {t: [0, 0, 0] for t in NAMES}
    for x in rows:
        if x["status"] == "final":
            m = x["sh"] - x["sa"]
            for t, s in ((x["home"], m), (x["away"], -m)):
                rec[t][0 if s > 0 else (1 if s < 0 else 2)] += 1
    off_r = pd.Series(rt["off"]).rank(ascending=False)
    def_r = pd.Series(rt["def"]).rank(ascending=False)
    out = {}
    for t in NAMES:
        w, l, ti = rec[t]
        o, dd = c.team_side(t, "off"), c.team_side(t, "def")
        unit = {}
        for side, tab in (("off", o), ("def", dd)):
            unit[side] = {k: dict(v=num(tab["Season"][k]), r=int(R[side][k][t]) if t in R[side][k] else None, l3=num(tab["Last 3"][k])) for _, k, _ in TEAM_M}
        out[t] = dict(name=NAMES[t], rec=f"{w}-{l}" + (f"-{ti}" if ti else ""), adj_off=round(rt["off"].get(t, 0), 1), adj_def=round(rt["def"].get(t, 0), 1),
                      adj_off_r=int(off_r.get(t, 16)), adj_def_r=int(def_r.get(t, 16)), unit=unit)
    return out


def track(rows):
    """Scored forecasts and pick results so far."""
    out = dict(games=[], props=[], bands=[], likely=None, clv=None)
    df = pd.DataFrame([x for x in rows if x["status"] == "final" and x.get("right") is not None])
    if len(df):
        df["err"] = [abs((x["bh"] - x["ba"]) - (x["sh"] - x["sa"])) for x in df.to_dict("records")]
        for wk, g in df.groupby("week"):
            out["games"].append(dict(week=int(wk), n=len(g), right=int(g.right.sum()), mae=round(float(g.err.mean()), 1)))
    f = os.path.join(OUT, "track_record.csv")
    if os.path.exists(f):
        tr = pd.read_csv(f, low_memory=False)
        tr["date"] = tr.date.astype(str)
        base = tr.groupby(["market", "threshold"]).hit.transform("mean")
        tr["b"], tr["bb"] = (tr.cal_p - tr.hit) ** 2, (base - tr.hit) ** 2
        top = tr[tr.cal_p >= .7].sort_values("threshold").groupby(["date", "pid", "market"]).tail(1)
        for dt, g in tr.groupby("date"):
            t1 = top[top.date == dt]
            out["props"].append(dict(date=dt, week=int(g.week.iloc[0]), n=len(g), brier=round(float(g.b.mean()), 4), base=round(float(g.bb.mean()), 4),
                                     likely_n=len(t1), likely_pred=round(float(t1.cal_p.mean()), 3) if len(t1) else None, likely_hit=round(float(t1.hit.mean()), 3) if len(t1) else None))
        band = pd.cut(tr.cal_p, [0, .3, .5, .7, .85, 1], labels=["Under 30%", "30-50%", "50-70%", "70-85%", "85%+"], include_lowest=True)
        for b, g in tr.groupby(band, observed=True):
            out["bands"].append(dict(band=str(b), n=len(g), pred=round(float(g.cal_p.mean()), 3), hit=round(float(g.hit.mean()), 3)))
        if len(top):
            out["likely"] = dict(n=len(top), pred=round(float(top.cal_p.mean()), 3), hit=round(float(top.hit.mean()), 3),
                                 moe=round(1.64 * math.sqrt(top.hit.mean() * (1 - top.hit.mean()) / len(top)), 3))
    try:
        import odds
        fr = [odds.clv(os.path.basename(p)[6:14]) for p in sorted(glob.glob(os.path.join(OUT, "picks_????????.csv")))]
        fr = [x for x in fr if x is not None and len(x)]
        if fr:
            cl = pd.concat(fr)
            v = cl[(cl.kind == "prop") & ~cl.flagged.astype(bool)]
            out["clv"] = dict(n=len(cl), mean=round(float(cl.clv.mean()), 3), pos=round(float((cl.clv > 0).mean()), 3),
                              props_n=len(v), props_mean=round(float(v.clv.mean()), 2) if len(v) else None, props_pos=round(float((v.clv > 0).mean()), 3) if len(v) else None)
    except Exception as ex:  # closing lines are optional; the hub still builds without them
        print("CLV skipped:", ex)
    return out


def current_week(rows):
    today = datetime.now().strftime("%Y%m%d")
    up = [x["week"] for x in rows if x["status"] != "final" and x["date"] >= today]
    return min(up) if up else max(x["week"] for x in rows)


ROOT = os.path.dirname(os.path.abspath(__file__))


def freshness():
    """One line per sport: is everything that should be in by now actually in? Printed on every rebuild so the scheduled runs can flag stale data.
    STALE when (a) a game that kicked off long enough ago still has no final score on file, (b) player stats trail the finished games,
    or (c) a finished slate with saved forecasts was never graded."""
    now = pd.Timestamp.now(tz="UTC")
    et = now.tz_convert("America/New_York")
    out = []
    try:  # NFL: results land within a day and a half of kickoff; reviews run Mon/Tue/Wed
        g = pd.read_csv(f"{ROOT}/data/games.csv", usecols=["season", "week", "gameday", "gametime", "result"], low_memory=False)
        g = g[g.season == 2026].copy()
        kick = pd.to_datetime(g.gameday + " " + g.gametime.fillna("13:00")).dt.tz_localize("America/New_York")
        late = g[g.result.isna() & (kick + pd.Timedelta(hours=36) < now)]
        done = g[g.result.notna()]
        last = int(done.week.max()) if len(done) else 0
        sw = int(pd.read_csv(f"{ROOT}/data/stats_player_week_2026.csv", usecols=["week"]).week.max())
        rw = int(pd.read_csv(f"{ROOT}/data/roster_weekly_2026.csv", usecols=["week"]).week.max())
        graded = set(pd.read_csv(f"{ROOT}/output/track_record.csv", usecols=["date"]).date.astype(str)) if os.path.exists(f"{ROOT}/output/track_record.csv") else set()
        cut = f"{et - pd.Timedelta(days=3):%Y%m%d}"
        ung = sorted(d for d in (os.path.basename(x)[12:20] for x in glob.glob(f"{ROOT}/output/predictions_2026*.csv")) if d < cut and d not in graded)
        msg = f"games final through week {last}; player stats through week {sw}; weekly rosters through week {rw}"
        msg += (f"; {len(late)} game(s) past kickoff with no result yet" if len(late) else "") + (f"; forecasts never graded: {', '.join(ung)}" if ung else "")
        out.append(("NFL", sw >= last and late.empty and not ung, msg))
    except Exception as ex:  # noqa: BLE001
        out.append(("NFL", False, f"check failed: {ex}"))
    try:  # CFB: weekly cadence (Sunday review, Thursday run), so allow two and a half days
        g = pd.read_csv(f"{ROOT}/cfb/data/games.csv", usecols=["game_id", "season", "stype", "week", "hs"])
        reg = g[(g.season == 2026) & (g.stype == 2)]
        fin = set(reg[reg.hs.notna()].game_id.astype(str))
        last = int(reg[reg.hs.notna()].week.max())
        ids = set(pd.read_csv(f"{ROOT}/cfb/data/players.csv.gz", usecols=["game_id"]).game_id)
        pw = int(reg[reg.game_id.isin(ids)].week.max())
        late, wk_end = 0, {}
        for x in glob.glob(f"{ROOT}/cfb/data/sb/2026_2_*.json"):
            wk = int(re.search(r"_(\d+)\.json$", x).group(1))
            for e in json.load(open(x)).get("events", []):
                t = pd.Timestamp(e["date"])
                wk_end[wk] = max(wk_end.get(wk, t), t)
                nm = e.get("status", {}).get("type", {}).get("name", "")
                if "CANCEL" in nm or "POSTPONE" in nm:
                    continue
                late += (t + pd.Timedelta(hours=60) < now) and str(e["id"]) not in fin
        ung = sorted(w for w in (int(re.search(r"_w(\d+)\.csv$", x).group(1)) for x in glob.glob(f"{ROOT}/cfb/output/predictions_w*.csv"))
                     if w in wk_end and wk_end[w] + pd.Timedelta(days=3) < now and not os.path.exists(f"{ROOT}/cfb/output/review_w{w}.json"))
        msg = f"games final through week {last}; player stats through week {pw}" + (f"; {late} game(s) played with no result on file" if late else "") + (f"; forecasts never graded: week {', '.join(map(str, ung))}" if ung else "")
        out.append(("CFB", pw >= last and not late and not ung, msg))
    except Exception as ex:  # noqa: BLE001
        out.append(("CFB", False, f"check failed: {ex}"))
    try:  # NBA: daily; a game that tipped 8+ hours ago must be final, and every past forecast night must be graded
        g = pd.read_csv(f"{ROOT}/nba/data/games.csv", dtype={"game_id": str})
        g = g[(g.season == g.season.max()) & g.stype.isin([2, 3, 5])]
        skip = set()
        for x in sorted(glob.glob(f"{ROOT}/nba/data/sb/*.json"))[-10:]:
            for e in json.load(open(x)):
                if any(k in str(e.get("status", "")).upper() for k in ("POSTPON", "CANCEL")):
                    skip.add(str(e["id"]))
        late = g[~g.completed & (pd.to_datetime(g.date, utc=True) + pd.Timedelta(hours=8) < now) & ~g.game_id.isin(skip)]
        done = g[g.completed]
        box = set(pd.read_csv(f"{ROOT}/nba/data/players.csv.gz", usecols=["game_id"], dtype={"game_id": str}).game_id)
        miss = done[~done.game_id.isin(box)]
        sh = {x[:-5] for x in os.listdir(f"{ROOT}/nba/data/shots")} if os.path.isdir(f"{ROOT}/nba/data/shots") else set()
        today, yday = f"{et:%Y-%m-%d}", f"{et - pd.Timedelta(days=1):%Y-%m-%d}"
        ung = sorted(d for d in (os.path.basename(x)[12:22] for x in glob.glob(f"{ROOT}/nba/output/predictions_*.csv"))
                     if d < today and not os.path.exists(f"{ROOT}/nba/output/review_{d}.json") and not (d == yday and et.hour < 11))
        msg = f"games final through {done.date_et.max()}; {len(done) - len(miss)}/{len(done)} with box scores; {len(set(done.game_id) & sh)}/{len(done)} with shot charts"
        msg += (f"; {len(late)} game(s) tipped 8+ hours ago with no final score on file (oldest {late.date_et.min()})" if len(late) else "") + (f"; forecasts never graded: {', '.join(ung)}" if ung else "")
        out.append(("NBA", miss.empty and late.empty and not ung, msg))
    except Exception as ex:  # noqa: BLE001
        out.append(("NBA", False, f"check failed: {ex}"))
    for sp, ok, msg in out:
        print(f"Data freshness {sp}: {'OK' if ok else 'STALE'} - {msg}")
    return out


def main():
    try:  # GitHub also runs this program: bring its newer outputs and data down before building
        import publish_site
        print("Sync with GitHub:", publish_site.pull())
    except Exception as e:  # noqa: BLE001  offline or no site repository: build from what is here
        print("sync skipped:", repr(e))
    d = E.load()
    c = S.Ctx(d)
    R = build_ranks(c)
    rep, summ = reports(), summaries()
    rows, rc = games(d, rep, summ)
    wk = current_week(rows)
    T = teams(c, R, rc[wk], rows)
    data = dict(season=SEASON, built=datetime.now().strftime("%a %b %-d, %-I:%M %p"), week=wk, rounds={str(k): v for k, v in ROUND.items()},
                mu=round(float(rc[wk]["mu"]), 2), metrics=[dict(label=a, key=k, fmt=f) for a, k, f in TEAM_M], games=rows, teams=T, track=track(rows), sd=SD)
    try:  # confident-winner tier and the spread rules tracked live at real prices (five-season study, Nov 2026)
        import spread_rules
        data["rules"] = spread_rules.build()
    except Exception as e:  # noqa: BLE001  the tracker must never stop the hub from building
        print("spread rules skipped:", repr(e))
        data["rules"] = None
    os.makedirs(os.path.join(HUB, "reports"), exist_ok=True)
    files = {}
    for dt, (src, rel) in sorted(rep.items()):
        copy_report(src, os.path.join(HUB, rel))
        files[rel] = os.path.join(HUB, rel)
    tpl = open(os.path.join(ROOT, "hub_template.html")).read()
    page = tpl.replace("/*DATA*/null", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))
    open(os.path.join(HUB, "index.html"), "w").write(page)
    import players  # Player Lookup data, loaded by the hub on demand
    files["players.json"] = players.build(os.path.join(HUB, "players.json"))
    import wrcb  # WR/CB matchups for the coming week, loaded by the hub on demand
    files["wrcb.json"] = wrcb.build(os.path.join(HUB, "wrcb.json"))
    import teampos  # Team vs Position tables, loaded by the hub on demand
    files["teampos.json"] = teampos.build(os.path.join(HUB, "teampos.json"))
    try:  # post-game review for every finished game (ESPN, fetched once per game)
        import nfl_reviews
        files["nflrev.json"] = nfl_reviews.build(os.path.join(HUB, "nflrev.json"))
    except Exception as e:  # noqa: BLE001  reviews are an extra: never block the hub
        print("NFL reviews skipped:", repr(e))
    import fantasy  # Fantasy page: point ranges from the latest simulated slates, loaded by the hub on demand
    files["fantasy.json"] = fantasy.build(os.path.join(HUB, "fantasy.json"))
    sys.path.insert(0, os.path.join(ROOT, "cfb"))
    import hub_cfb  # College football pages: cfb.json plus the week Game Books
    try:
        files.update(hub_cfb.build(HUB, copy_report))
    except Exception as e:  # a college data problem must not stop the NFL hub from building
        print("CFB hub data skipped:", repr(e))
    sys.path.insert(0, os.path.join(ROOT, "nba"))
    import hub_nba  # NBA pages: nba.json (tonight's lines, track record, player lookup)
    try:
        files.update(hub_nba.build(HUB))
    except Exception as e:  # an NBA data problem must not stop the NFL hub from building
        print("NBA hub data skipped:", repr(e))
    try:  # NHL pages: nhl.json (games, model numbers, standings, stat leaders, track record)
        sys.path.insert(0, os.path.join(ROOT, "nhl"))
        import hub_nhl
        files.update(hub_nhl.build(HUB))
    except Exception as e:  # noqa: BLE001  an NHL data problem must not stop the hub from building
        print("NHL hub data skipped:", repr(e))
    try:  # Champions League pages (built from TheBrogram/soccer on this computer; GitHub reuses the kept copy)
        import hub_ucl
        files.update(hub_ucl.build(HUB))
    except Exception as e:  # noqa: BLE001  a soccer data problem must not stop the hub from building
        print("UCL hub data skipped:", repr(e))
    try:  # Home page: parlay and picks of the day, typed in on this computer with picks_desk.py
        src = os.path.join(ROOT, "picks", "picks.json")
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(HUB, "picks.json"))
            files["picks.json"] = os.path.join(HUB, "picks.json")
    except Exception as e:  # noqa: BLE001  a picks problem must not stop the hub from building
        print("Picks skipped:", repr(e))
    try:  # NFL team logos for the hub's game tiles (the Game Books embed their own copies)
        os.makedirs(os.path.join(HUB, "logos"), exist_ok=True)
        for src in glob.glob(os.path.join(ROOT, "data", "logos", "*.png")):
            dst = os.path.join(HUB, "logos", os.path.basename(src))
            shutil.copy2(src, dst)
            files["logos/" + os.path.basename(src)] = dst
    except Exception as e:  # noqa: BLE001  missing logos must not stop the hub from building
        print("Logos skipped:", repr(e))
    try:  # the site's own logo (data/brand, made from TheBrogramLogo.png): top bar, footer, browser tab icon, share image
        os.makedirs(os.path.join(HUB, "brand"), exist_ok=True)
        for n in ("logo-96.png", "icon-32.png", "apple-touch-icon.png", "icon-512.png"):
            dst = os.path.join(HUB, "brand", n)
            shutil.copy2(os.path.join(ROOT, "data", "brand", n), dst)
            files["brand/" + n] = dst
    except Exception as e:  # noqa: BLE001
        print("Brand logo skipped:", repr(e))
    try:  # Home page board: upcoming games across every sport with each team's chance (tonight.json)
        import hub_tonight
        files.update(hub_tonight.build(HUB, data))
    except Exception as e:  # noqa: BLE001  a board problem must not stop the hub from building
        print("Tonight board skipped:", repr(e))
    try:  # one page per sport (nfl/, nba/, …) plus sitemap.xml and robots.txt
        import hub_pages
        files.update(hub_pages.build(HUB))
    except Exception as e:  # noqa: BLE001  the sport pages must not stop the hub from building
        print("Sport pages skipped:", repr(e))
    json.dump(files, open(os.path.join(HUB, "files.json"), "w"), indent=1)
    n = {k: sum(1 for x in rows if x["status"] == k) for k in ("final", "book", "lines", "early")}
    print(f"Hub: Week {wk} is current. {n['final']} final, {n['book']} with a Game Book, {n['lines']} with lines, {n['early']} early. "
          f"{sum(k.startswith('reports/') for k in files)} reports. {len(page) / 1e3:.0f} KB -> {os.path.join(HUB, 'index.html')}")
    freshness()


if __name__ == "__main__":
    main()
