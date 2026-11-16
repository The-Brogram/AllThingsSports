"""Render every game into output/nfl_<date>_report.html: a side-by-side Matchup tab, a three-tier Picks tab, and the full phase report."""
import html
import json
import math
import os
import pickle
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import engine as E
import scheme as S
import gamebook as GB
import lines as LN
from backtest import apply_cal, cal_key

from config import CACHE, COORD, DATE, DAY, HOLD_RUSH, NOTES, OUT, PICKS, TITLE, WEEK

QB_CHANGE = NOTES  # a team key flags its props; a key ending in "_" is display only


def qb_held(r):
    """Passing and receiving legs of a QB-change team: the new QB's yardage rests on a prior, not his own sample."""
    return r["mk"] in ("pass", "rec") and r["p"]["team"] in QB_CHANGE
PULLED = datetime.now().strftime("%A, %b %-d at %-I:%M %p")


def e(x):
    return html.escape(str(x))


def n(x, d=1):
    return "–" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{d}f}"


def pc(x, d=0):
    return "–" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{100 * x:.{d}f}%"


def sg(x, d=3):
    return "–" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:+.{d}f}"


def am(p):
    p = min(max(p, .001), .999)
    return f"{-100 * p / (1 - p):+.0f}" if p >= .5 else f"{100 * (1 - p) / p:+.0f}"


def be(o):
    return abs(o) / (abs(o) + 100) if o < 0 else 100 / (o + 100)


def payout(o):
    return 100 / abs(o) if o < 0 else o / 100


def T(head, rows, cap=None, cls=""):
    th = "".join(f"<th>{e(h)}</th>" for h in head)
    tr = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    c = f"<caption>{cap}</caption>" if cap else ""
    return f'<div class="tw"><table class="{cls}">{c}<thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>'


def phase(num, title, body):
    return f'<section class="ph"><h3><span class="pn">{e(num)}</span>{e(title)}</h3>{body}</section>'


def tag(cls, txt):
    return f'<span class="tag {cls}">{e(txt)}</span>'


def fx(p, k, side="over"):
    return p


class G:
    """Everything derived for one game."""

    def __init__(self, s, c, cal):
        self.s, self.c, self.cal = s, c, cal
        g = s["game"]
        self.g, self.h, self.a = g, g["home"], g["away"]
        self.pl = s["pl_a"] + s["pl_h"]
        self.rng = np.random.default_rng(int(g["id"]) % 99991)

    def cp(self, arr, k, mk, p=None):
        raw = float((arr >= k).mean())
        return float(apply_cal(np.array([raw]), self.cal[cal_key(self.cal, mk, p and p.get("tshare"))])[0])

    def side(self, t):
        return self.s["sh"] if t == self.h else self.s["sa"]

    def qb(self, t):
        return next((p for p in self.pl if p["team"] == t and p["pos"] == "QB"), None)

    def top(self, t, pos, key, k=1):
        x = sorted([p for p in self.pl if p["team"] == t and p["pos"] in pos], key=lambda z: -z[key].mean() if hasattr(z[key], "mean") else -z[key])
        return x[:k]

    def line(self, p, name):
        if p.get("espn") is None:
            return None
        v = self.s["props"].get((self.g["id"], int(p["espn"]), name))
        v = sorted(x for x in (v or []) if x is not None)
        return v[len(v) // 2] if v else None

    def miles(self, p, name):
        if p.get("espn") is None:
            return []
        return sorted(x for x in self.s["props"].get((self.g["id"], int(p["espn"]), name), []) if x is not None)


MK = {"rec": ("Total Receiving Yards (incl. overtime)", "Receiving Yards Milestones", "rec_y", "receiving yds"),
      "rush": ("Total Rushing Yards (incl. overtime)", "Rushing Yards Milestones", "rush_y", "rushing yds"),
      "pass": ("Total Passing Yards (incl. overtime)", "Passing Yards Milestones", "pass_y", "passing yds")}


BAND = {"rec": 50, "rush": 40, "pass": 0}
SHIFT = {"rb": set(), "pl": set()}


def find_shifts(c):
    """Snap-share swing of 20+ points in a player's latest game vs his earlier 2026 average."""
    sn = c.snaps[c.snaps.game_type == "REG"].sort_values("week")
    out = []
    for (t, name), x in sn.groupby(["team", "player"]):
        if len(x) < 3 or x.position.iloc[-1] not in ("RB", "WR", "TE", "FB"):
            continue
        last, prev = x.offense_pct.iloc[-1], x.offense_pct.iloc[:-1].mean()
        last, prev = (last, prev) if prev <= 1 and last <= 1 else (last / 100, prev / 100)
        if abs(last - prev) >= .20 and max(last, prev) >= .35:
            SHIFT["pl"].add(name)
            if x.position.iloc[-1] in ("RB", "FB"):
                SHIFT["rb"].add(t)
            out.append((t, name, x.position.iloc[-1], prev, last))
    return out
OFFSET, RAWZ = {}, {}


def value_rows(G_):
    rows = []
    for p in G_.pl:
        for mk, (main, _, key, lab) in MK.items():
            if key not in p:
                continue
            L = G_.line(p, main)
            if L is None or (mk != "pass" and L < 15):
                continue  # very low lines are mostly scramble noise
            arr = p[key]
            raw = float((arr > L).mean())
            c = float(apply_cal(np.array([raw]), G_.cal[cal_key(G_.cal, mk, p.get("tshare"))])[0])
            z = np.log(np.clip(c, .01, .99) / (1 - np.clip(c, .01, .99)))
            band = (mk, L >= BAND[mk])
            med = float(np.median(arr))
            shift = (mk == "rush" and p["team"] in SHIFT["rb"] and p["pos"] != "QB") or p["name"] in SHIFT["pl"]
            role = mk != "pass" and (abs(med - L) / L > .33 or shift)  # a large gap or fresh snap swing usually means role news history has not caught up to
            if not role:  # a stale role line would drag the whole band's tilt
                RAWZ.setdefault(band, []).append(z)
            z -= OFFSET.get(band, 0.0)  # keep only disagreement relative to how this market band sits overall
            sh = float(1 / (1 + np.exp(-.5 * z)))
            over = sh >= .5
            q = sh if over else 1 - sh
            flag = p["play_p"] < .9 or p["team"] in QB_CHANGE or role
            rows.append(dict(p=p, mk=mk, lab=lab, line=L, med=float(np.median(arr)), mean=float(arr.mean()), cal_over=c, over=over, prob=q, edge=q - be(-110), flag=flag, role=role, key=key,
                             minodds=am(max(q - .02, .02)), game=f"{G_.a}@{G_.h}", g=G_))
    return sorted(rows, key=lambda r: -r["edge"])


def miles_rows(G_):
    rows = []
    for p in G_.pl:
        for mk, (_, ms, key, lab) in MK.items():
            if key not in p:
                continue
            for k in G_.miles(p, ms):
                if mk == "rush" and k < 15:
                    continue
                prob = G_.cp(p[key], k, mk, p)
                rows.append(dict(p=p, mk=mk, lab=lab, k=k, prob=prob, key=key, game=f"{G_.a}@{G_.h}", g=G_))
    return rows


def team_compare(c, a, h, ranks):
    A, H = c.team_side(a, "off")["Season"], c.team_side(h, "off")["Season"]
    Ad, Hd = c.team_side(a, "def")["Season"], c.team_side(h, "def")["Season"]

    def rk(metric, t, side, hi=True):
        x = ranks[side][metric]
        return x.get(t)
    rows = []
    M = [("Points per game", "ppg", 1, 1), ("Yards per game", "ypg", 0, 1), ("Pass yards / game", "pass_y", 0, 1), ("Rush yards / game", "rush_y", 0, 1), ("Yards per play", "ypp", 2, 1), ("Yards per attempt", "ypa", 2, 1),
         ("Yards per carry", "ypc", 2, 1), ("Success rate", "success", -1, 1), ("EPA / play", "epa", 3, 1), ("Pass EPA / play", "pass_epa", 3, 1), ("Rush EPA / play", "rush_epa", 3, 1), ("3rd-down rate", "third", -1, 1), ("Red-zone TD rate", "rz_td", -1, 1),
         ("Turnovers / game", "to", 2, -1), ("Sacks / game", "sacks", 1, -1), ("Penalties / game", "penalties", 1, -1), ("Explosive pass (20+)", "expl_pass", -1, 1), ("Explosive rush (12+)", "expl_rush", -1, 1)]

    def f(v, d):
        return pc(v, 1) if d == -1 else n(v, d)
    for lab, k, d, sign in M:
        def cell(tm, side, v):
            r = ranks[side][k].get(tm)
            return f"{f(v, d)} <small>#{int(r)}</small>"
        rows.append([lab, cell(a, "off", A[k]), cell(h, "def", Hd[k]), cell(h, "off", H[k]), cell(a, "def", Ad[k])])
    return T(["Metric", f"{a} offense", f"{h} defense", f"{h} offense", f"{a} defense"], rows, "2026 regular season. # is the league rank, with #1 the best for that unit.")


def build_ranks(c):
    teams = sorted(c.p26.posteam.dropna().unique())
    off, de = {}, {}
    for t in teams:
        off[t] = c.team_side(t, "off")["Season"]
        de[t] = c.team_side(t, "def")["Season"]
    R = {"off": {}, "def": {}}
    low = {"to", "sacks", "penalties"}
    for k in off[teams[0]]:
        a, b = pd.Series({t: off[t][k] for t in teams}), pd.Series({t: de[t][k] for t in teams})
        R["off"][k] = a.rank(ascending=k in low).to_dict()
        R["def"][k] = b.rank(ascending=k not in {"to", "sacks", "penalties"}).to_dict()
    return R


def trends(c, t):
    o, d = c.team_side(t, "off"), c.team_side(t, "def")
    rows = []
    for lab in ("Season", "Last 5", "Last 3", "Home", "Away"):
        x, y = o[lab], d[lab]
        rows.append([f"{t} {lab}", x["g"], n(x["ppg"]), sg(x["epa"]), pc(x["success"], 1), n(x["ypp"], 2), n(y["ppg"]), sg(y["epa"]), pc(y["success"], 1), n(y["ypp"], 2)])
    return rows


def strengths(c, t, R):
    good, bad = [], []
    names = {"epa": "overall EPA", "pass_epa": "passing EPA", "rush_epa": "rushing EPA", "success": "success rate", "third": "3rd downs", "rz_td": "red-zone TDs", "expl_pass": "explosive passing", "expl_rush": "explosive runs"}
    for side, lab in (("off", "offense"), ("def", "defense")):
        for k, nm in names.items():
            r = R[side][k][t]
            if r <= 7:
                good.append((r, f"{nm} on {lab} (#{int(r)})"))
            elif r >= 26:
                bad.append((-r, f"{nm} on {lab} (#{int(r)})"))
    return [x[1] for x in sorted(good)[:5]], [x[1] for x in sorted(bad)[:5]]


def matchup_read(c, off, de):
    """Plain-language exploit list built from the numbers, for one offense versus one defense."""
    z = c.zones(de)
    out = []
    z = z.assign(diff=z.epa - z.epa_lg)
    v = z[(z.att >= 12)].sort_values("diff", ascending=False)
    for r in v.head(2).itertuples():
        if r.diff > .12:
            out.append(f"{de} is soft on {r.depth} throws to the {r.pass_location} ({sg(r.epa, 2)} EPA/att vs league {sg(r.epa_lg, 2)}, {r.att} attempts).")
    for r in v.tail(1).itertuples():
        if r.diff < -.12:
            out.append(f"Avoid {r.depth} throws to the {r.pass_location}: {de} allows {sg(r.epa, 2)} EPA/att vs league {sg(r.epa_lg, 2)} ({r.att} attempts).")
    pr, po = c.pressure(de), c.pressure(off)
    if pr["pr"] > pr["lg_pr"] * 1.12 and po["pr_allowed"] > pr["lg_pr"] * 1.05:
        out.append(f"Pressure mismatch favors {de}: its {pc(pr['pr'])} pressure rate meets a {off} line allowing {pc(po['pr_allowed'])} (league {pc(pr['lg_pr'])}).")
    elif pr["pr"] < pr["lg_pr"] * .9 and po["pr_allowed"] < pr["lg_pr"]:
        out.append(f"{off} should get clean pockets: {de} generates pressure on {pc(pr['pr'])} of dropbacks and {off} allows {pc(po['pr_allowed'])}.")
    for x in c.alignment_allowed(de):
        if x["yds"] > x["lg"] * 1.25:
            out.append(f"{de} gives up {n(x['yds'], 0)} yds/game to {x['role']} (league {n(x['lg'], 0)}).")
        elif x["yds"] < x["lg"] * .75:
            out.append(f"{de} limits {x['role']} to {n(x['yds'], 0)} yds/game (league {n(x['lg'], 0)}).")
    for x in c.run_def(de):
        if x["n"] >= 25 and x["ypc"] > x["lg_ypc"] + .6:
            out.append(f"Run lane: {x['zone']} gains {n(x['ypc'], 1)} YPC on {de} (league {n(x['lg_ypc'], 1)}).")
        if x["n"] >= 25 and x["ypc"] < x["lg_ypc"] - .6:
            out.append(f"Run stuffed: {x['zone']} gains only {n(x['ypc'], 1)} YPC on {de} (league {n(x['lg_ypc'], 1)}).")
    return out or ["No exploit in this matchup clears the sample and size filters."]


def weather_block(gm):
    g = gm["game"]
    h = g["home"]
    f = f"{E.DATA}/weather/{h}.json"
    kick = datetime.fromisoformat(g["kick"].replace("Z", "+00:00"))
    w = None
    if os.path.exists(f):
        ps = json.load(open(f))["properties"]["periods"]
        for p in ps:
            st = datetime.fromisoformat(p["startTime"]).astimezone(timezone.utc)
            if st <= kick < st + timedelta(hours=1):
                w = p
    return w, kick


def dist(a, b):
    la, lo = map(math.radians, COORD[a])
    lb, lp = map(math.radians, COORD[b])
    return 3959 * 2 * math.asin(math.sqrt(math.sin((lb - la) / 2) ** 2 + math.cos(la) * math.cos(lb) * math.sin((lp - lo) / 2) ** 2))


def scenarios(gg):
    s = gg.s
    m = s["margin"]
    a, b, c_ = m >= 8, m <= -8, np.abs(m) < 8
    rows = []
    who = []
    for t in (gg.h, gg.a):
        q = gg.qb(t)
        if q is not None:
            who.append((f"{q['name']} pass yds", q["pass_y"]))
        for lab, pos, key in (("rush yds", ("RB",), "rush_y"), ("rec yds", ("WR",), "rec_y"), ("rec yds", ("TE",), "rec_y")):
            for p in gg.top(t, pos, key):
                who.append((f"{p['name']} {lab}", p[key]))
    for lab, arr in who:
        rows.append([e(lab), n(arr[a].mean(), 0), n(arr[b].mean(), 0), n(arr[c_].mean(), 0), n(arr.mean(), 0)])
    probs = (a.mean(), b.mean(), c_.mean())
    head = ["Player stat (mean)", f"A: {gg.h} controls", f"B: {gg.a} controls", "C: close", "All sims"]
    sa, sh = s["sa"], s["sh"]
    ex = []
    for lab, mask in (("A", a), ("B", b), ("C", c_)):
        ex.append([lab, n(sh["att"][mask].mean(), 0), n(sh["car"][mask].mean(), 0), n(sa["att"][mask].mean(), 0), n(sa["car"][mask].mean(), 0)])
    return probs, T(head, rows), T(["Scenario", f"{gg.h} pass att", f"{gg.h} carries", f"{gg.a} pass att", f"{gg.a} carries"], ex, "Simulated volume by game script")


def scheme_block(c, de, off):
    cv = c.coverage25(de)
    pr = c.pressure(de)
    rows = [[e(r["cov"]), pc(r["rate"]), r["n"], sg(r["epa"], 2), n(r["ypa"], 1), r["td"], r["expl"]] for r in cv["rows"]]
    note = f"<p class='note'><b>2025 charting, used as a proxy.</b> No free 2026 man/zone feed exists. {de} ran man on {pc(cv['man'])}, single-high on {pc(cv['single'])}, and two-high on {pc(cv['two'])} of charted 2025 dropbacks."
    if cv["hc_change"]:
        note += f" <b>{de} changed head coaches ({e(c.hc25.get(de))} to {e(c.hc26.get(de))}), so treat this table as weak evidence.</b>"
    note += " Press rate and CB shadow rate are not in any free source and are left out rather than guessed.</p>"
    t1 = T(["Coverage", "Usage", "Dropbacks", "EPA/db", "Yds/att", "TD", "20+ plays"], rows)
    z = c.zones(de).assign(d=lambda x: x.epa - x.epa_lg)
    zr = [[e(r.depth), e(r.pass_location.title()), r.att, n(r.yds, 1), sg(r.epa, 2), sg(r.epa_lg, 2), f'<span class="{"bad" if r.d > .12 else ("good" if r.d < -.12 else "")}">{sg(r.d, 2)}</span>'] for r in z.itertuples()]
    t2 = T(["Depth", "Side", "Att", "Yds/att", "EPA/att", "League", "vs league"], zr, f"2026 passing against {de} by field zone. Positive means worse for the defense. Small cells are noisy.")
    pt = T(["Pressure rate", "League", "Blitz rate", "4-man rush", "5+ rush", "Light box (6-)", "Sack rate", "EPA when blitzing", "EPA without blitz", "QB hits"],
           [[pc(pr["pr"], 1), pc(pr["lg_pr"], 1), pc(pr["blitz"], 1), pc(pr["four"], 1), pc(pr["five_plus"], 1), pc(pr["box_light"], 1), pc(pr["sack_rate"], 1), sg(pr["epa_blitz"], 2), sg(pr["epa_non"], 2), int(pr["hits"])]], f"2026 FTN charting and PFR ({pr['n']} charted dropbacks).")
    ru = c.rushers(de)
    t3 = T(["Pass rusher", "G", "Pressures", "Sacks", "QB hits"], [[e(x["pfr_player_name"]), x["g"], int(x["pr"]), n(x["sk"], 1), int(x["hit"])] for x in ru])
    cr = c.corners(de)
    t4 = T(["Spot", "Defender", "Snap %", "G", "Targets", "Comp", "Yards", "Yds/tgt", "TD", "INT"],
           [[x["pos"], e(x["name"]), pc((x["snap"] or 0) / 1 if x["snap"] and x["snap"] <= 1 else (x["snap"] or 0) / 100), x.get("g", "–"), x.get("tg", "–"), x.get("cmp", "–"), x.get("yds", "–"), n(x.get("ypt"), 1), x.get("td", "–"), x.get("ints", "–")] for x in cr],
           "Secondary starters from the current ESPN depth chart, with 2026 PFR coverage numbers.")
    al = c.alignment_allowed(de)
    t5 = T(["Role", "Tgt/g", "Rec/g", "Yds/g", "League yds/g", "TD total"], [[e(x["role"]), n(x["tg"]), n(x["rec"]), n(x["yds"]), n(x["lg"]), int(x["td"])] for x in al], "Production allowed by target role per game. WR1 is whoever led that opponent's WRs in targets, which works as a stand-in for alignment splits.")
    rd = "".join(f"<li>{e(x)}</li>" for x in matchup_read(c, off, de))
    return f"<h4>{de} defense vs {off} offense: what to attack</h4><ul>{rd}</ul><h4>Coverage shell</h4>{t1}{note}<h4>Field zones</h4>{t2}<h4>Pressure</h4>{pt}{t3}<h4>Defenders</h4>{t4}{t5}"


def wr_block(c, gg, t, opp):
    rs = sorted([p for p in gg.pl if p["team"] == t and p["pos"] in ("WR", "TE", "RB") and p["tg"] >= 1.8], key=lambda z: -z["tg"])
    rows, rows2, blocks = [], [], ""
    for p in rs:
        row = pd.Series(p)
        w = c.wr_row(row, t)
        sp = w["snap"]
        sp = sp if sp is None or not math.isfinite(sp) else (sp if sp <= 1 else sp / 100)
        rows.append([e(p["name"]), p["pos"], w["g"], pc(sp), n(w["tg"]), pc(w["share"], 1), pc(w["air"], 1), pc(w["first"]), n(w["adot"]), n(w["rec"]), n(w["yds"]), n(w["ypt"]), n(w["yac"]), n(w["sep"]), w["rz"], w["ez"], w["td"]])
        rows2.append([e(p["name"]), pc(w["left"]), pc(w["mid"]), pc(w["right"]), pc(w["short"]), pc(w["inter"]), pc(w["deep"]), w["screen"], e(w["routes"])])
        rf = c.route_fit(p["player_id"], opp)
        cp = c.comps(row, opp)
        b = ""
        if rf:
            b += T(["Route (2025 sample)", f"{p['name'].split()[-1]} tgt", "His yds/tgt", f"{opp} tgt faced", f"{opp} yds/tgt", "League"], [[e(x["route"]), x["n"], n(x["ypt"], 1), x["dn"], n(x["dypt"], 1), n(x["lypt"], 1)] for x in rf])
        if cp and p["pos"] != "RB":
            b += "<p class='note'><b>Similar receivers vs " + opp + " in 2026</b> (same position, aDOT within 3 yds, 4+ targets): " + "; ".join(f"{e(x['name'])} Wk{x['week']} {x['rec']}/{x['tg']}, {x['yds']} yds{', ' + str(x['td']) + ' TD' if x['td'] else ''}" for x in cp[:7]) + f". Sample of {len(cp)}: context only.</p>"
        if b and p["tg"] >= 3.2:
            blocks += f"<details><summary>{e(p['name'])}: route fit and similar players</summary>{b}</details>"
    t1 = T(["Player", "Pos", "G", "Snap", "Tgt/g", "Tgt share", "Air share", "1st-read", "aDOT", "Rec/g", "Yds/g", "Yds/tgt", "YAC/rec", "Sep", "RZ tgt", "EZ tgt", "TD"], rows, f"{t} usage in 2026. 1st-read is the share of his targets charted as the first read. Sep is NGS average separation. Routes run per game are not in a free 2026 source, so snap share and targets stand in.")
    t2 = T(["Player", "Left", "Middle", "Right", "Short", "Inter", "Deep", "Screens", "Most-targeted routes (2025)"], rows2, "Target locations in 2026. Free data has no 2026 slot/wide alignment, so side and depth stand in for it. Check the route column against the opponent's soft zones above.")
    return t1 + t2 + blocks


def qb_block(c, gg, t, opp):
    q = gg.qb(t)
    if q is None:
        return "<p>No starting QB could be modeled.</p>"
    x = c.qb(q["name"], t)
    pr = c.pressure(opp)
    sim = gg.side(t)
    rw = lambda lab, d, ss=False: [lab, d["n"], pc(d["cmp"], 1), n(d["ypa"], 1), sg(d["epa"], 2), pc(d["sack"], 1)]
    r = [rw("Blitzed (2026)", x["blitz"]), rw("Not blitzed (2026)", x["non"]), rw("4-man rush (2026)", x["four"]), rw("Deep 20+ air (2026)", x["deep"]), rw("Play action (2026)", x["pa"]), rw("Out of pocket (2026)", x["oop"]),
         rw("Pressured (2025)", x["press25"]), rw("Clean pocket (2025)", x["clean25"]), rw("vs man (2025)", x["man25"]), rw("vs zone (2025)", x["zone25"]), rw("vs single-high (2025)", x["single25"]), rw("vs two-high (2025)", x["two25"])]
    head = f"<p><b>{e(q['name'])}</b>: {x['dbs']} dropbacks in 2026, {sg(x['epa'], 2)} EPA/db, CPOE {n(x['cpoe'], 1)}, time to throw {n(x['ttt'], 2)}s, intended air yards {n(x['air'], 1)}, sack rate {pc(x['sack'], 1)}, turnover-worthy throws {pc(x['iw'], 1)}. Pressure-to-sack rate in 2025: {pc(x['pts'], 1)}. 2026 under-pressure splits are not free, so 2025 stands in and blitz splits come from 2026 charting.</p>"
    t1 = T(["Situation", "Dropbacks", "Comp%", "Yds/att", "EPA/db", "Sack%"], r)
    if t in QB_CHANGE:
        head = f"<p class='warn'>{e(QB_CHANGE[t])}</p>" + head
    ol = c.oline(t, gg.c_status)
    t2 = T(["Spot", "Expected starter", "Status", "Snap % last 3", "Next up"], [[y["pos"], e(y["name"]), e(y["status"]), pc((y["snap3"] if y["snap3"] and y["snap3"] <= 1 else (y["snap3"] or 0) / 100)), e(y["backup"])] for y in ol], "Offensive line from the current depth chart.")
    o = c.pressure(t)
    ru = c.rushers(opp)
    top = ", ".join(f"{a['pfr_player_name']} ({int(a['pr'])} pressures)" for a in ru[:3])
    py = q["pass_y"]
    pj = T(["Att", "Comp", "Pass yds", "Sacks", "INT", "Pass TD", "10th-90th yds"], [[n(sim["att"].mean()), n(sim["comp"].mean()), n(py.mean(), 0), n(sim["sacks"].mean()), n(sim["ints"].mean()), n(sim["ptd"].mean()), f"{np.percentile(py, 10):.0f}-{np.percentile(py, 90):.0f}"]], "Simulated line")
    verdict = "volume and efficiency" if pr["pr"] > pr["lg_pr"] * 1.1 and o["pr_allowed"] > pr["lg_pr"] * 1.05 else ("mostly efficiency" if pr["pr"] > pr["lg_pr"] else "neither much")
    pv = f"<p>{t} line allows pressure on {pc(o['pr_allowed'])} of dropbacks and {opp} creates it on {pc(pr['pr'])} (league {pc(pr['lg_pr'])}). Main rushers: {e(top)}. Pressure looks likely to limit {verdict}. Individual tackle-vs-edge win rates are paid data and are not included.</p>"
    return head + t1 + t2 + pv + pj


def rb_block(c, gg, t, opp):
    rs = sorted([p for p in gg.pl if p["team"] == t and p["car"] >= 1.5], key=lambda z: -z["car"])
    rows, proj = [], []
    for p in rs:
        w = c.rb_row(pd.Series(p), t)
        f = lambda v: v if v is None or not math.isfinite(v) or v <= 1 else v / 100
        rows.append([e(p["name"]), p["pos"], w["g"], n(w["car"]), n(w["l3"]), pc(w["share"]), pc(f(w["snap"])), pc(f(w["snap3"])), n(w["ypc"], 2), pc(w["succ"]), pc(w["expl"]), n(w["ryoe"], 2), n(w["yco"], 2), w["btk"], w["rz"], w["i10"], w["i5"], w["third"], n(w["tg"]), w["td"]])
        y = p["rush_y"]
        proj.append([e(p["name"]), n(p["car"]), n(y.mean(), 0), f"{np.percentile(y, 10):.0f}-{np.percentile(y, 90):.0f}", n(p["tg"]), n(p["rec_y"].mean(), 0)])
    rd = c.run_def(opp)
    t3 = T(["Run type vs " + opp, "Carries", "YPC", "League", "EPA", "League", "Success", "12+ rate"], [[e(x["zone"]), x["n"], n(x["ypc"], 2), n(x["lg_ypc"], 2), sg(x["epa"], 2), sg(x["lg_epa"], 2), pc(x["succ"]), pc(x["expl"])] for x in rd])
    return T(["Back", "Pos", "G", "Car/g", "Last 3", "Rush share", "Snap", "Snap L3", "YPC", "Success", "12+", "RYOE/c", "YAC/c", "BrkTk", "RZ car", "In 10", "In 5", "3rd dn car", "Tgt/g", "TD"], rows, f"{t} backfield in 2026. RYOE is NGS rush yards over expected per carry. YAC/c is yards after contact (PFR).") + t3 + T(["Back", "Carries", "Rush yds", "10th-90th", "Tgt", "Rec yds"], proj, "Simulated workload") + "<p class='note'>Zone, gap, power and counter splits need paid charting and are left out. Inside and outside splits come from play-by-play run direction.</p>"


def td_block(gg):
    rows = []
    for p in sorted(gg.pl, key=lambda z: -z["atd"])[:14]:
        lo, hi = max(p["atd"] - .07, .01), min(p["atd"] + .07, .95)
        sh = "rush" if p["rush_td"] > p["rec_td"] else "receiving"
        rows.append([e(p["name"]), p["team"], p["pos"], pc(p["atd"]), am(p["atd"]), f"{pc(lo)}-{pc(hi)}", n(p["rush_td"], 2), n(p["rec_td"], 2), f"{pc(p['i10'])} of in-10 carries" if sh == "rush" else f"{pc(p['rz'])} of RZ targets"])
    return rows


def hit_block(c, gg):
    out = ""
    for t in (gg.a, gg.h):
        q = gg.qb(t)
        L = []
        if q is not None:
            L.append((q, "passy", [175, 200, 225, 250, 275, 300], q["pass_y"], gg.side(t)["att"].mean(), "pass"))
        for p in sorted([x for x in gg.pl if x["team"] == t and x["pos"] != "QB"], key=lambda z: -(z["rec_y"].mean() + z["rush_y"].mean()))[:6]:
            if p["rec_y"].mean() >= p["rush_y"].mean():
                L.append((p, "rec", [20, 30, 40, 50, 60, 70, 80, 100], p["rec_y"], max(p["tg"], 2), "rec"))
            else:
                L.append((p, "rush", [30, 40, 50, 60, 70, 80, 100], p["rush_y"], max(p["car"], 3), "rush"))
        for p, kind, ths, arr, vol, mk in L:
            rows, last = c.hitrates(p["player_id"], kind, ths, gg.h if t == gg.a else gg.a, t, vol)
            f = lambda q_: f"{q_[0]}/{q_[1]}"
            body = [[f"{r['k']}+", f(r["career"]), f(r["season"]), f(r["l10"]), f(r["l5"]), f(r["home"]), f(r["away"]), f(r["vol"]), f(r["tier"]), f"<b>{pc(gg.cp(arr, r['k'], mk, p))}</b>"] for r in rows]
            label = dict(passy="passing", rec="receiving", rush="rushing")[kind]
            out += T(["Threshold", "Career*", "2026", "Last 10", "Last 5", "2026 home", "2026 road", "Similar volume", "Vs similar D", "Model"], body, f"<b>{e(p['name'])}</b> ({t}) {label} yards. Last 5: {', '.join(map(str, last))}")
    return out + "<p class='note'>*Career means every game with a target, carry, or attempt since 2015, the start of this data pull. Only games with usage count, so a limited-snap game still counts in the denominator. Similar volume is within 20% of projected usage. Similar D uses the same 2026 pass or run EPA tier as this opponent. Model is the calibrated forecast, shown separately from history on purpose.</p>"


def prob_tables(gg):
    out = ""
    rec, rush, pa = [], [], []
    TH = {"rec": [20, 30, 40, 50, 60, 70, 80, 90, 100, 120], "rush": [20, 30, 40, 50, 60, 70, 80, 90, 100, 120], "pass": [150, 175, 200, 225, 250, 275, 300, 325]}
    for p in sorted(gg.pl, key=lambda z: -z["rec_y"].mean()):
        if p["rec_y"].mean() >= 10:
            rec.append([e(p["name"]), p["team"]] + [pc(gg.cp(p["rec_y"], k, "rec", p)) for k in TH["rec"]])
    for p in sorted(gg.pl, key=lambda z: -z["rush_y"].mean()):
        if p["rush_y"].mean() >= 10:
            rush.append([e(p["name"]), p["team"]] + [pc(gg.cp(p["rush_y"], k, "rush", p)) for k in TH["rush"]])
    for t in (gg.a, gg.h):
        q = gg.qb(t)
        if q is not None:
            pa.append([e(q["name"]), t] + [pc(gg.cp(q["pass_y"], k, "pass")) for k in TH["pass"]])
    return (T(["Receiver", "Tm"] + [f"{k}+" for k in TH["rec"]], rec, "Calibrated P(receiving yards at or above threshold). The under is 100% minus the cell.", "heat") +
            T(["Rusher", "Tm"] + [f"{k}+" for k in TH["rush"]], rush, "Calibrated P(rushing yards at or above threshold)", "heat") +
            T(["QB", "Tm"] + [f"{k}+" for k in TH["pass"]], pa, "Calibrated P(passing yards at or above threshold)", "heat"))


def player_proj(gg):
    rows = []
    for p in sorted(gg.pl, key=lambda z: (z["team"] != gg.a, -(z["rec_y"].mean() + z["rush_y"].mean() + (z["pass_y"].mean() if "pass_y" in z else 0)))):
        pyd = p.get("pass_y")
        tot = p["rec_y"] + p["rush_y"] + (pyd if pyd is not None else 0)
        if p["rec_y"].mean() + p["rush_y"].mean() < 7 and pyd is None:
            continue
        s_ = gg.side(p["team"])
        opp = f"{n(s_['att'].mean(), 0)} att / {n(p['car'])} car" if pyd is not None else f"{n(p['tg'])} tgt / {n(p['car'])} car"
        rows.append([e(p["name"]) + (f" {tag('warn', 'Q ' + pc(p['play_p']))}" if p["play_p"] < .99 else ""), p["team"], p["pos"], opp, n(pyd.mean(), 0) if pyd is not None else "–", n(p["rush_y"].mean(), 0), n(p["rec_y"].mean(), 0), n(p["rec"]), pc(p["atd"]), f"{np.percentile(tot, 10):.0f}-{np.percentile(tot, 90):.0f}"])
    return T(["Player", "Tm", "Pos", "Opportunities", "Pass yds", "Rush yds", "Rec yds", "Rec", "Anytime TD", "10th-90th total yds"], rows, "Means from 20,000 simulations. Q shows the chance he plays, and if he sits his share goes to teammates inside the same sim.")


def game_lines(gg):
    g, s = gg.g, gg.s
    m, tot = s["margin"], s["total"]
    h, a = gg.h, gg.a
    sp = g["spread_home"]
    res = []

    def cover(line):
        w, p = (m > line).mean(), (m == line).mean()
        return w, p
    ph = (m > 0).mean() + .5 * (m == 0).mean()
    rows = []
    bets = []
    for lab, q, o in ((f"{h} moneyline", ph, g["ml_h"]), (f"{a} moneyline", 1 - ph, g["ml_a"])):
        if o is None:
            continue
        rows.append([lab, f"{o:+.0f}", pc(be(o), 1), pc(q, 1), f"{(q * payout(o) - (1 - q)) * 100:+.1f}%"])
        bets.append((lab, q, o))
    wh, _ = cover(sp)
    lh, _ = (m < sp).mean(), 0
    hc = wh / max(wh + lh, 1e-9)
    tb = "spread"
    for lab, q, o in ((f"{h} {-sp:+g}", hc, g["sp_h"]), (f"{a} {sp:+g}", 1 - hc, g["sp_a"])):
        if o is None:
            continue
        rows.append([lab, f"{o:+.0f}", pc(be(o), 1), pc(q, 1), f"{(q * payout(o) - (1 - q)) * 100:+.1f}%"])
        bets.append((lab, q, o))
    ov, un = (tot > g["total"]).mean(), (tot < g["total"]).mean()
    po = ov / max(ov + un, 1e-9)
    for lab, q, o in ((f"Over {g['total']}", po, g["over_odds"]), (f"Under {g['total']}", 1 - po, g["under_odds"])):
        if o is None:
            continue
        rows.append([lab, f"{o:+.0f}", pc(be(o), 1), pc(q, 1), f"{(q * payout(o) - (1 - q)) * 100:+.1f}%"])
        bets.append((lab, q, o))
    t1 = T(["Selection", "DK price", "Break-even", "Model", "Est. EV"], rows, "Real DraftKings prices via ESPN. The sim is 90% market and 10% our line, so small gaps here are expected and are not edges.")
    fav = h if sp >= 0 else a
    sign = 1 if sp >= 0 else -1
    alts = []
    for x in [-14, -10.5, -7, -3.5, -2.5, 2.5, 3.5, 7, 10.5, 14]:
        alts.append([f"{h} {x:+g}", pc(float((m > -x).mean())), am(float((m > -x).mean())), f"{a} {-x:+g}", pc(float((m < -x).mean())), am(float((m < -x).mean()))])
    t2 = T(["Alt spread", "P", "Fair", "Other side", "P", "Fair"], alts, "Alternate spreads with fair prices. Alt prices are unavailable, so bet only at better than the fair price.")
    tl = g["total"]
    at = []
    for x in [tl - 10, tl - 7, tl - 3, tl, tl + 3, tl + 7, tl + 10]:
        o_ = float((tot > x).mean())
        at.append([f"{x:g}", pc(o_), am(o_), pc(1 - o_), am(max(1 - o_, .001))])
    t3 = T(["Alt total", "P over", "Fair over", "P under", "Fair under"], at)
    return t1, t2, t3, bets


MARGINS = [(2.5, "3+", "a field goal"), (3.5, "4+", "more than a field goal"), (5.5, "6+", "two field goals"), (6.5, "7+", "a touchdown"),
           (9.5, "10+", "a touchdown and a field goal, or three field goals"), (11.5, "12+", "four field goals"), (13.5, "14+", "two touchdowns")]


def margin_block(gg):
    """Chance the simulation's projected winner wins by each common margin, with the game's own spread marked."""
    m, g = gg.s["margin"], gg.g
    h, a = gg.h, gg.a
    ph = float((m > 0).mean() + .5 * (m == 0).mean())
    w, o, mw = (h, a, m) if ph >= .5 else (a, h, -m)
    pw = max(ph, 1 - ph)
    line = g["spread_home"] if w == h else -g["spread_home"]  # + = w favored by that many in the market
    rows = [(0, [f"{w} wins", "wins outright", pc(pw, 1), am(pw)])]
    have_line = line is not None and line == line and line != 0
    if have_line:
        sgn = f"-{abs(line):g}" if line > 0 else f"+{abs(line):g}"
        win, push = float((mw > line).mean()), float((mw == line).mean())
        what = f"wins by more than {line:g}" if line > 0 else f"wins, or loses by less than {abs(line):g}"
        rows.append((line, [f"<b>{w} {sgn}</b> <span class='tag mute'>game line</span>", what,
                            f"<b>{pc(win, 1)}</b>" + (f" <small>push {pc(push, 1)}</small>" if push >= .005 else ""), am(win / max(1 - push, 1e-9))]))
    for k, need, words in MARGINS:
        if have_line and abs(k - line) < 1e-9:
            continue
        q = float((mw > k).mean())
        rows.append((k, [f"{w} -{k:g}", f"wins by {need} ({words})", pc(q, 1), am(q)]))
    rows = [r for _, r in sorted(rows, key=lambda x: x[0])]
    rows.append([f"{o} wins", "the other team wins outright", pc(1 - pw, 1), am(1 - pw)])
    return T(["Margin", "Means", "Chance", "Fair odds"], rows,
             f"From all 20,000 simulated games. {w} is the simulation's projected winner; each row is the chance it wins by more than that margin. "
             "Fair odds have no vig, so a sportsbook price has to beat them to be worth it.")

def combos(gg, ms):
    def ev(r):
        return r["p"][r["key"]] >= r["k"]

    def pick(lo, hi, cnt):
        seen, out = set(), []
        for r in sorted([x for x in ms if lo <= x["prob"] <= hi and x["p"]["play_p"] >= .9 and not qb_held(x) and x["p"]["name"] not in SHIFT["pl"] and not (x["mk"] == "rush" and x["p"]["team"] in SHIFT["rb"])], key=lambda z: -z["prob"]):
            if r["p"]["name"] in seen:
                continue
            seen.add(r["p"]["name"])
            out.append(r)
            if len(out) == cnt:
                break
        return out
    html_ = ""
    for lab, lo, hi, cnt in (("Conservative", .80, .95, 2), ("Balanced", .62, .80, 3), ("Aggressive", .42, .62, 3)):
        L = pick(lo, hi, cnt)
        if len(L) < cnt:
            html_ += f"<p><b>{lab}:</b> not enough legs at this risk level with a posted milestone.</p>"
            continue
        arrs = [ev(r) for r in L]
        j = np.logical_and.reduce(arrs).mean()
        pr = np.prod([a.mean() for a in arrs])
        lift = j / max(pr, 1e-6)
        est = min(np.prod([r["prob"] for r in L]) * lift, min(r["prob"] for r in L))
        rel = "positively correlated" if lift > 1.04 else ("negatively correlated" if lift < .96 else "approximately independent")
        html_ += f"<p><b>{lab}:</b> " + " + ".join(f"{e(r['p']['name'])} {r['k']:g}+ {r['lab']} ({pc(r['prob'])})" for r in L) + f". Joint estimate <b>{pc(est, 1)}</b>, fair {am(est)}. These legs are {rel}: sim joint {pc(j, 1)} vs {pc(pr, 1)} if independent. Not guaranteed.</p>"
    return html_


def correlation(gg):
    rows = []
    for t in (gg.a, gg.h):
        q = gg.qb(t)
        wr = gg.top(t, ("WR", "TE"), "rec_y")
        rb = gg.top(t, ("RB",), "rush_y")
        opp = gg.h if t == gg.a else gg.a
        oq = gg.qb(opp)
        pairs = []
        if q is not None and wr:
            x, y = q["pass_y"] >= np.median(q["pass_y"]), wr[0]["rec_y"] >= np.median(wr[0]["rec_y"])
            pairs.append((f"{q['name']} over median + {wr[0]['name']} over median", x, y))
        if q is not None and rb:
            x, y = q["pass_y"] >= np.median(q["pass_y"]), rb[0]["rush_y"] >= np.median(rb[0]["rush_y"])
            pairs.append((f"{q['name']} over median + {rb[0]['name']} over median (same team)", x, y))
        if rb and oq is not None:
            x, y = rb[0]["rush_y"] >= np.median(rb[0]["rush_y"]), oq["pass_y"] >= np.median(oq["pass_y"])
            pairs.append((f"{rb[0]['name']} over median + {oq['name']} over median (opposing QB)", x, y))
        if rb and wr:
            s_ = gg.side(t)
            x, y = s_["rtds"][:, rb[0]["j"]] > 0, s_["ptds"][:, wr[0]["j"]] > 0
            pairs.append((f"{rb[0]['name']} rush TD + {wr[0]['name']} rec TD", x, y))
        for lab, x, y in pairs:
            j, pr = (x & y).mean(), x.mean() * y.mean()
            lf = j / max(pr, 1e-9)
            rows.append([e(lab), pc(x.mean()), pc(y.mean()), pc(j, 1), pc(pr, 1), n(lf, 2), "Positive" if lf > 1.04 else ("Negative" if lf < .96 else "About independent")])
    return T(["Pair", "Leg 1", "Leg 2", "Joint (sim)", "If independent", "Lift", "Correlation"], rows, "Joint outcomes counted directly from the simulation, not multiplied.")


# ---- side-by-side matchup view: away team always on the left, home on the right ----
PRED = '<span class="kind pred" title="Used by the model and shown to predict in the 2025-26 factor study">Predictive</span>'
CTX = '<span class="kind ctx" title="Did not add accuracy beyond the model in the factor study. Read it, but do not let it move a pick">Context</span>'
TEAM_M = [("EPA per play", "epa", 3), ("Success rate", "success", -1), ("Points per game", "ppg", 1), ("Yards per play", "ypp", 2), ("Pass EPA per play", "pass_epa", 3),
          ("Rush EPA per play", "rush_epa", 3), ("Explosive pass rate (20+)", "expl_pass", -1), ("Explosive run rate (12+)", "expl_rush", -1), ("3rd-down rate", "third", -1),
          ("Red-zone TD rate", "rz_td", -1), ("Turnovers per game", "to", 2), ("Sacks per game", "sacks", 1)]


def _fmt(v, d):
    return pc(v, 1) if d == -1 else (sg(v, d) if d == 3 else n(v, d))


def mrow(label, lv, rv, lr=None, rr=None, better=None):
    """One centred row. Bars grow outward from the label; rank 1 is a full bar. better: 'l', 'r' or None."""
    def side(v, r, cls):
        if r is None:  # nothing to rank, so no bar
            return f'<div class="{cls}{" win" if better == cls else ""}"><span class="v">{v}</span><span></span></div>'
        return f'<div class="{cls}{" win" if better == cls else ""}"><span class="v">{v}<i>#{int(r)}</i></span><span class="bar"><span style="width:{(33 - r) / 32 * 100:.0f}%"></span></span></div>'
    return f'<div class="mrow">{side(lv, lr, "l")}<div class="lab">{label}</div>{side(rv, rr, "r")}</div>'


def _better(lr, rr):
    return None if lr is None or rr is None or lr == rr else ("l" if lr < rr else "r")


def mblock(title, kind, rows, note=""):
    return f'<section class="mblock"><h4>{title} {kind}</h4>{"".join(rows)}{f"<p class=note>{note}</p>" if note else ""}</section>'


UNIT = {"off": ("offense", "gains"), "def": ("defense", "allows")}
SUBLAB = {"to": "offense loses · defense forces", "sacks": "offense allows · defense makes"}  # rows where the plain verbs would mislead


def side_head(lt, lside, rt, rside, mid="2026 season average<br># = league rank, 1 is best"):
    """Column headers that say which unit each side is, so a row reads without the note."""
    h = lambda t, s, cls: f'<div class="{cls}"><b>{t} {UNIT[s][0]}</b><small>what {t} {UNIT[s][1]}</small></div>'
    return f'<div class="mcols">{h(lt, lside, "l")}<div class="lab">{mid}</div>{h(rt, rside, "r")}</div>'


def unit_rows(c, R, lt, lside, rt, rside):
    L, Rr = c.team_side(lt, lside)["Season"], c.team_side(rt, rside)["Season"]
    out = [side_head(lt, lside, rt, rside)]
    for lab, k, d in TEAM_M:
        lr, rr = R[lside][k].get(lt), R[rside][k].get(rt)
        label = lab + (f"<small>{SUBLAB[k]}</small>" if k in SUBLAB else "")
        out.append(mrow(label, _fmt(L[k], d), _fmt(Rr[k], d), lr, rr, _better(lr, rr)))
    return out


UNIT_NOTE = "Green is the better-ranked unit in that row. It compares season form, not a projection for this game, and is not adjusted for opponents faced."
POS_ALLOWED = [("wr", "WR receiving yds"), ("te", "TE receiving yds"), ("rb_rec", "RB receiving yds"), ("rb_rush", "RB rushing yds")]


def allowed_view(c, gid, a, h):
    """Yards each defense gave up by position, with a Season / Last game / Last 5 games switch. Away team on the left."""
    G = {t: c.allowed_games(t) for t in (a, h)}
    avg = lambda rows, k: float(np.mean([r[k] for r in rows])) if rows else float("nan")
    head = lambda sub_a, sub_h: (f'<div class="mcols"><div class="l"><b>{a} defense allows</b><small>{sub_a}</small></div><div class="lab"></div>'
                                 f'<div class="r"><b>{h} defense allows</b><small>{sub_h}</small></div></div>')
    season = head(f"what {h}'s offense faces · {len(G[a])} games", f"what {a}'s offense faces · {len(G[h])} games") + "".join(
        mrow(f"{lab}<small>per game</small>", n(avg(G[a], k), 0), n(avg(G[h], k), 0)) for k, lab in POS_ALLOWED)
    la, lh = (G[a][-1] if G[a] else None), (G[h][-1] if G[h] else None)
    tag = lambda r: f"vs {r['opp']} · Week {r['week']}" if r else "no game yet"
    last = head(tag(la), tag(lh)) + "".join(mrow(lab, n(la[k], 0) if la else "–", n(lh[k], 0) if lh else "–") for k, lab in POS_ALLOWED)

    def five(t):
        rows = G[t][-5:]
        body = [[f"Wk {r['week']}", f"vs {r['opp']}"] + [n(r[k], 0) for k, _ in POS_ALLOWED] for r in reversed(rows)]
        body.append(["<b>Avg</b>", f"<b>{len(rows)} games</b>"] + [f"<b>{n(avg(rows, k), 0)}</b>" for k, _ in POS_ALLOWED])
        return f'<div class="pcol"><h5>{t} defense allowed</h5>{T(["Week", "Opp", "WR rec", "TE rec", "RB rec", "RB rush"], body)}</div>'
    l5 = f'<div class="pcols">{five(a)}{five(h)}</div>'
    sel = (f'<label class="win" for="win-{gid}">Show <select id="win-{gid}" data-win>'
           f'<option value="s">Season average</option><option value="1">Last game</option><option value="5">Last 5 games</option></select></label>')
    return (f'<section class="mblock"><h4>Yards allowed by position {CTX}{sel}</h4>'
            f'<div data-w="s">{season}</div><div data-w="1" hidden>{last}</div><div data-w="5" hidden>{l5}</div>'
            f'<p class="note">2026 regular season, what each defense gave up; bye weeks are skipped. Context only: in the factor study, '
            f'defense vs position added nothing once a player\'s volume was known.</p></section>')


def best_line(gg, p, mk, key, ths):
    """Highest threshold on our grid that the calibrated model still puts at 70%+."""
    ok = [(k, gg.cp(p[key], k, mk, p)) for k in ths]
    ok = [x for x in ok if x[1] >= .70]
    return ok[-1] if ok else None


def pos_groups(gg, t):
    out = {}
    for grp, pos in (("WR", ("WR",)), ("TE", ("TE",)), ("RB", ("RB", "FB", "HB"))):
        P = [p for p in gg.pl if p["team"] == t and p["pos"] in pos]
        out[grp] = dict(tg=sum(p["tg"] for p in P), rec=sum(float(p["rec_y"].mean()) for p in P), car=sum(p["car"] for p in P), rush=sum(float(p["rush_y"].mean()) for p in P))
    return out


def allowed(c, t):
    m = {x["role"]: x for x in c.alignment_allowed(t)}
    wr = [m[k] for k in ("WR1 (most targets)", "WR2", "WR3") if k in m]
    f = lambda xs: (sum(x["yds"] for x in xs), sum(x["lg"] for x in xs))
    return {"WR": f(wr), "TE": f([m["All TE"]] if "All TE" in m else []), "RB": f([m["All RB (receiving)"]] if "All RB (receiving)" in m else [])}


def player_card(gg, p):
    rows = []
    if p["pos"] == "QB":
        bl = best_line(gg, p, "pass", "pass_y", [150, 175, 200, 225, 250, 275, 300])
        rows.append(f"{n(p['pass_y'].mean(), 0)} pass yds" + (f" · <b>{bl[0]}+ {pc(bl[1])}</b>" if bl else ""))
    if p["rec_y"].mean() >= 10:
        bl = best_line(gg, p, "rec", "rec_y", [10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        rows.append(f"{n(p['tg'])} tgt, {n(p['rec_y'].mean(), 0)} rec yds" + (f" · <b>{bl[0]}+ {pc(bl[1])}</b>" if bl else ""))
    if p["rush_y"].mean() >= 15:
        bl = best_line(gg, p, "rush", "rush_y", [10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        rows.append(f"{n(p['car'])} car, {n(p['rush_y'].mean(), 0)} rush yds" + (f" · <b>{bl[0]}+ {pc(bl[1])}</b>" if bl else ""))
    flag = " <span class='tag warn'>Questionable</span>" if p["play_p"] < .9 else ""
    return f"<li><span class='pn2'>{e(p['name'])} <small>{p['pos']}</small>{flag}</span><span>{'<br>'.join(rows)}</span></li>"


def matchup_view(gg, c, R, s, probs):
    a, h, g = gg.a, gg.h, gg.g
    pa_, ph_ = float(np.mean(s["a_pts"])), float(np.mean(s["h_pts"]))
    pw_h = float((s["margin"] > 0).mean() + .5 * (s["margin"] == 0).mean())
    sp = g["spread_home"]
    imp_h, imp_a = (g["total"] + sp) / 2, (g["total"] - sp) / 2
    head = f"""<div class="mhead"><div class="tm l"><b>{a}</b><span>{e(c.record(a))}</span></div>
      <div class="mid"><div class="pts"><b>{pa_:.1f}</b><small>projected</small><b>{ph_:.1f}</b></div><div class="wp"><span style="width:{(1 - pw_h) * 100:.0f}%"></span></div><small>Win chance {a} {pc(1 - pw_h)} · {h} {pc(pw_h)}</small><small>DraftKings {e(g['details'])}, O/U {g['total']}</small></div>
      <div class="tm r"><b>{h}</b><span>{e(c.record(h))}</span></div></div>"""
    rt = R["rt"]
    rk = lambda side, t: pd.Series(rt[side]).rank(ascending=False).get(t)
    model = [mrow("Implied team points (market)", n(imp_a, 1), n(imp_h, 1), better=_better(-imp_a, -imp_h)),
             mrow("Adjusted offense, pts vs avg", f"{rt['off'].get(a, 0):+.1f}", f"{rt['off'].get(h, 0):+.1f}", rk("off", a), rk("off", h), _better(rk("off", a), rk("off", h))),
             mrow("Adjusted defense, pts vs avg", f"{rt['def'].get(a, 0):+.1f}", f"{rt['def'].get(h, 0):+.1f}", rk("def", a), rk("def", h), _better(rk("def", a), rk("def", h))),
             mrow("Projected pass yards", n(gg.side(a)["pass_y"].mean(), 0), n(gg.side(h)["pass_y"].mean(), 0)),
             mrow("Projected carries", n(gg.side(a)["car"].mean(), 1), n(gg.side(h)["car"].mean(), 1))]
    notes = []
    for t in (a, h):
        if t in QB_CHANGE:
            notes.append(f"{t}: {QB_CHANGE[t]}")
    B = mblock("Strength and script", PRED, model, "Adjusted ratings are opponent-adjusted points per game; rank 1 is best. The sim centers on the market, so implied points carry the most weight. " + e(" ".join(notes)))
    B += mblock(f"When {a} has the ball", CTX, unit_rows(c, R, a, "off", h, "def"), UNIT_NOTE)
    B += mblock(f"When {h} has the ball", CTX, unit_rows(c, R, a, "def", h, "off"), UNIT_NOTE)
    ga, gh = pos_groups(gg, a), pos_groups(gg, h)
    prow = []
    for grp in ("WR", "TE", "RB"):
        prow.append(mrow(f"{grp} projected targets", n(ga[grp]["tg"]), n(gh[grp]["tg"])))
        prow.append(mrow(f"{grp} projected rec yds", n(ga[grp]["rec"], 0), n(gh[grp]["rec"], 0)))
    prow.append(mrow("RB projected rush yds", n(ga["RB"]["rush"], 0), n(gh["RB"]["rush"], 0)))
    B += mblock("Position groups", PRED, prow, "Model projections, already adjusted for who is out and the expected script.")
    B += allowed_view(c, f"{a}-{h}".lower(), a, h)
    cards = ""
    for t in (a, h):
        P = sorted([p for p in gg.pl if p["team"] == t], key=lambda z: -(z["rec_y"].mean() + z["rush_y"].mean() + (z["pass_y"].mean() / 3 if "pass_y" in z else 0)))[:6]
        cards += f'<div class="pcol"><h5>{t}</h5><ul class="plist">{"".join(player_card(gg, p) for p in P)}</ul></div>'
    K = f'<section class="mblock"><h4>Key players {PRED}</h4><div class="pcols">{cards}</div><p class="note">Bold is the highest line on our grid the model still gives 70%+. That is the "most likely" tier, not a value claim.</p></section>'
    return head, B, K


def traps(gg, c, vals):
    """Lines where a popular read points one way and the model disagrees, using reads the factor study showed do not predict."""
    out = []
    soft = {}
    for t, opp in ((gg.a, gg.h), (gg.h, gg.a)):
        al = allowed(c, opp)
        soft[t] = {k: (v[0] / v[1] if v[1] else 1) for k, v in al.items()}
    for r in vals:
        p = r["p"]
        kind = {"rec": "rec", "rush": "rush", "pass": "passy"}[r["mk"]]
        k = int(math.ceil(r["line"]))
        rows, _ = c.hitrates(p["player_id"], kind, [k], gg.h if p["team"] == gg.a else gg.a, p["team"], 5)
        l5 = rows[0]["l5"]
        why = []
        if l5[1] >= 4 and l5[0] / l5[1] >= .8 and r["cal_over"] < .5:
            why.append(f"Hot streak: cleared {k}+ in {l5[0]} of his last {l5[1]}, but the model has the over at {pc(r['cal_over'])}. Last-5 hit rates were worse than the model in every market tested.")
        grp = {"WR": "WR", "TE": "TE", "RB": "RB", "FB": "RB"}.get(p["pos"])
        if r["mk"] == "rec" and grp and soft[p["team"]].get(grp, 1) >= 1.2 and r["cal_over"] < .5:
            why.append(f"Soft-matchup read: the opponent allows {pc(soft[p['team']][grp] - 1)} more {grp} yards than average, but the model has the over at {pc(r['cal_over'])}. Defense vs position did not predict in the study.")
        if p["team"] in QB_CHANGE and r["mk"] in ("rec", "pass") and r["cal_over"] < .55:
            why.append("New starting QB: these lines hit far less than history suggests (QB lines 26% vs 42% before the fix).")
        if why:
            out.append([e(p["name"]), f"Over {r['line']:g} {r['lab']}", pc(r["cal_over"]), " ".join(why)])
    return out


def checklist(gg, s):
    q = [x for t in (gg.a, gg.h) for x in s["inj"].get(t, []) if x["status"] in ("Questionable", "Doubtful")]
    items = ["Inactives post about 90 minutes before kickoff. Rerun the slate after they do, and drop any pick whose player or key teammate is out.",
             f"Confirm the starting QBs: {e((gg.qb(gg.a) or {}).get('name'))} and {e((gg.qb(gg.h) or {}).get('name'))}. A change moves every passing and receiving line.",
             "Check the line has not moved past the minimum price shown. If it has, skip the pick rather than chase it.",
             "Check the kickoff forecast for 15+ mph wind or heavy rain. The market total usually moves first; if it drops 3+ points, rerun."]
    if q:
        items.insert(1, "Questionable or doubtful: " + e(", ".join(f"{x['name']} ({x['pos']}, {x['status']})" for x in q[:10])) + ".")
    return "<ol class='check'>" + "".join(f"<li>{x}</li>" for x in items) + "</ol>"


PARLAY_RULES = """<ul><li><b>Two or three legs.</b> Each extra leg multiplies the risk: three 75% legs land together only about 42% of the time if independent.</li>
<li><b>Each leg at 70% or better</b> from the Most likely tier, and no questionable players or QB-change teams.</li>
<li><b>Use the joint estimate, not the product.</b> Same-game legs move together. A QB yardage over and his WR1 over are positively correlated; two receivers on one team partly compete for the same targets.</li>
<li><b>Price check:</b> a parlay is worth taking only if its payout beats the fair odds of the joint estimate.</li></ul>"""


def subtabs(gid, key, items):
    """Second-level tabs inside one Game Book tab: items = [(label, html), ...]. A single item needs no bar."""
    items = [(lab, body) for lab, body in items if body]
    if len(items) == 1:
        return items[0][1]
    bar = "".join(f'<button class="sb" data-s="{gid}-{key}-{i}">{lab}</button>' for i, (lab, _) in enumerate(items))
    return (f'<div class="subtabs" role="tablist">{bar}</div>'
            + "".join(f'<div class="sub" id="{gid}-{key}-{i}"{"" if i == 0 else " hidden"}>{body}</div>' for i, (_, body) in enumerate(items)))


def best_plays(gid, likely, tds, watch, links, note):
    """The strip that opens the Overview: the most likely lines, the top touchdown chances and what to watch out for.
    likely / tds: [(name, team, outcome, probability)]; watch: [html]; links: three (label, tab key, sub index or None)."""
    row = lambda nm, tm, what, p: (f'<div class="brow"><div><b>{e(nm)}</b> <small>{e(tm)}</small><span>{e(what)}</span></div>'
                                   f'<div class="bp"><b>{pc(p)}</b><i><u style="width:{min(max(p, 0), 1) * 100:.0f}%"></u></i></div></div>')
    def link(x):
        lab, tab, sub = x
        return f'<button class="blink" data-tab="{gid}-{tab}"' + (f' data-sub="{gid}-{tab}-{sub}"' if sub is not None else "") + f'>{lab} &rarr;</button>'
    cols = [("Most likely to hit", "".join(row(*r) for r in likely) or "<p class='note'>No posted line clears 70% without a flag.</p>", links[0], "bl"),
            ("Touchdown chances", "".join(row(*r) for r in tds) or "<p class='note'>No touchdown model for this game.</p>", links[1], "bt"),
            ("Watch out", ("<ul>" + "".join(f"<li>{w}</li>" for w in watch) + "</ul>") if watch else "<p class='note'>Nothing flagged for this game.</p>", links[2], "bw")]
    return (f'<section class="mblock best"><h4>Best plays {PRED}</h4><div class="bestg">'
            + "".join(f'<div class="bcol {cls}"><h5>{t}</h5>{body}{link(lk)}</div>' for t, body, lk, cls in cols)
            + f'</div><p class="note">{note}</p></section>')


def picks_view(gg, c, s, top10, cands, vals, ms):
    likely = T(["#", "Player", "Outcome", "Model %", "Fair odds", "Worst price to take"], [[i + 1, e(r["p"]["name"]), f"{r['k']:g}+ {r['lab']}", f"<b>{pc(r['prob'])}</b>", am(r["prob"]), am(max(r["prob"] - .02, .02))] for i, r in enumerate(top10)]) if top10 else "<p>No posted milestone clears 70% without a flag.</p>"
    value = T(["#", "Player", "Pick", "Model %", "Edge vs -110", "Minimum price"], [[i + 1, e(r["p"]["name"]), f"{'Over' if r['over'] else 'Under'} {r['line']:g} {r['lab']}", pc(r["prob"], 1), f"{r['edge'] * 100:+.1f}%", r["minodds"]] for i, r in enumerate(cands)]) if cands else "<p><b>Pass.</b> No main line clears a 3-point edge at an assumed -110.</p>"
    tr = traps(gg, c, vals)
    trap = T(["Player", "Popular side", "Model % over", "Why it looks good and why it isn't"], tr) if tr else "<p>No line here pairs a popular read with a model that disagrees.</p>"
    return f"""<section class="tier t1"><h4>1 · Most likely</h4><p class="note">Highest DraftKings milestone per player the model still puts at 70%+. Ranked by likelihood. Prices are not in the free feed, so short odds are likely.{" Passing and receiving lines for " + ", ".join(sorted(t for t in QB_CHANGE if not t.endswith("_"))) + " are held out because of the QB change." if any(t for t in QB_CHANGE if not t.endswith("_")) else ""}</p>{likely}</section>
    <section class="tier t2"><h4>2 · Value, only with a real price</h4><p class="note">Main lines where the model disagrees with the number. Edges assume -110 and are unproven against DraftKings; take one only at the minimum price or better.</p>{value}</section>
    <section class="tier t3"><h4>3 · Traps</h4><p class="note">Lines that look good on a streak, a soft matchup or old QB history, where the model says otherwise.</p>{trap}</section>
    <section class="tier"><h4>Before kickoff</h4>{checklist(gg, s)}</section>
    <section class="tier"><h4>Building a parlay</h4>{PARLAY_RULES}{combos(gg, ms)}<p class="note">Aggressive breaks the 70% rule on purpose, for comparison.</p></section>"""


def render_game(gm, c, R, cal):
    gg = G(gm, c, cal)
    g, h, a = gg.g, gg.h, gg.a
    gg.c_status = gm["c_status"]
    kick_w, kick = weather_block(gm)
    et = kick.astimezone(timezone(timedelta(hours=-5)))  # DST ended 2026-11-01 at 2 AM
    s = gm
    vals, ms = value_rows(gg), miles_rows(gg)
    t1, t2, t3, bets = game_lines(gg)
    probs, sc_t, sc_v = scenarios(gg)
    ph = float((s["margin"] > 0).mean() + .5 * (s["margin"] == 0).mean())
    win, pw = (h, ph) if ph >= .5 else (a, 1 - ph)
    pts_h, pts_a = float(np.mean(s["h_pts"])), float(np.mean(s["a_pts"]))
    rec_g = [r for r in vals if not r["flag"]]
    good_a, bad_a = strengths(c, a, R)
    good_h, bad_h = strengths(c, h, R)
    key = []
    inj_rows = []
    for t in (a, h):
        for x in s["inj"].get(t, []):
            inj_rows.append([t, e(x["name"]), x["pos"], e(x["status"]), e(x["injury"]), e(x["practice"].replace(" in Practice", "").replace(" Participation", "")), pc(x["play_p"]), x["updated"]])
    gid = f"{a}-{h}".lower()

    # A
    A = f"""<div class="exec"><div class="score"><div><span>{a}</span><b>{pts_a:.1f}</b></div><div><span>{h}</span><b>{pts_h:.1f}</b></div></div>
    <div class="exgrid"><div><small>Predicted winner</small><b>{win} {pc(pw)}</b></div><div><small>Market</small><b>{e(g['details'])}, O/U {g['total']}</b></div><div><small>Our independent line</small><b>{a} {s['model_a']:.1f}, {h} {s['model_h']:.1f}</b></div><div><small>Most likely script</small><b>{['A: ' + h + ' controls', 'B: ' + a + ' controls', 'C: Close game'][int(np.argmax(probs))]} ({pc(max(probs))})</b></div></div></div>
    <p><b>{a} strengths:</b> {e('; '.join(good_a) or 'none in the top 7')}. <b>Weaknesses:</b> {e('; '.join(bad_a) or 'none in the bottom 7')}.</p>
    <p><b>{h} strengths:</b> {e('; '.join(good_h) or 'none in the top 7')}. <b>Weaknesses:</b> {e('; '.join(bad_h) or 'none in the bottom 7')}.</p>"""
    for t in (a, h):
        for k in (t, t + "_"):  # "_" keys are notes only; they do not flag the team's props
            if k in QB_CHANGE:
                A += f"<p class='warn'>{e(QB_CHANGE[k])}</p>"
    outs = [x for t in (a, h) for x in s["inj"].get(t, []) if x["status"] in ("Out", "Doubtful", "Questionable") and x["pos"] in ("QB", "RB", "WR", "TE", "CB", "S", "DE", "LB", "OT", "G", "C", "DT", "EDGE", "OLB")]
    A += "<p><b>Key injuries and uncertainties:</b> " + e("; ".join(f"{x['name']} ({x['pos']}, {x['status']})" for x in outs[:12]) or "none listed") + ".</p>"

    # P1
    st = c.g26[(c.g26.home_team == h) & (c.g26.away_team == a)]
    roof, surf, rh, ra, div = "", "", "", "", ""
    if len(st):
        q = st.iloc[-1]
        roof, surf, rh, ra, div = q.roof, q.surface, q.home_rest, q.away_rest, q.div_game
    P1 = f"""<ul><li><b>Kickoff:</b> {et.strftime('%a %b %-d, %-I:%M %p')} ET. <b>Status:</b> {e(g['status'])}, upcoming.</li><li><b>Stadium:</b> {e(g['venue'])}, {e(g['city'])}. {a} at {h}.</li>
    <li><b>Records (verified from game results):</b> {a} {c.record(a)}, {h} {c.record(h)}. ESPN shows {e(g['rec_a'])} and {e(g['rec_h'])}.</li><li><b>Data through:</b> 2026 Week {WEEK - 1}. Games already played this week are not used, and no result from this game is used. Season: 2026 regular season, with last year's data only where labeled.</li>
    <li><b>Starting QBs:</b> {e((gg.qb(a) or {}).get('name'))} and {e((gg.qb(h) or {}).get('name'))}, from the current depth chart and matched to DraftKings props where posted.</li><li><b>Availability checked:</b> ESPN report pulled {PULLED}. Check inactives 90 minutes before kickoff.</li></ul>""" + T(["Tm", "Player", "Pos", "Status", "Injury", "Practice", "P(plays)", "Updated"], inj_rows, "All listed injuries. P(plays) is a model assumption based on status and practice participation.")
    # P2
    sa, sh = c.schedule_strength(a, R["rt"]), c.schedule_strength(h, R["rt"])
    P2 = team_compare(c, a, h, R) + T(["Split", "G", "PPG", "Off EPA", "Off success", "Y/play", "Opp PPG", "Def EPA", "Def success", "Opp Y/play"], trends(c, a) + trends(c, h), "Season vs recent form. Last-3 samples are small, so treat them as hints.") + \
        f"<p><b>Strength of schedule (ridge-adjusted, points vs average):</b> {a} opponents' offense {sa['opp_off']:+.1f}, defense {sa['opp_def']:+.1f}. {h} opponents' offense {sh['opp_off']:+.1f}, defense {sh['opp_def']:+.1f}. Adjusted team strength in points: {a} offense {R['rt']['off'].get(a, 0):+.1f}, defense {R['rt']['def'].get(a, 0):+.1f}; {h} offense {R['rt']['off'].get(h, 0):+.1f}, defense {R['rt']['def'].get(h, 0):+.1f}.</p>"
    # P3
    pa_, ph_ = c.pace(a), c.pace(h)
    P3 = T(["Team", "Plays/g", "Sec/snap (neutral)", "PROE", "PROE neutral", "Pass% leading 8+", "Pass% close", "Pass% trailing 8+", "Att/g", "Rush/g", "Drives/g", "TOP min"],
           [[t, n(p["plays"]), n(p["sec"]), pc(p["proe"], 1), pc(p["proe_neutral"], 1), pc(p["pass_lead"]), pc(p["pass_close"]), pc(p["pass_trail"]), n(p["att"]), n(p["rush"]), n(p["drives"]), n(p["top"])] for t, p in ((a, pa_), (h, ph_))]) + \
        f"<p><b>Scenario A, {h} controls (wins by 8+):</b> {pc(probs[0])}. <b>Scenario B, {a} controls:</b> {pc(probs[1])}. <b>Scenario C, one-score game:</b> {pc(probs[2])}. These sum to 100% and come from simulated final margins.</p>" + sc_v + sc_t
    P4 = scheme_block(c, h, a) + scheme_block(c, a, h)
    P5 = "".join(f"<h4>{t} pass catchers vs {o}</h4>" + wr_block(c, gg, t, o) for t, o in ((a, h), (h, a)))
    rank = sorted([p for p in gg.pl if p["pos"] != "QB" and p["tg"] >= 2], key=lambda z: -z["rec_y"].mean())[:12]
    P5 += T(["Rank", "Player", "Tm", "Exp tgt", "Proj yds", "40+", "60+", "80+"], [[i + 1, e(p["name"]), p["team"], n(p["tg"]), n(p["rec_y"].mean(), 0), pc(gg.cp(p["rec_y"], 40, "rec", p)), pc(gg.cp(p["rec_y"], 60, "rec", p)), pc(gg.cp(p["rec_y"], 80, "rec", p))] for i, p in enumerate(rank)], "Receiving options ranked by projection")
    P6 = "".join(f"<h4>{t} QB vs {o} rush</h4>" + qb_block(c, gg, t, o) for t, o in ((a, h), (h, a)))
    P7 = "".join(f"<h4>{t} run game vs {o}</h4>" + rb_block(c, gg, t, o) for t, o in ((a, h), (h, a)))
    rza, rzh = c.rz_team(a), c.rz_team(h)
    td = td_block(gg)
    P8 = T(["Team", "Plays inside 10", "Pass rate inside 10", "TD per play", "Opp pass TD rate allowed", "Opp rush TD rate allowed"],
           [[a, rza["plays"], pc(rza["pass_r"]), pc(rza["td_rate"]), pc(rzh["d_pass_td"]), pc(rzh["d_rush_td"])], [h, rzh["plays"], pc(rzh["pass_r"]), pc(rzh["td_rate"]), pc(rza["d_pass_td"]), pc(rza["d_rush_td"])]], "Goal-to-go tendencies in 2026. The opponent columns show what the other defense allows.") + \
        T(["Player", "Tm", "Pos", "Anytime TD", "Fair odds", "Plausible range", "Exp rush TD", "Exp rec TD", "Main route to a TD"], td, "TD model: simulated points give team TDs, then TDs split to players by red-zone role, weighted toward role over past TD luck. DK lists these markets, but ESPN's feed has no prices.")
    wtxt = ""
    if kick_w:
        wtxt = f"{kick_w['temperature']}°F, wind {e(kick_w['windSpeed'])} {e(kick_w['windDirection'])}, {kick_w['probabilityOfPrecipitation']['value'] or 0}% precipitation, {e(kick_w['shortForecast'])} (NWS hourly forecast)."
    else:
        wtxt = f"No outdoor NWS forecast used. ESPN lists {e((g.get('weather') or {}).get('displayValue', 'n/a'))}, {e((g.get('weather') or {}).get('temperature', ''))}°F."
    wind = int(str(kick_w["windSpeed"]).split()[0]) if kick_w else 0
    pop = (kick_w["probabilityOfPrecipitation"]["value"] or 0) if kick_w else 0
    adj = "No weather adjustment applied." if wind < 15 and pop < 50 else f"Watch this forecast: {'wind of 15+ mph trims deep passing' if wind >= 15 else 'rain risk can trim passing efficiency slightly'}. The market total should already reflect it, so no separate adjustment is applied."
    P9 = f"<ul><li><b>Forecast at kickoff:</b> {wtxt} {adj}</li><li><b>Roof and surface:</b> {e(roof if isinstance(roof, str) else 'not listed')}, {e(surf)}.</li><li><b>Rest:</b> {a} {ra} days, {h} {rh} days{'; divisional game' if div == 1 else ''}.</li><li><b>Travel:</b> {a} travels about {dist(a, h):,.0f} miles.</li><li><b>Home field:</b> the ridge fit puts it at {R['rt']['hfa']:.1f} points across the sample.</li><li><b>Head coaches:</b> {e(c.hc26.get(a, ''))} ({a}), {e(c.hc26.get(h, ''))} ({h}).</li></ul>"
    P10 = hit_block(c, gg)
    P11 = "<h4>B. Complete player projections</h4>" + player_proj(gg) + "<h4>C. Alternate-yardage probabilities</h4>" + prob_tables(gg)
    vrows = []
    for r in vals[:30]:
        p = r["p"]
        vrows.append([e(p["name"]), r["lab"], f"{r['line']:g}", n(r["mean"], 0), n(r["med"], 0), pc(r["cal_over"]), f"{'Over' if r['over'] else 'Under'} {pc(r['prob'], 1)}", f"{r['edge'] * 100:+.1f}%", r["minodds"], tag("warn", "Check role") if r["role"] else tag("warn", "Caution") if r["flag"] else (tag("good", "Candidate") if r["edge"] >= .03 else tag("mute", "Pass"))])
    msr = sorted(ms, key=lambda r: -r["prob"])
    P12 = T(["Player", "Market", "DK line", "Mean", "Median", "Cal. P over", "Shrunk side", "Edge vs -110", "Min odds", "Call"], vrows,
                                                                                    "<b>Lines are real DraftKings numbers, but ESPN's feed has no prop prices.</b> Edge assumes -110 and is not a verified EV. Shrunk side first removes the slate-wide tilt in that market, then pulls our probability halfway to the line (log-odds), because we cannot prove the model beats the prop market. Min odds is the worst price worth taking, with a 2-point cushion. Check role marks a median more than 33% from the line, or a 20-point snap swing in the latest game. Either usually means usage history lags a real role change, so we do not count it as an edge.")
    P13 = correlation(gg)
    # D
    bestm = {}
    for r in ms:
        if r["prob"] >= .7 and r["prob"] <= .985:
            k = (r["p"]["name"], r["mk"])
            if k not in bestm or r["k"] > bestm[k]["k"]:
                bestm[k] = r
    held = lambda r: r["p"]["name"] in SHIFT["pl"] or (r["mk"] == "rush" and r["p"]["team"] in SHIFT["rb"]) or r["p"]["play_p"] < .9 or qb_held(r)
    top10 = sorted([r for r in bestm.values() if not held(r)], key=lambda r: -r["prob"])[:10]
    D = T(["#", "Player", "Alternate outcome", "Calibrated P", "Fair odds", "2026 hit rate"], [[i + 1, e(r["p"]["name"]), f"{r['k']:g}+ {r['lab']}", pc(r["prob"]), am(r["prob"]), season_hit(c, r)] for i, r in enumerate(top10)], "Highest thresholds DK posts for each player that still clear 70%. Likely does not mean good value.")
    cands = [r for r in rec_g if r["edge"] >= .03][:5]
    Eh = T(["#", "Player", "Pick", "Prob", "Edge vs -110", "Min odds", "Uncertainty"], [[i + 1, e(r["p"]["name"]), f"{'Over' if r['over'] else 'Under'} {r['line']:g} {r['lab']}", pc(r["prob"], 1), f"{r['edge'] * 100:+.1f}%", r["minodds"], f"Sim 10th-90th {np.percentile(r['p'][r['key']], 10):.0f}-{np.percentile(r['p'][r['key']], 90):.0f}"] for i, r in enumerate(cands)]) if cands else "<p><b>Pass.</b> No player line clears a 3-point shrunk edge at an assumed -110.</p>"
    F = T(["#", "Player", "Anytime TD", "Fair odds", "Main route"], [[i + 1, r[0], r[3], r[4], r[8]] for i, r in enumerate(td[:5])])
    gb = sorted(bets, key=lambda b: -(b[1] * payout(b[2]) - (1 - b[1])))
    best = gb[0]
    bev = best[1] * payout(best[2]) - (1 - best[1])
    Gp = t1 + (f"<p><b>Call: {e(best[0])} at {best[2]:+.0f} ({bev * 100:+.1f}% EV).</b> Lean only. This gap comes from the 10% of the sim that is our own line, which lost to closing lines in the backtest, so treat it as unproven.</p>" if bev > .03 else "<p><b>Call: pass on game lines.</b> Our line does not beat the closing market in the backtest, and no side clears +3% EV.</p>") + \
        f"<p>Winner lean: {win} {pc(pw)}. Our independent line says {a} {s['model_a']:.1f}, {h} {s['model_h']:.1f} vs the market's {e(g['details'])}, O/U {g['total']}. Alternate spreads and totals are in Phase 12.</p>"
    H = combos(gg, ms)
    I = final_five(c, gg, cands, top10)
    H_ = {"A": A, "P1": P1}
    sections = [phase("A", "Executive summary", A), phase("1", "Game and data verification", P1), phase("2", "Team performance", P2), phase("3", "Pace and game script", P3), phase("4", "Defensive scheme", P4),
                phase("5", "Receiver matchups", P5), phase("6", "Quarterback vs pass rush", P6), phase("7", "Running game", P7), phase("8", "Touchdown model", P8), phase("9", "Weather and environment", P9),
                phase("10", "Historical hit rates", P10), phase("11", "Statistical model output", P11), phase("12", "Market and value", P12), phase("13", "Parlay correlation", P13),
                phase("D", "Top 10 highest-probability props", D), phase("E", "Top 5 value opportunities", Eh), phase("F", "Top 5 touchdown candidates", F), phase("G", "Game picks", Gp), phase("H", "Conservative, balanced, aggressive", H), phase("I", "Final ranked recommendations", I)]
    summ = dict(id=gid, event=g["id"], a=a, h=h, kick=et.strftime("%a %-I:%M %p"), venue=g["venue"], win=win, pw=pw, pa=pts_a, ph=pts_h, line=g["details"], total=g["total"], ma=s["model_a"], mh=s["model_h"], vals=vals, top10=top10, td=[(p["name"], p["team"], p["atd"]) for p in sorted(gg.pl, key=lambda z: -z["atd"])[:3]], script=probs, best=(best[0], best[2], bev))
    mhead, core, keyp = matchup_view(gg, c, R, s, probs)
    GS = c.d["games"]
    sec = lambda title, kind, body, note="": f'<section class="mblock"><h4>{title} {kind}</h4>{body}{f"<p class=note>{note}</p>" if note else ""}</section>'
    fold = lambda title, body: f'<details class="fold"><summary>{title}</summary>{body}</details>'
    chart, mtable, hist = GB.odds_movement(g, gid)

    # Overview
    nv_a, nv_h = GB.novig(g["ml_a"], g["ml_h"]) if g.get("ml_a") and g.get("ml_h") else (float("nan"), float("nan"))
    cell = lambda big, small: f"<div class='oc'><b>{big}</b><small>{small}</small></div>"
    orow = lambda t, nm, rec, spv, spo, ml, tl, to: (f"<div class='otm'><span class='mono'>{t}</span><div><b><span class='fn'>{e(nm)}</span><span class='ab'>{t}</span></b><small>{e(rec)}</small></div></div>"
                                                     f"{cell(GB._sp(spv), GB._ml(spo))}{cell(GB._ml(ml), 'to win')}{cell(tl, GB._ml(to))}")
    board = (f"<div class='board'><div class='bh'><span></span><span>Spread</span><span>Moneyline</span><span>Total</span></div>"
             f"<div class='br'>{orow(a, g['away_name'], c.record(a) + ' · away', g['spread_home'], g.get('sp_a'), g.get('ml_a'), 'O ' + format(g['total'], 'g'), g.get('over_odds'))}</div>"
             f"<div class='br'>{orow(h, g['home_name'], c.record(h) + ' · home', -g['spread_home'], g.get('sp_h'), g.get('ml_h'), 'U ' + format(g['total'], 'g'), g.get('under_odds'))}</div>"
             f"<p class='note'>DraftKings via ESPN, pulled {PULLED}. {et.strftime('%a %b %-d, %-I:%M %p')} ET · {e(g['venue'])}.</p></div>")
    split = lambda lab, pa_, ph_: (f"<div class='wpb'><div class='wl'><span>{a} <b>{pc(pa_)}</b></span><small>{lab}</small><span><b>{pc(ph_)}</b> {h}</span></div>"
                                   f"<div class='wbar2'><span class='{'f' if pa_ >= ph_ else 'u'}' style='width:{pa_ * 100:.1f}%'></span><span class='{'f' if ph_ > pa_ else 'u'}' style='width:{ph_ * 100:.1f}%'></span></div></div>")
    winp = sec("Win probability", PRED, split("Market, vig removed", nv_a, nv_h) + split("Our 20,000-game simulation", 1 - ph, ph) +
               f"<div class='proj'><div><small>Projected score</small><b>{a} {pts_a:.1f} – {pts_h:.1f} {h}</b></div><div><small>Projected total</small><b>{pts_a + pts_h:.1f}</b> <small>vs {g['total']:g}</small></div><div><small>Most likely script</small><b>{['A: ' + h + ' by 8+', 'B: ' + a + ' by 8+', 'C: one-score game'][int(np.argmax(probs))]} · {pc(max(probs))}</b></div></div>",
               "The simulation is centered 90% on the market and 10% on our own line, because our line alone did not beat closing numbers in the backtest.")
    winp += sec("Win margin", PRED, margin_block(gg))
    warn = "".join(f"<p class='warn'>{e(QB_CHANGE[k])}</p>" for t in (a, h) for k in (t, t + "_") if k in QB_CHANGE)
    sosb = sec("Strength of schedule", CTX, GB.sos_cards(c, R["rt"], a, h), "Rank 1 = toughest schedule in the league, from our opponent-adjusted ratings (each opponent's offense plus defense, in points per game against an average team). Red is a tough schedule, yellow average, green soft. In the dropdowns, Off and Def are each opponent's 2026 rank (1 = best unit), colored by how hard that unit is to face.")
    move = sec("Odds movement", CTX, chart, "From DraftKings' opening number to our snapshots to now. A big move usually means news (here, a QB change) more than betting volume. Bet and money percentages are not in any free feed, so they are not shown.")
    bars = mblock("2026 team comparison", CTX, [side_head_simple(a, h, "2026 regular season<br># = league rank, 1 is best")] + GB.team_bars(GS, a, h, mrow, _better), "Green is the better-ranked team in that row.")
    h2, h2s = GB.head_to_head(GS, a, h)
    h2b = sec("Head to head", CTX, (f"<p class='sub'>{e(h2s)}</p>" if h2s else "") + h2, "Since 1999, regular season and playoffs. Cover uses the closing spread from nflverse.")
    l10 = sec("Last 10 games", CTX, f"<div class='pcols'><div class='pcol'><h5>{a}</h5>{GB.last10(GS, a)}</div><div class='pcol'><h5>{h}</h5>{GB.last10(GS, h)}</div></div>", "Includes 2025 when 2026 has fewer than 10 games. ATS and O/U use closing lines.")
    cond = sec("Game conditions", CTX, P9)
    # Best plays: the picks people open the book for, first
    tr_ = traps(gg, c, vals)
    watch = [f"<b>Trap:</b> {r[0]} {r[1].lower()}. The model has it at {r[2]}." for r in tr_[:2]]
    flagged = sorted(t for t in (a, h) if t in QB_CHANGE)
    if flagged:
        watch.append(f"<b>{' and '.join(flagged)}:</b> passing and receiving lines are left out of Most likely (see this week's note below).")
    held_rb = sorted(t for t in (a, h) if t in SHIFT["rb"])
    if held_rb:
        watch.append(f"<b>{' and '.join(held_rb)}:</b> rushing lines are left out while the backfield roles are changing.")
    qd = [x for t in (a, h) for x in s["inj"].get(t, []) if x["status"] in ("Questionable", "Doubtful") and x["pos"] in ("QB", "RB", "WR", "TE")]
    if qd:
        watch.append("<b>Not certain to play:</b> " + e(", ".join(f"{x['name']} ({x['pos']}, {x['status'].lower()})" for x in qd[:5])) + ".")
    best_strip = best_plays(gid, [(r["p"]["name"], r["p"]["team"], f"{r['k']:g}+ {r['lab']}", r["prob"]) for r in top10[:3]],
                            [(nm, tm, "anytime touchdown", p_) for nm, tm, p_ in summ["td"]], watch,
                            [("All picks", "p", 0), ("Touchdown detail", "p", 1), ("Injuries", "pl", 1)],
                            "Most likely means likely to hit, not good value: the free feed has lines but no prices. From 20,000 simulated games.")
    overview = best_strip + board + warn + winp + sosb + bars + cond

    # Odds
    odds = (board + move + sec("Line movement", CTX, mtable) + sec("Game markets", PRED, t1 + t2 + t3) + sec("Game picks", PRED, Gp))

    # Offense & Defense
    offdef = mhead + core + sec("More detail", CTX, fold("Pace and game script", P3) + fold("Defensive scheme", P4) + fold("Receiver matchups", P5) + fold("Quarterback vs pass rush", P6) + fold("Running game", P7))

    # Offensive line vs defensive line
    try:
        from config import QB_START
        lnt = LN.tab(c, gg, s, a, h, WEEK, dict(mrow=mrow, better=_better, T=T, CTX=CTX, PRED=PRED, e=e, pc=pc, n=n, qb_change=QB_START))
    except Exception as ex:  # a missing file should not sink the whole book
        lnt = f"<p class='note'>The line matchup could not be built: {e(ex)}</p>"

    # Special teams
    try:
        who, kick, punt, ret = GB.special_teams(c, a, h, mrow, _better)
        kwarn = "".join(f"<p class='warn'>{e(t)} {e(x['pos'])} {e(x['name'])} is listed {e(x['status'])}. The season numbers below are mostly his, so this week's kicking may look different.</p>"
                        for t in (a, h) for x in s["inj"].get(t, []) if x["pos"] in ("PK", "K", "P") and x["status"] in ("Out", "Injured Reserve", "Doubtful"))
        stt = kwarn + mblock("Kicking", CTX, [who] + kick) + mblock("Punting", CTX, [who] + punt) + mblock("Returns and coverage", CTX, [who] + ret, "2026 regular season. Net punt average subtracts return yards and 20 yards per touchback. Return averages count only kicks that were returned.")
    except Exception as ex:  # a missing column should not sink the whole book
        stt = f"<p class='note'>Special teams could not be built: {e(ex)}</p>"

    # Splits
    splits = (sec("Against the spread and over/under", CTX, GB.splits_table(GS, a) + GB.splits_table(GS, h), "Closing lines from nflverse. Avg cover margin is how far the team beat (+) or missed (−) the spread on average.")
              + sec("Form by split", CTX, T(["Split", "G", "PPG", "Off EPA", "Off success", "Y/play", "Opp PPG", "Def EPA", "Def success", "Opp Y/play"], trends(c, a) + trends(c, h)), "Last-3 samples are small, so treat them as hints.")
              + sec("Unit by unit", CTX, team_compare(c, a, h, R)))

    # Lineups
    lineups = sec("Projected lineups", CTX, f"<div class='pcols'>{GB.lineup(c, a, s['inj'].get(a, []))}{GB.lineup(c, h, s['inj'].get(h, []))}</div>",
                  "Starters from the latest nflverse depth chart, with this week's injury report. A QB marked Starting (news) comes from the slate's QB override, which wins over a depth chart that has not caught up.")

    # Injuries
    injuries = sec("Injury report and data checks", CTX, P1)

    # Props
    picks = picks_view(gg, c, s, top10, cands, vals, ms)
    allprops = (keyp + sec("All prop tables", PRED, fold("Top 10 highest-probability props", D) + fold("Value opportunities", Eh) +
             fold("Historical hit rates", P10) + fold("Player projections and alternate lines", P11) + fold("Player main lines", P12) + fold("Parlay correlation", P13) + fold("Conservative, balanced, aggressive", H)))

    # Touchdowns
    tds = GB.touchdowns(c, a, h, gg.pl, s["inj"], mrow, _better, T, CTX, PRED) + sec("Simulation detail", PRED, fold("Goal-line tendencies and full TD model", P8))

    # Model
    model = sec("Executive summary", PRED, A) + sec("Final ranked recommendations", PRED, I) + sec("Scenarios", PRED, f"<p><b>A, {h} wins by 8+:</b> {pc(probs[0])}. <b>B, {a} wins by 8+:</b> {pc(probs[1])}. <b>C, one-score game:</b> {pc(probs[2])}.</p>" + sc_v + sc_t)

    tabs = [("o", "Overview", overview),
            ("m", "Matchup", subtabs(gid, "m", [("Offense & Defense", offdef), ("O-Line / D-Line", lnt), ("Special Teams", stt), ("Splits", splits), ("History", h2b + l10)])),
            ("pl", "Players", subtabs(gid, "pl", [("Lineups", lineups), ("Injuries", injuries)])),
            ("p", "Picks", subtabs(gid, "p", [("Picks", picks), ("Touchdowns", tds), ("All props", allprops)])),
            ("md", "Model", subtabs(gid, "md", [("Summary", model), ("Odds & lines", odds)]))]
    bar = "".join(f'<button class="gt" data-p="{gid}-{k}">{lab}</button>' for k, lab, _ in tabs)
    panes = "".join(f'<div id="{gid}-{k}" class="pane"{"" if k == "o" else " hidden"}>{body}</div>' for k, _, body in tabs)
    return f'<article id="{gid}" class="game" hidden><header class="gh"><h2>{e(g["away_name"])} at {e(g["home_name"])}</h2><p>{et.strftime("%A, %B %-d, %-I:%M %p")} ET · {e(g["venue"])}</p></header><div class="gtabs" role="tablist">{bar}</div>{panes}</article>', summ


def side_head_simple(a, h, mid):
    return f'<div class="mcols"><div class="l"><b>{a}</b></div><div class="lab">{mid}</div><div class="r"><b>{h}</b></div></div>'


def season_hit(c, r):
    kind = dict(rec="rec", rush="rush")[r["mk"]] if r["mk"] != "pass" else "passy"
    rows, _ = c.hitrates(r["p"]["player_id"], kind, [r["k"]], r["g"].a, r["p"]["team"], 5)
    q = rows[0]["season"]
    return f"{q[0]}/{q[1]}"


def final_five(c, gg, cands, top10):
    picks = []
    for r in cands:
        p = r["p"]
        kind = {"rec": "rec", "rush": "rush", "pass": "passy"}[r["mk"]]
        k = int(math.ceil(r["line"]))
        rows, _ = c.hitrates(p["player_id"], kind, [k], gg.h if p["team"] == gg.a else gg.a, p["team"], 5)
        hr = rows[0]["season"]
        ev_ = f"Projection {r['mean']:.0f} vs line {r['line']:g}"
        risk = "Game script turning against his role" if r["over"] else "One explosive play clears the under"
        picks.append(["Value", e(p["name"]), f"{'Over' if r['over'] else 'Under'} {r['line']:g} {r['lab']}", pc(r["prob"], 1), f"{hr[0]}/{hr[1]} at {k}+", ev_, risk, r["minodds"]])
    for r in top10:
        if len(picks) >= 5:
            break
        if any(e(r["p"]["name"]) == x[1] for x in picks):
            continue
        picks.append(["Probability", e(r["p"]["name"]), f"{r['k']:g}+ {r['lab']}", pc(r["prob"], 1), season_hit(c, r), "Clears the threshold in most simulated volume and efficiency outcomes", "Injury, early blowout, or snap reduction", am(max(r["prob"] - .02, .02))])
    return T(["Type", "Player", "Selection", "Est. hit", "2026 history", "Support", "Biggest risk", "Min odds"], picks[:5], "Value picks come first. Any remaining slots are the highest-probability outcomes and may have no value at market price.")


CSS = """/* Same look as the season hub: white field, green for the model's side and good results, red for the other side and warnings. Light in every theme. */
:root{--bg:#ffffff;--pan:#f5f7f6;--pan2:#edf0ef;--fg:#111615;--mut:#59626a;--dim:#838b91;--line:#e2e6e4;--line2:#cdd3d0;--acc:#0ea865;--acc2:#e1213b;--good:#0ea865;--bad:#e1213b;--warn:#c22e47;--head:#f5f7f6;--headfg:#111615;
--grn-d:rgba(14,168,101,.14);--red-d:rgba(225,33,59,.16);
--d:'Archivo','Arial Narrow',Impact,sans-serif;--b:'Hanken Grotesk',system-ui,-apple-system,sans-serif;--m:'JetBrains Mono',ui-monospace,Menlo,monospace;--r:14px;color-scheme:light}
*{box-sizing:border-box}body{background:var(--bg);color:var(--fg);font:15px/1.55 var(--b);margin:0;-webkit-font-smoothing:antialiased}
.top{position:relative;overflow:hidden;border-bottom:1px solid var(--line);padding-block:28px 22px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}
.top::before{content:"";position:absolute;inset:-40% -10% auto auto;width:60%;height:200%;background:radial-gradient(closest-side,rgba(14,168,101,.10),transparent 70%);pointer-events:none}
.top::after{content:"";position:absolute;left:0;right:0;bottom:-1px;height:1px;background:linear-gradient(90deg,var(--acc),transparent 45%,transparent 55%,var(--acc2))}
.top>*{position:relative}.top a{color:var(--mut)!important}
.top h1{font:900 clamp(34px,6.5vw,64px)/.95 var(--d);font-stretch:116%;text-transform:uppercase;letter-spacing:-.015em;margin:8px 0 10px;text-wrap:balance}.top p{margin:0;color:var(--mut);max-width:75ch}
nav{position:sticky;top:env(safe-area-inset-top,0px);z-index:5;background:rgba(255,255,255,.86);backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);border-bottom:1px solid var(--line);display:flex;gap:4px;overflow-x:auto;padding-block:10px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}
body::before{content:"";position:fixed;inset:0 0 auto;height:env(safe-area-inset-top,0px);z-index:10;background:var(--bg)}
nav button{font:600 14px var(--b);white-space:nowrap;border:1px solid var(--line);background:var(--pan);color:var(--mut);padding:7px 14px;border-radius:999px;cursor:pointer;transition:color .15s,border-color .15s}
nav button:hover{color:var(--fg);border-color:var(--line2)}
nav button[aria-pressed=true]{background:var(--fg);border-color:var(--fg);color:var(--bg)}:focus-visible{outline:2px solid var(--acc);outline-offset:2px}
main{max-width:1240px;margin:0 auto;padding-block:22px 72px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}
h2{font:800 clamp(28px,4.5vw,40px)/1.05 var(--d);font-stretch:112%;text-transform:uppercase;letter-spacing:-.01em;margin:0;text-wrap:balance}.gh p{color:var(--mut);margin:6px 0 16px}
.ph{background:var(--pan);border:1px solid var(--line);border-radius:var(--r);padding:16px 18px;margin-block:14px;min-width:0}
.ph h3{font:800 20px/1.2 var(--d);font-stretch:112%;text-transform:uppercase;margin:0 0 12px;display:flex;gap:10px;align-items:baseline}.pn{color:var(--acc2);font-family:var(--m);font-size:13px;font-weight:700}
h4{font:800 16px var(--d);font-stretch:112%;text-transform:uppercase;letter-spacing:.01em;margin:18px 0 8px;color:var(--fg)}
.tw{overflow-x:auto;margin-block:8px;max-width:100%}table{border-collapse:collapse;font:13px/1.4 var(--m);font-variant-numeric:tabular-nums;width:100%}
caption{caption-side:top;text-align:left;color:var(--mut);font:13px/1.45 var(--b);padding-bottom:8px}th{font:600 10.5px var(--m);text-transform:uppercase;letter-spacing:.07em;color:var(--dim);text-align:right;padding:8px;border-bottom:1px solid var(--line2);white-space:nowrap}
td{text-align:right;padding:7px 8px;border-bottom:1px solid var(--line);white-space:nowrap}th:first-child,td:first-child{text-align:left}td small{color:var(--dim)}tbody tr:hover td{background:var(--pan2)}
.note{color:var(--mut);font-size:13.5px;max-width:95ch}.warn{color:var(--warn);font-weight:600}.good{color:var(--good);font-weight:600}.bad{color:var(--bad);font-weight:600}
p.warn{background:var(--red-d);border:1px solid rgba(225,33,59,.35);border-radius:10px;padding:10px 14px;font-weight:500;color:var(--fg)}
.tag{display:inline-block;font:700 10.5px var(--m);text-transform:uppercase;letter-spacing:.06em;padding:2px 8px;border-radius:999px}.tag.good{color:var(--good);background:var(--grn-d)}.tag.warn{color:var(--warn);background:var(--red-d)}.tag.mute{color:var(--mut);border:1px solid var(--line2);font-weight:500}
.exec{display:flex;flex-wrap:wrap;gap:20px;align-items:center;margin-bottom:10px}.score{display:flex;gap:18px}.score div{display:flex;flex-direction:column;align-items:center}.score span{font:600 13px var(--m);color:var(--mut)}.score b{font:700 46px/1 var(--m)}
.exgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:10px;flex:1;min-width:0}.exgrid>div{background:var(--bg);border:1px solid var(--line);border-radius:10px;padding:8px 12px}.exgrid small{display:block;color:var(--dim);font:600 10px var(--m);text-transform:uppercase;letter-spacing:.08em}
details{margin-block:6px}summary{cursor:pointer;font-weight:600;color:var(--acc)}p,li{max-width:100ch}ul{padding-left:20px}a{color:var(--acc)}
.gtabs{display:inline-flex;gap:2px;background:var(--pan);border:1px solid var(--line);border-radius:999px;padding:3px;margin-bottom:16px;overflow-x:auto;max-width:100%}.gt{font:600 14px var(--b);background:none;border:0;border-radius:999px;padding:7px 16px;color:var(--mut);cursor:pointer;white-space:nowrap}
.gt[aria-selected=true]{color:var(--bg);background:var(--acc)}.gt:hover:not([aria-selected=true]){color:var(--fg)}
.mhead{display:grid;grid-template-columns:1fr minmax(0,1.5fr) 1fr;gap:14px;align-items:center;background:radial-gradient(120% 140% at 0% 0%,rgba(14,168,101,.10),transparent 55%),radial-gradient(120% 140% at 100% 100%,rgba(225,33,59,.10),transparent 55%),var(--pan);border:1px solid var(--line);border-radius:var(--r);color:var(--fg);padding:22px;margin-bottom:14px}
.mhead .tm b{display:block;font:900 clamp(32px,6vw,58px)/1 var(--d);font-stretch:118%}.mhead .tm span{font:500 12.5px var(--m);color:var(--mut)}.mhead .r{text-align:right}
.mhead .mid{display:grid;gap:8px;text-align:center}.mhead .pts{display:flex;justify-content:center;align-items:baseline;gap:14px}.mhead .pts b{font:700 36px/1 var(--m)}.mhead small{color:var(--mut);font-size:12.5px}
.wp{height:6px;background:var(--line2);border-radius:3px;overflow:hidden}.wp span{display:block;height:100%;background:var(--fg);border-radius:3px}
.mblock{background:var(--pan);border:1px solid var(--line);border-radius:var(--r);padding:16px 18px;margin-block:12px}.mblock h4{margin:0 0 10px;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
.lvh{display:flex;flex-wrap:wrap;gap:10px 18px;align-items:center;justify-content:space-between;padding:4px 0 12px;border-bottom:1px solid var(--line2);margin-bottom:6px}
.lvh small{display:block;font:600 10.5px var(--m);text-transform:uppercase;letter-spacing:.1em;color:var(--dim)}.lvh b{display:block;font:800 34px/1.1 var(--d);font-stretch:108%;font-variant-numeric:tabular-nums}.lvh b span{font:500 13px var(--m);color:var(--mut);margin-left:6px}
.lvt{font-size:12px;padding:5px 12px}.tag.bad{color:var(--bad);background:var(--red-d)}
.lsums{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,240px),1fr));gap:10px;margin-bottom:8px}
.lsum{background:var(--pan2);border:1px solid var(--line);border-radius:12px;padding:12px 14px;display:grid;gap:4px;justify-items:start}.lsum small{font:600 10.5px var(--m);text-transform:uppercase;letter-spacing:.1em;color:var(--dim)}
.lsum b{font:800 30px/1.1 var(--d);font-stretch:108%}.lsum em{font:600 12px var(--m);color:var(--warn);font-style:normal}
.kind{font:600 10px var(--m);text-transform:uppercase;letter-spacing:.1em;padding:3px 9px;border-radius:999px}.kind.pred{background:var(--grn-d);color:var(--acc)}.kind.ctx{border:1px solid var(--line2);color:var(--mut)}
.mrow{display:grid;grid-template-columns:1fr minmax(120px,200px) 1fr;gap:8px 18px;align-items:center;padding:7px 0;border-bottom:1px solid var(--line)}.mrow:last-of-type{border-bottom:0}
.mrow .lab{text-align:center;font-size:13px;color:var(--mut)}.mrow .l,.mrow .r{display:grid;grid-template-columns:auto 1fr;gap:10px;align-items:center}.mrow .l{grid-template-columns:1fr auto}
.mrow .l .v{order:2;text-align:right}.mrow .l .bar{order:1;direction:rtl}.mrow .v{font:600 14px var(--m);font-variant-numeric:tabular-nums;white-space:nowrap}.mrow .v i{font-style:normal;font-weight:400;color:var(--dim);font-size:11px;margin-left:5px}
.bar{height:6px;background:var(--line);border-radius:3px;overflow:hidden;display:block}.bar span{display:block;height:100%;border-radius:3px;background:var(--acc2);opacity:.55}.mrow .win .bar span{background:var(--acc);opacity:1}.mrow .win .v{color:var(--acc)}
.pcols{display:grid;grid-template-columns:1fr 1fr;gap:18px}.pcol{min-width:0}.pcol h5{font:800 22px var(--d);font-stretch:115%;margin:0 0 6px}.plist{list-style:none;padding:0;margin:0}.plist li{display:grid;gap:2px;padding:8px 0;border-bottom:1px solid var(--line);font-size:13.5px}.pn2{font-weight:600}
.tier{background:var(--pan);border:1px solid var(--line);border-radius:var(--r);padding:16px 18px;margin-block:12px;position:relative;overflow:hidden}.tier h4{margin-top:0}
.tier::before{content:"";position:absolute;left:0;right:0;top:0;height:3px;background:transparent}.t1::before{background:var(--good)}.t2::before{background:linear-gradient(90deg,var(--good),var(--bad))}.t3::before{background:var(--bad)}.check li{margin-block:4px}
.subtabs{display:flex;gap:4px 20px;overflow-x:auto;border-bottom:1px solid var(--line);margin:-4px 0 16px;scrollbar-width:none}.subtabs::-webkit-scrollbar{display:none}
.sb{font:600 13.5px var(--b);background:none;border:0;border-bottom:2px solid transparent;margin-bottom:-1px;padding:8px 2px;color:var(--mut);cursor:pointer;white-space:nowrap}.sb[aria-selected=true]{color:var(--fg);border-bottom-color:var(--acc)}.sb:hover:not([aria-selected=true]){color:var(--fg)}
.best{border-color:var(--line2)}.bestg{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,230px),1fr));gap:14px 22px}.bcol{display:flex;flex-direction:column;min-width:0;border-top:3px solid var(--line2);padding-top:10px}.bcol.bl{border-top-color:var(--good)}.bcol.bw{border-top-color:var(--bad)}
.bcol h5{margin:0 0 4px;font:700 11px var(--m);text-transform:uppercase;letter-spacing:.07em;color:var(--mut)}.brow{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:10px;align-items:center;padding:7px 0;border-bottom:1px solid var(--line)}
.brow b{font-weight:650}.brow small{color:var(--mut);font:500 11.5px var(--m)}.brow span{display:block;color:var(--mut);font-size:13px}.bp{text-align:right;min-width:64px}.bp b{font:700 17px var(--m);font-variant-numeric:tabular-nums}.bp i{display:block;height:4px;border-radius:2px;background:var(--line);margin-top:3px;overflow:hidden}.bp u{display:block;height:100%;background:var(--good);border-radius:2px}
.bcol.bt .bp u{background:var(--acc)}.bcol ul{margin:2px 0 0;padding-left:18px;font-size:13.5px}.bcol li{margin-block:5px}.blink{margin-top:auto;padding:10px 0 0;align-self:flex-start;background:none;border:0;color:var(--acc);font:600 13px var(--b);cursor:pointer}.blink:hover{text-decoration:underline}
.mcols{display:grid;grid-template-columns:1fr minmax(120px,200px) 1fr;gap:8px 18px;align-items:end;padding:2px 0 10px;border-bottom:1px solid var(--line2);margin-bottom:2px}
.mcols b{display:block;font:800 15px var(--d);font-stretch:112%;text-transform:uppercase;letter-spacing:.02em}.mcols small{display:block;font:500 11.5px var(--m);color:var(--mut)}
.mcols .l{text-align:right}.mcols .lab{text-align:center;font:500 10.5px/1.45 var(--m);color:var(--dim);text-transform:uppercase;letter-spacing:.05em}
.mrow .lab small{display:block;font-size:11px;color:var(--dim)}
label.win{margin-left:auto;display:inline-flex;align-items:center;gap:8px;font:600 10.5px var(--m);text-transform:uppercase;letter-spacing:.08em;color:var(--dim)}
label.win select{font:600 13px var(--b);text-transform:none;letter-spacing:0;padding:6px 10px;border:1px solid var(--line2);background:var(--bg);color:var(--fg);border-radius:999px;cursor:pointer}
@media (max-width:640px){th,td{padding:6px 4px}.pcol .tw table{font-size:12px}.mcols{grid-template-columns:1fr 1fr}.mcols .lab{grid-column:1/-1;grid-row:1;text-align:center}label.win{margin-left:0}}
.board{background:radial-gradient(120% 140% at 0% 0%,rgba(14,168,101,.08),transparent 55%),var(--pan);border:1px solid var(--line);border-radius:var(--r);padding:14px 18px 6px;margin-bottom:12px}
.board .bh,.board .br{display:grid;grid-template-columns:minmax(0,2fr) repeat(3,minmax(74px,1fr));gap:8px;align-items:center}
.board .bh span{font:600 10.5px var(--m);text-transform:uppercase;letter-spacing:.08em;color:var(--dim);text-align:center}.board .br{padding:8px 0;border-top:1px solid var(--line)}
.otm{display:flex;gap:12px;align-items:center;min-width:0}.otm .ab{display:none}.otm b{display:block;font:800 18px/1.1 var(--d);font-stretch:110%;text-transform:uppercase}.otm small{color:var(--mut);font:500 12px var(--m)}
.mono{flex:none;display:grid;place-items:center;width:42px;height:42px;border-radius:50%;background:var(--pan2);border:1px solid var(--line2);font:800 13px var(--d);font-stretch:120%;letter-spacing:.02em}
.oc{background:var(--bg);border:1px solid var(--line2);border-radius:10px;padding:8px 6px;text-align:center}.oc b{display:block;font:700 15px var(--m)}.oc small{color:var(--mut);font:500 11.5px var(--m)}
.wpb{margin-block:10px}.wl{display:flex;justify-content:space-between;align-items:baseline;gap:8px;font:500 13px var(--m)}.wl small{color:var(--dim);font:600 10.5px var(--m);text-transform:uppercase;letter-spacing:.08em;text-align:center}
.wbar2{display:flex;height:10px;border-radius:5px;overflow:hidden;gap:2px;margin-top:6px}.wbar2 .f{background:var(--acc)}.wbar2 .u{background:var(--acc2);opacity:.7}
.proj{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px;margin-top:14px}.proj>div{background:var(--bg);border:1px solid var(--line);border-radius:10px;padding:8px 12px}
.proj small{display:block;color:var(--dim);font:600 10px var(--m);text-transform:uppercase;letter-spacing:.08em}.proj b{font:700 15px var(--m)}.proj b+small{display:inline;text-transform:none;letter-spacing:0}
.segs{display:inline-flex;gap:2px;background:var(--bg);border:1px solid var(--line2);border-radius:999px;padding:3px;margin-bottom:8px}
.seg{font:600 13px var(--b);background:none;border:0;border-radius:999px;padding:6px 14px;color:var(--mut);cursor:pointer}.seg[aria-pressed=true]{background:var(--fg);color:var(--bg)}
.spark{width:100%;height:auto;max-width:760px;display:block}.spark .sv{font:700 12px var(--m);fill:var(--fg)}.spark .sl{font:500 10.5px var(--m);fill:var(--mut)}.sub{color:var(--mut);font-size:13px;margin:4px 0}
.h2{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:8px}.h2c{background:var(--bg);border:1px solid var(--line);border-radius:10px;padding:8px 12px;display:grid;gap:4px}
.h2c small{color:var(--dim);font:500 11px var(--m)}.h2s{display:grid;font:500 14px var(--m)}.h2s span{color:var(--mut)}.h2s .w{color:var(--fg);font-weight:700}
.chips{display:flex;gap:4px;flex-wrap:wrap;margin-bottom:8px}.res{display:grid;place-items:center;width:26px;height:26px;border-radius:7px;font:700 12px var(--m)}.res.w{background:var(--grn-d);color:var(--good)}.res.l{background:var(--red-d);color:var(--bad)}.res.t{border:1px solid var(--line2);color:var(--mut)}
.l10s{display:grid;grid-template-columns:repeat(3,1fr);gap:6px}.l10s>div{background:var(--bg);border:1px solid var(--line);border-radius:10px;padding:6px 10px}.l10s small{display:block;color:var(--dim);font:600 10px var(--m);text-transform:uppercase;letter-spacing:.08em}.l10s b{font:700 15px var(--m)}
.lu h5{font:800 22px var(--d);font-stretch:115%;margin:0}.lu h6{font:700 11px var(--m);text-transform:uppercase;letter-spacing:.08em;color:var(--dim);margin:14px 0 4px}.lu ul{list-style:none;padding:0;margin:0}
.lu li{display:grid;grid-template-columns:44px 1fr;gap:8px;padding:5px 0;border-bottom:1px solid var(--line);font-size:14px}.lu .pos{font:600 12px var(--m);color:var(--mut)}
.ij{font:700 10px var(--m);padding:1px 6px;border-radius:999px;margin-left:4px;vertical-align:1px}.ij.o{background:var(--red-d);color:var(--bad)}.ij.q{border:1px solid rgba(194,46,71,.5);color:var(--warn)}.ij.n{background:var(--grn-d);color:var(--good)}
details.fold{border-top:1px solid var(--line);padding:10px 0}details.fold>summary{font:800 15px var(--d);font-stretch:110%;text-transform:uppercase;color:var(--fg);list-style:none;display:flex;justify-content:space-between}
details.fold>summary::after{content:"+";color:var(--acc);font-family:var(--m)}details.fold[open]>summary::after{content:"–"}details.fold>summary::-webkit-details-marker{display:none}
.gtabs{display:flex;width:100%;border-radius:12px}.gt{flex:1 1 0;min-width:0;padding:7px 4px;font-size:13.5px;text-align:center}
@media (max-width:640px){.board{padding:12px}.board .bh,.board .br{grid-template-columns:minmax(0,1.3fr) repeat(3,minmax(58px,1fr));gap:5px}.mono{display:none}.otm b{font-size:22px}.otm .fn{display:none}.otm .ab{display:inline}.spark .sv{font-size:21px}.spark .sl{font-size:18px}.exgrid{flex-basis:100%;grid-template-columns:1fr 1fr}.oc b{font-size:13px}.l10s{grid-template-columns:repeat(3,1fr)}}
.lean{margin:10px 0 0;padding:8px 12px;border-left:3px solid var(--acc);background:var(--bg);border-radius:0 8px 8px 0;font-size:13.5px}
.tdl{list-style:none;padding:0;margin:0;counter-reset:none}.tdl li{display:grid;grid-template-columns:34px minmax(0,1fr) auto;gap:12px;align-items:center;padding:12px 0;border-bottom:1px solid var(--line)}.tdl li:last-child{border-bottom:0}
.tdl .rk{display:grid;place-items:center;width:30px;height:30px;border-radius:50%;background:var(--grn-d);color:var(--acc);font:800 15px var(--d)}.tdl li:first-child .rk{background:var(--acc);color:var(--bg)}
.tdl b{font-size:15.5px}.tdl small{color:var(--mut);font:500 12px var(--m)}.tdl p{margin:3px 0 0;color:var(--mut);font-size:13.5px}
.tdp{text-align:right;display:grid}.tdp b{font:700 22px var(--m);color:var(--acc)}.tdp small{color:var(--dim);font:600 10px var(--m);text-transform:uppercase;letter-spacing:.08em}.tdp span{color:var(--mut);font:500 11.5px var(--m);white-space:nowrap}
@media (max-width:640px){.tdl li{grid-template-columns:30px minmax(0,1fr)}.tdp{grid-column:2;text-align:left;grid-template-columns:auto auto;justify-content:start;gap:2px 10px;align-items:baseline}.tdp span{grid-column:1/-1}}
@media (prefers-reduced-motion:no-preference){html{scroll-behavior:smooth}}
@media (max-width:640px){.mrow{grid-template-columns:1fr 1fr;grid-template-areas:"lab lab" "l r";row-gap:4px}.mrow .lab{grid-area:lab}.mrow .l{grid-area:l}.mrow .r{grid-area:r}.mhead{grid-template-columns:1fr 1fr;padding:16px}.mhead .mid{grid-column:1/-1;grid-row:2}.pcols{grid-template-columns:1fr}}"""
FONTS = '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin><link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@62..125,500..900&family=Hanken+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap">'
JS = """const B=[...document.querySelectorAll('nav button')],V=[...document.querySelectorAll('.game,#slate')];
function go(i,s){V.forEach(v=>v.hidden=v.id!==i);B.forEach(b=>b.setAttribute('aria-pressed',b.dataset.t===i));if(s)scrollTo(0,0)}
B.forEach(b=>b.onclick=()=>{go(b.dataset.t,1);try{localStorage.setItem('nfl-tab',b.dataset.t)}catch(e){}});
document.querySelectorAll('[data-go]').forEach(a=>a.onclick=ev=>{ev.preventDefault();go(a.dataset.go,1)});
document.querySelectorAll('.game').forEach(g=>{const T=[...g.querySelectorAll('.gt')],P=[...g.querySelectorAll('.pane')];
const sel=id=>{P.forEach(p=>p.hidden=p.id!==id);T.forEach(b=>b.setAttribute('aria-selected',b.dataset.p===id))};T.forEach(b=>b.onclick=()=>sel(b.dataset.p));sel(T[0].dataset.p);P.forEach(p=>{const S=[...p.querySelectorAll('.sb')],U=[...p.querySelectorAll('.sub')];if(!S.length)return;const f=id=>{U.forEach(u=>u.hidden=u.id!==id);S.forEach(b=>b.setAttribute('aria-selected',b.dataset.s===id))};S.forEach(b=>b.onclick=()=>f(b.dataset.s));f(S[0].dataset.s)});g.querySelectorAll('[data-tab]').forEach(a=>a.onclick=()=>{sel(a.dataset.tab);const b=a.dataset.sub&&g.querySelector('.sb[data-s="'+a.dataset.sub+'"]');if(b)b.click();g.querySelector('.gtabs').scrollIntoView({block:'start'})})});
document.querySelectorAll('select[data-win]').forEach(s=>{const box=s.closest('.mblock'),f=()=>box.querySelectorAll('[data-w]').forEach(d=>d.hidden=d.dataset.w!==s.value);s.onchange=f;f()});
document.querySelectorAll('.om').forEach(o=>{const B=[...o.querySelectorAll('[data-om]')],f=k=>{o.querySelectorAll('[data-omv]').forEach(d=>d.hidden=d.dataset.omv!==k);B.forEach(b=>b.setAttribute('aria-pressed',b.dataset.om===k))};B.forEach(b=>b.onclick=()=>f(b.dataset.om));f('sp')});
let t=location.hash.slice(1);if(!V.some(v=>v.id===t)){try{t=localStorage.getItem('nfl-tab')}catch(e){}}go(V.some(v=>v.id===t)?t:'slate')"""


def reliab_svg(rows):
    W, H, p = 340, 300, 42
    x = lambda v: p + v * (W - p - 12)
    y = lambda v: H - p - v * (H - p - 14)
    g = "".join(f'<line x1="{x(v)}" x2="{x(v)}" y1="{y(0)}" y2="{y(1)}" stroke="var(--line)"/><line x1="{x(0)}" x2="{x(1)}" y1="{y(v)}" y2="{y(v)}" stroke="var(--line)"/><text x="{x(v)}" y="{H - p + 16}" text-anchor="middle" font-size="11" fill="var(--mut)">{int(v * 100)}</text><text x="{p - 7}" y="{y(v) + 4}" text-anchor="end" font-size="11" fill="var(--mut)">{int(v * 100)}</text>' for v in (0, .25, .5, .75, 1))
    d = "".join(f'<circle cx="{x(r["pred"]):.1f}" cy="{y(r["hit"]):.1f}" r="{3 + min(r["n"], 3000) ** .5 / 18:.1f}" fill="var(--acc)" fill-opacity=".8"/>' for r in rows)
    return f'<svg viewBox="0 0 {W} {H}" width="100%" style="max-width:{W}px" role="img" aria-label="Predicted against actual hit rate on the 2026 holdout">{g}<line x1="{x(0)}" y1="{y(0)}" x2="{x(1)}" y2="{y(1)}" stroke="var(--acc2)" stroke-dasharray="4 3"/>{d}<text x="{(W + p) / 2}" y="{H - 6}" text-anchor="middle" font-size="12" fill="var(--fg)">Predicted %</text><text transform="translate(12 {H / 2}) rotate(-90)" text-anchor="middle" font-size="12" fill="var(--fg)">Actual %</text></svg>'


def slate_page(sum_, bt, c, shifts=()):
    allv = sorted([v for s in sum_ for v in s["vals"] if not v["flag"]], key=lambda r: -r["edge"])
    side = pd.Series([v["over"] for v in allv]).mean()
    r1 = [[f'<a href="#" data-go="{s["id"]}">{s["a"]} at {s["h"]}</a>', s["kick"], e(s["line"]), s["total"], f'{s["a"]} {s["ma"]:.1f}, {s["h"]} {s["mh"]:.1f}', f'{s["pa"]:.1f}-{s["ph"]:.1f}', f'{s["win"]} {pc(s["pw"])}', pc(max(s["script"])), f'{e(s["best"][0])} {s["best"][1]:+.0f} ({100 * s["best"][2]:+.1f}%)'] for s in sum_]
    t1 = T(["Game", "Kickoff ET", "DK spread", "DK total", "Our line", "Blended score", "Winner", "Top script", "Best game-line EV"], r1, "Blended score is 90% market, 10% our model, because the backtest found no edge over closing lines. Click a game for its full report.")
    t2 = T(["#", "Player", "Game", "Selection", "Prob", "Edge vs -110", "Min odds"], [[i + 1, e(r["p"]["name"]), r["game"], f"{'Over' if r['over'] else 'Under'} {r['line']:g} {r['lab']}", pc(r["prob"], 1), f"{r['edge'] * 100:+.1f}%", r["minodds"]] for i, r in enumerate(allv[:15])], "Largest shrunk gaps from DK main lines. Excludes questionable players, QB-change teams, fresh role changes, and any gap over 33% of the line. Check the price: a line at -125 needs 55.6%, not 52.4%.")
    tops = sorted([r for s in sum_ for r in s["top10"]], key=lambda r: -r["prob"])[:15]
    t3 = T(["#", "Player", "Game", "Outcome", "Calibrated P", "Fair odds"], [[i + 1, e(r["p"]["name"]), r["game"], f"{r['k']:g}+ {r['lab']}", pc(r["prob"]), am(r["prob"])] for i, r in enumerate(tops)], "Highest-probability posted milestones across the slate. Prices are unknown and short odds are likely.")
    tds = sorted([(p, t, q) for s in sum_ for p, t, q in s["td"]], key=lambda x: -x[2])[:10]
    t4 = T(["#", "Player", "Tm", "Anytime TD", "Fair odds"], [[i + 1, e(p), t, pc(q), am(q)] for i, (p, t, q) in enumerate(tds)])
    m = bt["markets"]
    names = dict(rec="Receiving yards", rush="Rushing yards", atd="Anytime TD")
    names["pass"] = "Passing yards"
    t5 = T(["Market", "Holdout forecasts", "Model Brier", "Base-rate Brier", "Improvement", "MAE (yds)"], [[names[k], m[f"{k}_2026_holdout"]["n"], n(m[f"{k}_2026_holdout"]["brier_cal"], 4), n(m[f"{k}_2026_holdout"]["brier_base"], 4), pc(1 - m[f"{k}_2026_holdout"]["brier_cal"] / m[f"{k}_2026_holdout"]["brier_base"], 1), n(bt["mae"].get(f"{k}_2026"), 1)] for k in ("rec", "rush", "pass", "atd")], "2026 Weeks 4-8, scored with calibration fit only on 2025. Lower Brier is better.")
    mm, mt = bt["mae_margin"], bt["mae_total"]
    t6 = T(["Tm", "Player", "Pos", "Earlier 2026 snap %", "Latest game"], [[t, e(nm), ps, pc(a_), pc(b_)] for t, nm, ps, a_, b_ in shifts])
    return f"""<div id="slate" hidden><h2>Slate</h2><p class="note">{len(sum_)} game{'s' if len(sum_) != 1 else ''} on {DAY:%A, %b %-d, %Y}. Data pulled {PULLED}. Lines move and inactives post about 90 minutes before kickoff, so rerun after they post.</p>
    <section class="ph"><h3>All games</h3>{t1}</section>
    <section class="ph"><h3>What the model can and cannot do</h3><div style="display:flex;flex-wrap:wrap;gap:20px;align-items:flex-start"><div style="flex:1 1 420px;min-width:0">{t5}
    <p><b>Player forecasts are well calibrated.</b> On the holdout, predicted buckets landed within about 3 points of actual hit rates. The test used who actually played, so live forecasts will be slightly worse because inactives are unknown in advance.</p>
    <p><b>Game lines show no edge.</b> Closing spread MAE was {mm['market']:.2f} points vs {mm['model']:.2f} for our line, and the fitted weight on our line was about zero. Closing totals: {mt['market']:.2f} vs {mt['model']:.2f}. So game sides default to pass.</p>
    <p><b>Prop edges are unproven.</b> Calibration shows the probabilities are honest. It does not show they beat DraftKings, because no historical prop lines were available to test against. The edges here are leads to price-check.</p></div><div style="flex:0 1 340px;min-width:0">{reliab_svg(bt['reliability_2026'])}<p class="note">2026 holdout, all markets. Dots near the dashed line mean a forecast of X% hit about X% of the time. Bigger dots hold more forecasts.</p></div></div></section>
    <section class="ph"><h3>Top line gaps across the slate</h3>{t2}<p class="note">Before re-centering, the sim leaned under on most receiving lines: its means match DraftKings but its medians run lower. That tilt is treated as model shape, not opportunity, so these gaps show relative disagreement only. Over share after re-centering: {pc(side)}.</p></section>
    <section class="ph"><h3>Fresh role changes</h3><p class="note">These players' snap share moved 20+ points in their latest game. Season usage lags these moves, so their lines are not counted as edges and projections for them may be off. In the backfields listed, all rushing lines are held out. Example: Coleman took 49% of DEN snaps in Week 7 to Dobbins's 23%, and Mason took 64% of MIN snaps to Jones's 25%.</p>{t6}</section><section class="ph"><h3>Highest-probability alternates</h3>{t3}</section><section class="ph"><h3>Top touchdown candidates</h3>{t4}</section>
    <section class="ph"><h3>Data gaps</h3><ul><li><b>Prop prices:</b> ESPN's DraftKings feed gives lines and milestone levels but no prices. Moneyline, spread and total prices are real. Prop edges assume -110, and each pick shows a minimum acceptable price.</li>
    <li><b>2026 coverage:</b> man/zone and coverage shells have no free 2026 feed. 2025 charting stands in, labeled every time, and flagged when a head coach changed. Coordinator changes are not tracked.</li><li><b>Alignment and routes:</b> 2026 slot/wide alignment and routes run are paid data. Target side, depth and 2025 route targets stand in. Press rate, shadow rate, tackle-vs-edge win rate, and run-scheme splits are left out.</li>
    <li><b>Under-pressure splits:</b> team pressure rates are 2026 (PFR). Individual QB pressure splits are 2025. Blitz splits are 2026 (FTN).</li><li><b>Injury odds:</b> P(plays) values are assumptions from status and practice, not fitted estimates.</li></ul>
    <p><a href="https://github.com/nflverse/nflverse-data">nflverse</a> supplied play-by-play, FTN charting, PFR advanced stats, NGS, snap counts, depth charts and rosters. ESPN supplied the schedule, injuries and DraftKings lines. <a href="https://www.weather.gov">NWS</a> supplied forecasts.</p>
    <p class="note">These are estimates, and no selection is guaranteed. After the games, run review.py to score every forecast (Phase 15).</p></section></div>"""


def save_picks(sums):
    """Every flagged-or-not prop gap and each game's best line, with the line taken, so review.py can score closing line value."""
    rows = []
    for s_ in sums:
        for r in s_["vals"]:
            p = r["p"]
            rows.append(dict(date=DATE, event=r["g"].g["id"], game=r["game"], kind="prop", player=p["name"], athlete=p.get("espn"), market=MK[r["mk"]][0], side="over" if r["over"] else "under",
                             line=r["line"], price=None, prob=r["prob"], edge=r["edge"], flagged=r["flag"]))
        lab, price, ev = s_["best"]
        rows.append(dict(date=DATE, event=s_["event"], game=f'{s_["a"]}@{s_["h"]}', kind="game", player=None, athlete=None, market=lab, side=None, line=None, price=price, prob=None, edge=ev, flagged=ev <= .03))
    # closing line value is measured from the first run, so a rerun after inactives does not overwrite it
    pd.DataFrame(rows).to_csv(PICKS.replace(".csv", "_final.csv") if os.path.exists(PICKS) else PICKS, index=False)


def main():
    data = pickle.load(open(CACHE, "rb"))
    d = E.load()
    c = S.Ctx(d)
    R = build_ranks(c)
    R["rt"] = data["ratings"]
    arts, sums = [], []
    shifts = find_shifts(c)
    slate_t = {t for s in data["res"] for t in (s["game"]["home"], s["game"]["away"])}
    R["shifts"] = sorted([x for x in shifts if x[0] in slate_t])
    SHIFT["rb"].update(HOLD_RUSH)  # backfield news that usage history cannot see
    print(R["shifts"])
    for s in data["res"]:  # first pass sets the slate-wide tilt per market, so a shared bias is not read as edge
        s["c_status"] = data["stat_espn"]
        value_rows(G(s, c, data["cal"]))
    OFFSET.update({k: float(np.median(v)) for k, v in RAWZ.items() if len(v) >= 5})  # too few lines to call it a tilt
    for s in data["res"]:
        s["c_status"] = data["stat_espn"]
        a, b = render_game(s, c, R, data["cal"])
        arts.append(a)
        sums.append(b)
        print("rendered", b["id"], flush=True)
    btn = '<button data-t="slate">Slate</button>' + "".join(f'<button data-t="{x["id"]}">{x["a"]} at {x["h"]}</button>' for x in sums)
    page = f"""<meta charset="utf-8"><title>{TITLE}</title><meta name="claude:viewport-fit-sides" content="cover"><link rel="preconnect" href="https://fonts.googleapis.com">{FONTS}<style>{CSS}</style><style>{GB.logo_css()}</style>
    <div class="top"><h1>{TITLE}</h1><p>Full matchup report, 20,000-game simulation, and calibrated player forecasts for {DAY:%A, %B %-d, %Y}.</p></div><nav>{btn}</nav><main>{slate_page(sums, data['bt'], c, R['shifts'])}{''.join(arts)}</main><script>{JS}</script>"""
    with open(OUT, "w") as f:
        f.write(page)
    save_picks(sums)
    save_summary(sums)
    print(OUT, len(page) / 1e6, "MB")


def save_summary(sums):
    """Small per-slate summary the season hub (season.py) reads: score, win chance, Most likely tier."""
    out = [dict(id=x["id"], a=x["a"], h=x["h"], win=x["win"], pw=float(x["pw"]), pa=float(x["pa"]), ph=float(x["ph"]), ma=float(x["ma"]), mh=float(x["mh"]),
                line=x["line"], total=float(x["total"]), likely=[dict(player=r["p"]["name"], team=r["p"]["team"], outcome=f"{r['k']:g}+ {r['lab']}", prob=float(r["prob"])) for r in x["top10"]],
                td=[dict(player=p, team=t, prob=float(q)) for p, t, q in x["td"]]) for x in sums]
    with open(os.path.join(os.path.dirname(OUT), f"summary_{DATE}.json"), "w") as f:
        json.dump(dict(date=DATE, week=WEEK, built=datetime.now().strftime("%Y-%m-%d %H:%M"), games=out), f, indent=1)


if __name__ == "__main__":
    main()
