"""Render the week's Game Books (Top 25 + SEC, ACC, Big Ten, Big 12) into output/cfb_w<week>_report.html,
plus output/summary_w<week>.json, the per-game numbers the season hub reads."""
import json
import os
import pickle
import re
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import cfb_model as C
import run_cfb as R
import report as P
from backtest import apply_cal
from report import e, n, pc, am, T, phase, tag, scenarios, correlation, td_block, game_lines, combos, SHIFT, CSS, reliab_svg, PRED, CTX, side_head_simple, subtabs, best_plays

ET = ZoneInfo("America/New_York")
WK = R.week()
OUT = f"output/cfb_w{WK}_report.html"
TH = {"rec": [20, 30, 40, 50, 60, 70, 80, 90, 100, 120], "rush": [20, 30, 40, 50, 60, 70, 80, 90, 100, 120], "pass": [150, 175, 200, 225, 250, 275, 300, 325]}
LAB = {"rec": "receiving yds", "rush": "rushing yds", "pass": "passing yds"}
KEY = {"rec": "rec_y", "rush": "rush_y", "pass": "pass_y"}
NOTE = R.hand(WK).get("note", {})
BUILT = ""
SIMILAR = .34


class G(P.G):
    def cp(self, arr, k, mk, p=None):
        return hi(super().cp(arr, k, mk, p))


def hi(p):
    """The holdout's 84% and 93% buckets hit 80% and 89%, so high probabilities take a haircut."""
    return p - min(.04, max(p - .76, 0) * .5)


def fair(arr, cal, target):
    """The number where the calibrated chance of going over equals target."""
    grid = np.linspace(.01, .99, 197)
    c = hi_v(apply_cal(grid, cal))
    raw = grid[np.argmin(np.abs(c - target))]
    return float(np.quantile(arr, 1 - raw))


def hi_v(p):
    return np.array([hi(x) for x in p])


def half(x):
    return round(x * 2) / 2 - (.5 if round(x * 2) % 2 == 0 else 0)


def success(x):
    return ((x.down == 1) & (x.yds >= .5 * x.dist)) | ((x.down == 2) & (x.yds >= .7 * x.dist)) | ((x.down >= 3) & (x.yds >= x.dist))


def rk_(v):
    return "–" if pd.isna(v) else f"#{v:.0f}"


class Ctx:
    def __init__(self, d):
        self.d = d
        g = d["games"]
        self.g = g[(g.season == 2026) & (g.stype == 2)].copy()
        self.g["dt"] = pd.to_datetime(self.g.date, utc=True)
        l_ = d["plays"]
        l_ = l_[l_.game_id.isin(self.g.game_id)].copy()
        l_["ok"] = success(l_)
        l_["wk"] = l_.game_id.map(self.g.set_index("game_id").week)
        self.l = l_
        self.pl = d["players"][d["players"].season == 2026].copy()
        self.pl = self.pl.merge(self.g[["game_id", "home", "away", "week"]].rename(columns={"week": "wk"}), on="game_id")
        self.tab = self.table()
        self.rk = self.tab.rank(ascending=False)
        for k in ("opp_ppg", "opp_ypp", "opp_succ", "opp_ypa", "opp_ypc", "opp_expl", "sack_allowed", "opp_third", "giveaways"):
            self.rk[k] = self.tab[k].rank()
        teams = sorted(set(self.g.home) | set(self.g.away) | set(d["games"].home) | set(d["games"].away))
        self.tab = self.tab.reindex(self.tab.index.union(teams))   # FCS opponents: no FBS row, shown as "–"
        self.rk = self.rk.reindex(self.tab.index)

    def line(self, x):
        p = x[x.pass_play == 1]
        r = x[x.pass_play == 0]
        third = x[x.down == 3]
        return dict(plays=len(x), ypp=x.yds.mean(), succ=x.ok.mean(), ypa=p.yds.mean(), ypc=r.yds.mean(), expl=(x.yds >= 20).mean(), third=(third.yds >= third.dist).mean(), sack=p.sack.mean(), pass_r=len(p) / max(len(x), 1))

    def table(self):
        rows = {}
        fbs = self.d["fbs"]
        for t in sorted(fbs):
            gh, ga = self.g[self.g.home == t], self.g[self.g.away == t]
            n_ = len(gh) + len(ga)
            if n_ < 3:
                continue
            o, de = self.line(self.l[self.l.off == t]), self.line(self.l[self.l.de == t])
            tov = self.l[self.l.ty.isin(["Pass Interception Return", "Interception Return Touchdown", "Fumble Recovery (Opponent)", "Fumble Return Touchdown"])]
            rows[t] = dict(ppg=(gh.hs.sum() + ga.aw.sum()) / n_, opp_ppg=(gh.aw.sum() + ga.hs.sum()) / n_, ypp=o["ypp"], opp_ypp=de["ypp"], succ=o["succ"], opp_succ=de["succ"], ypa=o["ypa"], opp_ypa=de["ypa"], ypc=o["ypc"], opp_ypc=de["ypc"], expl=o["expl"], opp_expl=de["expl"],
                           third=o["third"], opp_third=de["third"], sack_made=de["sack"], sack_allowed=o["sack"], giveaways=(tov.off == t).sum() / n_, takeaways=(tov.de == t).sum() / n_, plays=o["plays"] / n_, pass_r=o["pass_r"], g=n_)
        return pd.DataFrame(rows).T

    def compare(self, a, h):
        L = [("Points per game", "ppg", 1), ("Points allowed", "opp_ppg", 1), ("Yards per play", "ypp", 2), ("Yards per play allowed", "opp_ypp", 2), ("Success rate", "succ", 3), ("Success rate allowed", "opp_succ", 3), ("Yards per dropback", "ypa", 2), ("Dropback yards allowed", "opp_ypa", 2), ("Yards per carry", "ypc", 2), ("Carry yards allowed", "opp_ypc", 2),
             ("20+ yard play rate", "expl", 3), ("20+ rate allowed", "opp_expl", 3), ("Third-down rate", "third", 3), ("Third-down rate allowed", "opp_third", 3), ("Sack rate made", "sack_made", 3), ("Sack rate allowed", "sack_allowed", 3), ("Takeaways/g", "takeaways", 2), ("Giveaways/g", "giveaways", 2)]
        rows = []
        for lab, k, d in L:
            rows.append([lab, n(self.tab.at[a, k], d), f"{rk_(self.rk.at[a, k])}", n(self.tab.at[h, k], d), f"{rk_(self.rk.at[h, k])}"])
        return T(["2026 regular season", a, "FBS rank", h, "FBS rank"], rows, f"Raw 2026 results, not opponent-adjusted. Ranks are out of {len(self.tab)} FBS teams, and 1 is always best.")

    def recent(self, t):
        rows = []
        ws = sorted(set(self.l[self.l.off == t].wk))[-3:]
        for lab, q in (("Season", self.l), ("Last 3", self.l[self.l.wk.isin(ws)])):
            o, de = self.line(q[q.off == t]), self.line(q[q.de == t])
            rows.append([t, lab, n(o["ypp"], 2), pc(o["succ"], 1), n(de["ypp"], 2), pc(de["succ"], 1)])
        return rows

    def strengths(self, t):
        names = dict(ppg="scoring", ypp="yards per play", opp_ypp="yards per play allowed", ypa="dropback yards", opp_ypa="dropback yards allowed", ypc="yards per carry", opp_ypc="carry yards allowed", opp_expl="explosive plays allowed", expl="explosive plays", sack_made="sack rate", sack_allowed="sack rate allowed", takeaways="takeaways", giveaways="giveaways")
        r = self.rk.loc[t, list(names)].sort_values()
        good = [f"{names[k]} (#{v:.0f})" for k, v in r.items() if v <= 30][:4]
        bad = [f"{names[k]} (#{v:.0f})" for k, v in r.items() if v >= 100][-3:]
        return good, bad

    def pace(self, t):
        x = self.l[self.l.off == t]
        close, lead, trail = x[x.sd.abs() <= 7], x[x.sd >= 8], x[x.sd <= -8]
        ers = x[(x.down.isin([1, 2])) & (x.sd.abs() <= 7)]
        return [t, n(self.tab.at[t, "plays"], 1), pc(self.tab.at[t, "pass_r"]), pc(ers.pass_play.mean()), pc(close.pass_play.mean()), pc(lead.pass_play.mean()), pc(trail.pass_play.mean())]

    def zones(self, de):
        """What this defense gives up by throw location. Only games with charted text have depth and side."""
        x = self.l[(self.l.de == de) & (self.l.pass_play == 1) & self.l.comp.notna()]
        lg = self.l[(self.l.pass_play == 1) & self.l.comp.notna()]
        rows = []
        for lab, f in (("Short", lambda q: q.depth == "short"), ("Deep", lambda q: q.depth == "deep"), ("Left", lambda q: q.side == "left"), ("Middle", lambda q: q.side == "middle"), ("Right", lambda q: q.side == "right")):
            q, b = x[f(x)], lg[f(lg)]
            if len(q) >= 12:
                rows.append([lab, len(q), pc(q.comp.mean()), n(q.yds.mean(), 2), n(b.yds.mean(), 2), f"{q.yds.mean() - b.yds.mean():+.2f}"])
        return rows

    def run_dirs(self, off, de):
        lg = self.l[self.l.rside.notna()]
        rows = []
        for s_ in ("left", "middle", "right"):
            o, q = lg[(lg.off == off) & (lg.rside == s_)], lg[(lg.de == de) & (lg.rside == s_)]
            if len(o) >= 10 and len(q) >= 10:
                rows.append([s_.title(), len(o), n(o.yds.mean(), 2), len(q), n(q.yds.mean(), 2)])
        return rows

    def front(self, t):
        x = self.pl[self.pl.team == t]
        g_ = self.tab.at[t, "g"]
        return [t, n(x.sacks.sum() / g_, 1), n(x.tfl.sum() / g_, 1), n(x.hur.sum() / g_, 1), n(x.pd.sum() / g_, 1), pc(self.tab.at[t, "sack_made"], 1), f"{rk_(self.rk.at[t, 'sack_made'])}"]

    def exploits(self, off, de):
        out = []
        for lab, ko, kd in (("passing game", "ypa", "opp_ypa"), ("run game", "ypc", "opp_ypc"), ("explosive plays", "expl", "opp_expl"), ("third downs", "third", "opp_third")):
            ro, rd = self.rk.at[off, ko], self.rk.at[de, kd]
            if ro <= 45 and rd >= 85:
                out.append(f"<b>{off} {lab} edge:</b> offense #{ro:.0f} against defense #{rd:.0f}.")
            elif ro >= 85 and rd <= 45:
                out.append(f"<b>{de} defense edge on the {lab}:</b> defense #{rd:.0f} against offense #{ro:.0f}.")
        if self.rk.at[de, "sack_made"] <= 40 and self.rk.at[off, "sack_allowed"] >= 85:
            out.append(f"<b>Protection mismatch:</b> {de} sack rate {rk_(self.rk.at[de, 'sack_made'])} against {off} protection {rk_(self.rk.at[off, 'sack_allowed'])}.")
        z = {r[0]: float(r[5]) for r in self.zones(de)}
        if z:
            k = max(z, key=z.get)
            if z[k] >= .7:
                out.append(f"<b>Target zone:</b> {de} allows {z[k]:+.2f} yards per throw versus FBS average on {k.lower()} throws in charted games.")
        return out or ["No matchup clears the rank filters: neither side has a large statistical edge here."]

    def rest(self, t, kick):
        x = self.g[((self.g.home == t) | (self.g.away == t)) & (self.g.dt < pd.Timestamp(kick))]
        return (pd.Timestamp(kick) - x.dt.max()).days if len(x) else None

    def receivers(self, gg, t):
        x = self.pl[self.pl.team == t]
        wks = sorted(x.wk.unique())
        last3 = wks[-3:]
        tt, t3 = x.targets.sum(), x[x.wk.isin(last3)].targets.sum()
        rows = []
        for p in sorted([q for q in gg.pl if q["team"] == t and q["pos"] != "QB" and q["tg"] >= 1], key=lambda z: -z["rec_y"].mean())[:7]:
            y = x[x.pid == p["player_id"]]
            rows.append([e(p["name"]), p["pos"], len(y), pc(y.targets.sum() / max(tt, 1)), pc(y[y.wk.isin(last3)].targets.sum() / max(t3, 1)), pc(y.rec.sum() / max(y.targets.sum(), 1)), n(y.rec_y.sum() / max(y.rec.sum(), 1)), pc(y.deep_t.sum() / max(y.targets.sum(), 1)), int(y.rz_tgt.sum()), n(p["tg"]), n(p["rec_y"].mean(), 0)])
        return T(["Player", "Role", "G", "Tgt share", "Last 3", "Catch%", "Y/rec", "Deep tgt%", "RZ tgt", "Exp tgt", "Proj yds"], rows, f"{t} pass catchers in 2026. Role comes from usage, so tight ends show as WR. Routes, alignment and yards after catch are not available.")

    def qb(self, gg, t, opp):
        q = gg.qb(t)
        if q is None:
            return ""
        x = self.pl[(self.pl.team == t) & (self.pl.pid == q["player_id"]) & (self.pl.att > 0)]
        l5 = ", ".join(f"{r.pass_y:.0f}" for r in x.sort_values("wk").tail(5).itertuples()) or "none"
        s_ = gg.side(t)
        rows = [[e(q["name"]), len(x), n(x.att.mean(), 1), pc(x.comp.sum() / max(x.att.sum(), 1)), n(x.pass_y.sum() / max(x.att.sum(), 1), 1), int(x.pass_td.sum()), int(x.ints.sum()), pc(self.tab.at[t, "sack_allowed"], 1), n(s_["att"].mean(), 0)]]
        return T(["QB", "G", "Att/g", "Comp%", "Y/att", "TD", "INT", "Team sack rate allowed", "Sim att"], rows) + f"<p>Last 5 passing games: {l5}. Against {opp}: dropback yards allowed {rk_(self.rk.at[opp, 'opp_ypa'])}, sack rate {rk_(self.rk.at[opp, 'sack_made'])}. Starter's rushing line in the box score counts sacks as negative carries.</p>"

    def backs(self, gg, t):
        x = self.pl[self.pl.team == t]
        rows = []
        tc = x.car.sum()
        lw = sorted(x.wk.unique())[-3:]
        t3 = x[x.wk.isin(lw)].car.sum()
        lp = self.l[(self.l.off == t) & self.l.rusher.notna()].copy()
        lp["rid"] = lp.rusher.astype(float).astype(int)
        for p in sorted([q for q in gg.pl if q["team"] == t and q["car"] >= 2], key=lambda z: -z["rush_y"].mean())[:5]:
            y = x[x.pid == p["player_id"]]
            q = lp[lp.rid == p["player_id"]]
            rows.append([e(p["name"]), p["pos"], len(y), n(y.car.sum() / max(len(y), 1), 1), pc(y.car.sum() / max(tc, 1)), pc(y[y.wk.isin(lw)].car.sum() / max(t3, 1)), n(y.rush_y.sum() / max(y.car.sum(), 1), 1), pc((q.yds >= 10).mean()) if len(q) else "–", int(y.i10_car.sum()), int(y.i5_car.sum()), int(y.rush_td.sum()), n(p["car"], 1), n(p["rush_y"].mean(), 0)])
        return T(["Rusher", "Role", "G", "Car/g", "Car share", "Last 3", "YPC", "10+ rate", "In 10", "In 5", "TD", "Sim car", "Proj yds"], rows, f"{t} ground game in 2026")

    def hits(self, gg, t, opp):
        out = ""
        x = self.pl[self.pl.team == t]
        L = []
        q = gg.qb(t)
        if q is not None:
            L.append((q, "pass", "pass_y", [150, 175, 200, 225, 250, 275], "att"))
        for p in sorted([z for z in gg.pl if z["team"] == t], key=lambda z: -(z["rec_y"].mean() + z["rush_y"].mean()))[:5]:
            if q is not None and p["name"] == q["name"]:
                L.append((p, "rush", "rush_y", [20, 30, 40, 50, 60, 80], "car"))
            elif p["rec_y"].mean() >= p["rush_y"].mean():
                L.append((p, "rec", "rec_y", [20, 30, 40, 50, 60, 80], "targets"))
            else:
                L.append((p, "rush", "rush_y", [30, 40, 50, 60, 80, 100], "car"))
        for p, mk, col, ths, vol in L:
            y = x[(x.pid == p["player_id"]) & (x[vol] > 0)].sort_values("wk")
            if y.empty:
                continue
            key_ = "opp_ypa" if mk != "rush" else "opp_ypc"
            tier = self.rk[key_]
            mine = tier.get(opp, 68)
            oppo = np.where(y.team == y.home, y.away, y.home)
            sim_d = np.array([abs(tier.get(o, 999) - mine) <= 35 for o in oppo])
            homeg = (y.team == y.home).values
            f = lambda m, k_: f"{int((y[m][col] >= k_).sum())}/{int(m.sum())}"
            arr = p[KEY[mk]]
            rows = [[f"{k_}+", f(np.ones(len(y), bool), k_), f(np.arange(len(y)) >= len(y) - 5, k_), f(homeg, k_), f(~homeg, k_), f(sim_d, k_), f"<b>{pc(gg.cp(arr, k_, mk))}</b>"] for k_ in ths]
            out += T(["Threshold", "2026", "Last 5", "Home", "Road", "Vs similar D", "Model"], rows, f"<b>{e(p['name'])}</b> ({t}) {LAB[mk]}. Log: {', '.join(f'{v:.0f}' for v in y[col])}")
        return out


def milestones(gg, g_lab):
    ms = []
    for p in gg.pl:
        for mk in ("rec", "rush", "pass"):
            arr = p.get(KEY[mk])
            if arr is None or arr.mean() < 12:
                continue
            for k in TH[mk]:
                ms.append(dict(p=p, mk=mk, lab=LAB[mk], k=k, prob=gg.cp(arr, k, mk), key=KEY[mk], game=g_lab, g=gg))
    return ms


def fair_rows(gg, cal):
    rows, out = [], []
    for p in sorted(gg.pl, key=lambda z: (z["team"] != gg.a, -(z["rec_y"].mean() + z["rush_y"].mean()))):
        for mk in ("pass", "rec", "rush"):
            arr = p.get(KEY[mk])
            if arr is None or arr.mean() < 15:
                continue
            ln, ov, un = fair(arr, cal[mk], .5), fair(arr, cal[mk], .55), fair(arr, cal[mk], .45)
            lowc = p["name"] in gg.low
            out.append(dict(p=p, mk=mk, ln=ln, ov=ov, un=un, low=lowc))
            rows.append([e(p["name"]) + (" " + tag("warn", "Low confidence") if lowc or p["play_p"] < .9 else ""), p["team"], LAB[mk], n(arr.mean(), 0), f"{half(ln):g}", f"{half(ov):g} or lower", f"{half(un):g} or higher"])
    return rows, out


# ---------- tabbed Game Book (same layout as the NFL books) ----------
def _sec(title, kind, body, note=""):
    return f'<section class="mblock"><h4>{title} {kind}</h4>{body}{f"<p class=note>{note}</p>" if note else ""}</section>'


def _fold(title, body):
    return f'<details class="fold"><summary>{title}</summary>{body}</details>'


def _mrow(label, lv, rv, lr=None, rr=None, N=136):
    """Centred row like the NFL books; bars scale to the FBS rank (1 = full bar), the better rank is green."""
    better = None if lr is None or rr is None or lr == rr else ("l" if lr < rr else "r")
    def side(v, r, cls):
        if r is None:
            return f'<div class="{cls}"><span class="v">{v}</span><span></span></div>'
        return f'<div class="{cls}{" win" if better == cls else ""}"><span class="v">{v}<i>#{int(r)}</i></span><span class="bar"><span style="width:{(N + 1 - r) / N * 100:.0f}%"></span></span></div>'
    return f'<div class="mrow">{side(lv, lr, "l")}<div class="lab">{label}</div>{side(rv, rr, "r")}</div>'


def _unit_head(lt, ls, rt, rs, mid):
    w = {"off": ("offense", "gains"), "def": ("defense", "allows")}
    hh = lambda t, k, cls: f'<div class="{cls}"><b>{t} {w[k][0]}</b><small>what {t} {w[k][1]}</small></div>'
    return f'<div class="mcols">{hh(lt, ls, "l")}<div class="lab">{mid}</div>{hh(rt, rs, "r")}</div>'


def _fmt(v, d):
    return "–" if v is None or (isinstance(v, float) and not np.isfinite(v)) else (pc(v, 1) if d == -1 else n(v, d))


OFFDEF = [("Points per game", "ppg", "opp_ppg", 1), ("Yards per play", "ypp", "opp_ypp", 2), ("Success rate", "succ", "opp_succ", -1), ("Yards per dropback", "ypa", "opp_ypa", 2),
          ("Yards per carry", "ypc", "opp_ypc", 2), ("20+ yard play rate", "expl", "opp_expl", -1), ("Third-down conversion", "third", "opp_third", -1),
          ("Sack rate<small>offense allows · defense makes</small>", "sack_allowed", "sack_made", -1), ("Turnovers / game<small>offense gives · defense takes</small>", "giveaways", "takeaways", 2)]
TEAMBAR = [("Points per game", "ppg", 1), ("Points allowed", "opp_ppg", 1), ("Yards per play", "ypp", 2), ("Yards per play allowed", "opp_ypp", 2), ("Success rate", "succ", -1),
           ("Success rate allowed", "opp_succ", -1), ("Third-down rate", "third", -1), ("Takeaways / game", "takeaways", 2), ("Giveaways / game", "giveaways", 2), ("Plays / game", "plays", 1)]


def _results(c, t, k=6):
    """Last k games with score, ATS (closing spread) and O/U."""
    x = c.g[((c.g.home == t) | (c.g.away == t)) & c.g.hs.notna()].sort_values("dt").tail(k)
    rows = []
    for r in x.itertuples():
        hm = r.home == t
        us, th = (r.hs, r.aw) if hm else (r.aw, r.hs)
        opp = r.away if hm else r.home
        ats = ou = "–"
        if pd.notna(r.spread_line):
            m = (r.hs - r.aw) - r.spread_line
            m = m if hm else -m
            ats = "Push" if m == 0 else ("Cover" if m > 0 else "Miss")
        if pd.notna(r.total_line):
            tt = r.hs + r.aw
            ou = "Push" if tt == r.total_line else ("Over" if tt > r.total_line else "Under")
        rows.append([f"Wk {r.week}", ("vs " if hm else "@ ") + opp, f"{'W' if us > th else 'L' if us < th else 'T'} {int(us)}-{int(th)}", ats, ou])
    return T(["Week", "Opp", "Result", "ATS", "O/U"], rows) if rows else "<p class='note'>No games yet.</p>"


def _ats(c, t):
    x = c.g[((c.g.home == t) | (c.g.away == t)) & c.g.hs.notna() & c.g.spread_line.notna()]
    w = l = p = 0
    for r in x.itertuples():
        m = (r.hs - r.aw) - r.spread_line
        m = m if r.home == t else -m
        w, l, p = w + (m > 0), l + (m < 0), p + (m == 0)
    return f"{w}-{l}" + (f"-{p}" if p else "")


def _rz(c):
    """Red-zone units: trips per game (drives that reach the 20), TD rate on trips, red-zone pass rate and TDs, for every FBS team, with ranks."""
    l = c.l
    rz = l[l.yl <= 20]
    dr = rz.groupby(["game_id", "drive", "off", "de"]).agg(td=("dres", lambda v: (v == "TD").any())).reset_index()
    out = {}
    for side, col in (("off", "off"), ("def", "de")):
        g_ = c.tab.g
        trips = dr.groupby(col).size() / g_
        tdp = dr.groupby(col).td.mean()
        q = rz.groupby(col)
        prate = q.pass_play.mean()
        ptd = rz[rz.ty == "Passing Touchdown"].groupby(col).size()
        rtd = rz[rz.ty == "Rushing Touchdown"].groupby(col).size()
        df = pd.DataFrame(dict(trips=trips, td_pct=tdp, pass_rate=prate, ptd=ptd, rtd=rtd)).reindex(c.tab.index).fillna({"ptd": 0, "rtd": 0})
        df["ptd_g"], df["rtd_g"] = df.ptd / c.tab.g, df.rtd / c.tab.g
        asc = side == "def"
        rk = pd.DataFrame({k: df[k].rank(ascending=asc) for k in ("trips", "td_pct", "ptd_g", "rtd_g")})
        out[side] = (df, rk)
    return out


def _injury_view(c, a, h, s):
    """Key injuries ranked by how much of the offense the player carried, then the full list."""
    big, rest = [], []
    for t in (a, h):
        x = c.pl[c.pl.team == t]
        tot = max(float((x.rush_y + x.rec_y + x.pass_y * .5).sum()), 1)
        for r in s["inj"].get(t, []):
            nm = r.get("matched")
            share, line = 0.0, ""
            if nm:
                y = x[x.name == nm]
                if len(y):
                    share = float((y.rush_y + y.rec_y + y.pass_y * .5).sum()) / tot
                    gp = y.game_id.nunique()
                    parts = []
                    if y.att.sum() > 20:
                        parts.append(f"{y.pass_y.sum() / gp:.0f} pass yds/g, {int(y.pass_td.sum())} TD")
                    if y.car.sum() > 10:
                        parts.append(f"{y.rush_y.sum() / gp:.0f} rush yds/g")
                    if y.rec.sum() > 5:
                        parts.append(f"{y.rec.sum() / gp:.1f} rec, {y.rec_y.sum() / gp:.0f} yds/g")
                    tds = int(y.rush_td.sum() + y.rec_td.sum())
                    if tds:
                        parts.append(f"{tds} TD")
                    line = f"{gp} games · " + ", ".join(parts) if parts else f"{gp} games"
            st = r["status"].split(" - ")[0]
            row = dict(t=t, name=r["name"] if not nm else nm, pos=r["pos"], st=st, why=r["status"].split(" - ")[-1] if " - " in r["status"] else "", date=r["date"], note=r["note"], share=share, line=line, stale=r.get("stale"), pp=r["play_p"])
            (big if share >= .05 and not r.get("stale") else rest).append(row)
    big.sort(key=lambda z: -z["share"])
    card = lambda z: (f"<div class='injc {'out' if z['st'] in ('Out', 'Doubtful') else 'q'}'><div><b>{e(z['name'])}</b> <small>{z['t']} · {e(z['pos'])}</small></div>"
                      f"<div class='ist'>{e(z['st'])}{(' · ' + e(z['why'])) if z['why'] else ''} · plays {pc(z['pp'])}</div>"
                      f"<div class='iln'>{e(z['line'])} · {pc(z['share'])} of {z['t']}'s yards</div><p>{e(z['note'])}</p><small>Listed {e(z['date'])}</small></div>")
    css = ("<style>.injg{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,300px),1fr));gap:12px}.injc{border:1px solid var(--line);border-left:4px solid var(--warn,#d97706);border-radius:10px;padding:12px;display:grid;gap:4px}"
           ".injc.out{border-left-color:var(--bad,#e1213b)}.injc small{color:var(--mut)}.injc .ist{font-weight:600}.injc .iln{font-family:var(--m,monospace);font-size:12.5px}.injc p{margin:4px 0 0;font-size:13.5px}</style>")
    head = css + (f"<div class='injg'>{''.join(card(z) for z in big)}</div>" if big else "<p>No injured player carried 5%+ of either offense's yards this season.</p>")
    cols = "".join(f"<div class='pcol'><h5>{t}</h5>" + (T(["Player", "Pos", "Status", "Plays", "Listed"], [[e(z['name']) + (" " + tag("mute", "Stale") if z["stale"] else ""), e(z["pos"]), e(z["st"]), pc(z["pp"]), e(z["date"])] for z in big + rest if z["t"] == t]) if any(z["t"] == t for z in big + rest) else "<p class='note'>None listed.</p>") + "</div>" for t in (a, h))
    return head, f"<div class='pcols'>{cols}</div>", big


def _td_tab(c, gg, a, h, s, td):
    """Touchdowns: each red-zone offense vs the other red-zone defense, player red-zone usage, and the TD model's top candidates per team."""
    RZ = _rz(c)
    N = len(c.tab)
    html = ""
    for o, d in ((a, h), (h, a)):
        O, Ro = RZ["off"][0].loc[o], RZ["off"][1].loc[o]
        D, Rd = RZ["def"][0].loc[d], RZ["def"][1].loc[d]
        rk = lambda v: None if pd.isna(v) else int(v)
        rows = [_unit_head(o, "off", d, "def", f"2026 red zone (inside the 20)<br># = FBS rank, 1 is best"),
                _mrow("Red-zone trips / game", n(O.trips, 1), n(D.trips, 1), rk(Ro.trips), rk(Rd.trips), N),
                _mrow("Trips ending in a TD", pc(O.td_pct), pc(D.td_pct), rk(Ro.td_pct), rk(Rd.td_pct), N),
                _mrow("Passing TDs / game inside the 20", n(O.ptd_g, 2), n(D.ptd_g, 2), rk(Ro.ptd_g), rk(Rd.ptd_g), N),
                _mrow("Rushing TDs / game inside the 20", n(O.rtd_g, 2), n(D.rtd_g, 2), rk(Ro.rtd_g), rk(Rd.rtd_g), N),
                _mrow("Pass rate inside the 20", pc(O.pass_rate), pc(D.pass_rate))]
        html += _sec(f"{o} red-zone offense vs {d} red-zone defense", CTX, "".join(rows))
    # player usage near the goal line
    cols = ""
    for t in (a, h):
        x = c.pl[c.pl.team == t].groupby("name").agg(g=("game_id", "nunique"), rz=("rz_tgt", "sum"), i10=("i10_car", "sum"), i5=("i5_car", "sum"), rtd=("rush_td", "sum"), ctd=("rec_td", "sum")).reset_index()
        x["tds"] = x.rtd + x.ctd
        x = x[(x.rz + x.i10 + x.tds) > 0].sort_values(["tds", "rz", "i10"], ascending=False).head(8)
        cols += f"<div class='pcol'><h5>{t}</h5>" + T(["Player", "TDs", "Rush / rec", "Inside-10 carries", "RZ targets"], [[e(r.name), int(r.tds), f"{int(r.rtd)} / {int(r.ctd)}", int(r.i10), int(r.rz)] for r in x.itertuples()]) + "</div>"
    html += _sec("Who gets the ball near the goal line", CTX, f"<div class='pcols'>{cols}</div>", "2026 season totals. Red-zone targets and inside-10 carries come from charted play text, so a few games may be missing.")
    # TD model per team
    cols = ""
    for t in (a, h):
        rows = [r for r in td if r[1] == t][:6]
        cols += f"<div class='pcol'><h5>{t}</h5>" + (T(["Player", "Pos", "Anytime TD", "Fair odds", "Main route"], [[r[0], r[2], r[3], r[4], r[8]] for r in rows]) if rows else "<p class='note'>None.</p>") + "</div>"
    html += _sec("Anytime touchdown chances", PRED, f"<div class='pcols'>{cols}</div>", "From 20,000 simulated games: team TDs from simulated points, split by each player's red-zone role. Holdout calibration: Brier .1178 against .1352 base rate. No TD prices were available, so fair odds are what our chance implies.")
    return html


def render(gm, c, cal, wx, rt):
    gg = G(gm, c, cal)
    g, a, h, s = gg.g, gg.a, gg.h, gm
    kick = pd.Timestamp(g["kick"]).tz_convert(ET)
    gg.low = {r[1] for r in s["roles"]} | {gg.qb(t)["name"] for t in (a, h) if s["qb_flag"].get(t) is not None and gg.qb(t)} | {gg.qb(t)["name"] for t in (a, h) if t in NOTE and gg.qb(t)}
    SHIFT["pl"] = gg.low
    SHIFT["rb"] = set()
    t1, t2, t3, bets = game_lines(gg)
    t1 = t1.replace("The sim is 90% market and 10% our line, so small gaps here are expected and are not edges.", "The sim's margin and total are centered on the market, so model % sits near break-even by design and is not an edge.")
    t2 = t2.replace("Alt prices are unavailable, so bet only at better than the fair price.", "Alt prices are unavailable, so only take a number better than the fair price.")
    probs, sc_t, sc_v = scenarios(gg)
    ph = float((s["margin"] > 0).mean() + .5 * (s["margin"] == 0).mean())
    win, pw = (h, ph) if ph >= .5 else (a, 1 - ph)
    pa_, ph_ = float(np.mean(s["a_pts"])), float(np.mean(s["h_pts"]))
    ra = f"#{g['rank_a']} " if g["rank_a"] else ""
    rh = f"#{g['rank_h']} " if g["rank_h"] else ""
    gid = f"{a}-{h}".lower()
    lab = f"{a} at {h}" if not g["neutral"] else f"{a} vs {h}"
    fr, fairs = fair_rows(gg, cal)
    ms = milestones(gg, lab)
    good_a, bad_a = c.strengths(a)
    good_h, bad_h = c.strengths(h)
    inj = [[t, e(x["name"]), x["pos"], e(x["status"]), e(x["date"]), "Yes" if x.get("matched") else "No", pc(x["play_p"]) + (" " + tag("mute", "Stale note") if x.get("stale") else ""), e(x["note"][:70])] for t in (a, h) for x in s["inj"].get(t, [])]
    w = wx.get(h, {})
    if w.get("indoor"):
        wt, wadj = "Indoors, so weather does not apply.", ""
    elif "temp" in w:
        wt = f"{w['temp']}°F, {e(w['wind'])} {e(w['dir'])}, {w['rain'] or 0}% rain chance, {e(w['short'])}. Three hours in: {w['temp_end']}°F, {e(w['wind_end'])}, {w['rain_end'] or 0}% rain."
        wadj = "Rain risk can trim passing efficiency, and the market total should already carry it, so no separate adjustment is made." if max(w["rain"] or 0, w["rain_end"] or 0) >= 50 else "No weather adjustment."
    else:
        wt, wadj = "Forecast unavailable.", ""
    # A
    A = f"""<div class="exec"><div class="score"><div><span>{a}</span><b>{pa_:.1f}</b></div><div><span>{h}</span><b>{ph_:.1f}</b></div></div>
    <div class="exgrid"><div><small>Predicted winner</small><b>{win} {pc(pw)}</b></div><div><small>Market</small><b>{e(g['details'])}, O/U {g['total']}</b></div><div><small>Our independent line</small><b>{a} {s['model_a']:.1f}, {h} {s['model_h']:.1f}</b></div><div><small>Most likely script</small><b>{['A: ' + h + ' controls', 'B: ' + a + ' controls', 'C: One-score game'][int(np.argmax(probs))]} ({pc(max(probs))})</b></div></div></div>
    <p><b>{a} strengths:</b> {e('; '.join(good_a) or 'none in the top 30')}. <b>Weaknesses:</b> {e('; '.join(bad_a) or 'none in the bottom third')}.</p><p><b>{h} strengths:</b> {e('; '.join(good_h) or 'none in the top 30')}. <b>Weaknesses:</b> {e('; '.join(bad_h) or 'none in the bottom third')}.</p>"""
    for t in (a, h):
        if t in NOTE:
            A += f"<p class='warn'>{e(NOTE[t])}</p>"
    A += "<h4>Win margin</h4>" + P.margin_block(gg)
    outs = [x for t in (a, h) for x in s["inj"].get(t, []) if x.get("matched") and not x.get("stale")]
    A += "<p><b>Key injuries among players with 2026 stats:</b> " + e("; ".join(f"{x['name']} ({x['team']}, {x['status'].split(' - ')[0]})" for x in outs[:12]) or "none listed") + ".</p>"
    P1 = f"""<ul><li><b>Kickoff:</b> {kick.strftime('%a %b %-d, %-I:%M %p')} ET. <b>Status:</b> {e(g['status'])}.</li><li><b>Site:</b> {e(g['venue'])}, {e(g['city'])}{', neutral site' if g['neutral'] else ''}. {'Conference game.' if g['conf'] else 'Non-conference game.'}</li>
    <li><b>Records:</b> {a} {e(g['rec_a'])}, {h} {e(g['rec_h'])} (ESPN). <b>AP rank:</b> {a} {g['rank_a'] or 'unranked'}, {h} {g['rank_h'] or 'unranked'}.</li><li><b>Starting QBs modeled:</b> {e((gg.qb(a) or {}).get('name'))} and {e((gg.qb(h) or {}).get('name'))}. College teams do not publish a required starter list.</li>
    <li><b>Data through:</b> Week {WK - 1} of 2026. Nothing from this game is used.</li></ul>""" + (T(["Tm", "Player", "Pos", "Status", "Listed", "Has 2026 stats", "P(plays)", "Note"], inj, "Covers injury list, which is unofficial. P(plays) is an assumption from status. A player listed earlier who has played since is marked stale and treated as available.") if inj else "<p>No injuries listed by Covers for either team.</p>")
    P2 = c.compare(a, h) + T(["Team", "Split", "Yds/play", "Success", "Allowed yds/play", "Allowed success"], c.recent(a) + c.recent(h), "Season vs last three games. Three games is a small sample.") + \
        f"<p><b>Adjusted team strength in points vs average (ridge fit on 2026):</b> {a} offense {rt['off'].get(a, 0):+.1f}, defense {rt['def'].get(a, 0):+.1f}. {h} offense {rt['off'].get(h, 0):+.1f}, defense {rt['def'].get(h, 0):+.1f}. Positive is better on both sides of the ball.</p>"
    P3 = T(["Team", "Plays/g", "Pass rate", "Early down, close", "Within 7", "Leading 8+", "Trailing 8+"], [c.pace(a), c.pace(h)], "Offensive tendencies. Sacks count as pass plays. Clock data per snap is not available.") + \
        f"<p><b>Scenario A, {h} wins by 8+:</b> {pc(probs[0])}. <b>Scenario B, {a} wins by 8+:</b> {pc(probs[1])}. <b>Scenario C, one-score game:</b> {pc(probs[2])}. These sum to 100% and come from simulated final margins.</p>" + sc_v + sc_t
    P4 = "<p class='note'>Man/zone rates, coverage shells, blitz rate and pressure rate are not in any free college feed. They are not estimated here. What follows is what each defense has allowed by throw location, plus box-score pass-rush production.</p>"
    for de, off in ((h, a), (a, h)):
        z = c.zones(de)
        P4 += f"<h4>{de} defense vs {off} offense</h4>" + (T(["Throw zone", "Throws", "Comp%", "Yds/throw", "FBS avg", "Vs avg"], z, f"Charted throws {de} has faced") if z else "<p>Too few charted throws to split by zone.</p>")
        P4 += "<ul>" + "".join(f"<li>{x}</li>" for x in c.exploits(off, de)) + "</ul>"
    P4 += T(["Team", "Sacks/g", "TFL/g", "Hurries/g", "PD/g", "Sack rate", "FBS rank"], [c.front(a), c.front(h)], "Box-score production. Hurries are recorded unevenly from school to school.")
    P5 = "".join(f"<h4>{t}</h4>" + c.receivers(gg, t) for t in (a, h))
    P6 = "".join(f"<h4>{t} QB vs {o}</h4>" + c.qb(gg, t, o) for t, o in ((a, h), (h, a)))
    P7 = ""
    for t, o in ((a, h), (h, a)):
        rd = c.run_dirs(t, o)
        P7 += f"<h4>{t} run game vs {o}</h4>" + c.backs(gg, t) + (T(["Direction", f"{t} carries", f"{t} YPC", f"{o} faced", f"{o} YPC allowed"], rd, "Run direction, from games with charted text") if rd else "")
    roles = [f"{e(nm)} ({tm}) went from {pc(b_)} of touches to {pc(a_)} in the latest game" for tm, nm, b_, a_ in s["roles"]]
    if roles:
        P7 += "<p class='warn'>Role changes: " + "; ".join(roles) + ". Season shares may lag these moves. Their numbers are marked low confidence.</p>"
    td = td_block(gg)
    P8 = T(["Player", "Tm", "Pos", "Anytime TD", "Fair odds", "Range", "Exp rush TD", "Exp rec TD", "Main route"], td, "TD model: simulated points give team TDs, then TDs split by red-zone role. Holdout calibration: Brier .1178 against .1352 base rate. No TD prices were available.")
    rs = c.rest(a, g["kick"]), c.rest(h, g["kick"])
    P9 = f"<ul><li><b>Forecast at kickoff:</b> {wt} {wadj}</li><li><b>Rest:</b> {a} {rs[0]} days, {h} {rs[1]} days.</li><li><b>Home field:</b> {'none, neutral site' if g['neutral'] else 'about ' + n(rt['hfa'], 1) + ' points in the 2026 fit'}.</li><li><b>Not tracked:</b> travel distance, altitude, crowd, and coach or coordinator changes.</li></ul>"
    P10 = "".join(f"<h4>{t}</h4>" + c.hits(gg, t, o) for t, o in ((a, h), (h, a))) + "<p class='note'>Counts are 2026 only, in games with usage. Vs similar D uses opponents within 35 FBS ranks of this defense on yards allowed. Model is the calibrated forecast, kept separate from history.</p>"
    P11 = "<h4>B. Complete player projections</h4>" + P.player_proj(gg).replace("Means from 20,000 simulations.", "Raw means from 20,000 simulations. QB passing means run high: use the calibrated fair lines in Phase 12.").replace("if he sits his share goes to teammates inside the same sim", "absences are handled inside the same sim") + "<h4>C. Alternate-yardage probabilities</h4>" + P.prob_tables(gg)
    P12 = "<h4>Game markets</h4>" + t1 + t2 + t3 + "<h4>Fair player lines</h4>" + T(["Player", "Tm", "Market", "Raw mean", "Fair line", "Over only at", "Under only at"], fr, "<b>No college prop lines or prices were available.</b> Fair line is the calibrated 50/50 number. Over and under columns are the numbers that clear 55%, which leaves a small cushion over the 52.4% needed at -110. Treat these as limits, and shop your book against them.")
    P13 = correlation(gg)
    best = {}
    for r in ms:
        if .70 <= r["prob"] <= .9 and r["p"]["play_p"] >= .9 and r["p"]["name"] not in gg.low:
            k = (r["p"]["name"], r["mk"])
            if k not in best or r["k"] > best[k]["k"]:
                best[k] = r
    top = sorted(best.values(), key=lambda r: -r["prob"])[:10]
    D = T(["#", "Player", "Outcome", "Calibrated P", "Fair odds", "Min odds"], [[i + 1, e(r["p"]["name"]), f"{r['k']:g}+ {r['lab']}", pc(r["prob"]), am(r["prob"]), am(max(r["prob"] - .03, .02))] for i, r in enumerate(top)], "Highest threshold on our ladder that clears 70% for each player. Likely does not mean good value. Low-confidence players are excluded.") if top else "<p>No outcome clears 70% among players with a stable role.</p>"
    Eh = "<p><b>Pass: EV cannot be measured without a posted line and price.</b> The numbers to check are below: a value bet exists only if your book posts a line past these limits.</p>" + T(["#", "Player", "Market", "Fair line", "Over only at", "Under only at"], [[i + 1, e(r["p"]["name"]), LAB[r["mk"]], f"{half(r['ln']):g}", f"{half(r['ov']):g} or lower", f"{half(r['un']):g} or higher"] for i, r in enumerate(sorted([x for x in fairs if not x["low"] and x["p"]["play_p"] >= .9 and x["mk"] != "pass"], key=lambda z: -z["ln"])[:5])])
    F = T(["#", "Player", "Anytime TD", "Fair odds", "Main route"], [[i + 1, r[0], r[3], r[4], r[8]] for i, r in enumerate(td[:5])])
    gb = sorted(bets, key=lambda b: -(b[1] * P.payout(b[2]) - (1 - b[1])))
    bev = gb[0][1] * P.payout(gb[0][2]) - (1 - gb[0][1])
    Gp = f"<p><b>Call: pass on game lines.</b> The sim sits on the market, and our own line lost to it in the backtest (margin MAE 13.06 vs 11.84), so no side is recommended. Best listed EV is {e(gb[0][0])} at {gb[0][2]:+.0f} ({bev * 100:+.1f}%), which is noise.</p><p>Winner lean: {win} {pc(pw)}. Our independent line: {a} {s['model_a']:.1f}, {h} {s['model_h']:.1f}.</p>"
    H = combos(gg, ms).replace("with a posted milestone", "on our ladder")
    fin = [["Probability", e(r["p"]["name"]), f"{r['k']:g}+ {r['lab']}", pc(r["prob"]), am(max(r["prob"] - .03, .02)), "Early blowout, injury, or a smaller role"] for r in top[:5]]
    I = T(["Type", "Player", "Selection", "Est. hit", "Min odds", "Biggest risk"], fin, "Ranked by calibrated probability. These are not value picks, since no price is known.") if fin else "<p>No selection clears the bar.</p>"
    secs = [("A", "Executive summary", A), ("1", "Game and data verification", P1), ("2", "Team performance", P2), ("3", "Pace and game script", P3), ("4", "Defense and scheme evidence", P4), ("5", "Receiver matchups", P5), ("6", "Quarterback vs pass rush", P6), ("7", "Running game", P7), ("8", "Touchdown model", P8), ("9", "Weather and environment", P9),
            ("10", "Historical hit rates", P10), ("11", "Statistical model output", P11), ("12", "Market and fair lines", P12), ("13", "Parlay correlation", P13), ("D", "Top 10 highest-probability outcomes", D), ("E", "Top 5 value opportunities", Eh), ("F", "Top 5 touchdown candidates", F), ("G", "Game picks", Gp), ("H", "Conservative, balanced, aggressive", H), ("I", "Final ranked recommendations", I)]
    summ = dict(id=gid, espn=g["id"], kick_utc=g["kick"], conf_a=g.get("conf_a"), conf_h=g.get("conf_h"), spread_home=g["spread_home"], ml_a=g.get("ml_a"), ml_h=g.get("ml_h"), name_a=g["away_name"], name_h=g["home_name"], rec_a=g["rec_a"], rec_h=g["rec_h"], venue=g["venue"], neutral=g["neutral"], a=a, h=h, ra=g["rank_a"], rh=g["rank_h"], kick=kick.strftime("%-I:%M %p"), line=g["details"], total=g["total"], ma=s["model_a"], mh=s["model_h"], pa=pa_, ph=ph_, win=win, pw=pw, script=probs, top=[dict(r, game=lab) for r in top], td=[(p["name"], p["team"], p["atd"]) for p in sorted(gg.pl, key=lambda z: -z["atd"])[:3]], w=(w.get("rain") or 0, w.get("rain_end") or 0), notes=[t for t in (a, h) if t in NOTE])
    N = len(c.tab)
    rk = lambda t, k: None if t not in c.rk.index or pd.isna(c.rk.at[t, k]) else int(c.rk.at[t, k])
    tv = lambda t, k: c.tab.at[t, k] if t in c.tab.index else None
    # Overview
    nv_a, nv_h = (P.GB.novig(g["ml_a"], g["ml_h"]) if g.get("ml_a") and g.get("ml_h") else (float("nan"), float("nan"))) if hasattr(P, "GB") else (float("nan"), float("nan"))
    cell = lambda big, small: f"<div class='oc'><b>{big}</b><small>{small}</small></div>"
    sp = lambda v: "–" if v is None else (f"{v:+g}" if v else "PK")
    ml = lambda o: "–" if o is None or (isinstance(o, float) and not np.isfinite(o)) else f"{o:+.0f}"
    orow = lambda t, nm, rk_, rec, spv, spo, mlv, tl, to: (f"<div class='otm'><span class='mono'>{t}</span><div><b><span class='fn'>{rk_}{e(nm)}</span><span class='ab'>{t}</span></b><small>{e(rec)}</small></div></div>"
                                                          f"{cell(sp(spv), ml(spo))}{cell(ml(mlv), 'to win')}{cell(tl, ml(to))}")
    board = (f"<div class='board'><div class='bh'><span></span><span>Spread</span><span>Moneyline</span><span>Total</span></div>"
             f"<div class='br'>{orow(a, g['away_name'], ra, (g['rec_a'] or '') + ' · ' + ('neutral' if g['neutral'] else 'away') + ' · ATS ' + _ats(c, a), g['spread_home'], g.get('sp_a'), g.get('ml_a'), 'O ' + format(g['total'], 'g') if g.get('total') else '–', g.get('over_odds'))}</div>"
             f"<div class='br'>{orow(h, g['home_name'], rh, (g['rec_h'] or '') + ' · ' + ('neutral' if g['neutral'] else 'home') + ' · ATS ' + _ats(c, h), -g['spread_home'] if g['spread_home'] is not None else None, g.get('sp_h'), g.get('ml_h'), 'U ' + format(g['total'], 'g') if g.get('total') else '–', g.get('under_odds'))}</div>"
             f"<p class='note'>DraftKings via ESPN, pulled {BUILT}. {kick.strftime('%a %b %-d, %-I:%M %p')} ET · {e(g['venue'])}. ATS is this season against closing spreads.</p></div>")
    split = lambda lab, pa2, ph2: (f"<div class='wpb'><div class='wl'><span>{a} <b>{pc(pa2)}</b></span><small>{lab}</small><span><b>{pc(ph2)}</b> {h}</span></div>"
                                   f"<div class='wbar2'><span class='{'f' if pa2 >= ph2 else 'u'}' style='width:{pa2 * 100:.1f}%'></span><span class='{'f' if ph2 > pa2 else 'u'}' style='width:{ph2 * 100:.1f}%'></span></div></div>")
    mk_split = split("Market, vig removed", nv_a, nv_h) if np.isfinite(nv_a) else ""
    winp = _sec("Win probability", PRED, mk_split + split("Our 20,000-game simulation", 1 - ph, ph) +
                f"<div class='proj'><div><small>Projected score</small><b>{a} {pa_:.1f} – {ph_:.1f} {h}</b></div><div><small>Projected total</small><b>{pa_ + ph_:.1f}</b> <small>vs {g['total']:g}</small></div><div><small>Most likely script</small><b>{['A: ' + h + ' by 8+', 'B: ' + a + ' by 8+', 'C: one-score game'][int(np.argmax(probs))]} · {pc(max(probs))}</b></div></div>"
                if g.get("total") else "", "The simulation is centered on the market, because our own line did not beat closing numbers in the backtest.")
    winp += _sec("Win margin", PRED, P.margin_block(gg))
    warn = "".join(f"<p class='warn'>{e(NOTE[t])}</p>" for t in (a, h) if t in NOTE)
    bars = [side_head_simple(a, h, f"2026 regular season<br># = FBS rank of {N}, 1 is best")] + [_mrow(lab, _fmt(tv(a, k), d), _fmt(tv(h, k), d), rk(a, k), rk(h, k), N) for lab, k, d in TEAMBAR]
    sw = lambda t: (lambda gb: f"<p><b>{t} strengths:</b> {e('; '.join(gb[0]) or 'none in the top 30')}. <b>Weaknesses:</b> {e('; '.join(gb[1]) or 'none in the bottom third')}.</p>")(c.strengths(t))
    teamb = _sec("2026 team comparison", CTX, "".join(bars) + sw(a) + sw(h), "Green is the better-ranked team in that row. Raw 2026 results, not adjusted for opponents.") +         f"<p class='note'><b>Adjusted strength (points vs an average FBS team):</b> {a} offense {rt['off'].get(a, 0):+.1f}, defense {rt['def'].get(a, 0):+.1f} · {h} offense {rt['off'].get(h, 0):+.1f}, defense {rt['def'].get(h, 0):+.1f}. Positive is better on both sides.</p>"
    last = _sec("Last 6 games", CTX, f"<div class='pcols'><div class='pcol'><h5>{a}</h5>{_results(c, a)}</div><div class='pcol'><h5>{h}</h5>{_results(c, h)}</div></div>", "This season. ATS and O/U use closing lines.")
    inj_cards, inj_full, inj_big = _injury_view(c, a, h, s)
    # Best plays: the picks first, then the detail
    watch = [f"<b>{e(z['name'])}</b> ({z['t']} {e(z['pos'])}) is listed {e(z['st'].lower())} (as of {e(z['date'])}). He has produced {pc(z['share'])} of {z['t']}'s yards." for z in inj_big[:3]]
    watch += [f"<b>{t} quarterback:</b> checked by hand this week (see the note below)." for t in (a, h) if t in NOTE]
    best_strip = best_plays(gid, [(r["p"]["name"], r["p"]["team"], f"{r['k']:g}+ {r['lab']}", r["prob"]) for r in top[:3]],
                            [(p["name"], p["team"], "anytime touchdown", p["atd"]) for p in sorted(gg.pl, key=lambda z: -z["atd"])[:3]], watch,
                            [("All picks", "p", 0), ("Touchdown detail", "p", 1), ("Injuries", "pl", 0)],
                            "Most likely means likely to hit, not good value. No sportsbook posts college prop lines in the free feeds, so each line is our own: the highest mark the player clears in 70%+ of 20,000 simulated games.")
    overview = best_strip + board + warn + winp + _sec("Key injuries", CTX, inj_cards) + teamb + _sec("Game conditions", CTX, P9)
    # Offense & Defense
    od = ""
    for o, d in ((a, h), (h, a)):
        rows = [_unit_head(o, "off", d, "def", "2026 season<br># = FBS rank, 1 is best")]
        for lab, ko, kd, dd in OFFDEF:
            rows.append(_mrow(lab, _fmt(tv(o, ko), dd), _fmt(tv(d, kd), dd), rk(o, ko), rk(d, kd), N))
        od += _sec(f"{o} offense vs {d} defense", CTX, "".join(rows) + "<ul>" + "".join(f"<li>{x}</li>" for x in c.exploits(o, d)) + "</ul>", "Green is the better-ranked side of each matchup.")
    od += _sec("Recent form", CTX, T(["Team", "Split", "Yds/play", "Success", "Allowed yds/play", "Allowed success"], c.recent(a) + c.recent(h)), "Season vs last three games. Three games is a small sample.")
    # Injuries
    injuries = _sec("Key injuries", CTX, inj_cards, "Ranked by the share of the team's yards the player produced this season (passing yards count half). Only players listed by Covers appear; college teams are not required to report.") + \
        _sec("Full injury report", CTX, inj_full, "Covers injury list, which is unofficial. Plays = our assumed chance he suits up, from his status. A listing older than the team's last game is marked stale and he is treated as available.")
    # Touchdowns
    tds = _td_tab(c, gg, a, h, s, td) + _sec("Simulation detail", PRED, _fold("Full touchdown model", P8))
    # Picks and model
    picks = _sec("Most likely", PRED, D) + _sec("Final ranked recommendations", PRED, I) + _sec("Building a parlay", PRED, H)
    allprops = _sec("All prop tables", PRED, _fold("Historical hit rates", P10) + _fold("Player projections and alternate lines", P11) + _fold("Market and fair lines", P12) + _fold("Parlay correlation", P13) + _fold("Value limits", Eh))
    model = _sec("Executive summary", PRED, A) + _sec("Game picks", PRED, Gp)
    tabs = [("o", "Overview", overview),
            ("m", "Matchup", subtabs(gid, "m", [("Offense & Defense", od), ("Pace & pass defense", _sec("Pace and game script", CTX, P3) + _sec("Pass defense by throw zone and pass rush", CTX, P4)), ("Last 6 games", last)])),
            ("pl", "Players", subtabs(gid, "pl", [("Injuries", injuries), ("Receivers", _sec("Receivers", CTX, P5)), ("Quarterbacks", _sec("Quarterbacks", CTX, P6)), ("Running game", _sec("Running game", CTX, P7))])),
            ("p", "Picks", subtabs(gid, "p", [("Picks", picks), ("Touchdowns", tds), ("All props", allprops)])),
            ("md", "Model", subtabs(gid, "md", [("Summary", model), ("Data checks", _sec("Data checks", CTX, P1))]))]
    bar = "".join(f'<button class="gt" data-p="{gid}-{k}">{lab}</button>' for k, lab, _ in tabs)
    panes = "".join(f'<div id="{gid}-{k}" class="pane"{"" if k == "o" else " hidden"}>{body}</div>' for k, _, body in tabs)
    return f'<article id="{gid}" class="game" hidden><header class="gh"><h2>{ra}{e(g["away_name"])} {"vs" if g["neutral"] else "at"} {rh}{e(g["home_name"])}</h2><p>{kick.strftime("%A, %B %-d, %-I:%M %p")} ET · {e(g["venue"])}</p></header><div class="gtabs" role="tablist">{bar}</div>{panes}</article>', summ


def slate_page(sums, bt):
    def nm(s, side):
        r = s["r" + side]
        return (f"#{r} " if r else "") + s[side]
    r1 = [[f'<a href="#" data-go="{s["id"]}">{nm(s, "a")} {"vs" if s["neutral"] else "at"} {nm(s, "h")}</a>', s["kick"], e(s["line"]), s["total"], f'{s["a"]} {s["ma"]:.1f}, {s["h"]} {s["mh"]:.1f}', f'{s["pa"]:.1f}-{s["ph"]:.1f}', f'{s["win"]} {pc(s["pw"])}', pc(max(s["script"]))] for s in sums]
    t1 = T(["Game", "Kick ET", "DK spread", "DK total", "Our line", "Sim score", "Winner", "Top script"], r1, "Sim score is centered on the market. Our line is the independent rating, which lost to the close in the backtest. Click a game for its full report.")
    tops = sorted([r for s in sums for r in s["top"]], key=lambda r: -r["prob"])[:15]
    t2 = T(["#", "Player", "Game", "Outcome", "Calibrated P", "Fair odds", "Min odds"], [[i + 1, e(r["p"]["name"]), e(r["game"]), f"{r['k']:g}+ {r['lab']}", pc(r["prob"]), am(r["prob"]), am(max(r["prob"] - .03, .02))] for i, r in enumerate(tops)], "Stable-role players only. Prices are unknown, and short odds are likely.")
    tds = sorted([x for s in sums for x in s["td"]], key=lambda x: -x[2])[:10]
    t3 = T(["#", "Player", "Tm", "Anytime TD", "Fair odds"], [[i + 1, e(p), t, pc(q), am(q)] for i, (p, t, q) in enumerate(tds)])
    q = T(["Team", "Situation"], [[t, e(NOTE[t])] for t in NOTE])
    m = bt["markets"]
    names = dict(rec="Receiving yards", rush="Rushing yards", atd="Anytime TD")
    names["pass"] = "Passing yards"
    t4 = T(["Market", "Holdout forecasts", "Model Brier", "Base-rate Brier", "Improvement", "MAE (yds)"], [[names[k], m[k]["n"], n(m[k]["brier"], 4), n(m[k]["base"], 4), pc(1 - m[k]["brier"] / m[k]["base"], 1), n(bt["mae"].get(k), 1)] for k in ("rec", "rush", "pass", "atd")], "2026 Weeks 7-9, with calibration fit on Weeks 4-6. Lower Brier is better.")
    mm, mt = bt["mae_margin"], bt["mae_total"]
    src = [("ESPN", "https://www.espn.com/college-football/"), ("NWS", "https://www.weather.gov"), ("Covers injuries", "https://www.covers.com/sport/football/ncaaf/injuries"), ] + [(x[0], x[1]) for x in R.hand(WK).get("sources", [])]
    links = ", ".join(f'<a href="{u}">{x}</a>' for x, u in src)
    return f"""<div id="slate" hidden><h2>Week {WK} slate</h2><p class="note">{len(sums)} games with an AP Top 25 team or an SEC, ACC, Big Ten or Big 12 team. AP ranks are from the latest poll. Lines and injuries were pulled {BUILT}. Confirm starters before you act.</p>
    <section class="ph"><h3>All games</h3>{t1}</section>
    <section class="ph"><h3>Quarterback situations</h3>{q}<p class="note">Starters on other teams come from the latest game played, skipping anyone listed out, not a confirmed announcement.</p></section>
    <section class="ph"><h3>What the model can and cannot do</h3><div style="display:flex;flex-wrap:wrap;gap:20px;align-items:flex-start"><div style="flex:1 1 420px;min-width:0">{t4}
    <p><b>Player probabilities are close to honest.</b> Holdout buckets landed within about 3 points of actual hit rates, except the top two, which ran 4 points high. Those high numbers now take a haircut, fit on the same holdout, so that correction has not been tested on new games.</p>
    <p><b>Game lines show no edge.</b> Spread MAE was {mm['market']:.2f} points for the close vs {mm['model']:.2f} for our line. Total MAE was {mt['market']:.2f} vs {mt['model']:.2f}, across {bt['n_games']} games. The best-fit weight on our line was negative, so every game side here is a pass.</p>
    <p><b>Prop value is unknown.</b> No college prop lines were available, so no prop EV is claimed. The small sample covers six weeks of one season.</p></div><div style="flex:0 1 340px;min-width:0">{reliab_svg(bt['reliability'])}<p class="note">2026 holdout, all markets. Dots near the dashed line mean a forecast of X% hit about X% of the time.</p></div></div></section>
    <section class="ph"><h3>Highest-probability outcomes</h3>{t2}</section><section class="ph"><h3>Top touchdown candidates</h3>{t3}</section>
    <section class="ph"><h3>Data gaps</h3><ul><li><b>Prop lines and prices:</b> none found. Each game gives fair lines and limits instead. Moneyline, spread and total prices are real DraftKings numbers via ESPN.</li><li><b>Scheme:</b> no man/zone, coverage shell, blitz or pressure rate, routes, alignment or snap counts exist in free college data. Throw location appears only in games with charted text. Nothing here is inferred from a coach's reputation.</li>
    <li><b>Positions:</b> roles come from usage, so tight ends show as WR.</li><li><b>Injuries:</b> college has no required injury report. Covers is unofficial, and P(plays) values are assumptions. In the backtest 35% of eligible players did not log a stat, so absences are a large risk.</li><li><b>Sample:</b> 2026 only for market-tied testing, because ESPN holds few 2025 lines.</li></ul>
    <p>Sources: {links}.</p><p class="note">These are estimates, and no selection is guaranteed. After the games, run review_cfb.py to score every forecast (Phase 15).</p></section></div>"""


def main():
    global BUILT
    data = pickle.load(open(R.cache(WK), "rb"))
    BUILT = pd.Timestamp(data["built"]).strftime("%a %b %-d at %-I:%M %p")
    d = C.load()
    c = Ctx(d)
    wx = json.load(open(f"output/weather_w{WK}.json"))
    arts, sums = [], []
    for s in data["res"]:
        a, b = render(s, c, data["cal"], wx, data["ratings"])
        arts.append(a)
        sums.append(b)
        print("rendered", b["id"], flush=True)
    btn = '<button data-t="slate">Slate</button>' + "".join(f'<button data-t="{x["id"]}">{x["a"]} {"vs" if x["neutral"] else "at"} {x["h"]}</button>' for x in sums)
    js = P.JS.replace("wk8", f"cfb{WK}")
    page = f"""<meta charset="utf-8"><title>College Game Book</title><meta name="claude:viewport-fit-sides" content="cover"><link rel="preconnect" href="https://fonts.googleapis.com"><link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700&family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;600&display=swap"><style>{CSS}</style>
    <div class="top"><h1>College Game Book</h1><p>Full matchup reports, 20,000-game simulations, and calibrated player forecasts for every Week {WK} game with an AP Top 25 team or an SEC, ACC, Big Ten or Big 12 team.</p></div><nav>{btn}</nav><main>{slate_page(sums, data['bt'])}{''.join(arts)}</main><script>{js}</script>"""
    open(OUT, "w").write(page)
    keep = ("id", "espn", "kick_utc", "conf_a", "conf_h", "spread_home", "ml_a", "ml_h", "name_a", "name_h", "rec_a", "rec_h", "venue", "neutral", "a", "h", "ra", "rh", "line", "total", "ma", "mh", "pa", "ph", "win", "pw", "script", "td", "notes")
    games = [{k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in x.items() if k in keep} for x in sums]
    for x in games:
        x["script"] = [float(v) for v in x["script"]]
        x["td"] = [[p, t, float(q)] for p, t, q in x["td"]]
    json.dump(dict(week=WK, built=data["built"], games=games), open(f"output/summary_w{WK}.json", "w"), indent=0, default=str)
    print(OUT, round(len(page) / 1e6, 2), "MB")


if __name__ == "__main__":
    main()
