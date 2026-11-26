"""NFL Standings tab: python hub_standings.py -> output/hub/nflst.json (season.py runs it on every rebuild).

Three views, all computed from data already on this computer:
  Divisions     W-L-T, win %, division and conference records, points, differential and streak (nflverse games.csv finals)
  Playoff race  seeds 1-7 per conference, then the teams in the hunt, ranked with the NFL's tiebreakers through
                strength of schedule (head-to-head, division, common games, conference, strength of victory, strength of
                schedule). Ties still level after those steps fall back to point differential, and each row says which step
                decided it. Strength of schedule uses games played so far (the official step uses the full schedule).
  Leaders       season player leaders (nflverse weekly player stats) and team leaders (nflverse play-by-play)

ESPN's standings feed is read only as a cross-check of the seeds and for its clinch/elimination marks; any seed that
disagrees is printed so a tiebreak mistake can't hide.
"""
import json
import os
import subprocess
from collections import defaultdict

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
SEASON = 2026
DIVS = {
    "AFC East": ["BUF", "MIA", "NE", "NYJ"], "AFC North": ["BAL", "CIN", "CLE", "PIT"],
    "AFC South": ["HOU", "IND", "JAX", "TEN"], "AFC West": ["DEN", "KC", "LAC", "LV"],
    "NFC East": ["DAL", "NYG", "PHI", "WAS"], "NFC North": ["CHI", "DET", "GB", "MIN"],
    "NFC South": ["ATL", "CAR", "NO", "TB"], "NFC West": ["ARI", "LA", "SF", "SEA"],
}
DIV = {t: d for d, ts in DIVS.items() for t in ts}
CONF = {t: d[:3] for t, d in DIV.items()}
ESPN_AB = {"LAR": "LA", "WSH": "WAS"}


def pct(w, l, t):
    g = w + l + t
    return (w + 0.5 * t) / g if g else 0.0


def rec_str(w, l, t):
    return f"{w}-{l}" + (f"-{t}" if t else "")


class Season:
    def __init__(self, g):
        self.g = g  # finals: away, home, sa, sh, week, gameday
        self.res = defaultdict(list)  # team -> [(opp, +1/0/-1, pf, pa, week, date)]
        for r in g.itertuples():
            s = (r.sh > r.sa) - (r.sh < r.sa)
            self.res[r.home].append((r.away, s, r.sh, r.sa, r.week, r.gameday, True))
            self.res[r.away].append((r.home, -s, r.sa, r.sh, r.week, r.gameday, False))

    def wlt(self, t, opps=None):
        w = l = ti = 0
        for o, s, *_ in self.res[t]:
            if opps is not None and o not in opps:
                continue
            w += s > 0
            l += s < 0
            ti += s == 0
        return w, l, ti

    def p(self, t, opps=None):
        return pct(*self.wlt(t, opps))

    def opps(self, t):
        return {o for o, *_ in self.res[t]}

    def sov(self, t):  # combined record of the teams beaten (each win counts)
        w = l = ti = 0
        for o, s, *_ in self.res[t]:
            if s > 0:
                a, b, c = self.wlt(o)
                w, l, ti = w + a, l + b, ti + c
        return pct(w, l, ti)

    def sos(self, t):  # combined record of every opponent played (each game counts)
        w = l = ti = 0
        for o, *_ in self.res[t]:
            a, b, c = self.wlt(o)
            w, l, ti = w + a, l + b, ti + c
        return pct(w, l, ti)

    def diff(self, t):
        return sum(pf - pa for _, _, pf, pa, *_ in self.res[t])


def best(S, group, kind, div_order=None):
    """The top team of a tied group and the tiebreak step that separated it. kind: 'div' or 'wc' (wild card / seeding)."""
    group = list(group)
    if len(group) == 1:
        return group[0], None
    if kind == "wc" and div_order:  # first keep only the highest-ranked club from each division
        keep = []
        for t in group:
            if not any(DIV[u] == DIV[t] and div_order.index(u) < div_order.index(t) for u in group):
                keep.append(t)
        if len(keep) < len(group):
            if len(keep) == 1:
                return keep[0], "division tiebreak"
            group = keep
    steps = []
    if kind == "div" or len(group) == 2:
        steps.append(("head-to-head", lambda t: S.p(t, set(group) - {t}) if S.wlt(t, set(group) - {t}) != (0, 0, 0) else None))
    else:  # 3+ clubs for a wild card: head-to-head only on a sweep
        def sweep(t):
            others = set(group) - {t}
            met = [o for o in others if S.wlt(t, {o}) != (0, 0, 0)]
            if len(met) < len(others):
                return None
            w, l, ti = S.wlt(t, others)
            return 1 if l == 0 and ti == 0 else (-1 if w == 0 and ti == 0 else 0)
        steps.append(("head-to-head sweep", sweep))
    if kind == "div":
        steps.append(("division record", lambda t: S.p(t, set(DIVS[DIV[t]]) - {t})))
    common = set.intersection(*[S.opps(t) for t in group]) - set(group)
    ncom = min(sum(1 for o, *_ in S.res[t] if o in common) for t in group)
    com = ("common games", lambda t: S.p(t, common) if common and (kind == "div" or ncom >= 4) else None)
    conf = ("conference record", lambda t: S.p(t, {u for u in CONF if CONF[u] == CONF[t]} - {t}))
    steps += [com, conf] if kind == "div" else [conf, com]
    steps += [("strength of victory", S.sov), ("strength of schedule", S.sos),
              ("point differential", S.diff)]
    for name, f in steps:
        v = {t: f(t) for t in group}
        if any(x is None for x in v.values()):
            if name.startswith("head-to-head") and kind == "div":
                v = {t: (x if x is not None else 0.5) for t, x in v.items()}  # unplayed head-to-head counts as level
            else:
                continue
        top = max(v.values())
        lead = [t for t in group if abs(v[t] - top) < 1e-9]
        if len(lead) == 1:
            return lead[0], name
        if len(lead) < len(group):
            t, why = best(S, lead, kind, div_order)  # restart with the clubs still tied
            return t, why or name
    return sorted(group)[0], "still level (alphabetical; the NFL would use further steps)"


def order(S, teams, kind, div_order=None):
    """Rank teams by win % and break each level group with best()."""
    out, why = [], {}
    left = list(teams)
    while left:
        top = max(S.p(t) for t in left)
        tied = [t for t in left if abs(S.p(t) - top) < 1e-9]
        t, r = best(S, tied, kind, div_order)
        if len(tied) > 1:
            why[t] = r
        out.append(t)
        left.remove(t)
    return out, why


def streak(S, t):
    rs = sorted(S.res[t], key=lambda x: x[5])
    if not rs:
        return ""
    k = {1: "W", -1: "L", 0: "T"}[rs[-1][1]]
    n = 0
    for x in reversed(rs):
        if {1: "W", -1: "L", 0: "T"}[x[1]] != k:
            break
        n += 1
    return f"{k}{n}"


def espn():
    """ESPN's seeds and clinch marks, only to cross-check. Empty when offline."""
    try:
        raw = subprocess.run(["curl", "-s", "--compressed", "--max-time", "20",
                              f"https://site.api.espn.com/apis/v2/sports/football/nfl/standings?season={SEASON}&seasontype=2"],
                             capture_output=True, text=True, timeout=30).stdout
        out = {}
        for c in json.loads(raw).get("children", []):
            for e in c["standings"]["entries"]:
                ab = e["team"]["abbreviation"]
                ab = ESPN_AB.get(ab, ab)
                st = {s["name"]: s for s in e["stats"]}
                out[ab] = dict(seed=int(float(st["playoffSeed"]["value"])) if "playoffSeed" in st and st["playoffSeed"].get("value") is not None else None,
                               clinch=st.get("clincher", {}).get("displayValue"),
                               rec=st.get("overall", {}).get("displayValue"))
        return out
    except Exception as ex:  # noqa: BLE001
        print("ESPN standings cross-check skipped:", repr(ex))
        return {}


def standings(g):
    S = Season(g)
    T = {}
    for t in DIV:
        w, l, ti = S.wlt(t)
        dv = S.wlt(t, set(DIVS[DIV[t]]) - {t})
        cf = S.wlt(t, {u for u in CONF if CONF[u] == CONF[t]} - {t})
        pf = sum(x[2] for x in S.res[t])
        pa = sum(x[3] for x in S.res[t])
        home = [0, 0, 0]
        away = [0, 0, 0]
        for x in S.res[t]:
            box = home if x[6] else away
            box[0 if x[1] > 0 else 1 if x[1] < 0 else 2] += 1
        T[t] = dict(ab=t, conf=CONF[t], div=DIV[t], w=w, l=l, t=ti, pct=round(pct(w, l, ti), 3), gp=w + l + ti,
                    dr=rec_str(*dv), cr=rec_str(*cf), home=rec_str(*home), away=rec_str(*away),
                    pf=pf, pa=pa, diff=pf - pa, strk=streak(S, t),
                    sov=round(S.sov(t), 3), sos=round(S.sos(t), 3),
                    last5="".join({1: "W", -1: "L", 0: "T"}[x[1]] for x in sorted(S.res[t], key=lambda x: x[5])[-5:]))
    divs, dorder = [], {}
    for d, ts in DIVS.items():
        o, why = order(S, ts, "div")
        dorder[d] = o
        lead = o[0]
        for i, t in enumerate(o):
            T[t]["drank"] = i + 1
            if t in why:
                T[t]["dwhy"] = why[t]
        for i, t in enumerate(o[1:], 1):  # games behind the division leader
            T[t]["gb"] = ((T[lead]["w"] - T[t]["w"]) + (T[t]["l"] - T[lead]["l"])) / 2
        divs.append(dict(name=d, conf=d[:3], teams=o))
    race = []
    for c in ("AFC", "NFC"):
        flat = [t for d in DIVS if d.startswith(c) for t in dorder[d]]
        leaders = [dorder[d][0] for d in DIVS if d.startswith(c)]
        # division champions are seeded by the wild-card steps among themselves
        lo, lwhy = order(S, leaders, "wc", flat)
        rest = [t for t in flat if t not in leaders]
        # wild cards: one at a time, a division's lower team becomes eligible only after the teams above it are placed
        wc, wwhy, left = [], {}, list(rest)
        while left:
            top = max(S.p(t) for t in left)
            tied = [t for t in left if abs(S.p(t) - top) < 1e-9]
            t, r = best(S, tied, "wc", flat)
            if len(tied) > 1:
                wwhy[t] = r
            wc.append(t)
            left.remove(t)
        seeds = lo + wc
        seven = seeds[6]
        rows = []
        for i, t in enumerate(seeds):
            x = dict(ab=t, seed=i + 1, kind="div" if i < 4 else ("wc" if i < 7 else "hunt"))
            why = (lwhy if i < 4 else wwhy).get(t)
            if why:
                x["why"] = why
            if i >= 7:
                x["back"] = ((T[seven]["w"] - T[t]["w"]) + (T[t]["l"] - T[seven]["l"])) / 2
            rows.append(x)
        race.append(dict(conf=c, rows=rows))
    return T, divs, race


def leaders():
    st = pd.read_csv(os.path.join(ROOT, "data", f"stats_player_week_{SEASON}.csv"), low_memory=False)
    st = st[st.season_type == "REG"]
    cols = ["completions", "attempts", "passing_yards", "passing_tds", "passing_interceptions", "sacks_suffered", "passing_epa",
            "carries", "rushing_yards", "rushing_tds", "receptions", "targets", "receiving_yards", "receiving_tds",
            "def_tackles_solo", "def_tackle_assists", "def_tackles_for_loss", "def_sacks", "def_qb_hits", "def_interceptions",
            "def_pass_defended", "def_fumbles_forced", "fg_made", "fg_att", "fantasy_points_ppr"]
    agg = st.groupby("player_id").agg(n=("player_display_name", "last"), tm=("team", "last"), pos=("position", "last"),
                                        gp=("week", "nunique"), **{c: (c, "sum") for c in cols})
    agg["scrim"] = agg.rushing_yards + agg.receiving_yards
    agg["tds"] = agg.rushing_tds + agg.receiving_tds
    agg["tkl"] = agg.def_tackles_solo + agg.def_tackle_assists
    agg["ypa"] = (agg.passing_yards / agg.attempts.where(agg.attempts > 0)).round(1)
    agg["ypc"] = (agg.rushing_yards / agg.carries.where(agg.carries > 0)).round(1)
    agg["cmp"] = (100 * agg.completions / agg.attempts.where(agg.attempts > 0)).round(1)
    # stat key -> (source column, minimum to qualify for the rate stats)
    P = {"pyd": "passing_yards", "ptd": "passing_tds", "ypa": "ypa", "cmp": "cmp", "pepa": "passing_epa",
         "ryd": "rushing_yards", "rtd": "rushing_tds", "ypc": "ypc", "rec": "receptions", "tgt": "targets", "recyd": "receiving_yards",
         "rectd": "receiving_tds", "scrim": "scrim", "tds": "tds", "sk": "def_sacks", "dint": "def_interceptions", "tkl": "tkl",
         "tfl": "def_tackles_for_loss", "qbh": "def_qb_hits", "pd": "def_pass_defended", "ff": "def_fumbles_forced", "fg": "fg_made",
         "ppr": "fantasy_points_ppr"}
    weeks = int(st.week.max()) if len(st) else 0
    qual = {"ypa": agg.attempts >= 14 * weeks, "cmp": agg.attempts >= 14 * weeks, "ypc": agg.carries >= 6.25 * weeks,
            "pepa": agg.attempts >= 14 * weeks}
    out = {}
    for k, c in P.items():
        a = agg[agg[c].notna() & (agg[c] != 0)]
        if k in qual:
            a = a[qual[k].reindex(a.index).fillna(False)]
        a = a.sort_values([c, "gp"], ascending=[False, True])
        out[k] = [dict(n=r.n, tm=r.tm, pos=r.pos, gp=int(r.gp), v=round(float(r[c]), 1) if isinstance(r[c], float) and r[c] % 1 else int(r[c]))
                  for _, r in a.head(15).iterrows()]
    return out, weeks


def team_leaders(T):
    p = pd.read_csv(os.path.join(ROOT, "data", f"play_by_play_{SEASON}.csv.gz"), low_memory=False,
                    usecols=["season_type", "posteam", "defteam", "play_type", "yards_gained", "epa", "success", "pass", "rush",
                             "interception", "fumble_lost", "sack", "third_down_converted", "third_down_failed",
                             "game_id", "drive", "yardline_100", "touchdown", "td_team"])
    p = p[p.season_type == "REG"]
    sc = p[(p["pass"] == 1) | (p["rush"] == 1)]  # scrimmage plays, scrambles and sacks included (nflverse pass/rush flags)
    off = sc.groupby("posteam").agg(plays=("epa", "size"), yds=("yards_gained", "sum"), epa=("epa", "mean"), sr=("success", "mean"))
    dfn = sc.groupby("defteam").agg(plays=("epa", "size"), yds=("yards_gained", "sum"), epa=("epa", "mean"), sr=("success", "mean"))
    pe = sc[sc["pass"] == 1].groupby("posteam").epa.mean()
    re_ = sc[sc["rush"] == 1].groupby("posteam").epa.mean()
    dpe = sc[sc["pass"] == 1].groupby("defteam").epa.mean()
    dre = sc[sc["rush"] == 1].groupby("defteam").epa.mean()
    give = p.groupby("posteam").agg(i=("interception", "sum"), f=("fumble_lost", "sum"))
    take = p.groupby("defteam").agg(i=("interception", "sum"), f=("fumble_lost", "sum"))
    sacks = p.groupby("defteam").sack.sum()
    t3 = p.groupby("posteam").agg(c=("third_down_converted", "sum"), f=("third_down_failed", "sum"))
    d3 = p.groupby("defteam").agg(c=("third_down_converted", "sum"), f=("third_down_failed", "sum"))
    # red zone: drives that reached the 20 and ended in an offensive touchdown
    rz = p[(p.yardline_100 <= 20) & p.posteam.notna()].assign(otd=lambda x: (x.touchdown == 1) & (x.td_team == x.posteam))
    rzd = rz.groupby(["posteam", "game_id", "drive"]).otd.any().reset_index(name="td")
    rzo = rzd.groupby("posteam").td.agg(["sum", "size"])
    out = []
    for t, x in T.items():
        g = x["gp"] or 1
        r = dict(ab=t, gp=x["gp"], pf=x["pf"], pa=x["pa"], ppg=round(x["pf"] / g, 1), papg=round(x["pa"] / g, 1), diff=x["diff"])
        if t in off.index:
            o = off.loc[t]
            r.update(ypp=round(o.yds / o.plays, 2), ypg=round(o.yds / g, 1), epa=round(o.epa, 3), sr=round(100 * o.sr, 1),
                     pepa=round(pe.get(t, float("nan")), 3), repa=round(re_.get(t, float("nan")), 3))
        if t in dfn.index:
            dd = dfn.loc[t]
            r.update(dypp=round(dd.yds / dd.plays, 2), dypg=round(dd.yds / g, 1), depa=round(dd.epa, 3), dsr=round(100 * dd.sr, 1),
                     dpepa=round(dpe.get(t, float("nan")), 3), drepa=round(dre.get(t, float("nan")), 3))
        gv = int(give.loc[t].sum()) if t in give.index else 0
        tk = int(take.loc[t].sum()) if t in take.index else 0
        r.update(give=gv, take=tk, to=tk - gv, sacks=int(sacks.get(t, 0)))
        if t in t3.index and t3.loc[t].sum():
            r["d3"] = round(100 * t3.loc[t].c / t3.loc[t].sum(), 1)
        if t in d3.index and d3.loc[t].sum():
            r["dd3"] = round(100 * d3.loc[t].c / d3.loc[t].sum(), 1)
        if t in rzo.index and rzo.loc[t, "size"]:
            r["rz"] = round(100 * rzo.loc[t, "sum"] / rzo.loc[t, "size"], 1)
        out.append({k: (None if isinstance(v, float) and v != v else v) for k, v in r.items()})
    return out


def build(dst):
    g = pd.read_csv(os.path.join(ROOT, "data", "games.csv"), low_memory=False)
    g = g[(g.season == SEASON) & (g.game_type == "REG") & g.result.notna()].rename(
        columns={"away_team": "away", "home_team": "home", "away_score": "sa", "home_score": "sh"})
    g[["sa", "sh"]] = g[["sa", "sh"]].astype(int)
    T, divs, race = standings(g)
    E = espn()
    if E:
        for c in race:
            for x in c["rows"][:7]:
                e = E.get(x["ab"], {})
                if e.get("seed") and e["seed"] != x["seed"] and e.get("rec") == rec_str(T[x["ab"]]["w"], T[x["ab"]]["l"], T[x["ab"]]["t"]):
                    print(f"Standings check: {x['ab']} seed {x['seed']} here, {e['seed']} at ESPN")
        for t, e in E.items():
            if t in T and e.get("clinch") and e.get("rec") == rec_str(T[t]["w"], T[t]["l"], T[t]["t"]):
                T[t]["clinch"] = e["clinch"]
    pl, weeks = leaders()
    out = dict(season=SEASON, through=int(g.week.max()) if len(g) else 0, finals=len(g), last=str(g.gameday.max()) if len(g) else None,
               stats_through=weeks, teams=T, divs=divs, race=race, players=pl, tl=team_leaders(T),
               espn=bool(E))
    with open(dst, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    return dst


if __name__ == "__main__":
    print(build(os.path.join(ROOT, "output", "hub", "nflst.json")))
