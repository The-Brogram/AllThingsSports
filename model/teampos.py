"""Team vs Position: what each defense has given up, per game, to opposing QBs, RBs, WRs and TEs.

For every regular-season game, the stats of all opposing players at a position are added together (two QBs in one game
count as one game's worth), then averaged over the games that defense has played. Passer rating is computed from the
season totals, not averaged game by game. Fantasy points are built in the page from these averages, so Standard,
Half PPR and PPR all come from the same numbers:
  0.04 per pass yard, 4 per pass TD, -2 per INT, 0.1 per rush/receiving yard, 6 per rush/receiving TD,
  2 per two-point conversion, -2 per fumble lost, plus 0 / 0.5 / 1 per reception.
Writes output/hub/teampos.json. Run on its own or through season.py.
"""
import json
import os
import pandas as pd
from engine import DATA

ROOT = os.path.dirname(os.path.abspath(__file__))
POS = ["QB", "RB", "WR", "TE"]
SRC = dict(att="attempts", cmp="completions", py="passing_yards", ptd="passing_tds", int="passing_interceptions",
           car="carries", ry="rushing_yards", rtd="rushing_tds", tgt="targets", rec="receptions", ryd="receiving_yards", rctd="receiving_tds")
TIMEFRAMES = [(2026, "2026 year to date average"), (2025, "2025 full season average")]


def rating(att, cmp, yds, td, it):
    if att <= 0:
        return None
    c = lambda v: min(max(v, 0), 2.375)
    return round((c((cmp / att - .3) * 5) + c((yds / att - 3) * .25) + c(td / att * 20) + c(2.375 - it / att * 25)) / 6 * 100, 1)


def season(y):
    s = pd.read_csv(f"{DATA}/stats_player_week_{y}.csv", low_memory=False)
    s = s[(s.season_type == "REG") & s.position_group.isin(POS) & s.opponent_team.notna()].copy()
    s["fl"] = s[[c for c in ("rushing_fumbles_lost", "receiving_fumbles_lost", "sack_fumbles_lost") if c in s]].fillna(0).sum(axis=1)
    s["two"] = s[[c for c in ("passing_2pt_conversions", "rushing_2pt_conversions", "receiving_2pt_conversions") if c in s]].fillna(0).sum(axis=1)
    for k, c in SRC.items():
        s[k] = s[c].fillna(0)
    keys = list(SRC) + ["fl", "two"]
    # games each defense played (from every offensive player row, so a game with no RB stats still counts)
    games = s.groupby("opponent_team").apply(lambda x: x[["season", "week"]].drop_duplicates().shape[0], include_groups=False)
    out = {}
    for pos in POS:
        x = s[s.position_group == pos].groupby("opponent_team")[keys].sum()
        rows = []
        for t, g in games.items():
            tot = x.loc[t] if t in x.index else pd.Series(0.0, index=keys)
            r = {k: round(float(tot[k]) / g, 2) for k in keys}
            r.update(team=t, g=int(g), rate=rating(tot.att, tot.cmp, tot.py, tot.ptd, tot.int) if pos == "QB" else None)
            rows.append(r)
        out[pos] = rows
    wk = int(s.week.max())
    g = pd.read_csv(f"{DATA}/games.csv", low_memory=False)
    g = g[(g.season == y) & (g.game_type == "REG") & (g.week == wk)]
    partial = bool(g.result.isna().any())  # e.g. Thursday's game is in, Sunday's are not
    return dict(season=y, through=wk - 1 if partial else wk, partial=wk if partial else None, pos=out)


KEEP = dict(QB=40, RB=80, WR=110, TE=50)


def sos(y):
    """Strength of schedule inputs: every team's opponent each week, and the season's fantasy players with their
    scoring parts summed (the page turns these into Standard / Half / PPR points and grades each opponent)."""
    g = pd.read_csv(f"{DATA}/games.csv", low_memory=False)
    g = g[(g.season == y) & (g.game_type == "REG")]
    teams = sorted(set(g.home_team) | set(g.away_team))
    weeks = int(g.week.max())
    sched = {t: [None] * weeks for t in teams}
    for r in g.itertuples():
        done = 0 if pd.isna(r.result) else 1
        sched[r.home_team][r.week - 1] = [r.away_team, 1, done]
        sched[r.away_team][r.week - 1] = [r.home_team, 0, done]
    s = pd.read_csv(f"{DATA}/stats_player_week_{y}.csv", low_memory=False)
    s = s[(s.season_type == "REG") & s.position_group.isin(POS)].copy()
    z = lambda c: s[c].fillna(0) if c in s else 0
    s["rec_"] = z("receptions"); s["yd_"] = z("rushing_yards") + z("receiving_yards"); s["td_"] = z("rushing_tds") + z("receiving_tds")
    s["py_"] = z("passing_yards"); s["ptd_"] = z("passing_tds"); s["int_"] = z("passing_interceptions")
    s["fl_"] = z("rushing_fumbles_lost") + z("receiving_fumbles_lost") + z("sack_fumbles_lost")
    s["two_"] = z("passing_2pt_conversions") + z("rushing_2pt_conversions") + z("receiving_2pt_conversions")
    for k, c in SRC.items():
        s[k + "_"] = z(c)
    s = s.sort_values("week")
    cols = ["rec_", "yd_", "td_", "py_", "ptd_", "int_", "fl_", "two_"]
    xcols = ["att_", "cmp_", "car_", "ry_", "rtd_", "tgt_", "ryd_", "rctd_"]
    # team targets in each game, for target share
    tt = s.groupby(["team", "week"]).tgt_.sum()
    s["ttg"] = [tt.get((t, w), 0) for t, w in zip(s.team, s.week)]
    # offensive snap share from the snap counts (PFR ids, matched through the roster)
    snap = {}
    try:
        sc = pd.read_csv(f"{DATA}/snap_counts_{y}.csv", low_memory=False)
        sc = sc[sc.game_type == "REG"]
        rs = pd.read_csv(f"{DATA}/roster_{y}.csv", low_memory=False)[["gsis_id", "pfr_id"]].dropna().drop_duplicates("pfr_id")
        sc = sc.merge(rs, left_on="pfr_player_id", right_on="pfr_id")
        snap = {(g, w): p for g, w, p in zip(sc.gsis_id, sc.week, sc.offense_pct)}
    except Exception:
        pass
    s["snap"] = [snap.get((p, w)) for p, w in zip(s.player_id, s.week)]
    wk = {pid: [[int(r.week), r.opponent_team, int(r.team == r.opponent_team)] + [round(float(getattr(r, c)), 1) for c in cols] for r in x.itertuples()]
          for pid, x in s.groupby("player_id")}
    a = s.groupby("player_id").agg(n=("player_display_name", "last"), pos=("position_group", "last"), tm=("team", "last"), wk=("week", "max"),
                                   gp=("week", "nunique"), ttg=("ttg", "sum"), snap=("snap", "mean"), **{c: (c, "sum") for c in cols + xcols}).reset_index()
    a["ppr"] = a.rec_ + .1 * a.yd_ + 6 * a.td_ + .04 * a.py_ + 4 * a.ptd_ - 2 * a.int_ - 2 * a.fl_ + 2 * a.two_
    try:  # newer team / injured reserve from the weekly rosters
        ro = pd.read_csv(f"{DATA}/roster_weekly_{y}.csv", low_memory=False)
        ro = ro[ro.week == ro.week.max()].drop_duplicates("gsis_id").set_index("gsis_id")
        rw = int(ro.week.max())
    except Exception:
        ro, rw = None, 0
    out = []
    for pos in POS:
        x = a[a.pos == pos]
        keep = set(x.nlargest(KEEP[pos], "ppr").player_id) | set(x[x.gp >= 2].assign(pg=x.ppr / x.gp).nlargest(KEEP[pos], "pg").player_id)
        for r in x[x.player_id.isin(keep)].itertuples():
            tm, st = r.tm, ""
            if ro is not None and r.player_id in ro.index:
                q = ro.loc[r.player_id]
                if rw > r.wk and isinstance(q.team, str) and q.team in sched:
                    tm = q.team
                st = {"RES": "IR", "CUT": "Cut", "RET": "Retired", "INA": "Inactive"}.get(q.status, "")
            if st in ("Cut", "Retired"):
                continue
            out.append([r.player_id, r.n, pos, tm, int(r.gp)] + [round(float(getattr(r, c)), 1) for c in cols] + [st]
                       + [round(float(getattr(r, c)), 1) for c in xcols]
                       + [None if pd.isna(r.snap) else round(float(r.snap), 3), round(r.tgt_ / r.ttg, 3) if r.ttg else None, wk.get(r.player_id, [])])
    # team records so far, for overall strength of schedule (opponents' win %)
    rec = {t: [0, 0, 0] for t in teams}
    for r in g[g.result.notna()].itertuples():
        h = 0 if r.result > 0 else 1 if r.result < 0 else 2
        a = {0: 1, 1: 0, 2: 2}[h]
        rec[r.home_team][h] += 1
        rec[r.away_team][a] += 1
    return dict(season=y, weeks=weeks, sched=sched, rec=rec, wkeys=["wk", "opp", "x", "rec", "yd", "td", "py", "ptd", "int", "fl", "two"],
                pkeys=["id", "n", "pos", "tm", "gp", "rec", "yd", "td", "py", "ptd", "int", "fl", "two", "st",
                       "att", "cmp", "car", "ry", "rtd", "tgt", "ryd", "rctd", "snap", "tsh", "wl"], players=out)


def build(out=os.path.join(ROOT, "output", "hub", "teampos.json")):
    data = dict(built=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
                timeframes=[dict(key=str(y), label=lab, **season(y)) for y, lab in TIMEFRAMES], sos=sos(TIMEFRAMES[0][0]))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    print(f"Team vs position: {', '.join(f'{t['season']} thru wk {t['through']}' for t in data['timeframes'])} -> {out}")
    return out


if __name__ == "__main__":
    build()
