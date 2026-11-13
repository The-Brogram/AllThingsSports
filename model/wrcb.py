"""WR/CB matchups from free data only: which receivers face a corner they should beat, and which face a lockdown.

Paid charting (PFF, Fantasy Points) counts every route and every corner's alignment from film. We do not have that,
so three columns are ESTIMATES and are labeled that way on the page:
  routes run      = offensive snaps x the team's dropback rate that game (TEs x 0.883). Checked on 2025 against
                    nflverse participation (who was on the field for every dropback): WR r = 0.98, TE r = 0.96.
  receiver side   = where his targets went (play-by-play pass_location: left / middle / right, offense's view).
  corner side     = where the plays he made went (pass breakups and tackles on targeted throws). The offense's right
                    is the defense's left, so offense-right plays = LCB, middle = slot (SCB), offense-left = RCB.
                    On 2026 depth charts, listed LCBs make 58% of these plays to the offense's right, RCBs 54% to the left.
Real (charted) numbers: targets, receptions, yards and TDs for receivers (nflverse) and allowed by each corner (PFR).

Matchup: each receiver's target-side mix is spread over the opposing corners by how much each one plays that side,
weighted by his recent snap share. The FP per coverage snap those corners allow, shrunk to the league average
(K_CB coverage snaps), is compared with the receiver's own FP per route (shrunk, K_WR routes):
  ADVANTAGE score = receiver percentile - corner lockdown percentile  (WR at +20 or more, CB at -20 or less, else Even).
Thresholds were set before the backtest. Run: venv/bin/python wrcb.py  -> output/hub/wrcb.json
"""
import json
import os
import numpy as np
import pandas as pd
from engine import DATA

ROOT = os.path.dirname(os.path.abspath(__file__))
TE_ROUTE = 0.883      # TEs are on the field for fewer dropbacks than their snap share says (2025 participation check)
K_WR, K_CB = 60, 100  # routes / coverage snaps of league-average prior in the shrunk rates
K_SIDE_WR, K_SIDE_CB = 20, 10
EDGE = 20
SIDES = ["L", "M", "R"]
CB_POS = ("CB", "DB")


def fp(rec, yds, td):
    return rec + 0.1 * yds + 6 * td


def load(season):
    p = pd.read_csv(f"{DATA}/play_by_play_{season}.csv.gz", low_memory=False,
                    usecols=["game_id", "week", "season_type", "posteam", "defteam", "play_type", "qb_dropback", "pass_location", "receiver_player_id",
                             "pass_defense_1_player_id", "pass_defense_2_player_id", "solo_tackle_1_player_id", "assist_tackle_1_player_id", "tackle_with_assist_1_player_id"])
    p = p[(p.season_type == "REG") & p.play_type.isin(["pass", "run"])]
    tm = p.groupby(["game_id", "posteam"]).agg(plays=("play_type", "size"), db=("qb_dropback", "sum")).reset_index()
    tm["dbr"] = tm.db / tm.plays
    s = pd.read_csv(f"{DATA}/snap_counts_{season}.csv")
    s = s[s.game_type == "REG"]
    ro = pd.read_csv(f"{DATA}/roster_{season}.csv", low_memory=False)[["pfr_id", "gsis_id"]].dropna().drop_duplicates("pfr_id")
    s = s.merge(ro, left_on="pfr_player_id", right_on="pfr_id", how="left")
    st = pd.read_csv(f"{DATA}/stats_player_week_{season}.csv", low_memory=False)
    st = st[st.season_type == "REG"]
    pf = pd.read_csv(f"{DATA}/advstats_week_def_{season}.csv")
    pf = pf[pf.game_type == "REG"]
    return dict(season=season, p=p, tm=tm, s=s, st=st, pf=pf)


def receivers_games(D):
    """One row per WR/TE game: estimated routes, team dropbacks, targets and receiving PPR."""
    s = D["s"][D["s"].position.isin(["WR", "TE"]) & (D["s"].offense_snaps > 0)].dropna(subset=["gsis_id"])
    s = s.merge(D["tm"], left_on=["game_id", "team"], right_on=["game_id", "posteam"])
    s["routes"] = s.offense_snaps * s.dbr * np.where(s.position == "TE", TE_ROUTE, 1.0)
    st = D["st"][["player_id", "week", "player_display_name", "targets", "receptions", "receiving_yards", "receiving_tds"]]
    s = s.merge(st, left_on=["gsis_id", "week"], right_on=["player_id", "week"], how="left")
    for c in ("targets", "receptions", "receiving_yards", "receiving_tds"):
        s[c] = s[c].fillna(0)
    s["fp"] = fp(s.receptions, s.receiving_yards, s.receiving_tds)
    s["name"] = s.player_display_name.fillna(s.player)
    return s[["game_id", "week", "team", "opponent", "gsis_id", "name", "position", "offense_snaps", "offense_pct", "db", "routes", "targets", "fp"]]


def corners_games(D):
    """One row per CB game: estimated coverage snaps and PFR targets / receptions / yards / TDs allowed."""
    s = D["s"][D["s"].position.isin(CB_POS) & (D["s"].defense_snaps > 0)]
    s = s.merge(D["tm"].rename(columns={"posteam": "opponent"}), on=["game_id", "opponent"])
    s["cov"] = s.defense_snaps * s.dbr
    pf = D["pf"][["pfr_game_id", "pfr_player_id", "def_targets", "def_completions_allowed", "def_yards_allowed", "def_receiving_td_allowed"]]
    s = s.merge(pf, on=["pfr_game_id", "pfr_player_id"], how="left")
    for c in ("def_targets", "def_completions_allowed", "def_yards_allowed", "def_receiving_td_allowed"):
        s[c] = s[c].fillna(0)
    s["fpa"] = fp(s.def_completions_allowed, s.def_yards_allowed, s.def_receiving_td_allowed)
    return s.rename(columns={"player": "name", "def_targets": "tga"})[
        ["game_id", "week", "team", "opponent", "pfr_player_id", "gsis_id", "name", "position", "defense_snaps", "defense_pct", "cov", "tga", "fpa",
         "def_completions_allowed", "def_yards_allowed", "def_receiving_td_allowed"]]


def side_counts(D, thru):
    """Target side for receivers, and side of the plays each defender made on targeted throws (offense's view)."""
    p = D["p"][(D["p"].week <= thru) & D["p"].pass_location.notna()]
    loc = p.pass_location.str[0].str.upper()
    rec = pd.crosstab(p.receiver_player_id, loc)
    cols = ["pass_defense_1_player_id", "pass_defense_2_player_id", "solo_tackle_1_player_id", "assist_tackle_1_player_id", "tackle_with_assist_1_player_id"]
    t = p[p.receiver_player_id.notna()]
    d = pd.concat([pd.DataFrame(dict(id=t[c], s=t.pass_location.str[0].str.upper())) for c in cols]).dropna()
    dfn = pd.crosstab(d.id, d.s)
    tg = loc[p.receiver_player_id.notna()].value_counts(normalize=True)
    return rec.reindex(columns=SIDES, fill_value=0), dfn.reindex(columns=SIDES, fill_value=0), tg.reindex(SIDES).fillna(1 / 3)


def _pct(v, ref):
    ref = np.sort(np.asarray(ref, float))
    return float(np.searchsorted(ref, v, side="right") / len(ref) * 100) if len(ref) else 50.0


class Book:
    """Everything through week `thru` of a season, ready to score any game the following week."""

    def __init__(self, D, thru, roles=None, RC=None):
        self.D, self.thru, self.roles = D, thru, roles or {}
        R, C = RC if RC is not None else (receivers_games(D), corners_games(D))
        self.R, self.C = R[R.week <= thru], C[C.week <= thru]
        r = self.R.groupby("gsis_id").agg(name=("name", "last"), team=("team", "last"), pos=("position", "last"), g=("game_id", "nunique"),
                                         routes=("routes", "sum"), db=("db", "sum"), tgt=("targets", "sum"), fp=("fp", "sum"))
        self.lg_tpr, self.lg_fpr = r.tgt.sum() / r.routes.sum(), r.fp.sum() / r.routes.sum()
        r["tprr"], r["fprr"] = r.tgt / r.routes, r.fp / r.routes
        r["fprr_s"] = (r.fp + K_WR * self.lg_fpr) / (r.routes + K_WR)
        r["route_pct"] = r.routes / r.db
        self.r = r
        c = self.C.groupby("pfr_player_id").agg(name=("name", "last"), team=("team", "last"), gsis_id=("gsis_id", "last"), g=("game_id", "nunique"),
                                               cvs=("cov", "sum"), tga=("tga", "sum"), fpa=("fpa", "sum"), rec=("def_completions_allowed", "sum"),
                                               yds=("def_yards_allowed", "sum"), td=("def_receiving_td_allowed", "sum"), snaps=("defense_snaps", "sum"))
        self.lg_ctpr, self.lg_cfpr = c.tga.sum() / c.cvs.sum(), c.fpa.sum() / c.cvs.sum()
        c["tprr"], c["fprr"] = c.tga / c.cvs, c.fpa / c.cvs
        c["tprr_s"] = (c.tga + K_CB * self.lg_ctpr) / (c.cvs + K_CB)
        c["fprr_s"] = (c.fpa + K_CB * self.lg_cfpr) / (c.cvs + K_CB)
        self.c = c
        rs, ds, self.lg_side = side_counts(D, thru)
        self.rs = rs
        # defenders: involvement share per side relative to what an average corner shows, then a role prior when we have one
        cb_ids = set(c.gsis_id.dropna())
        ds = ds[ds.index.isin(cb_ids)]
        self.cb_lg = (ds.sum() / ds.values.sum()).reindex(SIDES)
        self.ds = ds
        self.role_prior = {}
        if self.roles:
            al = {g: self._cb_raw(g) for g in ds.index}
            for role in ("LCB", "NB", "RCB"):
                m = [al[g] for g, ro in self.roles.items() if ro == role and g in al]
                if m:
                    self.role_prior[role] = np.mean(m, axis=0)
        # pools for percentiles: regulars only
        self.r_pool = r[(r.routes >= 60) & (r.pos.isin(["WR", "TE"]))].fprr_s.values
        self.c_pool = c[c.cvs >= 80].fprr_s.values

    def _cb_raw(self, gid):
        n = self.ds.loc[gid].values.astype(float) if gid in self.ds.index else np.zeros(3)
        rel = (n + K_SIDE_CB * self.cb_lg.values) / (n.sum() + K_SIDE_CB) / self.cb_lg.values
        return rel / rel.sum()

    def cb_align(self, gid):
        """Share of his work at offense-left / middle / offense-right, as rel. to an average corner (sums to 1)."""
        n = self.ds.loc[gid].values.astype(float) if gid in self.ds.index else np.zeros(3)
        prior = self.role_prior.get(self.roles.get(gid))
        if prior is None:
            prior = np.ones(3) / 3
        rel = n / self.cb_lg.values
        tot = n.sum()
        return (rel / rel.sum() * tot + K_SIDE_CB * prior) / (tot + K_SIDE_CB) if tot else prior

    def rec_side(self, gid):
        n = self.rs.loc[gid].values.astype(float) if gid in self.rs.index else np.zeros(3)
        v = (n + K_SIDE_WR * self.lg_side.values) / (n.sum() + K_SIDE_WR)
        # left vs right target mix is not a stable trait (2025 odd/even weeks r = 0.18 / 0.02); middle is (r = 0.39).
        # So only outside vs middle is used: the outside part is split evenly between the two outside corners.
        return np.array([(1 - v[1]) / 2, v[1], (1 - v[1]) / 2])

    def recent(self, df, team, key, n=3):
        """Players' average snap share over the team's last n games."""
        x = df[df.team == team]
        wk = sorted(x.week.unique())[-n:]
        x = x[x.week.isin(wk)]
        col = "offense_pct" if "offense_pct" in x else "defense_pct"
        avg = x.groupby(key)[col].sum() / len(wk)
        last = x[x.week == wk[-1]].groupby(key)[col].sum().reindex(avg.index).fillna(0)
        return np.maximum(avg, last)  # a starter back from injury, or one who just took the job, counts at his latest share

    def score(self, off, dfn, rec_ids=None, cb_ids=None, min_route=0.30):
        """Matchup rows for `off` receivers against `dfn` corners. rec_ids / cb_ids restrict to who is active."""
        rr = self.R[self.R.team == off]
        wk = sorted(rr.week.unique())[-3:]
        rr = rr[rr.week.isin(wk)].groupby("gsis_id").agg(routes=("routes", "sum"), db=("db", "sum"))
        share = (rr.routes / rr.db).sort_values(ascending=False)
        cs = self.recent(self.C, dfn, "pfr_player_id")
        cbs = [k for k in cs.index if cs[k] >= 0.35 and k in self.c.index and (cb_ids is None or k in cb_ids)]
        cbs = sorted(cbs, key=lambda k: -cs[k])[:4]
        if not cbs:
            return [], []
        A = np.array([self.cb_align(self.c.loc[k, "gsis_id"]) for k in cbs])  # rows: corners, cols: offense L/M/R
        W = A * np.array([cs[k] for k in cbs])[:, None]
        W = W / W.sum(axis=0, keepdims=True)
        fprr_c = np.array([self.c.loc[k, "fprr_s"] for k in cbs])
        tprr_c = np.array([self.c.loc[k, "tprr_s"] for k in cbs])
        rows = []
        for gid, sh in share.items():
            if sh < min_route or gid not in self.r.index or (rec_ids is not None and gid not in rec_ids):
                continue
            x = self.r.loc[gid]
            side = self.rec_side(gid)
            face = W @ side  # weight on each corner
            f_fp, f_tp = float(face @ fprr_c), float(face @ tprr_c)
            rp = _pct(x.fprr_s, self.r_pool)
            cp = 100 - _pct(f_fp, self.c_pool)
            sc = rp - cp
            i = int(np.argmax(face))
            tn = self.rs.loc[gid].values.astype(float) if gid in self.rs.index else np.zeros(3)
            rows.append(dict(id=gid, name=x["name"], pos=x.pos, route=round(float(sh), 3), side=[round(float(v), 3) for v in side],
                             tsplit=[round(float(v / tn.sum()), 3) for v in tn] if tn.sum() else None, tn=int(tn.sum()),
                             tprr=round(float(x.tprr), 3), fprr=round(float(x.fprr), 3), fprr_s=round(float(x.fprr_s), 3), routes=round(float(x.routes)),
                             tgt=int(x.tgt), vs=self.c.loc[cbs[i], "name"], vs_id=cbs[i], vs_w=round(float(face[i]), 3),
                             mix=[[self.c.loc[k, "name"], round(float(w), 3)] for k, w in sorted(zip(cbs, face), key=lambda kv: -kv[1])],
                             face_fprr=round(f_fp, 3), face_tprr=round(f_tp, 3), lift=round(f_fp / self.lg_cfpr, 3),
                             rp=round(rp), cp=round(cp), score=round(sc), adv="WR" if sc >= EDGE else "CB" if sc <= -EDGE else "Even"))
        crow = []
        for k in cbs:
            x = self.c.loc[k]
            al = self.cb_align(x.gsis_id)
            raw = self.ds.loc[x.gsis_id].values if x.gsis_id in self.ds.index else np.zeros(3)
            crow.append(dict(id=k, name=x["name"], role=self.roles.get(x.gsis_id, ""), snap=round(float(cs[k]), 3),
                             align=[round(float(al[2]), 3), round(float(al[1]), 3), round(float(al[0]), 3)],  # LCB, SCB, RCB
                             plays=int(raw.sum()), cov=round(float(x.cvs)), tga=int(x.tga), rec=int(x.rec), yds=int(x.yds), td=int(x.td),
                             tprr=round(float(x.tprr), 3), fprr=round(float(x.fprr), 3), fprr_s=round(float(x.fprr_s), 3),
                             lock=round(100 - _pct(x.fprr_s, self.c_pool))))
        return rows, crow


def roles_2026():
    d = pd.read_csv(f"{DATA}/depth_charts_2026.csv", low_memory=False)
    d = d[d.dt == d.groupby("team").dt.transform("max")]
    d = d[d.pos_abb.isin(["LCB", "RCB", "NB"])].sort_values("pos_rank").drop_duplicates("gsis_id")
    return dict(zip(d.gsis_id, d.pos_abb))


def injuries(week):
    i = pd.read_csv(f"{DATA}/injuries_2026.csv", low_memory=False)
    i = i[(i.season_type == "REG") & (i.week == week)]
    return dict(zip(i.gsis_id, i.report_status.fillna("")))


def build(out=os.path.join(ROOT, "output", "hub", "wrcb.json"), season=2026):
    D = load(season)
    g0 = pd.read_csv(f"{DATA}/games.csv", low_memory=False)
    g0 = g0[(g0.season == season) & (g0.game_type == "REG")]
    left = g0[g0.result.isna()].week
    thru = int(left.min() - 1) if len(left) else int(g0.week.max())  # last fully played week; the coming week may have started (TNF)
    B = Book(D, thru, roles_2026())
    g = pd.read_csv(f"{DATA}/games.csv", low_memory=False)
    g = g[(g.season == season) & (g.week == thru + 1) & (g.game_type == "REG")].sort_values(["gameday", "gametime", "game_id"])
    inj = injuries(thru + 1)
    pfr2g = dict(zip(B.c.index, B.c.gsis_id))
    out_ok = lambda gid: inj.get(gid, "") not in ("Out", "Doubtful")
    rec_ok = {k for k in B.r.index if out_ok(k)}
    cb_ok = {k for k in B.c.index if out_ok(pfr2g.get(k))}
    games = []
    for r in g.itertuples():
        side = {}
        for off, dfn, key in ((r.away_team, r.home_team, "away"), (r.home_team, r.away_team, "home")):
            rows, crow = B.score(off, dfn, rec_ok, cb_ok)
            for x in rows:
                x["inj"] = inj.get(x["id"], "")
            for x in crow:
                x["inj"] = inj.get(pfr2g.get(x["id"]), "")
            side[key] = dict(off=off, dfn=dfn, wr=rows, cb=crow)
        games.append(dict(id=r.game_id, day=pd.Timestamp(r.gameday).strftime("%a %b %-d"), time=r.gametime, a=r.away_team, h=r.home_team, **side))
    bt = None
    bp = os.path.join(ROOT, "output", "wrcb_backtest.json")
    if os.path.exists(bp):
        bt = json.load(open(bp)).get("summary")
    data = dict(season=season, week=thru + 1, through=thru, built=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"), edge=EDGE,
                lg=dict(wr_fprr=round(B.lg_fpr, 3), wr_tprr=round(B.lg_tpr, 3), cb_fprr=round(B.lg_cfpr, 3), cb_tprr=round(B.lg_ctpr, 3)),
                games=games, backtest=bt)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    n = sum(len(x[k]["wr"]) for x in games for k in ("away", "home"))
    print(f"WR/CB: week {thru + 1}, {len(games)} games, {n} receivers -> {out}")
    return out


if __name__ == "__main__":
    build()
