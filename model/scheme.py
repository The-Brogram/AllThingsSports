"""Descriptive matchup data: team efficiency, scheme, pressure, field zones, personnel, hit rates."""
import glob
import numpy as np
import pandas as pd
import engine as E

DATA = E.DATA
THIS = 2026


class Ctx:
    def __init__(self, data):
        self.d = data
        p = data["pbp"]
        self.p26 = p[(p.season == THIS) & (p.season_type == "REG")].copy()
        self.p25 = p[(p.season == 2025) & (p.season_type == "REG")].copy()
        ftn = pd.read_csv(f"{DATA}/ftn_charting_2026.csv", low_memory=False)
        self.p26 = self.p26.merge(ftn[["nflverse_game_id", "nflverse_play_id", "n_blitzers", "n_pass_rushers", "n_defense_box", "is_play_action",
                                       "is_screen_pass", "is_motion", "is_qb_out_of_pocket", "is_interception_worthy", "read_thrown"]],
                                  left_on=["game_id", "play_id"], right_on=["nflverse_game_id", "nflverse_play_id"], how="left")
        part = pd.read_csv(f"{DATA}/pbp_participation_2025.csv", low_memory=False,
                           usecols=["nflverse_game_id", "play_id", "route", "defense_man_zone_type", "defense_coverage_type", "was_pressure", "time_to_throw"])
        self.p25 = self.p25.merge(part, left_on=["game_id", "play_id"], right_on=["nflverse_game_id", "play_id"], how="left")
        self.apass = pd.read_csv(f"{DATA}/advstats_week_pass_2026.csv")
        self.arush = pd.read_csv(f"{DATA}/advstats_week_rush_2026.csv")
        self.adef = pd.read_csv(f"{DATA}/advstats_week_def_2026.csv")
        self.arec = pd.read_csv(f"{DATA}/advstats_week_rec_2026.csv")
        self.ngsp = pd.read_csv(f"{DATA}/ngs_passing.csv.gz")
        self.ngsr = pd.read_csv(f"{DATA}/ngs_receiving.csv.gz")
        self.ngsu = pd.read_csv(f"{DATA}/ngs_rushing.csv.gz")
        dc = pd.read_csv(f"{DATA}/depth_charts_2026.csv", low_memory=False)
        self.dc = dc[dc.dt == dc.groupby("team").dt.transform("max")]
        sn = data["snaps"]
        self.snaps = sn[(sn.season == THIS)]
        allst = [pd.read_csv(f, usecols=E.STAT_COLS, low_memory=False) for f in sorted(glob.glob(f"{DATA}/stats_player_week_20*.csv"))]
        h = pd.concat(allst)
        self.hist = h[h.season_type == "REG"].rename(columns={"player_display_name": "name"})
        g = data["games"]
        self.g26 = g[g.season == THIS]
        self.teams_rank = self._rank()
        self.lg_zone = self._zones(self.p26)
        c26, c25 = self.g26, g[g.season == 2025]
        self.hc26 = {**dict(zip(c26.home_team, c26.home_coach))}
        self.hc25 = {**dict(zip(c25.home_team, c25.home_coach))}

    def _rank(self):
        p = self.p26[self.p26.play_type.isin(["pass", "run"])]
        off = p.groupby("posteam").epa.mean()
        de = p.groupby("defteam").epa.mean()
        pd_ = p[p.pass_attempt == 1].groupby("defteam").epa.mean()
        rd = p[p.rush_attempt == 1].groupby("defteam").epa.mean()
        return dict(off=off.rank(ascending=False).to_dict(), de=de.rank().to_dict(), passd=pd_.rank().to_dict(), rushd=rd.rank().to_dict())

    @staticmethod
    def _zones(p):
        x = p[(p.pass_attempt == 1) & (p.sack == 0) & p.air_yards.notna() & p.pass_location.notna()].copy()
        x["depth"] = pd.cut(x.air_yards, [-99, 9.5, 19.5, 99], labels=["Short 0-9", "Inter 10-19", "Deep 20+"])
        return x.groupby(["depth", "pass_location"], observed=True).agg(att=("epa", "size"), epa=("epa", "mean"), yds=("yards_gained", "mean")).reset_index()

    def team_side(self, tm, side):
        """Efficiency table. side='off' uses plays by tm; side='def' uses plays against tm."""
        col = "posteam" if side == "off" else "defteam"
        p = self.p26[self.p26[col] == tm]
        out = {}
        wks = sorted(p.week.unique())
        for lab, x in (("Season", p), ("Last 5", p[p.week.isin(wks[-5:])]), ("Last 3", p[p.week.isin(wks[-3:])]),
                       ("Home", p[p.home_team == tm]), ("Away", p[p.away_team == tm])):
            out[lab] = self._eff(x, side, tm)
        return out

    def _eff(self, x, side, tm):
        s = x[x.play_type.isin(["pass", "run"]) & (x.two_point_attempt != 1)]
        n = max(x.game_id.nunique(), 1)
        pa, ru = s[(s.pass_attempt == 1)], s[(s.rush_attempt == 1)]
        real = pa[pa.sack == 0]
        tdn = s[s.down == 3]
        dr = x[x.drive.notna()].groupby(["game_id", "fixed_drive"]).agg(rz=("drive_inside20", "max"), res=("fixed_drive_result", "first")).reset_index()
        rzd = dr[dr.rz == 1]
        pts = (x.groupby("game_id").apply(lambda g: g.posteam_score_post.max() if False else 0, include_groups=False).sum())
        g = x.drop_duplicates("game_id")
        col = "total_home_score"
        if side == "off":
            ppg = np.mean([r.home_score if r.home_team == tm else r.away_score for r in g.itertuples()]) if len(g) else np.nan
        else:
            ppg = np.mean([r.away_score if r.home_team == tm else r.home_score for r in g.itertuples()]) if len(g) else np.nan
        return dict(g=n, ppg=ppg, ypg=s.yards_gained.sum() / n, pass_y=real.yards_gained.sum() / n + pa[pa.sack == 1].yards_gained.sum() / n,
                    rush_y=ru.yards_gained.sum() / n, ypp=s.yards_gained.mean(), ypa=real.yards_gained.mean(), ypc=ru.yards_gained.mean(),
                    success=s.success.mean(), epa=s.epa.mean(), pass_epa=pa.epa.mean(), rush_epa=ru.epa.mean(),
                    third=tdn.third_down_converted.sum() / max(len(tdn), 1), rz_td=(rzd.res == "Touchdown").mean() if len(rzd) else np.nan,
                    to=(s.interception.sum() + s.fumble_lost.sum()) / n, sacks=pa.sack.sum() / n, sack_rate=pa.sack.mean(), penalties=x[x.penalty == 1].shape[0] / n,
                    expl_pass=(real.yards_gained >= 20).mean(), expl_rush=(ru.yards_gained >= 12).mean(), plays=len(s) / n)

    def pace(self, tm):
        p = self.p26[(self.p26.posteam == tm) & self.p26.play_type.isin(["pass", "run"])].sort_values(["game_id", "play_id"])
        p = p.assign(gap=p.groupby(["game_id", "drive"]).game_seconds_remaining.diff(-1))
        n = p[(p.score_differential.abs() <= 7) & (p.qtr <= 3) & (p.gap > 0) & (p.gap < 60)]
        s = lambda m: p[m]
        lead, trail, close = p.score_differential >= 8, p.score_differential <= -8, p.score_differential.abs() <= 7
        g = max(p.game_id.nunique(), 1)
        dr = self.p26[(self.p26.posteam == tm)].dropna(subset=["fixed_drive"]).drop_duplicates(["game_id", "fixed_drive"])
        def top(v):
            try:
                a, b = str(v).split(":")
                return int(a) * 60 + int(b)
            except Exception:
                return 0
        return dict(plays=len(p) / g, sec=n.gap.mean(), proe=p.pass_oe.mean(), proe_neutral=p[close & (p.qtr <= 3)].pass_oe.mean() if "pass_oe" in p else np.nan,
                    pass_lead=p[lead].pass_attempt.mean(), pass_trail=p[trail].pass_attempt.mean(), pass_close=p[close].pass_attempt.mean(),
                    att=p.pass_attempt.sum() / g, rush=p.rush_attempt.sum() / g, drives=len(dr) / g, top=dr.drive_time_of_possession.map(top).sum() / g / 60)

    def schedule_strength(self, tm, r):
        g = self.g26[self.g26.result.notna() & ((self.g26.home_team == tm) | (self.g26.away_team == tm))]
        opp = [a if h == tm else h for h, a in zip(g.home_team, g.away_team)]
        return dict(opp_off=float(np.mean([r["off"].get(o, 0) for o in opp])), opp_def=float(np.mean([r["def"].get(o, 0) for o in opp])), n=len(opp), record=self.record(tm))

    def record(self, tm):
        g = self.g26[self.g26.result.notna() & ((self.g26.home_team == tm) | (self.g26.away_team == tm))]
        w = l = tie = 0
        for r in g.itertuples():
            m = r.result if r.home_team == tm else -r.result
            w += m > 0; l += m < 0; tie += m == 0
        return f"{w}-{l}" + (f"-{tie}" if tie else "")

    def pressure(self, tm):
        a = self.apass[self.apass.opponent == tm]
        o = self.apass[self.apass.team == tm]
        p = self.p26[(self.p26.defteam == tm) & (self.p26.qb_dropback == 1)]
        k = p[p.n_pass_rushers.fillna(0) > 0]
        bl = k[k.n_blitzers > 0]
        non = k[k.n_blitzers == 0]
        def t(df):
            d = self.p26[self.p26.posteam.isin(df.team.unique()) & (self.p26.defteam.isin(df.opponent.unique()))]
            return df.times_pressured.sum() / max(df.times_pressured.sum() / df.times_pressured_pct.replace(0, np.nan).mean(), 1) if False else None
        drop = {}
        db_d = self.p26[(self.p26.defteam == tm) & (self.p26.qb_dropback == 1)].shape[0]
        db_o = self.p26[(self.p26.posteam == tm) & (self.p26.qb_dropback == 1)].shape[0]
        lg_db = self.p26[self.p26.qb_dropback == 1].shape[0]
        lg_pr = self.apass.times_pressured.sum() / max(lg_db, 1)
        return dict(pr=a.times_pressured.sum() / max(db_d, 1), pr_allowed=o.times_pressured.sum() / max(db_o, 1), lg_pr=lg_pr, blitz=len(bl) / max(len(k), 1),
                    four=(k.n_pass_rushers == 4).mean() if len(k) else np.nan, five_plus=(k.n_pass_rushers >= 5).mean() if len(k) else np.nan,
                    box_light=(p.n_defense_box <= 6).mean(), epa_blitz=bl.epa.mean(), epa_non=non.epa.mean(), sacks=p.sack.sum(), hits=a.times_hit.sum(), n=len(k),
                    sack_rate=p.sack.mean())

    def qb(self, name, team):
        x = self.p26[(self.p26.passer_player_name.notna()) & (self.p26.passer.notna()) & (self.p26.posteam == team)]
        ids = self.d["stats"]
        pid = ids[(ids.name == name)].player_id.iloc[-1]
        x = x[x.passer_player_id == pid]
        k = x[x.n_pass_rushers.fillna(0) > 0]
        bl, nb = k[k.n_blitzers > 0], k[k.n_blitzers == 0]
        real = x[(x.sack == 0) & (x.pass_attempt == 1)]
        deep = real[real.air_yards >= 20]
        pa = x[x.is_play_action == True]
        oop = x[x.is_qb_out_of_pocket == True]
        ng = self.ngsp[(self.ngsp.player_gsis_id == pid) & (self.ngsp.season == THIS) & (self.ngsp.week == 0)]
        if ng.empty:
            ng = self.ngsp[(self.ngsp.player_gsis_id == pid) & (self.ngsp.season == THIS) & (self.ngsp.week > 0)]
        ttt = ng.avg_time_to_throw.mean() if len(ng) else np.nan
        air = ng.avg_intended_air_yards.mean() if len(ng) else np.nan
        ap = self.apass[self.apass.team == team]
        y25 = self.p25[(self.p25.passer_player_id == pid) & (self.p25.qb_dropback == 1)]
        pr25, cl25 = y25[y25.was_pressure == True], y25[y25.was_pressure == False]
        man = y25[y25.defense_man_zone_type == "MAN_COVERAGE"]
        zone = y25[y25.defense_man_zone_type == "ZONE_COVERAGE"]
        cov = y25.defense_coverage_type.fillna("")
        one = y25[cov.isin(["COVER_1", "COVER_3", "COVER_0"])]
        two = y25[cov.isin(["COVER_2", "COVER_4", "COVER_6", "2_MAN"])]
        f = lambda z: dict(n=len(z), epa=z.epa.mean(), ypa=z[z.sack == 0].yards_gained.mean(),
                           cmp=z[z.sack == 0].complete_pass.mean(), sack=z.sack.mean())
        return dict(pid=pid, epa=x.epa.mean(), dbs=len(x), cpoe=x.cpoe.mean(), blitz=f(bl), non=f(nb), four=f(k[k.n_pass_rushers == 4]), deep=f(deep), pa=f(pa), oop=f(oop), ttt=ttt, air=air,
                    iw=(x.is_interception_worthy == True).sum() / max(len(real), 1), sack=x.sack.mean(), press25=f(pr25), clean25=f(cl25), man25=f(man), zone25=f(zone),
                    single25=f(one), two25=f(two), pts=(pr25.sack.mean() if len(pr25) else np.nan))

    def coverage25(self, tm):
        y = self.p25[(self.p25.defteam == tm) & (self.p25.qb_dropback == 1)]
        v = y[y.defense_coverage_type.notna() & (y.defense_coverage_type != "BLOWN")]
        res = dict(n=len(v), man=(v.defense_man_zone_type == "MAN_COVERAGE").sum() / max(len(v), 1), hc_change=self.hc25.get(tm) != self.hc26.get(tm))
        rows = []
        for c, x in v.groupby("defense_coverage_type"):
            if len(x) >= 8:
                rows.append(dict(cov=c.replace("COVER_", "Cover ").replace("_", " ").title().replace("Cover", "Cover"), rate=len(x) / len(v), n=len(x), epa=x.epa.mean(),
                                 ypa=x[x.sack == 0].yards_gained.mean(), td=int(x.pass_touchdown.sum()), expl=int((x.yards_gained >= 20).sum())))
        res["rows"] = sorted(rows, key=lambda r: -r["rate"])
        cov = v.defense_coverage_type
        res["single"] = cov.isin(["COVER_0", "COVER_1", "COVER_3"]).sum() / max(len(v), 1)
        res["two"] = cov.isin(["COVER_2", "COVER_4", "COVER_6", "2_MAN"]).sum() / max(len(v), 1)
        return res

    def zones(self, tm):
        z = self._zones(self.p26[self.p26.defteam == tm]).merge(self.lg_zone, on=["depth", "pass_location"], suffixes=("", "_lg"))
        return z

    def alignment_allowed(self, tm):
        """Yards per game to each target rank by defense's opponents. Rank is per game among WRs."""
        s = self.d["stats"]
        x = s[(s.season == THIS) & (s.opponent_team == tm) & (s.season_type == "REG")]
        rows = {}
        for wk, g in x.groupby("week"):
            wr = g[g.position == "WR"].sort_values("targets", ascending=False)
            for lab, r in (("WR1 (most targets)", wr.iloc[:1]), ("WR2", wr.iloc[1:2]), ("WR3", wr.iloc[2:3])):
                rows.setdefault(lab, []).append((r.targets.sum(), r.receptions.sum(), r.receiving_yards.sum(), r.receiving_tds.sum()))
            for lab, q in (("All TE", g[g.position == "TE"]), ("All RB (receiving)", g[g.position == "RB"])):
                rows.setdefault(lab, []).append((q.targets.sum(), q.receptions.sum(), q.receiving_yards.sum(), q.receiving_tds.sum()))
        L = {}
        for wk, g in self.d["stats"][(self.d["stats"].season == THIS) & (self.d["stats"].season_type == "REG")].groupby(["week", "opponent_team"]):
            wr = g[g.position == "WR"].sort_values("targets", ascending=False)
            for lab, r in (("WR1 (most targets)", wr.iloc[:1]), ("WR2", wr.iloc[1:2]), ("WR3", wr.iloc[2:3])):
                L.setdefault(lab, []).append(r.receiving_yards.sum())
            L.setdefault("All TE", []).append(g[g.position == "TE"].receiving_yards.sum())
            L.setdefault("All RB (receiving)", []).append(g[g.position == "RB"].receiving_yards.sum())
        return [dict(role=k, tg=np.mean([a[0] for a in v]), rec=np.mean([a[1] for a in v]), yds=np.mean([a[2] for a in v]), td=sum(a[3] for a in v), lg=np.mean(L[k])) for k, v in rows.items()]

    def allowed_games(self, tm):
        """Yards each position gained against tm's defense, one row per 2026 game, oldest first.
        WR/TE/RB receiving yards and RB rushing yards (RB carries only, so QB scrambles and WR sweeps are left out)."""
        s = self.d["stats"]
        x = s[(s.season == THIS) & (s.opponent_team == tm) & (s.season_type == "REG")]
        out = []
        for wk, g in x.groupby("week"):
            pos = lambda p: g[g.position == p]
            out.append(dict(week=int(wk), opp=g.team.mode().iloc[0], wr=float(pos("WR").receiving_yards.sum()), te=float(pos("TE").receiving_yards.sum()),
                            rb_rec=float(pos("RB").receiving_yards.sum()), rb_rush=float(pos("RB").rushing_yards.sum())))
        return out

    def corners(self, tm):
        a = self.adef[self.adef.team == tm].groupby("pfr_player_name").agg(
            g=("week", "nunique"), tg=("def_targets", "sum"), cmp=("def_completions_allowed", "sum"), yds=("def_yards_allowed", "sum"),
            td=("def_receiving_td_allowed", "sum"), ints=("def_ints", "sum"), pr=("def_pressures", "sum"), sk=("def_sacks", "sum"), hur=("def_times_hurried", "sum"), bl=("def_times_blitzed", "sum")).reset_index()
        d = self.dc[(self.dc.team == tm) & (self.dc.pos_abb.isin(["LCB", "RCB", "NB", "FS", "SS"])) & (self.dc.pos_rank == 1)].drop_duplicates(["pos_abb"])
        rows = []
        for r in d.itertuples():
            m = a[a.pfr_player_name == r.player_name]
            sn = self.snaps[(self.snaps.team == tm) & (self.snaps.player == r.player_name)]
            row = dict(pos=r.pos_abb, name=r.player_name, snap=sn.defense_pct.mean() if len(sn) else np.nan)
            if len(m):
                z = m.iloc[0]
                row.update(g=int(z.g), tg=int(z.tg), cmp=int(z.cmp), yds=int(z.yds), td=int(z.td), ints=int(z.ints), ypt=z.yds / max(z.tg, 1))
            rows.append(row)
        order = {"LCB": 0, "RCB": 1, "NB": 2, "FS": 3, "SS": 4}
        return sorted(rows, key=lambda r: order[r["pos"]])

    def rushers(self, tm):
        a = self.adef[self.adef.team == tm].groupby("pfr_player_name").agg(pr=("def_pressures", "sum"), sk=("def_sacks", "sum"), hit=("def_times_hitqb", "sum"), g=("week", "nunique")).reset_index()
        return a.sort_values("pr", ascending=False).head(4).to_dict("records")

    def oline(self, tm, status):
        d = self.dc[(self.dc.team == tm) & (self.dc.pos_abb.isin(["LT", "LG", "C", "RG", "RT"]))].sort_values("pos_rank")
        rows = []
        for pos in ["LT", "LG", "C", "RG", "RT"]:
            x = d[(d.pos_abb == pos)].drop_duplicates("player_name").head(2)
            if x.empty:
                continue
            a = x.iloc[0]
            sn = self.snaps[(self.snaps.team == tm) & (self.snaps.player == a.player_name)].sort_values("week")
            rows.append(dict(pos=pos, name=a.player_name, status=status.get(int(a.espn_id) if pd.notna(a.espn_id) else 0, "Active"),
                             backup=x.iloc[1].player_name if len(x) > 1 else "-", snap3=sn.offense_pct.tail(3).mean() if len(sn) else np.nan))
        return rows

    def run_def(self, tm):
        p = self.p26[(self.p26.defteam == tm) & (self.p26.rush_attempt == 1) & (self.p26.qb_scramble != 1)]
        lg = self.p26[(self.p26.rush_attempt == 1) & (self.p26.qb_scramble != 1)]
        def zone(x):
            ins = x[(x.run_location == "middle") | x.run_gap.isin(["guard"])]
            out = x[x.run_gap.isin(["tackle", "end"])]
            return [("Inside (middle / guard)", ins), ("Outside (tackle / end)", out)]
        rows = []
        for (lab, a), (_, b) in zip(zone(p), zone(lg)):
            rows.append(dict(zone=lab, n=len(a), ypc=a.yards_gained.mean(), epa=a.epa.mean(), succ=a.success.mean(), expl=(a.yards_gained >= 12).mean(), lg_ypc=b.yards_gained.mean(), lg_epa=b.epa.mean()))
        return rows

    def wr_row(self, p, tm):
        pid = p.player_id
        s = self.d["stats"]
        st = s[(s.player_id == pid) & (s.season == THIS) & (s.season_type == "REG") & (s.team == tm)]
        g = max(len(st), 1)
        tmst = s[(s.team == tm) & (s.season == THIS) & (s.week.isin(st.week))]
        ttg, tair = tmst.targets.sum(), tmst.receiving_air_yards.sum()
        x = self.p26[(self.p26.receiver_player_id == pid) & (self.p26.pass_attempt == 1)]
        rz, ez = (x.yardline_100 <= 20).sum(), ((x.yardline_100 - x.air_yards.fillna(0)) <= 0).sum()
        sn = self.snaps[(self.snaps.team == tm) & (self.snaps.player == p["name"])]
        sc = st.receiving_yards_after_catch.sum() / max(st.receptions.sum(), 1)
        ng = self.ngsr[(self.ngsr.player_gsis_id == pid) & (self.ngsr.season == THIS)]
        loc = x.pass_location.value_counts(normalize=True).to_dict()
        air = x.air_yards.dropna()
        y25 = self.p25[(self.p25.receiver_player_id == pid)]
        routes = y25.route.value_counts().head(4)
        rt = [f"{k.title()} {v}" for k, v in routes.items()]
        scr = (x.is_screen_pass == True).sum()
        read1 = (x.read_thrown.astype(str).isin(["1", "1.0"])).sum()
        return dict(name=p["name"], pos=p.pos, g=len(st), snap=sn.offense_pct.mean() if len(sn) else np.nan, tg=st.targets.sum() / g, share=st.targets.sum() / max(ttg, 1),
                    air=st.receiving_air_yards.sum() / max(tair, 1), adot=st.receiving_air_yards.sum() / max(st.targets.sum(), 1), rec=st.receptions.sum() / g, yds=st.receiving_yards.sum() / g,
                    ypt=st.receiving_yards.sum() / max(st.targets.sum(), 1), yac=sc, rz=int(rz), ez=int(ez), sep=ng.avg_separation.mean() if len(ng) else np.nan,
                    left=loc.get("left", 0), mid=loc.get("middle", 0), right=loc.get("right", 0), short=(air < 10).mean() if len(air) else 0, inter=((air >= 10) & (air < 20)).mean() if len(air) else 0,
                    deep=(air >= 20).mean() if len(air) else 0, routes=", ".join(rt) if rt else "n/a", screen=int(scr), first=read1 / max(len(x), 1), td=int(st.receiving_tds.sum()))

    def route_fit(self, pid, opp):
        y = self.p25[(self.p25.receiver_player_id == pid) & self.p25.route.notna()]
        if len(y) < 15:
            return []
        lg = self.p25[self.p25.route.notna() & (self.p25.pass_attempt == 1)]
        de = lg[lg.defteam == opp]
        rows = []
        for r, x in y.groupby("route"):
            if len(x) < 4:
                continue
            d, l = de[de.route == r], lg[lg.route == r]
            if len(d) < 6:
                continue
            rows.append(dict(route=r.title(), n=len(x), ypt=x.yards_gained.mean(), dn=len(d), dypt=d.yards_gained.mean(), lypt=l.yards_gained.mean()))
        return sorted(rows, key=lambda r: -r["n"])[:5]

    def comps(self, p, opp):
        """Similar receivers (aDOT within 3 yards, same position) who faced this defence in 2026."""
        s = self.d["stats"]
        pa = s[(s.player_id == p.player_id) & (s.season == THIS)]
        ad = pa.receiving_air_yards.sum() / max(pa.targets.sum(), 1)
        x = s[(s.season == THIS) & (s.opponent_team == opp) & (s.position == p.pos) & (s.targets >= 4) & (s.season_type == "REG")]
        sn = s[(s.season == THIS)].groupby("player_id").agg(a=("receiving_air_yards", "sum"), t=("targets", "sum"))
        sn["adot"] = sn.a / sn.t.clip(1)
        x = x.assign(adot=x.player_id.map(sn.adot), ts=x.player_id.map(sn.t / 7))
        x = x[(x.adot - ad).abs() <= 3].sort_values("week")
        return [dict(name=r.name, week=int(r.week), tg=int(r.targets), rec=int(r.receptions), yds=int(r.receiving_yards), td=int(r.receiving_tds)) for r in x.itertuples()]

    def rb_row(self, p, tm):
        pid = p.player_id
        s = self.d["stats"]
        st = s[(s.player_id == pid) & (s.season == THIS) & (s.season_type == "REG") & (s.team == tm)]
        g = max(len(st), 1)
        t = s[(s.team == tm) & (s.season == THIS) & (s.week.isin(st.week))]
        x = self.p26[(self.p26.rusher_player_id == pid) & (self.p26.rush_attempt == 1)]
        sn = self.snaps[(self.snaps.team == tm) & (self.snaps.player == p["name"])].sort_values("week")
        ad = self.arush[(self.arush.pfr_player_name == p["name"]) & (self.arush.team == tm)]
        ng = self.ngsu[(self.ngsu.player_gsis_id == pid) & (self.ngsu.season == THIS)]
        ngw = ng[ng.week > 0] if (ng.week > 0).any() else ng
        ry = ngw.rush_yards_over_expected.sum() / max(ngw.rush_attempts.sum(), 1) if len(ngw) and ngw.rush_yards_over_expected.notna().any() else np.nan
        last3 = st.sort_values("week").tail(3)
        d3 = x[x.down == 3]
        tmp = self.p26[self.p26.posteam == tm]
        return dict(name=p["name"], pos=p.pos, g=len(st), car=st.carries.sum() / g, l3=last3.carries.mean() if len(last3) else 0, share=st.carries.sum() / max(t.carries.sum(), 1), snap=sn.offense_pct.mean() if len(sn) else np.nan,
                    snap3=sn.offense_pct.tail(3).mean() if len(sn) else np.nan, ypc=st.rushing_yards.sum() / max(st.carries.sum(), 1), succ=x.success.mean(), expl=(x.yards_gained >= 12).mean(),
                    ryoe=ry, yco=ad.rushing_yards_after_contact.sum() / max(ad.carries.sum(), 1) if len(ad) else np.nan, btk=int(ad.rushing_broken_tackles.sum()) if len(ad) else 0,
                    i10=int((x.yardline_100 <= 10).sum()), i5=int((x.yardline_100 <= 5).sum()), rz=int((x.yardline_100 <= 20).sum()), tg=st.targets.sum() / g, td=int(st.rushing_tds.sum()),
                    third=int(len(d3)))

    def rz_team(self, tm):
        p = self.p26[(self.p26.posteam == tm) & (self.p26.yardline_100 <= 10) & self.p26.play_type.isin(["pass", "run"])]
        d = self.p26[(self.p26.defteam == tm) & (self.p26.yardline_100 <= 10) & self.p26.play_type.isin(["pass", "run"])]
        return dict(plays=len(p), pass_r=p.pass_attempt.mean(), td_rate=p.touchdown.mean(), d_pass_td=d[d.pass_attempt == 1].touchdown.mean(), d_rush_td=d[d.rush_attempt == 1].touchdown.mean(), d_plays=len(d))

    def hitrates(self, pid, kind, ths, opp, team, proj_vol):
        col = dict(rec="receiving_yards", rush="rushing_yards", passy="passing_yards")[kind]
        vol = dict(rec="targets", rush="carries", passy="attempts")[kind]
        h = self.hist[(self.hist.player_id == pid)].sort_values(["season", "week"])
        h = h[h[vol] > 0]
        cur = h[h.season == THIS]
        g = self.d["games"]
        hm = g[g.season == THIS][["week", "home_team"]]
        cur2 = cur.merge(g[(g.season == THIS)][["week", "home_team", "away_team"]].assign(), on="week", how="left")
        cur2 = cur2[(cur2.home_team == cur2.team) | (cur2.away_team == cur2.team)]
        hg = cur2[cur2.home_team == cur2.team]
        ag = cur2[cur2.away_team == cur2.team]
        wl = h[(h[vol] >= proj_vol * .8) & (h[vol] <= proj_vol * 1.2)]
        rk = self.teams_rank["rushd" if kind == "rush" else "passd"]
        tier = lambda x: 0 if rk.get(x, 16) <= 11 else (1 if rk.get(x, 16) <= 21 else 2)
        tv = cur[np.array([tier(o) == tier(opp) for o in cur.opponent_team], dtype=bool)]  # an array, so a player with no games this season gives an empty frame, not zero columns
        rows = []
        for k in ths:
            f = lambda d: (int((d[col] >= k).sum()), len(d))
            rows.append(dict(k=k, career=f(h), season=f(cur), l10=f(h.tail(10)), l5=f(h.tail(5)), home=f(hg), away=f(ag), vol=f(wl), tier=f(tv)))
        return rows, [int(v) for v in h.tail(5)[col]]
