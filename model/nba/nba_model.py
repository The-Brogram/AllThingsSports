"""NBA game and player simulation.

For a game on date d, only games before d are used.
1. Game script: final margin and total come from the DraftKings spread and total plus last season's residuals
   (actual minus market), so the sim is centered on the market. Team points follow from margin and total.
2. Minutes: each player's expected minutes are a recency-weighted average of his minutes in games played for this team.
   Minutes of players ruled out are handed to the rest of the rotation in proportion to their minutes, so the team adds up to 240.
   In each simulated game, starters lose minutes as the margin grows (blowouts) and every player gets minute-to-minute noise.
3. Rates: per-minute points, rebounds, assists, threes, steals, blocks and turnovers, recency weighted and shrunk toward the
   position average. Opponent: what the defense allows per minute versus league average, shrunk. Pace: the game total versus
   the team's normal game total.
4. Points are split from the simulated team points by each player's minutes x scoring rate, so a teammate's absence raises
   everyone else's share. For rebounds, assists, threes, steals, blocks and turnovers, a regular's absence would otherwise shrink the team
   total (his minutes go to teammates at their own rates), so the total is pulled halfway back to its full-strength level (PARAMS["vac"]). Every stat is drawn from a negative binomial around its mean (overdispersion fit in backtest_nba.py).
"""
import os
import numpy as np
import pandas as pd

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
STATS = ["pts", "reb", "ast", "fg3", "stl", "blk", "tov"]
COMBO = {"pr": ("pts", "reb"), "pa": ("pts", "ast"), "ra": ("reb", "ast"), "pra": ("pts", "reb", "ast")}
HL_MIN = 8.0     # games, half-life for minutes
HL_RATE = 20.0   # games, half-life for per-minute rates
K_RATE = 30.0    # minutes of position-average prior in each rate (tuned vs DraftKings lines, Nov 2026)
K_OPP = 12.0     # games of league-average prior in each opponent factor
ELAST = {"reb": .5, "ast": .7, "fg3": .7, "stl": .4, "blk": .4, "tov": .6}  # how much a stat moves with game pace
# noise fit in backtest_nba.py --grid on DraftKings-priced lines, Apr-Nov 2026 (log loss)
PARAMS = dict(disp={"pts": 7.2, "reb": 3.6, "ast": 2.8, "fg3": 2.0, "stl": 2.4, "blk": 1.2, "tov": 3.2}, min_sd=.28, blow_start=.0065, blow_free=8.0,
              # when a regular is out, pull the team's rebounds/assists/... halfway back to full-strength level. Passed factor_nba.py on Nov 12, 2026:
              # tuned on Jan-Jun 2026, then on unseen Oct 20+ games the affected players' log loss went .7169 -> .7138 (0 = off)
              vac={s: .5 for s in ("reb", "ast", "fg3", "stl", "blk", "tov")})


def pgroup(pos):
    p = str(pos or "")
    return "C" if p.startswith("C") else ("G" if p in ("PG", "SG", "G") else "F")


def load():
    g = pd.read_csv(f"{D}/games.csv")
    p = pd.read_csv(f"{D}/players.csv.gz", dtype={"pid": str})
    g["game_id"] = g.game_id.astype(str)
    p["game_id"] = p.game_id.astype(str)
    p = p.merge(g[["game_id", "date_et", "season", "stype"]], on="game_id")
    p["pg"] = p.pos.map(pgroup)
    p = p.sort_values(["date_et", "game_id"])
    return dict(games=g, players=p)


def residuals(g):
    """(actual margin - market margin, actual total - market total) pairs for the game script."""
    x = g[g.completed & g.spread_home.notna() & g.total.notna() & g.hs.notna()]
    return np.c_[x.spread_home.values, (x.hs - x["as"] - x.spread_home).values, (x.hs + x["as"] - x.total).values]


def wavg(v, hl):
    w = 0.5 ** (np.arange(len(v))[::-1] / hl)
    return float(np.dot(w, v) / w.sum()), w


class Ctx:
    """Everything known before date d."""

    def __init__(self, d, date):
        self.date = date
        p = d["players"]
        self.p = p[p.date_et < date]
        played = self.p[self.p["min"] > 0]
        cur = played[played.season == played.season.max()] if len(played) else played
        self.played = played
        tot = played.groupby("pg")[STATS + ["min"]].sum()
        self.pos_rate = {k: (tot[STATS].T / tot["min"]).T.loc[k].to_dict() for k in tot.index}
        lg = cur[STATS + ["min"]].sum()
        self.lg_rate = (lg[STATS] / lg["min"]).to_dict()
        g = d["games"]
        self.g = g[(g.date_et < date) & g.completed]
        # opponent factors: per-minute stats allowed to each position, vs league, current season (last season fills early)
        opp = played.groupby(["opp", "pg", "season"])[STATS + ["min"]].sum().reset_index()
        self.opp = {}
        seas = played.season.max() if len(played) else None
        for (o, pg), x in opp.groupby(["opp", "pg"]):
            c = x[x.season == seas]
            prv = x[x.season != seas]
            m = c["min"].sum() + .25 * prv["min"].sum()
            fac = {}
            for s in STATS:
                v = c[s].sum() + .25 * prv[s].sum()
                base = self.pos_rate[pg][s] if pg in self.pos_rate else self.lg_rate[s]
                ng = m / 48.0
                fac[s] = ((v / max(m, 1)) / base * ng + K_OPP) / (ng + K_OPP) if base > 0 else 1.0
            self.opp[(o, pg)] = fac
        tg = self.g.assign(tot=self.g.hs + self.g["as"])
        cs = tg[tg.season == tg.season.max()]
        self.team_total = {t: float(pd.concat([cs[cs.home == t].tot, cs[cs.away == t].tot]).tail(20).mean()) for t in set(cs.home) | set(cs.away)}
        self.lg_total = float(cs.tot.mean()) if len(cs) else 228.0
        # minutes by rank on the team that night (1st man, 2nd man, ...), for spreading 240 minutes over a crowded rotation
        r = played[["game_id", "team", "min"]].tail(60000).copy()
        r["rk"] = r.groupby(["game_id", "team"])["min"].rank(ascending=False, method="first")
        t = r.groupby("rk")["min"].sum() / r.drop_duplicates(["game_id", "team"]).shape[0]
        self.tmpl = np.array([float(t.get(i + 1, 0.0)) for i in range(20)])

    def roster(self, team, avail=None, listed_only=False):
        """Expected minutes and per-minute rates for the team's players. A player's history follows him across teams.
        Candidates: everyone who has appeared for this team this season, plus anyone in `avail` (new signings, opening night).
        avail: {pid: probability of playing}. Players not listed default to 1 if they played in the team's last 3 games
        (0 otherwise), or to 0 when listed_only (the backtest passes the game's active list)."""
        x = self.p[self.p.team == team]
        seas = self.p.season.max() if len(self.p) else None
        cur = x[x.season == seas]
        cand = set(cur.pid) | {k for k, v in (avail or {}).items() if v > 0}
        if not cand:
            return pd.DataFrame()
        tg = cur.drop_duplicates("game_id").game_id.tolist()
        last3, last10 = set(tg[-3:]), set(tg[-10:])
        rows, vac, vac_min = [], dict.fromkeys(STATS, 0.0), 0.0
        for pid, y in self.p[self.p.pid.isin(cand)].groupby("pid"):
            pl = y[y["min"] > 0]
            if len(pl) == 0:
                continue
            # minutes from his time with this team once he has 3+ games here, else his recent games anywhere
            here = pl[(pl.team == team) & (pl.season == seas)]
            q = (here if len(here) >= 3 else pl).tail(25)
            em, _ = wavg(q["min"].values, HL_MIN)
            r = pl.tail(60)
            _, w = wavg(r["min"].values, HL_RATE)
            mins = float(np.dot(w, r["min"].values))
            pg = pgroup(y.pos.dropna().iloc[-1] if y.pos.notna().any() else None)
            pr = self.pos_rate.get(pg, self.lg_rate)
            rate = {s: (float(np.dot(w, r[s].fillna(0).values)) + pr[s] * K_RATE) / (mins + K_RATE) for s in STATS}
            pp = 0.0 if listed_only else float(bool(set(pl.game_id) & last3))
            if avail is not None and pid in avail:
                pp = avail[pid]
            if pp <= 0:
                # a regular who is out tonight (12+ minutes, in 3+ of the team's last 10 games): his usual production is vacated
                if em >= 12 and len(set(here.game_id) & last10) >= 3:
                    vac_min += em
                    for s in STATS:
                        vac[s] += em * rate[s]
                continue
            st = float(q.starter.tail(5).mean())
            rows.append(dict(pid=pid, name=y.name.iloc[-1], pos=y.pos.dropna().iloc[-1] if y.pos.notna().any() else None, pg=pg, exp_min=em, starter=st, play_p=pp,
                             g=len(pl), last_date=pl.date_et.iloc[-1], **{"r_" + s: rate[s] for s in STATS}))
        df = pd.DataFrame(rows)
        df = df[df.play_p > 0].reset_index(drop=True) if len(df) else df
        df.attrs["vac"], df.attrs["vac_min"] = vac, vac_min
        return df


W_OWN = 0.2  # weight on a player's own minutes vs the rank template


def fill_minutes(df, tmpl=None, total=240.0, cap=42.0):
    """Expected minutes for the players who play: blend each player's own recent minutes with the league's minutes for his
    rank in this rotation, then scale so the team sums to `total`, capped per player."""
    m = (df.exp_min * df.play_p).values.astype(float)
    if tmpl is not None and len(m):
        rk = np.argsort(np.argsort(-m))
        m = W_OWN * m + (1 - W_OWN) * tmpl[np.minimum(rk, len(tmpl) - 1)] * df.play_p.values
    for _ in range(8):
        s = m.sum()
        if s <= 0:
            break
        m = np.minimum(m * total / s, cap)
    return m


def sim_team(rng, ctx, df, team, opp, team_pts, margin, total_draw, n, P=PARAMS):
    """Simulate every player's stat line for one team. Returns {stat: array (n, players)}, plus minutes."""
    k = len(df)
    base_min = fill_minutes(df, ctx.tmpl)
    play = rng.random((n, k)) < df.play_p.values  # questionable players: in or out per simulated game
    # blowouts: past `blow_free` points, starters lose minutes and the bench picks them up
    over = np.maximum(np.abs(margin) - P["blow_free"], 0)[:, None]
    st = df.starter.values[None, :]
    fac = 1 - P["blow_start"] * over * st + P["blow_start"] * over * (1 - st) * .8
    mins = base_min[None, :] * np.clip(fac, .4, 1.6) * np.exp(rng.normal(0, P["min_sd"], (n, k)))
    mins = mins * play
    s = mins.sum(1, keepdims=True)
    mins = np.minimum(mins * (240.0 / np.where(s > 0, s, 1)), 48)
    pace = (total_draw / max(ctx.team_total.get(team, ctx.lg_total) * .5 + ctx.team_total.get(opp, ctx.lg_total) * .5, 150))[:, None]
    out = {"min": mins}
    oppf = {s_: np.array([ctx.opp.get((opp, pg), {}).get(s_, 1.0) for pg in df.pg]) for s_ in STATS}
    # points: split the simulated team total
    w = mins * df.r_pts.values[None, :] * oppf["pts"][None, :]
    mu_pts = team_pts[:, None] * w / np.maximum(w.sum(1, keepdims=True), 1e-9)
    out["pts"] = nb(rng, mu_pts, P["disp"]["pts"])
    # when a regular is out, his minutes go to teammates at THEIR rates, so the team's rebounds, assists, ... sag. vac[stat] pulls the
    # team total back toward its full-strength level (1 = all the way, 0 = off); tested in factor_nba.py
    lam, vac, vmin = P.get("vac") or {}, df.attrs.get("vac") or {}, df.attrs.get("vac_min", 0.0)
    em = (df.exp_min * df.play_p).values
    for s_ in STATS[1:]:
        mu = mins * df["r_" + s_].values[None, :] * oppf[s_][None, :] * pace ** ELAST[s_]
        if lam.get(s_) and vmin > 0:
            r = df["r_" + s_].values
            now = float(np.dot(base_min, r)) / 240.0
            full = (float(np.dot(em, r)) + vac.get(s_, 0.0)) / max(em.sum() + vmin, 1e-9)
            if now > 0:
                mu = mu * float(np.clip(1 + lam[s_] * (full / now - 1), .8, 1.4))
        out[s_] = nb(rng, mu, P["disp"][s_])
    out["fg3"] = np.minimum(out["fg3"], out["pts"] // 3)
    for c, parts in COMBO.items():
        out[c] = sum(out[x] for x in parts)
    out["dd"] = ((out["pts"] >= 10).astype(int) + (out["reb"] >= 10) + (out["ast"] >= 10) >= 2).astype(float)
    return out


def nb(rng, mu, k):
    """Negative binomial with mean mu and size k (variance mu + mu^2 / k)."""
    mu = np.maximum(mu, 1e-6)
    lam = rng.gamma(k, mu / k)
    return rng.poisson(lam).astype(float)


def sim_game(rng, ctx, res, home, away, spread_home, total, n, avail_h=None, avail_a=None, P=PARAMS, listed_only=False):
    q = res[np.abs(np.abs(res[:, 0]) - abs(spread_home)) <= 4]
    q = q if len(q) >= 150 else res
    e = q[rng.integers(0, len(q), n)]
    margin = spread_home + e[:, 1]
    tot = np.maximum(total + e[:, 2], np.abs(margin) + 150)
    hp, ap = (tot + margin) / 2, (tot - margin) / 2
    dh, da = ctx.roster(home, avail_h, listed_only), ctx.roster(away, avail_a, listed_only)
    if dh.empty or da.empty:
        return None
    sh = sim_team(rng, ctx, dh, home, away, hp, margin, tot, n, P)
    sa = sim_team(rng, ctx, da, away, home, ap, -margin, tot, n, P)
    return dict(margin=margin, total=tot, h_pts=hp, a_pts=ap, dfh=dh, dfa=da, sh=sh, sa=sa)
