"""Core NFL projection engine.

Pipeline: load -> walk-forward team ratings -> game script regression ->
player opportunity/efficiency -> correlated Monte Carlo.
The same `simulate_game` is used by backtest.py and report.py.
"""
import os
import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
STAT_COLS = ["player_id", "player_display_name", "position", "season", "week", "season_type", "team",
             "opponent_team", "completions", "attempts", "passing_yards", "passing_tds", "passing_interceptions",
             "sacks_suffered", "carries", "rushing_yards", "rushing_tds", "receptions", "targets",
             "receiving_yards", "receiving_tds", "receiving_air_yards", "receiving_yards_after_catch"]
SEASON_T = 25  # week index spacing, so 2026 W1 is 25 after 2025 W1
OTHER_T, OTHER_C = 0.07, 0.02  # share going to players below the modeling cutoff
DEFAULT_SHARE = {"WR": .11, "TE": .09, "RB": .06, "FB": .02}
K_T = float(os.environ.get("K_T", 18))  # prior strength in decay-weighted team targets
KAPPA_T = float(os.environ.get("KAPPA_T", 26))
SIGMA_P = float(os.environ.get("SIGMA_P", .12))
# observed game YPA residuals already include per-catch sampling noise, which the sim draws separately;
# only the remaining latent share is added, otherwise variance is double counted
LATENT = float(os.environ.get("LATENT", .35))
# factor study switches (comma list): posdef = no defence-vs-position factor, oppdef = opponent rated league average,
# recency = equal weights over the usage window instead of decay. posdef is off by default: the Nov 2026 factor study
# found it added nothing. Set ABLATE= and FIX= (empty) to reproduce the pre-Step-3 model.
ABLATE = set(filter(None, os.environ.get("ABLATE", "posdef").split(",")))
# Step 3 fixes, on by default after passing the backtest: qbeff = rebuild pass efficiency for a new starting QB,
# total = tie yards per attempt to the implied team total (TOT_K), vac = absent teammates' targets go to depth players
# rather than the remaining starters (VAC_K share of them)
FIX = set(filter(None, os.environ.get("FIX", "qbeff,total,vac").split(",")))
TOT_K = float(os.environ.get("TOT_K", .4))
VAC_K = float(os.environ.get("VAC_K", 1.0))
QB_A, QB_BACK = 150.0, 0.90  # attempts of prior weight; an unknown QB starts at 90% of league yards per attempt


def tidx(season, week):
    return (season - 2025) * SEASON_T + week


def load(seasons=(2025, 2026)):
    d = {}
    d["games"] = pd.read_csv(f"{DATA}/games.csv", low_memory=False)
    d["pbp"] = pd.concat([pd.read_csv(f"{DATA}/play_by_play_{y}.csv.gz", low_memory=False) for y in seasons])
    st = pd.concat([pd.read_csv(f"{DATA}/stats_player_week_{y}.csv", usecols=STAT_COLS) for y in seasons])
    st = st[st.team.notna()].rename(columns={"player_display_name": "name"})
    st["t"] = tidx(st.season, st.week)
    d["stats"] = st[st.season_type.isin(["REG", "POST"])]
    d["snaps"] = pd.concat([pd.read_csv(f"{DATA}/snap_counts_{y}.csv") for y in seasons])
    d["roster"] = pd.read_csv(f"{DATA}/roster_2026.csv", low_memory=False)
    d["prior_share"] = prior_shares()
    d["team_games"] = team_games(d["pbp"])
    d["usage"] = red_zone_usage(d["pbp"])
    d["pools"] = yard_pools(d["pbp"])
    d["pos_def"] = position_defense(d["pbp"], d["stats"])
    return d


def prior_shares():
    """{season: {player: share of team targets in games he played the previous season}}"""
    st = pd.concat([pd.read_csv(f"{DATA}/stats_player_week_{y}.csv", usecols=["player_id", "season", "week", "season_type", "team", "targets", "carries"]) for y in (2024, 2025)])
    st = st[(st.season_type == "REG") & st.team.notna()]
    tot = st.groupby(["season", "week", "team"]).targets.transform("sum")
    g = st.assign(tt=tot).groupby(["season", "player_id"]).agg(t=("targets", "sum"), tt=("tt", "sum"), n=("week", "size"))
    g = g[g.n >= 4]
    out = {}
    for (y, pid), r in g.iterrows():
        out.setdefault(y + 1, {})[pid] = r.t / max(r.tt, 1)
    return out


def team_games(pbp):
    p = pbp[pbp.season_type.isin(["REG", "POST"]) & pbp.posteam.notna() & pbp.defteam.notna()].copy()
    p["t"] = tidx(p.season, p.week)
    p = p[(p.play_type.isin(["pass", "run"])) & (p.two_point_attempt != 1)]
    g = p.groupby(["game_id", "t", "posteam", "defteam"]).agg(
        att=("pass_attempt", lambda s: 0), epa=("epa", "sum"), plays=("epa", "size"),
        home_team=("home_team", "first"), hs=("home_score", "first"), aw=("away_score", "first"),
        sacks=("sack", "sum"), ints=("interception", "sum"), pass_td=("pass_touchdown", "sum"),
        rush_td=("rush_touchdown", "sum"), xpass=("xpass", "mean"), pass_oe=("pass_oe", "mean"),
    ).reset_index().rename(columns={"posteam": "team", "defteam": "opp"})
    pa = p[(p.pass_attempt == 1) & (p.sack == 0)].groupby(["game_id", "posteam"]).agg(
        att=("pass_attempt", "sum"), rec_yds=("receiving_yards", "sum")).reset_index().rename(columns={"posteam": "team"})
    ra = p[(p.rush_attempt == 1) | (p.qb_scramble == 1)].groupby(["game_id", "posteam"]).agg(
        car=("yards_gained", "size"), rush_yds=("yards_gained", "sum")).reset_index().rename(columns={"posteam": "team"})
    g = g.drop(columns="att").merge(pa, on=["game_id", "team"]).merge(ra, on=["game_id", "team"])
    g["home"] = (g.team == g.home_team).astype(int)
    g["pf"] = np.where(g.home == 1, g.hs, g.aw)
    g["pa"] = np.where(g.home == 1, g.aw, g.hs)
    g["db"] = g.att + g.sacks
    g["margin"] = g.pf - g.pa
    g["epp"] = g.epa / g.plays
    return g.dropna(subset=["pf"])


def red_zone_usage(pbp):
    p = pbp[pbp.season_type.isin(["REG", "POST"])].copy()
    p["t"] = tidx(p.season, p.week)
    tg = p[(p.pass_attempt == 1) & p.receiver_player_id.notna()]
    rz_t = tg[tg.yardline_100 <= 20].groupby(["t", "posteam", "receiver_player_id"]).size().rename("rz_tgt")
    ez_t = tg[tg.yardline_100 - tg.air_yards.fillna(0) <= 0].groupby(["t", "posteam", "receiver_player_id"]).size().rename("ez_tgt")
    ru = p[(p.rush_attempt == 1) & p.rusher_player_id.notna()]
    i10 = ru[ru.yardline_100 <= 10].groupby(["t", "posteam", "rusher_player_id"]).size().rename("i10_car")
    i5 = ru[ru.yardline_100 <= 5].groupby(["t", "posteam", "rusher_player_id"]).size().rename("i5_car")
    for s in (rz_t, ez_t, i10, i5):
        s.index.names = ["t", "team", "player_id"]
    return pd.concat([rz_t, ez_t, i10, i5], axis=1).fillna(0).reset_index()


def yard_pools(pbp):
    """Empirical per-reception and per-carry yardage by role; preserves real skew/tails."""
    p = pbp[pbp.season_type == "REG"]
    c = p[(p.complete_pass == 1) & p.receiving_yards.notna()]
    pools = {
        "short": c[c.air_yards < 6].receiving_yards.values,
        "mid": c[(c.air_yards >= 6) & (c.air_yards < 13)].receiving_yards.values,
        "deep": c[c.air_yards >= 13].receiving_yards.values,
        "RB": p[(p.rush_attempt == 1) & (p.qb_scramble != 1) & (p.qb_dropback != 1)].yards_gained.dropna().values,
        "QB": p[p.qb_scramble == 1].yards_gained.dropna().values,
    }
    return {k: v.astype(float) for k, v in pools.items()}


def position_defense(pbp, stats):
    pos = stats.drop_duplicates("player_id").set_index("player_id").position
    p = pbp[(pbp.pass_attempt == 1) & pbp.receiver_player_id.notna() & pbp.season_type.isin(["REG", "POST"])].copy()
    p["t"] = tidx(p.season, p.week)
    p["pos"] = p.receiver_player_id.map(pos)
    p["yds"] = p.receiving_yards.fillna(0)
    return p[["t", "defteam", "pos", "yds"]]


def season_of(t):
    return 2025 + (t - 1) // SEASON_T


def decay(ts, t, r=0.90, prior_mult=0.45):
    w = r ** (t - ts)
    return np.where((t - ts) >= 12, w * prior_mult, w)


def ratings(tg, t, lam=4.0, epa_mix=0.5):
    """Opponent-adjusted offence/defence (ridge). y mixes points with EPA so one fluky score counts less."""
    h = tg[tg.t < t]
    teams = sorted(set(h.team) | set(h.opp))
    ix = {x: i for i, x in enumerate(teams)}
    n, m = len(h), len(teams)
    mu = h.pf.mean()
    y = (1 - epa_mix) * h.pf.values + epa_mix * (mu + h.epa.values * 0.9)
    X = np.zeros((n, 2 * m + 1))
    for r, (a, b, hm) in enumerate(zip(h.team, h.opp, h.home)):
        X[r, ix[a]] = 1
        X[r, m + ix[b]] = -1
        X[r, 2 * m] = hm - 0.5
    w = decay(h.t.values, t)
    A = (X.T * w) @ X + lam * np.eye(X.shape[1])
    A[2 * m, 2 * m] -= lam
    b = (X.T * w) @ (y - mu)
    c = np.linalg.solve(A, b)
    return {"mu": mu, "off": {k: c[i] for k, i in ix.items()}, "def": {k: c[m + i] for k, i in ix.items()}, "hfa": c[2 * m]}


def model_line(r, home, away, neutral=False):
    h = 0 if neutral else r["hfa"] / 2
    ph = r["mu"] + h + r["off"].get(home, 0) - r["def"].get(away, 0)
    pa = r["mu"] - h + r["off"].get(away, 0) - r["def"].get(home, 0)
    return ph, pa


def wavg(s, w):
    return float(np.sum(s * w) / max(np.sum(w), 1e-9))


def team_prior(tg, team, t):
    """Decay-weighted offensive and defensive rates known before time t, shrunk toward league."""
    h = tg[tg.t < t]
    lg = h[h.t >= t - 40] if len(h[h.t >= t - 40]) > 40 else h
    L = dict(pf=lg.pf.mean(), db=lg.db.mean(), car=lg.car.mean(), ypa=lg.rec_yds.sum() / lg.att.sum(), ypc=lg.rush_yds.sum() / lg.car.sum(),
             sack=lg.sacks.sum() / lg.db.sum(), intr=lg.ints.sum() / lg.att.sum(),
             pr=lg.pass_td.sum() / max(lg.pass_td.sum() + lg.rush_td.sum(), 1))
    o, d = h[h.team == team], h[h.opp == team]
    wo, wd = decay(o.t.values, t), decay(d.t.values, t)
    k = 3.0
    so, sd = wo.sum(), wd.sum()

    def sh(val, lv, s):
        return (val * s + lv * k) / (s + k)
    out = {
        "db": sh(wavg(o.db, wo), L["db"], so), "car": sh(wavg(o.car, wo), L["car"], so),
        "ypa": sh(wavg(o.rec_yds, wo) / max(wavg(o.att, wo), 1), L["ypa"], so),
        "ypc": sh(wavg(o.rush_yds, wo) / max(wavg(o.car, wo), 1), L["ypc"], so),
        "sack": sh(wavg(o.sacks, wo) / max(wavg(o.db, wo), 1), L["sack"], so),
        "intr": sh(wavg(o.ints, wo) / max(wavg(o.att, wo), 1), L["intr"], so),
        "ptd_share": sh(o.pass_td.values.dot(wo) / max((o.pass_td + o.rush_td).values.dot(wo), 1), L["pr"], so),
        "d_db": sh(wavg(d.db, wd), L["db"], sd), "d_car": sh(wavg(d.car, wd), L["car"], sd),
        "d_ypa": sh(wavg(d.rec_yds, wd) / max(wavg(d.att, wd), 1), L["ypa"], sd),
        "d_ypc": sh(wavg(d.rush_yds, wd) / max(wavg(d.car, wd), 1), L["ypc"], sd),
        "d_sack": sh(wavg(d.sacks, wd) / max(wavg(d.db, wd), 1), L["sack"], sd),
        "d_intr": sh(wavg(d.ints, wd) / max(wavg(d.att, wd), 1), L["intr"], sd),
        "n": len(o),
    }
    out["L"] = L
    return out


def script_rows(data, tmin, tmax):
    """Team-game rows with walk-forward priors, for fitting the script regressions."""
    tg, games = data["team_games"], data["games"]
    g = games[games.total_line.notna()].set_index("game_id")
    rows = []
    for r in tg[(tg.t >= tmin) & (tg.t <= tmax)].itertuples():
        if r.game_id not in g.index:
            continue
        a, b = team_prior(tg, r.team, r.t), team_prior(tg, r.opp, r.t)
        if a["n"] < 3 or b["n"] < 3:
            continue
        s, tot = g.at[r.game_id, "spread_line"], g.at[r.game_id, "total_line"]
        m = s if r.home else -s
        rows.append(dict(t=r.t, db=r.db, car=r.car, pf=r.pf, margin=r.margin, exp_pf=(tot + m) / 2, exp_m=m,
                         ypa=r.rec_yds / max(r.att, 1), ypc=r.rush_yds / max(r.car, 1),
                         p_db=a["db"], o_db=b["d_db"], p_car=a["car"], o_car=b["d_car"],
                         p_ypa=a["ypa"], o_ypa=b["d_ypa"], p_ypc=a["ypc"], o_ypc=b["d_ypc"],
                         L_ypa=a["L"]["ypa"], L_ypc=a["L"]["ypc"], L_db=a["L"]["db"], L_car=a["L"]["car"]))
    return pd.DataFrame(rows)


def fit_script(rows):
    def ols(X, y):
        X = np.column_stack([np.ones(len(X)), X])
        b = np.linalg.lstsq(X, y, rcond=None)[0]
        return b, float(np.std(y - X @ b))
    f = {}
    m, ex = rows.margin.values, rows.pf.values - rows.exp_pf.values
    f["db"] = ols(np.column_stack([rows.p_db - rows.L_db, rows.o_db - rows.L_db, m, rows.pf]), rows.db.values)
    f["car"] = ols(np.column_stack([rows.p_car - rows.L_car, rows.o_car - rows.L_car, m, rows.pf]), rows.car.values)
    f["ypa"] = ols(np.column_stack([np.log(rows.p_ypa / rows.L_ypa), np.log(rows.o_ypa / rows.L_ypa), ex]), np.log(rows.ypa.clip(1)).values - np.log(rows.L_ypa).values)
    f["ypc"] = ols(np.column_stack([np.log(rows.p_ypc / rows.L_ypc), np.log(rows.o_ypc / rows.L_ypc), ex]), np.log(rows.ypc.clip(1.5)).values - np.log(rows.L_ypc).values)
    return f


def residual_pairs(games):
    """Real (margin - spread, total - line) pairs: keeps key numbers and fat tails without assuming normality."""
    g = games[(games.season >= 2012) & games.result.notna() & games.spread_line.notna()]
    return np.column_stack([(g.result - g.spread_line).values, (g.total - g.total_line).values])


def player_table(data, team, t, opp, active=None, availability=None):
    """Opportunity shares and efficiency for one team. active: restrict to these ids (backtest)."""
    st = data["stats"]
    h = st[(st.team == team) & (st.t < t)]
    cur = h[h.season == season_of(t)].t.unique()
    # stay inside the current season once 4 games exist, so departed players cannot hold shares
    tgames = sorted(cur)[-10:] if len(cur) >= 4 else sorted(h.t.unique())[-10:]
    h = h[h.t.isin(tgames)]
    if h.empty:
        return pd.DataFrame()
    tot = h.groupby("t").agg(tt=("targets", "sum"), tc=("carries", "sum")).reset_index()
    h = h.merge(tot, on="t").merge(
        data["usage"].rename(columns={"t": "t"}), how="left", on=["t", "team", "player_id"]).fillna(
        {"rz_tgt": 0, "ez_tgt": 0, "i10_car": 0, "i5_car": 0})
    h["w"] = np.ones(len(h)) if "recency" in ABLATE else decay(h.t.values, t, r=0.86, prior_mult=0.5)
    rz = h.groupby("t").agg(trz=("rz_tgt", "sum"), ti10=("i10_car", "sum")).reset_index()
    h = h.merge(rz, on="t")
    # season_tg counts only games the player appeared in; missed games do not dilute shares
    rows = []
    last = max(tgames)
    for pid, x in h.groupby("player_id"):
        pos = x.position.iloc[-1]
        if pos not in ("WR", "TE", "RB", "FB", "QB", "HB"):
            continue
        if active is not None and pid not in active:
            continue
        w = x.w.values
        wt, wc = w.dot(x.tt), w.dot(x.tc)
        # shrink toward his own prior-year share, not a flat number, so stars keep star shares
        p0 = data["prior_share"].get(season_of(t), {}).get(pid, DEFAULT_SHARE.get(pos, .06))
        ts = (w.dot(x.targets) + K_T * p0) / (wt + K_T) if pos != "QB" else 0.0
        cs = (w.dot(x.carries) + 0.4) / (wc + 6.0)
        rzs = (w.dot(x.rz_tgt) + 0.3) / (w.dot(x.trz) + 8.0)
        i10s = (w.dot(x.i10_car) + 0.15) / (w.dot(x.ti10) + 5.0)
        tgt, rec, ry = x.targets.sum(), x.receptions.sum(), x.receiving_yards.sum()
        car, rush = x.carries.sum(), x.rushing_yards.sum()
        lg_cr = {"WR": .64, "TE": .71, "RB": .78}.get(pos, .65)
        lg_ypr = {"WR": 12.8, "TE": 10.4, "RB": 7.6}.get(pos, 10.0)
        cr = (rec + lg_cr * 30) / (tgt + 30)
        ypr = (ry + lg_ypr * 15) / (rec + 15)
        ypc = (rush + ({"QB": 5.3}.get(pos, 4.2)) * 60) / (car + 60)
        adot = x.receiving_air_yards.sum() / max(tgt, 1)
        rows.append(dict(player_id=pid, name=x.name.iloc[-1], pos=pos, g=len(x), tshare=ts, cshare=cs, rz=rzs, i10=i10s,
                         cr=cr, ypr=ypr, ypc=ypc, adot=adot,
                         pool="RB" if False else ("short" if adot < 5.5 else ("mid" if adot < 11.5 else "deep")),
                         attw=w.dot(x.attempts), last_seen=x.t.max()))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["play_p"] = 1.0
    if availability is not None:
        df["play_p"] = [availability.get((team, p), availability.get(p, 1.0)) for p in df.player_id]
        df = df[df.play_p > 0].copy()
    # QB: starter is the recent attempts leader among available QBs
    qbs = df[df.pos == "QB"].sort_values(["last_seen", "attw"], ascending=False)
    named = (availability or {}).get(("qb", team))
    if named is not None and named in set(qbs.player_id):
        qbs = qbs[qbs.player_id == named]  # live: depth chart / market names the starter, not last week's passer
    df["starter_qb"] = False
    if len(qbs):
        df.loc[qbs.index[0], "starter_qb"] = True
        df = df[(df.pos != "QB") | df.starter_qb].copy()
    # drop fringe bodies so the Other bucket carries them
    df = df[(df.tshare >= .022) | (df.cshare >= .035) | (df.pos == "QB")].copy()
    # position defence factor (yards per target allowed vs this position vs all positions)
    pdf = data["pos_def"]
    h2 = pdf[(pdf.defteam == opp) & (pdf.t < t)]
    w2 = decay(h2.t.values, t)
    fac = {}
    for p in ("WR", "TE", "RB"):
        s = h2.pos.values == p
        nn = w2[s].sum()
        lg = pdf[(pdf.pos == p) & (pdf.t < t)].yds.mean()
        allow = (w2[s].dot(h2.yds.values[s]) + lg * 45) / (nn + 45)
        fac[p] = allow / lg
    base = np.mean(list(fac.values()))
    df["pos_fac"] = 1.0 if "posdef" in ABLATE else [fac.get(p, base) / base for p in df.pos]
    return df.reset_index(drop=True)


def sim_offense(rng, data, tp, op, df, pf, margin, exp_pf, fit, n, kappa_t=None, kappa_c=18.0, sigma_p=None):
    """Simulate one offence given the game's points & margin draws."""
    L = tp["L"]
    kappa_t = KAPPA_T if kappa_t is None else kappa_t
    sigma_p = SIGMA_P if sigma_p is None else sigma_p
    b, s = fit["db"]
    db = b[0] + b[1] * (tp["db"] - L["db"]) + b[2] * (op["d_db"] - L["db"]) + b[3] * margin + b[4] * pf + rng.normal(0, s, n)
    b, s = fit["car"]
    car = b[0] + b[1] * (tp["car"] - L["car"]) + b[2] * (op["d_car"] - L["car"]) + b[3] * margin + b[4] * pf + rng.normal(0, s, n)
    db = np.maximum(np.rint(db), 8).astype(int)
    car = np.maximum(np.rint(car), 6).astype(int)
    srate = np.clip((tp["sack"] + op["d_sack"]) / 2, .02, .14)
    sacks = rng.binomial(db, srate)
    att = db - sacks
    b, s = fit["ypa"]
    ypa = L["ypa"] * np.exp(b[0] + b[1] * np.log(tp["ypa"] / L["ypa"]) + b[2] * np.log(op["d_ypa"] / L["ypa"]) + b[3] * (pf - exp_pf) + rng.normal(0, s * LATENT, n))
    if "total" in FIX:
        ypa = ypa * np.exp(TOT_K * (exp_pf - L["pf"]) / L["pf"])
    b, s = fit["ypc"]
    ypc = L["ypc"] * np.exp(b[0] + b[1] * np.log(tp["ypc"] / L["ypc"]) + b[2] * np.log(op["d_ypc"] / L["ypc"]) + b[3] * (pf - exp_pf) + rng.normal(0, s * LATENT, n))

    m = len(df)
    avail = (rng.random((n, m)) < df.play_p.values[None, :]).astype(float)
    avail[:, df.pos.values == "QB"] = 1.0
    ts = df.tshare.values.copy()
    oth_t = OTHER_T  # fixed: absent teammates' targets go to the active players, not to Other
    ts = ts / ts.sum() * (1 - oth_t)
    cs = df.cshare.values.copy()
    oth_c = OTHER_C
    cs = cs / cs.sum() * (1 - oth_c)

    def dm(pm, tot, kappa):
        g = rng.gamma(np.maximum(pm * kappa, 1e-3), 1.0)
        g = g / g.sum(1, keepdims=True)
        return rng.multinomial(tot, g)

    tg_tot = rng.binomial(att, 0.955)
    oth_n = np.full(n, oth_t)
    if "vac" in FIX:
        # targets of players missing from the table (inactive) or out in this draw: part stays with depth players
        gone = max(0.0, (1 - OTHER_T) - df.tshare.values.sum()) + (ts[None, :] * (1 - avail)).sum(1)
        oth_n = oth_n + VAC_K * gone
    pm = np.column_stack([ts[None, :] * avail, oth_n])
    targets = dm(pm / pm.sum(1, keepdims=True), tg_tot, kappa_t)[:, :m + 1]
    cr = df.cr.values
    rec_m = rng.binomial(targets[:, :m], cr)
    rec_o = rng.binomial(targets[:, m], .68)
    # scale player means so the team's per-attempt yardage matches the matchup-adjusted forecast
    pop = rng.lognormal(-sigma_p ** 2 / 2, sigma_p, (n, m))
    ypr_i = df.ypr.values * df.pos_fac.values
    exp_team = (ts * cr * ypr_i).sum() * .955 + oth_t * .955 * .68 * 9.5
    base = L["ypa"] * np.exp(fit["ypa"][0][0] + fit["ypa"][0][1] * np.log(tp["ypa"] / L["ypa"]) + fit["ypa"][0][2] * np.log(op["d_ypa"] / L["ypa"]))
    scale_team = ypa / exp_team
    rec_y = np.zeros((n, m))
    pools = data["pools"]
    for j in range(m):
        pool = pools[df["pool"].iat[j]]
        k = int(rec_m[:, j].max())
        if k == 0:
            continue
        draw = rng.choice(pool, (n, k))
        mask = np.arange(k)[None, :] < rec_m[:, j][:, None]
        rec_y[:, j] = (draw * mask).sum(1) * (ypr_i[j] / pool.mean()) * pop[:, j] * scale_team
    o_y = rec_o * 9.5 * scale_team

    pmc = np.column_stack([cs[None, :] * avail, np.full(n, oth_c)])
    cars = dm(pmc / pmc.sum(1, keepdims=True), car, kappa_c)
    exp_ypc = (cs * df.ypc.values).sum() + oth_c * 5
    scale_r = ypc / exp_ypc
    rush_y = np.zeros((n, m))
    for j in range(m):
        qb = df.pos.iat[j] == "QB"
        pool = pools["QB" if qb else "RB"]
        k = int(cars[:, j].max())
        if k == 0:
            continue
        draw = rng.choice(pool, (n, k))
        mask = np.arange(k)[None, :] < cars[:, j][:, None]
        rush_y[:, j] = (draw * mask).sum(1) * (df.ypc.values[j] / pool.mean()) * rng.lognormal(-.12 ** 2 / 2, .12, n) * scale_r

    # touchdowns: team count from points, split pass/rush, then to players by red-zone role and live volume
    lam = np.maximum(pf, 0) * 0.1035
    td = np.minimum(rng.poisson(lam), np.floor(np.maximum(pf, 0) / 6).astype(int))
    ptd = rng.binomial(td, np.clip(tp["ptd_share"], .35, .8))
    rtd = td - ptd
    vt = targets[:, :m] / np.maximum(ts * tg_tot.mean(), .3)
    vc = cars[:, :m] / np.maximum(cs * car.mean(), .3)
    wr = (0.55 * df.rz.values + 0.45 * ts)[None, :] * np.clip(vt, 0, 2.5) * (targets[:, :m] > 0)
    wc = (0.70 * df.i10.values + 0.30 * cs)[None, :] * np.clip(vc, 0, 2.5) * (cars[:, :m] > 0)
    wr = np.column_stack([wr, np.full(n, .03)])
    wc = np.column_stack([wc, np.full(n, .01)])
    ptds = rng.multinomial(ptd, wr / wr.sum(1, keepdims=True))[:, :m]
    rtds = rng.multinomial(rtd, wc / wc.sum(1, keepdims=True))[:, :m]
    comp = rec_m.sum(1) + rec_o
    pass_y = rec_y.sum(1) + o_y
    ints = rng.binomial(att, np.clip((tp["intr"] + op["d_intr"]) / 2, .004, .05))
    return dict(db=db, att=att, car=car, sacks=sacks, comp=comp, pass_y=pass_y, ptd=ptd, rtd=rtd, ints=ints,
                targets=targets[:, :m], rec=rec_m, rec_y=rec_y, cars=cars[:, :m], rush_y=rush_y, ptds=ptds, rtds=rtds)


def qb_adjust(data, team, t, df, tp):
    """New starter: scale team yards per attempt by his own shrunk YPA relative to the QB who built the team rate."""
    st = data["stats"]
    if df.empty or not df.starter_qb.any():
        return tp
    new = df.loc[df.starter_qb, "player_id"].iat[0]
    h = st[(st.team == team) & (st.t < t) & (st.attempts > 0)]
    if h.empty:
        return tp
    last = h[h.t == h.t.max()].sort_values("attempts").player_id.iat[-1]
    if last == new:
        return tp
    L = tp["L"]

    def own(pid, base):
        x = st[(st.player_id == pid) & (st.t < t)]
        return (x.passing_yards.sum() + base * QB_A) / (x.attempts.sum() + QB_A)
    r = own(new, L["ypa"] * QB_BACK) / own(last, L["ypa"])
    return {**tp, "ypa": tp["ypa"] * r}


def simulate_game(rng, data, home, away, spread_home, total, t, fit, resid, n=20000,
                  active_h=None, active_a=None, avail=None, wmodel=None, ratings_cache=None, model_lines=None):
    """spread_home: market home margin (+ = home favoured). wmodel: weight on our line vs market."""
    r = ratings_cache or ratings(data["team_games"], t)
    ph, pa = model_lines or model_line(r, home, away)
    w = 0.10 if wmodel is None else wmodel
    M = w * (ph - pa) + (1 - w) * spread_home
    T = w * (ph + pa) + (1 - w) * total
    e = resid[rng.integers(0, len(resid), n)]
    margin = M + e[:, 0]
    tot = np.maximum(T + e[:, 1], np.abs(margin))
    h_pts, a_pts = (tot + margin) / 2, (tot - margin) / 2
    h_tp, a_tp = team_prior(data["team_games"], home, t), team_prior(data["team_games"], away, t)
    dfh = player_table(data, home, t, away, active_h, avail)
    dfa = player_table(data, away, t, home, active_a, avail)
    if "qbeff" in FIX:
        h_tp, a_tp = qb_adjust(data, home, t, dfh, h_tp), qb_adjust(data, away, t, dfa, a_tp)
    if "oppdef" in ABLATE:
        def flat(tp):
            L = tp["L"]
            return {**tp, "d_db": L["db"], "d_car": L["car"], "d_ypa": L["ypa"], "d_ypc": L["ypc"], "d_sack": L["sack"], "d_intr": L["intr"]}
        h_tp, a_tp = flat(h_tp), flat(a_tp)
    sh = sim_offense(rng, data, h_tp, a_tp, dfh, h_pts, margin, (T + M) / 2, fit, n)
    sa = sim_offense(rng, data, a_tp, h_tp, dfa, a_pts, -margin, (T - M) / 2, fit, n)
    return dict(M=M, T=T, model_h=ph, model_a=pa, margin=margin, total=tot, h_pts=h_pts, a_pts=a_pts,
                dfh=dfh, dfa=dfa, sh=sh, sa=sa, h_tp=h_tp, a_tp=a_tp)
