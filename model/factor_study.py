"""Factor study: which inputs actually predict player outcomes, and which add nothing.

  python factor_study.py sims [tag]   walk-forward backtest rows; tag = base or an ABLATE switch (posdef, oppdef, recency)
  python factor_study.py report       score ablations and candidate factors, write output/factor_study.json
  python factor_study.py fixes        score Step 3 fixes against base, write output/step3_fixes.json

The engine's ABLATE and FIX environment switches pick the model variant. Their defaults are the current model,
so `sims base` runs it; the Nov 2026 study was run with ABLATE= FIX= (the pre-Step-3 model) as base.

Two questions, both scored out of sample on real results:
  1. Ablation: does a factor already in the model help? Re-run the backtest with it switched off.
  2. Candidate: does a factor outside the model add anything? Fit hit ~ model probability + factor on one
     season and score the other. If the model already uses the information, the factor adds nothing.
Uncertainty is a bootstrap over games, because thresholds and teammates within one game move together.
"""
import json
import os
import sys
import zlib
import numpy as np
import pandas as pd
import backtest as B
import engine as E
from config import COORD

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "fs")
WEEKS = [(2025, w) for w in range(5, 19)] + [(2026, w) for w in range(4, 9)]
STAT = {"rec": "receiving_yards", "rush": "rushing_yards", "pass": "passing_yards"}


def sims(tag):
    os.makedirs(OUT, exist_ok=True)
    df, gl = B.run(n=4000, seed=11, weeks=WEEKS)
    df.to_pickle(f"{OUT}/rows_{tag}.pkl")
    gl.to_csv(f"{OUT}/games_{tag}.csv", index=False)


def long_rows(df):
    """One row per player, market and threshold."""
    cols = ["season", "week", "game_id", "team", "opp", "pid", "pos", "mkt", "tshare", "cshare", "proj", "act"]
    out = [dict(zip(cols, r[:-1]), k=k, p=p) for r in df[cols + ["probs"]].itertuples(index=False) for k, p in r[-1].items()]
    L = pd.DataFrame(out)
    L["y"] = (L.act >= L.k).astype(int)
    L["t"] = E.tidx(L.season, L.week)
    return L


def km(a, b):
    (y1, x1), (y2, x2) = np.radians(COORD[a]), np.radians(COORD[b])
    h = np.sin((y2 - y1) / 2) ** 2 + np.cos(y1) * np.cos(y2) * np.sin((x2 - x1) / 2) ** 2
    return 12742 * np.arcsin(np.sqrt(h))


def features(L, d):
    """Pre-game candidate factors for each row. Missing values stay NaN; the fit adds a missing flag."""
    st, games, sn = d["stats"], d["games"].set_index("game_id"), d["snaps"]
    st = st[st.season_type == "REG"].copy()
    st["any_td"] = st.rushing_tds + st.receiving_tds
    # recent form: share of his last 5 / 10 games at or above this threshold, and last-5 average
    key = L[["pid", "t", "mkt"]].drop_duplicates()
    hist = {pid: x.sort_values("t") for pid, x in st[st.player_id.isin(set(L.pid))].groupby("player_id")}
    form = {}
    for pid, t, m in key.itertuples(index=False):
        x = hist.get(pid)
        col = STAT.get(m, "any_td")
        v = x[x.t < t][col].values if x is not None else np.array([])
        form[(pid, t, m)] = (v[-5:], v[-10:])
    def hr(r, j):
        v = form[(r.pid, r.t, r.mkt)][j]
        return np.nan if len(v) < 3 else (v >= r.k).mean()
    L["l5_hit"] = [hr(r, 0) for r in L.itertuples()]
    L["l10_hit"] = [hr(r, 1) for r in L.itertuples()]
    L["l5_n"] = [len(form[(r.pid, r.t, r.mkt)][0]) for r in L.itertuples()]
    avg5 = {k: (v[0].mean() if len(v[0]) >= 3 else np.nan) for k, v in form.items()}
    L["form_vs_model"] = [np.log((avg5[(r.pid, r.t, r.mkt)] + 5) / (r.proj + 5)) if r.mkt != "atd" else np.nan for r in L.itertuples()]

    # defence vs position, season to date (the "allows the most to TEs" stat), as a ratio to league and a rank
    pos = L.pos.replace({"HB": "RB", "FB": "RB"})
    L["pgrp"] = np.where(L.mkt == "pass", "PASS", np.where(L.mkt == "rush", np.where(pos == "QB", "QBR", "RB"), pos))
    st["pg"] = st.position.replace({"HB": "RB", "FB": "RB"})
    allowed = pd.concat([
        st[st.pg.isin(["WR", "TE", "RB"])].groupby(["opponent_team", "season", "t", "pg"]).receiving_yards.sum().rename("y").reset_index(),
        st[st.pg == "RB"].groupby(["opponent_team", "season", "t"]).rushing_yards.sum().rename("y").reset_index().assign(pg="RB_RUSH"),
        st[st.pg == "QB"].groupby(["opponent_team", "season", "t"]).rushing_yards.sum().rename("y").reset_index().assign(pg="QBR"),
        st.groupby(["opponent_team", "season", "t"]).passing_yards.sum().rename("y").reset_index().assign(pg="PASS")])
    grp = np.where((L.mkt == "rush") & (L.pgrp == "RB"), "RB_RUSH", L.pgrp)
    L["dgrp"] = grp
    dvp, rank = {}, {}
    for (s, t), _ in L.groupby(["season", "t"]):
        a = allowed[(allowed.season == s) & (allowed.t < t)]
        if a.t.nunique() < 3:
            continue
        per = a.groupby(["pg", "opponent_team"]).y.mean()
        for g, x in per.groupby(level=0):
            x = x.droplevel(0)
            r = (x / x.mean()).to_dict()
            rk = x.rank(ascending=False).to_dict()  # 1 = allows the most
            for tm in x.index:
                dvp[(t, g, tm)], rank[(t, g, tm)] = r[tm], rk[tm]
    L["dvp"] = [dvp.get((t, g, o), np.nan) for t, g, o in zip(L.t, L.dgrp, L.opp)]
    L["dvp_top8"] = [np.nan if (t, g, o) not in rank else float(rank[(t, g, o)] <= 8) for t, g, o in zip(L.t, L.dgrp, L.opp)]

    # overall opponent defence (points-based ratings the model already uses for its own line)
    rd = {t: E.ratings(d["team_games"], t)["def"] for t in L.t.unique()}
    L["opp_def"] = [-rd[t].get(o, 0) for t, o in zip(L.t, L.opp)]  # higher = worse defence

    # game environment from the closing market, weather, home, rest, travel, starting QB change
    gi = games.loc[L.game_id]
    home = (gi.home_team.values == L.team.values)
    m = np.where(home, gi.spread_line.values, -gi.spread_line.values)
    L["home"] = home.astype(float)
    L["spread_fav"] = m
    L["imp_total"] = (gi.total_line.values + m) / 2
    indoor = gi.roof.isin(["dome", "closed"]).values
    L["wind"] = np.where(indoor, 0, gi.wind.values)
    L["cold"] = np.where(indoor, 0, (gi.temp.values < 40).astype(float))
    L.loc[~indoor & gi.temp.isna().values, "cold"] = np.nan
    L["rest"] = np.where(home, gi.home_rest.values, gi.away_rest.values) - np.where(home, gi.away_rest.values, gi.home_rest.values)
    L["travel"] = [0.0 if h else km(tm, ht) / 1000 for h, tm, ht in zip(home, L.team, gi.home_team.values)]
    qb_now = np.where(home, gi.home_qb_id.values, gi.away_qb_id.values)
    g_all = d["games"][d["games"].result.notna()].sort_values(["season", "week"])
    prev_qb = {}
    for g in g_all.itertuples():
        for tm, q in ((g.home_team, g.home_qb_id), (g.away_team, g.away_qb_id)):
            prev_qb.setdefault(tm, []).append((E.tidx(g.season, g.week), q))
    def last_qb(tm, t):
        x = [q for tt, q in prev_qb.get(tm, []) if tt < t]
        return x[-1] if x else None
    L["qb_change"] = [float(last_qb(tm, t) not in (None, q)) for tm, t, q in zip(L.team, L.t, qb_now)]

    # vacated volume: last-4-game share of targets (carries for rushing) held by teammates who did not play
    vac = {}
    for (tm, s, w, t), _ in L.groupby(["team", "season", "week", "t"]):
        h = st[(st.team == tm) & (st.t < t)]
        last4 = sorted(h.t.unique())[-4:]
        h = h[h.t.isin(last4)]
        on = set(sn[(sn.season == s) & (sn.week == w) & (sn.team == tm) & (sn.offense_snaps > 0)].player)
        out = ~h.name.isin(on)
        vac[(tm, t)] = (h.targets[out].sum() / max(h.targets.sum(), 1), h.carries[out].sum() / max(h.carries.sum(), 1))
    L["vacated"] = [vac[(tm, t)][1 if m == "rush" else 0] for tm, t, m in zip(L.team, L.t, L.mkt)]
    return L


FACTORS = {"l5_hit": "Last-5 hit rate at this line", "l10_hit": "Last-10 hit rate at this line",
           "form_vs_model": "Last-5 average vs our projection", "dvp": "Opponent yards allowed to this position (ratio to league)",
           "dvp_top8": "Opponent is a top-8 'worst vs position' defence", "opp_def": "Opponent overall defence rating",
           "imp_total": "Team implied total (market)", "spread_fav": "Point spread (team favoured by)",
           "wind": "Wind (mph, outdoors)", "cold": "Cold (<40F, outdoors)", "home": "Home team", "rest": "Rest days vs opponent",
           "travel": "Travel distance (1000 km)", "qb_change": "Starting QB changed from last game", "vacated": "Teammate volume vacated by absences"}


# weather is only recorded for part of 2026, so it is scored on games where it is known rather than with a missing flag
COMPLETE = {"wind", "cold"}


def logit(p):
    p = np.clip(p, .003, .997)
    return np.log(p / (1 - p))


def fit_lr(X, y, lam=1e-3):
    b = np.zeros(X.shape[1])
    for _ in range(50):
        q = 1 / (1 + np.exp(-X @ b))
        g = X.T @ (y - q) - lam * b
        H = (X.T * (q * (1 - q))) @ X + lam * np.eye(len(b))
        step = np.linalg.solve(H, g)
        b += step
        if np.abs(step).max() < 1e-8:
            break
    return b


def design(x, f=None, stats=None):
    z = logit(x.p.values)
    if f is None:
        return np.column_stack([np.ones(len(x)), z]), None
    v = x[f].values.astype(float)
    miss = np.isnan(v)
    if stats is None:
        stats = (np.nanmean(v), np.nanstd(v) or 1.0)
    v = np.where(miss, 0, (v - stats[0]) / stats[1])
    cols = [np.ones(len(x)), z, v] + ([miss.astype(float)] if miss.any() else [])
    return np.column_stack(cols), stats


def ll(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def boot(diff, gid, n=2000, seed=5):
    """Mean of a per-row difference with a 90% interval from resampling whole games."""
    g = pd.DataFrame({"d": diff, "g": gid}).groupby("g").d.agg(["sum", "size"])
    w = np.random.default_rng(seed).multinomial(len(g), np.full(len(g), 1 / len(g)), n)
    m = (w @ g["sum"].values) / (w @ g["size"].values)
    return float(diff.mean()), float(np.quantile(m, .05)), float(np.quantile(m, .95))


def add_value(L, f, tr, te):
    """Log-loss gain (per 1000 rows) on the test rows from adding factor f to the calibrated model."""
    if f in COMPLETE:
        L = L[L[f].notna()]
        tr, te = tr[L.index], te[L.index]
    a, b = L[tr.values], L[te.values]
    if len(a) < 200 or len(b) < 200 or a[f].nunique() < 2:
        return None
    Xa, _ = design(a)
    Xb, _ = design(b)
    p0 = 1 / (1 + np.exp(-Xb @ fit_lr(Xa, a.y.values)))
    Xa2, s = design(a, f)
    Xb2, _ = design(b, f, s)
    if Xb2.shape[1] != Xa2.shape[1]:  # missing flag only in one season
        k = min(Xa2.shape[1], Xb2.shape[1])
        Xa2, Xb2 = Xa2[:, :k], Xb2[:, :k]
    coef = fit_lr(Xa2, a.y.values)
    p1 = 1 / (1 + np.exp(-Xb2 @ coef))
    gain = (ll(p0, b.y.values) - ll(p1, b.y.values)) * 1000
    m, lo, hi = boot(gain, b.game_id.values)
    return dict(gain=m, lo=lo, hi=hi, coef=float(coef[2]), n=len(b), pp_shift=float(np.abs(p1 - p0).mean() * 100))


def low_usage(x):
    """Depth players: under 15% of targets for receiving, under 30% of carries for rushing."""
    return np.where(x.mkt == "rush", x.cshare < LOW_C, x.tshare < LOW_T)


LOW_T, LOW_C = .15, .30


def cal_design(x, tiered):
    X = design(x)[0]
    if not tiered:
        return X
    lo = low_usage(x).astype(float)
    return np.column_stack([X, lo, lo * X[:, 1]])


def calibrated(L, tiered=False):
    """Calibrated probability: map fit on 2025 applied to 2026, and on 2026 applied to 2025 (each out of sample).
    tiered adds a separate intercept and slope for low-usage players in receiving and rushing."""
    pc = np.zeros(len(L))
    for m, x in L.groupby("mkt"):
        tr = tiered and m in ("rec", "rush")
        for train, test in ((2025, 2026), (2026, 2025)):
            a, ib = x[x.season == train], (L.mkt == m) & (L.season == test)
            pc[ib.values] = 1 / (1 + np.exp(-cal_design(L[ib], tr) @ fit_lr(cal_design(a, tr), a.y.values)))
    return pc


def paired(j, a, b):
    """Log-loss gain per 1000 lines of column b over column a (+ = b better), by market and season."""
    out = {}
    for m, x in list(j.groupby("mkt")) + [("all", j)]:
        for s in (2025, 2026):
            y = x[x.season == s]
            gain = (ll(y[a].values, y.y.values) - ll(y[b].values, y.y.values)) * 1000
            mm, lo, hi = boot(gain, y.game_id.values)
            out[f"{m}_{s}"] = dict(gain=mm, lo=lo, hi=hi, n=len(y))
    return out


def fixes(tags=("qbeff", "total", "total7", "vac", "vac7", "vac10", "posdef", "combo", "combo10")):
    """Step 3: each fix against the base run on the same random draws, plus its effect on the rows it targets."""
    d = E.load()
    base = features(long_rows(pd.read_pickle(f"{OUT}/rows_base.pkl")), d)
    base["pc"] = calibrated(base)
    base["pc_tier"] = calibrated(base, tiered=True)
    key = ["game_id", "pid", "mkt", "k"]
    res = {"lowusage_cal": paired(base, "pc", "pc_tier"), "fix": {}, "targets": {}}
    for name in TARGETS:
        x = base[target_mask(name, base).values]
        res["targets"][name] = {"n": int(len(x)), "hit": float(x.y.mean()), "base": float(x.pc.mean()), "tiered_cal": float(x.pc_tier.mean())}
    for tag in tags:
        f = f"{OUT}/rows_{tag}.pkl"
        if not os.path.exists(f):
            continue
        A = long_rows(pd.read_pickle(f))
        A["pc"] = calibrated(A)
        j = base.merge(A[key + ["pc"]], on=key, suffixes=("", "_fix"))
        res["fix"][tag] = paired(j, "pc", "pc_fix")
        for name in TARGETS:
            res["targets"][name][tag] = float(j[target_mask(name, j).values].pc_fix.mean())
    json.dump(res, open(f"{OUT}/../step3_fixes.json", "w"), indent=1, default=float)
    return res


TARGETS = ("qb", "low_total_pass", "vacated_rec", "low_usage")


def target_mask(name, x):
    return {"qb": x.qb_change == 1, "low_total_pass": (x.mkt == "pass") & (x.imp_total <= 19.75),
            "vacated_rec": (x.mkt == "rec") & (x.vacated > .275),
            "low_usage": pd.Series(low_usage(x), index=x.index) & x.mkt.isin(["rec", "rush"])}[name]


def report():
    d = E.load()
    base = long_rows(pd.read_pickle(f"{OUT}/rows_base.pkl"))
    base = features(base, d)
    base["pc"] = calibrated(base)
    base.to_pickle(f"{OUT}/long_base.pkl")
    res = {"rows": int(len(base)), "games": int(base.game_id.nunique()), "weeks": f"2025 W5-18, 2026 W4-8"}

    # 1. ablations of factors inside the model, paired with the base run on the same draws
    key = ["game_id", "pid", "mkt", "k"]
    res["ablation"] = {}
    for tag in ("posdef", "oppdef", "recency"):
        f = f"{OUT}/rows_{tag}.pkl"
        if not os.path.exists(f):
            continue
        A = long_rows(pd.read_pickle(f))
        A["pc"] = calibrated(A)
        j = base[key + ["season", "y", "pc"]].merge(A[key + ["pc"]], on=key, suffixes=("", "_off"))
        out = {}
        for m, x in list(j.groupby("mkt")) + [("all", j)]:
            for s in (2025, 2026):
                y = x[x.season == s]
                gain = (ll(y.pc_off.values, y.y.values) - ll(y.pc.values, y.y.values)) * 1000  # + = factor helps
                mm, lo, hi = boot(gain, y.game_id.values)
                out[f"{m}_{s}"] = dict(gain=mm, lo=lo, hi=hi, n=len(y))
        res["ablation"][tag] = out

    # 2. candidate factors: added on top of the calibrated model, fit one season, score the other
    res["added"] = {}
    for m, x in base.groupby("mkt"):
        for f in FACTORS:
            if x[f].notna().sum() < 200 or x[f].nunique() < 2:
                continue
            # forward (2025 -> 2026), reverse, and a pooled two-fold split by game
            fold = x.game_id.map(lambda g: zlib.crc32(g.encode()) % 2) == 0
            res["added"][f"{m}|{f}"] = {"fwd": add_value(x, f, x.season == 2025, x.season == 2026), "rev": add_value(x, f, x.season == 2026, x.season == 2025),
                                        "cv1": add_value(x, f, fold, ~fold), "cv2": add_value(x, f, ~fold, fold)}

    # 3. what the factor looks like on its own: model miss (hit rate minus calibrated probability) by factor bucket
    res["buckets"] = {}
    for m, x in base.groupby("mkt"):
        for f in FACTORS:
            v = x[f]
            if v.notna().sum() < 200 or v.nunique() < 2:
                continue
            q = pd.qcut(v.rank(method="first"), 4, labels=False) if v.nunique() > 4 else v
            g = x.assign(b=q).dropna(subset=["b"]).groupby("b").agg(lo=(f, "min"), hi=(f, "max"), n=("y", "size"), hit=("y", "mean"), pred=("pc", "mean"))
            res["buckets"][f"{m}|{f}"] = g.reset_index().to_dict("records")

    # 4. last-5 / last-10 hit rate used alone versus the model, on the same rows
    res["alone"] = {}
    for m, x in base[base.mkt != "atd"].groupby("mkt"):
        x = x[x.l10_hit.notna() & (x.l5_n >= 5)]
        for s in (2025, 2026):
            y = x[x.season == s]
            sh5 = (y.l5_hit * 5 + y.groupby("k").y.transform("mean") * 2) / 7  # light shrink so 0/5 is not 0%
            res["alone"][f"{m}_{s}"] = dict(n=len(y), model=float(((y.pc - y.y) ** 2).mean()), l5=float(((y.l5_hit - y.y) ** 2).mean()),
                                            l5_shrunk=float(((sh5 - y.y) ** 2).mean()), l10=float(((y.l10_hit - y.y) ** 2).mean()),
                                            base=float(((y.groupby("k").y.transform("mean") - y.y) ** 2).mean()))

    # 5. calibration where it matters for picks: by usage tier and in the high band
    def rel(x):
        b = pd.cut(x.pc, [0, .1, .3, .5, .7, .85, 1], include_lowest=True)
        return x.groupby(b, observed=True).agg(n=("y", "size"), pred=("pc", "mean"), hit=("y", "mean")).reset_index().assign(pc=lambda r: r.pc.astype(str)).to_dict("records")
    rec = base[base.mkt == "rec"]
    tier = pd.cut(rec.tshare, [0, .08, .15, .22, 1], labels=["<8%", "8-15%", "15-22%", "22%+"])
    res["tiers_rec"] = {str(t): rel(x) for t, x in rec.groupby(tier, observed=True)}
    res["rel_all"] = {m: rel(x) for m, x in base.groupby("mkt")}
    res["rel_all"]["all"] = rel(base)
    json.dump(res, open(f"{OUT}/../factor_study.json", "w"), indent=1, default=float)
    return res


if __name__ == "__main__":
    if sys.argv[1] == "sims":
        sims(sys.argv[2] if len(sys.argv) > 2 else "base")
    elif sys.argv[1] == "fixes":
        fixes()
    else:
        report()
