"""College model. Reuses the NFL simulator; everything college-specific (data shape, shares, script, injuries) lives here."""
import html
import json
import os
import re
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import engine as E  # noqa: E402

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(os.path.join(D, "..", "output"), exist_ok=True)
E.K_T = 18.0


def t_of(season, week, stype=2):
    return E.tidx(season, week if stype == 2 else 17 + (week or 1))


def load():
    g = pd.read_csv(f"{D}/games.csv")
    g["t"] = [t_of(a, b, c) for a, b, c in zip(g.season, g.week, g.stype)]
    p = pd.read_csv(f"{D}/plays.csv.gz", low_memory=False).merge(g[["game_id", "t", "season", "week", "home", "away", "neutral"]], on="game_id")
    pl = pd.read_csv(f"{D}/players.csv.gz").merge(g[["game_id", "t", "season", "week", "date"]], on="game_id")
    pl["opp"] = pl.game_id.map(dict(zip(g.game_id, zip(g.home, g.away))))
    pl["opp"] = [b if tm == a else a for tm, (a, b) in zip(pl.team, pl.opp)]
    d = dict(games=g, plays=p, players=pl)
    d["team_games"] = team_games(g, p)
    d["pools"] = pools(p)
    d["fbs"] = set(pd.concat([g[g.spread_line.notna()].home, g[g.spread_line.notna()].away]).value_counts().loc[lambda s: s >= 6].index)
    # position by usage: college box scores carry no positions
    tot = pl[pl.season >= 2025].groupby("pid").agg(att=("att", "sum"), car=("car", "sum"), tg=("targets", "sum"))
    d["pos"] = {i: ("QB" if r.att >= max(r.car, 8) or (r.att >= 4 and r.tg == 0) else ("RB" if r.car >= r.tg * .9 and r.car >= 3 else "WR")) for i, r in tot.iterrows()}
    d["prior"] = prior_share(pl)
    return d


def team_games(g, p):
    s = p[p.ty != "x"]
    rows = []
    a = s.groupby(["game_id", "off"]).agg(plays=("yds", "size"), sacks=("sack", "sum"), att=("comp", "count"), comp=("comp", "sum"), success=("yds", "size"))
    rec = s[(s.pass_play == 1) & (s.sack == 0)].groupby(["game_id", "off"]).yds.sum().rename("rec_yds")
    ru = s[(s.pass_play == 0)].groupby(["game_id", "off"]).agg(car=("yds", "size"), rush_yds=("yds", "sum"))
    ty = s.assign(pt=(s.ty == "Passing Touchdown"), rt=(s.ty == "Rushing Touchdown"), it=s.ty.str.contains("Interception")).groupby(["game_id", "off"]).agg(pass_td=("pt", "sum"), rush_td=("rt", "sum"), ints=("it", "sum"))
    x = a.join(rec).join(ru).join(ty).fillna(0).reset_index().rename(columns={"off": "team"})
    x["db"] = x.att + x.sacks
    gi = g.set_index("game_id")
    x = x[x.game_id.isin(gi.index)]
    x["t"] = x.game_id.map(gi.t)
    hm, aw, hs, asc, ne = x.game_id.map(gi.home), x.game_id.map(gi.away), x.game_id.map(gi.hs), x.game_id.map(gi.aw), x.game_id.map(gi.neutral)
    x["home"] = (x.team == hm).astype(int)
    x["opp"] = np.where(x.home == 1, aw, hm)
    x["pf"] = np.where(x.home == 1, hs, asc)
    x["pa"] = np.where(x.home == 1, asc, hs)
    x["margin"] = x.pf - x.pa
    x["neutral"] = ne.astype(bool)
    return x.dropna(subset=["opp"])


def pools(p):
    c = p[(p.comp == 1) & (p.sack == 0)]
    qb_like = p[(p.scramble == 1)]
    return {"short": c[c.depth == "short"].yds.values.astype(float), "mid": c.yds.values.astype(float), "deep": c[c.depth == "deep"].yds.values.astype(float),
            "RB": p[(p.pass_play == 0) & (p.scramble != 1) & p.rusher.notna()].yds.values.astype(float), "QB": (qb_like.yds.values.astype(float) if len(qb_like) > 200 else p[(p.pass_play == 0)].yds.values.astype(float))}


def prior_share(pl):
    x = pl[pl.season == 2025]
    tt = x.groupby(["game_id", "team"]).targets.transform("sum")
    g = x.assign(tt=tt).groupby("pid").agg(t=("targets", "sum"), tt=("tt", "sum"), n=("game_id", "size"))
    g = g[g.n >= 4]
    return (g.t / g.tt.clip(1)).to_dict()


def ratings(d, t, lam=2.5):
    """Margin/points ridge. Non-FBS teams share one rating so an FCS blowout is not read as a great team."""
    tg = d["team_games"]
    h = tg[tg.t < t].copy()
    f = lambda s: s if s in d["fbs"] else "FCS"
    teams = sorted(set(map(f, h.team)) | set(map(f, h.opp)))
    ix = {x: i for i, x in enumerate(teams)}
    m = len(teams)
    X = np.zeros((len(h), 2 * m + 1))
    for r, (a, b, hm, ne) in enumerate(zip(h.team, h.opp, h.home, h.neutral)):
        X[r, ix[f(a)]] = 1
        X[r, m + ix[f(b)]] = -1
        X[r, 2 * m] = 0 if ne else hm - .5
    w = E.decay(h.t.values, t, r=.93, prior_mult=.5)
    mu = float((h.pf * w).sum() / w.sum())
    A = (X.T * w) @ X + lam * np.eye(X.shape[1])
    A[2 * m, 2 * m] -= lam
    c = np.linalg.solve(A, (X.T * w) @ (h.pf.clip(upper=56).values - mu))
    return {"mu": mu, "off": {k: c[i] for k, i in ix.items()}, "def": {k: c[m + i] for k, i in ix.items()}, "hfa": float(c[2 * m])}


def line(r, home, away, neutral):
    g = lambda k, s: k.get(s, k.get("FCS", 0))
    hf = 0 if neutral else r["hfa"] / 2
    return r["mu"] + hf + g(r["off"], home) - g(r["def"], away), r["mu"] - hf + g(r["off"], away) - g(r["def"], home)


def residuals(g):
    x = g[g.spread_line.notna() & g.total_line.notna()]
    return np.column_stack([x.spread_line.abs(), (x.hs - x.aw - x.spread_line).values * np.sign(x.spread_line.replace(0, 1)), (x.hs + x.aw - x.total_line).values, np.sign(x.spread_line.replace(0, 1))])


QB_PICK = {}


def table(d, team, t, opp, availability=None):
    """Who is eligible and their shares. No hindsight: same rule live and in the backtest."""
    pl = d["players"]
    h = pl[(pl.team == team) & (pl.t < t)]
    cur = h[h.season == E.season_of(t)]
    if cur.t.nunique() >= 3:
        h = cur
    keep = sorted(h.t.unique())[-10:]
    h = h[h.t.isin(keep)]
    if h.empty:
        return pd.DataFrame()
    recent = keep[-3:]
    tot = h.groupby("t").agg(tt=("targets", "sum"), tc=("car", "sum"), trz=("rz_tgt", "sum"), ti10=("i10_car", "sum")).reset_index()
    h = h.merge(tot, on="t")
    h["w"] = E.decay(h.t.values, t, r=.86, prior_mult=.5)
    rows = []
    for pid, x in h.groupby("pid"):
        if not ((x.targets + x.car + x.att) > 0).any():
            continue
        if not x.t.isin(recent).any():
            continue  # absent three straight games: treat as gone, as we would live
        pos = d["pos"].get(pid, "WR")
        w = x.w.values
        p0 = d["prior"].get(pid, {"WR": .10, "RB": .05}.get(pos, 0)) if E.season_of(t) == 2026 else {"WR": .10, "RB": .05}.get(pos, 0)
        ts = (w.dot(x.targets) + E.K_T * p0) / (w.dot(x.tt) + E.K_T) if pos != "QB" else 0.0
        cs = (w.dot(x.car) + .4) / (w.dot(x.tc) + 6)
        tg, rc, ry, car, ru = x.targets.sum(), x.rec.sum(), x.rec_y.sum(), x.car.sum(), x.rush_y.sum()
        deep = x.deep_t.sum() / max(tg, 1)
        cr, lg_y = ({"RB": .76}.get(pos, .62), {"RB": 7.8}.get(pos, 13.0))
        rows.append(dict(player_id=pid, name=x.name.dropna().iloc[-1] if x.name.notna().any() else str(pid), pos=pos, g=len(x), tshare=ts, cshare=cs, rz=(w.dot(x.rz_tgt) + .3) / (w.dot(x.trz) + 8), i10=(w.dot(x.i10_car) + .15) / (w.dot(x.ti10) + 5),
                         cr=min((rc + cr * 30) / (max(tg, rc) + 30), .92), ypr=(ry + lg_y * 15) / (rc + 15), ypc=(ru + {"QB": 4.2}.get(pos, 4.7) * 60) / (car + 60), adot=deep,
                         pool="deep" if deep >= .3 else ("short" if ry / max(rc, 1) < 9 else "mid"), attw=w.dot(x.att), last_att=x[x.t == x.t.max()].att.sum(), last_seen=x.t.max()))
    df = pd.DataFrame(rows)
    df["play_p"] = 1.0
    av = availability or {}
    if av:
        df["play_p"] = [av.get((team, n_), 1.0) for n_ in df.name]
        df = df[df.play_p > 0].copy()
    pick = QB_PICK.get(team)  # reported starter beats usage; option QBs out-carry their attempts and read as RB
    if pick:
        df.loc[df.name == pick, "pos"] = "QB"
    qb = df[df.pos == "QB"].assign(is_pick=lambda y: y.name == pick).sort_values(["is_pick", "last_seen", "last_att", "attw"], ascending=False)
    df["starter_qb"] = False
    if len(qb):
        df.loc[qb.index[0], "starter_qb"] = True
    df = df[(df.pos != "QB") | df.starter_qb].copy()
    df = df[(df.tshare >= .03) | (df.cshare >= .05) | (df.pos == "QB")].copy()
    df["pos_fac"] = 1.0
    if not (df.pos == "QB").any():
        return pd.DataFrame()
    return df.reset_index(drop=True)


def script_rows(d, tmin, tmax):
    data = dict(team_games=d["team_games"], games=d["games"])
    return E.script_rows(data, tmin, tmax).replace([np.inf, -np.inf], np.nan).dropna()


class Sim:
    def __init__(self, d, fit, res, wmodel):
        self.d, self.fit, self.res, self.w = d, fit, res, wmodel

    def game(self, rng, home, away, spread, total, neutral, t, r, n, avail=None):
        ph, pa = line(r, home, away, neutral)
        M = self.w * (ph - pa) + (1 - self.w) * spread
        T = self.w * (ph + pa) + (1 - self.w) * total
        q = self.res[np.abs(self.res[:, 0] - abs(spread)) <= 6]
        q = q if len(q) >= 80 else self.res
        e = q[rng.integers(0, len(q), n)]
        margin = M + e[:, 1] * (np.sign(spread) or 1)
        tot = np.maximum(T + e[:, 2], np.abs(margin))
        h_pts, a_pts = (tot + margin) / 2, (tot - margin) / 2
        tg = self.d["team_games"]
        ht, at = E.team_prior(tg, home, t), E.team_prior(tg, away, t)
        dh, da = table(self.d, home, t, away, avail), table(self.d, away, t, home, avail)
        if dh.empty or da.empty:
            return None
        sh = E.sim_offense(rng, self.d, ht, at, dh, h_pts, margin, (T + M) / 2, self.fit, n)
        sa = E.sim_offense(rng, self.d, at, ht, da, a_pts, -margin, (T - M) / 2, self.fit, n)
        return dict(M=M, T=T, model_h=ph, model_a=pa, margin=margin, total=tot, h_pts=h_pts, a_pts=a_pts, dfh=dh, dfa=da, sh=sh, sa=sa, h_tp=ht, a_tp=at)


def covers_injuries():
    """Parse the Covers NCAAF injury page into {team name: rows}."""
    t = open(f"{D}/covers_inj.html", encoding="utf8", errors="ignore").read()
    t = re.sub(r"<script.*?</script>|<style.*?</style>", "", t, flags=re.S)
    s = html.unescape(re.sub(r"[\s|]*\|[\s|]*", "|", re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "|", t))))
    out = {}
    for m in re.finditer(r"\|([^|]{3,30})\|([^|]{3,30})\|(?:\(\d+\)\|)?Player\|POS\|Status\|", s):
        team = m.group(1).strip()
        nx = re.search(r"\|[^|]{3,30}\|[^|]{3,30}\|(?:\(\d+\)\|)?Player\|POS\|Status\|", s[m.end():])
        blk = s[m.end():m.end() + nx.start()] if nx else s[m.end():m.end() + 6000]
        rows = []
        for r in re.finditer(r"([A-Z][\w'.]*\.? ?[A-Z][\w'.\- ]*)\|([A-Z]{1,4})\|((?:Out|Questionable|Doubtful|Probable|Day-to-Day|Suspended|Injured|Game Time Decision|IR)[^|]*)\|\( ?([^)]*)\)\|([^|]*)", blk):
            rows.append(dict(name=re.sub(r"\s+", " ", r.group(1)).strip(), pos=r.group(2), status=r.group(3).strip(), date=r.group(4).strip(), note=r.group(5).strip()))
        out[team] = rows
    return out


def status_p(st):
    s = st.lower()
    if s.startswith(("out", "ir", "suspended", "injured")):
        return 0.0
    if s.startswith("doubtful"):
        return .15
    if s.startswith("questionable") or "decision" in s:
        return .6
    if s.startswith("probable") or s.startswith("day"):
        return .85
    return .8
