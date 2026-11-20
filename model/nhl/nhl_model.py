"""NHL game simulation, centred on the DraftKings price.

How it works:
- The market's no-vig moneyline (OT and shootout included) and its total are turned into two scoring rates (home and away goals
  per 60 minutes). fit() finds the rates whose simulated games reproduce the market's win chance and over chance.
- sim() then plays each game minute by minute:
  - score effects: the trailing team presses and the leading team sits back;
  - the goalie pull: an empty net late when down by one or two;
  - overtime: 5 minutes 3-on-3 in the regular season, then a shootout. In the playoffs, sudden-death 20-minute periods.
- Settlement follows DraftKings: a shootout adds one goal to the winner's score and to the game total, so a shootout or OT win
  never covers -1.5.
- From that one distribution come the puck line, the chance of OT and the regulation-only splits. Player props are built on the
  same simulated goals.

The research study (briefs/nhl/research.md) found that nothing beats the closing moneyline or total, so the model does not move
them: the win chance and the over chance equal the market's. What the simulation adds is everything the market price implies
but does not post (OT chance, regulation splits, the puck line from the same game, and player props tied to the same game).
"""
import numpy as np

# Score effects and goalie pull: shape parameters, tuned in backtest_nhl.py on 2021-24 and checked on 2024-25 onward
# Score effects, goalie pull, overtime: tuned in backtest_nhl.py on 2021-22 to 2023-24 (refit 2026-11-20 after the skeptic review, to
# match the overtime rate, the share of overtimes that reach a shootout, and empty-net goals per game), checked on 2024-25 onward.
P = dict(trail=0.277, lead=0.070,           # trailing team scores 28% more often, leading team 7% less
         en_min=57.0,                       # goalie pulled for the last 3 minutes when down 1 (the last 4 when down 2)
         en_for=0.052, en_against=0.213,    # goals per minute with the net empty: trailing team, leading team
         ot_rate=0.254, so_skill=0.5)       # 3-on-3 OT goals per minute (both teams), shootout edge from team strength
MIN = 60


def implied(a):
    """Implied chance from an American price."""
    a = float(a)
    return 100 / (a + 100) if a > 0 else -a / (-a + 100)


def novig(a, b):
    pa, pb = implied(a), implied(b)
    return pa / (pa + pb)


class Draws:
    """Fixed random numbers for one game, so fit() moves smoothly as the rates change (common random numbers)."""

    def __init__(self, n, seed):
        r = np.random.default_rng(seed)
        self.n = n
        self.u = r.random((MIN, n, 2))
        self.u2 = r.random((MIN, n, 2))   # a second goal in the same minute (rare, keeps the counts Poisson-like)
        self.ot = r.exponential(1.0, n)
        self.side = r.random(n)
        self.so = r.random(n)


def sim(mu_h, mu_a, d, playoff=False, p=P):
    """Play n games. mu_* are 60-minute goal rates at even score. Returns a dict of arrays."""
    n = d.n
    h = np.zeros(n, int)
    a = np.zeros(n, int)
    enh = np.zeros(n, int)
    ena = np.zeros(n, int)
    bh, ba = mu_h / MIN, mu_a / MIN
    for t in range(MIN):
        diff = h - a
        rh = np.where(diff < 0, bh * (1 + p["trail"]), np.where(diff > 0, bh * (1 - p["lead"]), bh))
        ra = np.where(diff > 0, ba * (1 + p["trail"]), np.where(diff < 0, ba * (1 - p["lead"]), ba))
        pull_h = ((diff == -1) & (t >= p["en_min"] - 0.5)) | ((diff == -2) & (t >= p["en_min"] - 1.5))
        pull_a = ((diff == 1) & (t >= p["en_min"] - 0.5)) | ((diff == 2) & (t >= p["en_min"] - 1.5))
        rh = np.where(pull_h, p["en_for"], np.where(pull_a, p["en_against"], rh))
        ra = np.where(pull_a, p["en_for"], np.where(pull_h, p["en_against"], ra))
        ph, pa = 1 - np.exp(-rh), 1 - np.exp(-ra)
        gh = (d.u[t, :, 0] < ph).astype(int) + (d.u2[t, :, 0] < ph * ph / 2)
        ga = (d.u[t, :, 1] < pa).astype(int) + (d.u2[t, :, 1] < pa * pa / 2)
        enh += gh * pull_a
        ena += ga * pull_h
        h += gh
        a += ga
    hreg, areg = h.copy(), a.copy()
    tie = h == a
    share = mu_h / (mu_h + mu_a)
    if playoff:
        rate = (mu_h + mu_a) / MIN            # 5-on-5 sudden death until someone scores
        so = np.zeros(n, bool)
        ot_goal = tie
    else:
        rate = p["ot_rate"] * (mu_h + mu_a) / 6.0
        ot_goal = tie & (d.ot / rate < 5)
        so = tie & ~ot_goal
    home_ot = d.side < share
    h = h + (ot_goal & home_ot)
    a = a + (ot_goal & ~home_ot)
    so_home = d.so < 0.5 + p["so_skill"] * (share - 0.5)
    h = h + (so & so_home)
    a = a + (so & ~so_home)
    return dict(h=h, a=a, hreg=hreg, areg=areg, ot=tie, so=so, enh=enh, ena=ena, hw=h > a)


def p_over(s, line):
    tot = s["h"] + s["a"]
    over, under = (tot > line).mean(), (tot < line).mean()
    return over / max(over + under, 1e-9)


def fit(p_home, line, po, d, playoff=False, p=P):
    """Rates (mu_h, mu_a) whose simulation matches the market: P(home wins incl OT) = p_home and P(over line) = po."""
    tot, share = max(line, 4.0) + 0.15, 0.5 + (p_home - 0.5) * 0.9
    for _ in range(12):
        s = sim(tot * share, tot * (1 - share), d, playoff, p)
        e1 = s["hw"].mean() - p_home
        e2 = p_over(s, line) - po
        if abs(e1) < 0.002 and abs(e2) < 0.002:
            break
        share = float(np.clip(share - e1 / 1.25, 0.2, 0.8))
        tot = float(np.clip(tot - e2 / 0.17, 2.5, 11))
    return tot * share, tot * (1 - share)


def summary(s, pl_line_home=None, total=None):
    """Probabilities the hub shows, from one simulated game."""
    h, a = s["h"], s["a"]
    m = h - a
    out = dict(ph=float(s["hw"].mean()), pot=float(s["ot"].mean()), pso=float(s["so"].mean()),
               rh=float((s["hreg"] > s["areg"]).mean()), ra=float((s["hreg"] < s["areg"]).mean()),
               xh=float((s["h"] - (s["so"] & s["hw"])).mean()), xa=float((s["a"] - (s["so"] & ~s["hw"])).mean()))
    out["pa"] = 1 - out["ph"]
    out["en"] = float((s["enh"] + s["ena"]).mean())   # empty-net goals per game
    out["home_m15"] = float((m >= 2).mean())     # home -1.5 covers
    out["away_m15"] = float((m <= -2).mean())    # away -1.5 covers
    if total is not None:
        out["po"] = p_over(s, total)
    return out


# ---------- player props ----------
LEAGUE_SH = 0.105          # league shooting % used to steady small-sample goal rates
SHOT_DISP = 8.0            # negative-binomial size for skater shots (variance = m + m^2/8)
SAVE_DISP = 30.0           # for team shots against a goalie
BLEND = 0.35               # starting weight on the model vs the no-vig price for two-way props; refit once ~2 weeks of props are graded


def team_goals(s, side):
    """Goals a team scored in each simulated game, shootout goal excluded (props never count it)."""
    if side == "h":
        return s["h"] - (s["so"] & s["hw"])
    return s["a"] - (s["so"] & ~s["hw"])


def nb(rng, mean, size, n):
    mean = max(mean, 1e-3)
    return rng.negative_binomial(size, size / (size + mean), n)


def skater(rng, s, side, gp, g, a, sog, team_gf_pg, team_xg):
    """Simulated goals, assists, points and shots for one skater across the same games as s."""
    n = len(s["h"])
    k = team_goals(s, side)
    gp = max(gp, 1)
    est_g = (g + LEAGUE_SH * sog) / 2 / gp            # half the player's goals, half shots x league shooting %
    tg = max(team_gf_pg, 1.5)
    sg = min(est_g / tg, 0.45)
    sa = min(a / gp / tg, 0.6)
    tot = min(sg + sa, 0.9)
    pts = rng.binomial(k, tot)
    goals = rng.binomial(pts, sg / tot if tot > 0 else 0)
    shots = nb(rng, sog / gp * (team_xg / tg) ** 0.3, SHOT_DISP, n)
    shots = np.maximum(shots, goals)
    return dict(goal=goals, ast=pts - goals, pts=pts, sog=shots)


def goalie(rng, s, side, opp_sf_pg, own_sa_pg):
    """Simulated saves for the goalie of team `side` (goals into an empty net do not count against the goalie)."""
    n = len(s["h"])
    opp = "a" if side == "h" else "h"
    ga = team_goals(s, opp) - (s["ena"] if side == "h" else s["enh"])
    ga = np.maximum(ga, 0)
    shots = np.maximum(nb(rng, (opp_sf_pg + own_sa_pg) / 2, SAVE_DISP, n), ga)
    return shots - ga


def lg(p):
    p = min(max(p, 1e-3), 1 - 1e-3)
    return np.log(p / (1 - p))


def blend(p_model, nv):
    if nv is None:
        return p_model
    z = BLEND * lg(p_model) + (1 - BLEND) * lg(nv)
    return float(1 / (1 + np.exp(-z)))
