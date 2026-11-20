"""NHL player props, v2 (after the skeptic review of 2026-11-20).

What changed from v1:
- Each player's rates are steadied toward their own position's league rate (forwards and defencemen shoot very differently), weighted
  by how many shots or minutes they have, instead of a flat 50/50 pull toward one league shooting %.
- Last season counts too (at half weight), so a player with 12 games this season is not judged on 12 games alone.
- Playing time drives everything: expected shots, goals and assists come from per-60-minute rates times projected ice time (the average
  of their last 10 games), with power-play time rated separately.
- Goals and points come from the same simulated games as the team totals: each simulated team goal goes to the player in proportion
  to their expected share, so a game the market expects to be low-scoring lowers every player's chances together.

All inputs come from games before the one being priced (players.csv.gz, built by fetch_history.py; refreshed each run).
"""
import csv
import gzip
import os
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PRIOR_SHOTS = {"F": 60.0, "D": 120.0}     # shooting % prior strength, in shots
PRIOR_MIN = {"F": 300.0, "D": 400.0}      # per-60 rate prior strength, in minutes played
LAST_SEASON_W = 0.5
SHOT_DISP = 9.0
SAVE_DISP = 30.0


def season_of(date):
    y, m = int(date[:4]), int(date[5:7])
    return y + 1 if m >= 8 else y   # ESPN-style season = the year it ends


class History:
    """Every skater and goalie game from players.csv.gz, for pre-game features."""

    def __init__(self, path=None):
        path = path or os.path.join(HERE, "data", "players.csv.gz")
        self.games = defaultdict(list)      # aid -> [(date, row)]
        self.team_games = defaultdict(list)  # team -> [(date, gf, sog_for, sog_against)]
        tg = defaultdict(lambda: dict(sog={}, g={}))
        for r in csv.DictReader(gzip.open(path, "rt")):
            for k in ("g", "a", "sog", "toi", "pptoi"):
                r[k] = int(r[k] or 0)
            self.games[r["aid"]].append((r["date"], r))
            if r["pos"] in ("F", "D"):
                t = tg[(r["gid"], r["date"])]
                t["sog"][r["tm"]] = t["sog"].get(r["tm"], 0) + r["sog"]
                t["g"][r["tm"]] = t["g"].get(r["tm"], 0) + r["g"]
        for (gid, date), t in tg.items():
            ts = list(t["sog"])
            if len(ts) == 2:
                for a, b in (ts, ts[::-1]):
                    self.team_games[a].append((date, t["g"][a], t["sog"][a], t["sog"][b]))
        for v in self.games.values():
            v.sort(key=lambda x: x[0])
        for v in self.team_games.values():
            v.sort(key=lambda x: x[0])
        self.league = self._league()

    def _league(self):
        """Position rates per 60 minutes and shooting %, from the 2025-26 regular season only (so the test months never feed the priors)."""
        acc = defaultdict(lambda: np.zeros(5))
        for v in self.games.values():
            for d, r in v:
                if r["pos"] in ("F", "D") and "2025-10-01" <= d < "2026-04-20":
                    acc[r["pos"]] += [r["g"], r["a"], r["sog"], r["toi"], r["pptoi"]]
        out = {}
        for pos, (g, a, s, toi, pp) in acc.items():
            h = toi / 3600
            out[pos] = dict(sh=g / s, sog60=s / h, a60=a / h, g60=g / h)
        return out

    def skater(self, aid, before, pos_hint=None):
        """Pre-game expectations for one skater: projected minutes and per-game shots, goals and assists (before team context)."""
        rows = [r for d, r in self.games.get(aid, []) if d < before and r["pos"] in ("F", "D")]
        if not rows:
            return None
        pos = rows[-1]["pos"]
        cur = season_of(before)
        w = np.array([1.0 if season_of(r["date"]) == cur else LAST_SEASON_W if season_of(r["date"]) == cur - 1 else 0.0 for r in rows])
        if w.sum() == 0:
            return None
        g = np.array([r["g"] for r in rows]) @ w
        a = np.array([r["a"] for r in rows]) @ w
        s = np.array([r["sog"] for r in rows]) @ w
        mins = np.array([r["toi"] for r in rows]) @ w / 60
        L = self.league[pos]
        sh = (g + PRIOR_SHOTS[pos] * L["sh"]) / (s + PRIOR_SHOTS[pos])
        sog60 = (s + PRIOR_MIN[pos] / 60 * L["sog60"]) / (mins / 60 + PRIOR_MIN[pos] / 60)
        a60 = (a + PRIOR_MIN[pos] / 60 * L["a60"]) / (mins / 60 + PRIOR_MIN[pos] / 60)
        recent = [r["toi"] for r in rows[-10:] if r["toi"] >= 180]
        if not recent:
            return None
        toi = float(np.mean(recent)) / 60
        e_sog = sog60 * toi / 60
        return dict(pos=pos, toi=toi, e_sog=e_sog, e_g=e_sog * sh, e_a=a60 * toi / 60, n=len(rows))

    def team(self, tm, before, last=30):
        """Average goals, shots for and shots against over a team's last games (this season and last)."""
        rows = [x for x in self.team_games.get(tm, []) if x[0] < before][-last:]
        if len(rows) < 5:
            return None
        a = np.array([x[1:] for x in rows], float).mean(0)
        return dict(gf=a[0], sf=a[1], sa=a[2])


def team_goals(s, side):
    if side == "h":
        return s["h"] - (s["so"] & s["hw"])
    return s["a"] - (s["so"] & ~s["hw"])


def nb(rng, mean, size, n):
    mean = max(mean, 1e-3)
    return rng.negative_binomial(size, size / (size + mean), n)


def skater_draws(rng, s, side, f, team_avg_gf):
    """Goals, assists, points and shots for one skater across the simulated games."""
    k = team_goals(s, side)
    ek = max(float(k.mean()), 0.3)
    ctx = ek / max(team_avg_gf, 1.5)               # this game's expected team goals vs the team's usual
    eg, ea = f["e_g"] * ctx, f["e_a"] * ctx
    sg = min(eg / ek, 0.6)
    sp = min((eg + ea) / ek, 0.95)
    pts = rng.binomial(k, sp)
    goals = rng.binomial(pts, sg / sp if sp > 0 else 0)
    shots = np.maximum(nb(rng, f["e_sog"] * ctx ** 0.5, SHOT_DISP, len(k)), goals)
    return dict(goal=goals, ast=pts - goals, pts=pts, sog=shots)


def saves_draws(rng, s, side, opp_sf, own_sa):
    opp = "a" if side == "h" else "h"
    ga = np.maximum(team_goals(s, opp) - (s["ena"] if side == "h" else s["enh"]), 0)
    shots = np.maximum(nb(rng, (opp_sf + own_sa) / 2, SAVE_DISP, len(ga)), ga)
    return shots - ga
