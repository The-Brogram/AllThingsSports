"""Head-to-head: should the simulation be centered on the sportsbook's line or on our own team-vs-team model?

Run: venv/bin/python matchup_backtest.py
Walk-forward over every scored week (2025 weeks 5-18, 2026 weeks 4-9): each week uses only earlier games.
  market  the sim's margin and total come from the closing spread and total
  blend   90% market, 10% our line (what the live model uses)
  teams   100% our own line: opponent-adjusted offense and defense ratings (points + EPA, recent games weighted more), no sportsbook input
Scores player forecasts (raw sim probabilities, before calibration) and the game lines themselves. Writes output/matchup_backtest.json.
"""
import json
import time
import numpy as np
import pandas as pd
import backtest as B
import engine as E

WEEKS = [(2025, w) for w in range(5, 19)] + [(2026, w) for w in range(4, 10)]
CONFIGS = [("market", 0.0), ("blend", 0.10), ("teams", 1.0)]


def ll(p, y):
    p = np.clip(p, 1e-3, 1 - 1e-3)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def players(df, gl):
    L = B.score(df, gl)
    L["grp"] = L.mkt.replace({"rec_low": "rec"})
    out = {}
    for k, x in L.groupby("grp"):
        out[k] = dict(n=int(len(x)), logloss=ll(x.p.values, x.y.values), brier=float(np.mean((x.p - x.y) ** 2)))
    out["all"] = dict(n=int(len(L)), logloss=ll(L.p.values, L.y.values), brier=float(np.mean((L.p - L.y) ** 2)))
    for k, x in df[df.mkt != "atd"].groupby("mkt"):
        out[k]["mae_yards"] = float((x.proj - x.act).abs().mean())
    return out


def lines(gl):
    g = gl.dropna(subset=["spread", "tline", "result", "total"]).copy()
    o = dict(games=int(len(g)), mae_margin=dict(market=float((g.result - g.spread).abs().mean()), teams=float((g.result - g.mh).abs().mean())),
             mae_total=dict(market=float((g.total - g.tline).abs().mean()), teams=float((g.total - g.mt).abs().mean())))
    nz = g[g.result != 0]
    o["winner"] = dict(market=float(((nz.spread > 0) == (nz.result > 0))[nz.spread != 0].mean()), teams=float(((nz.mh > 0) == (nz.result > 0)).mean()))
    ats = []
    for cut in (0, 2, 3, 5):
        x = g[((g.mh - g.spread).abs() >= cut) & (g.result != g.spread)]
        home = x.mh > x.spread                      # our line likes the home side more than the market does
        cover = np.where(home, x.result > x.spread, x.result < x.spread)
        ats.append(dict(gap=cut, n=int(len(x)), cover=float(cover.mean()) if len(x) else None))
    o["ats_when_we_disagree"] = ats
    x = g[(g.mt - g.tline).abs() >= 3]
    x = x[x.total != x.tline]
    o["totals_when_we_disagree_3"] = dict(n=int(len(x)), right=float(np.where(x.mt > x.tline, x.total > x.tline, x.total < x.tline).mean()) if len(x) else None)
    return o


if __name__ == "__main__":
    orig = E.simulate_game
    res = {}
    for name, w in CONFIGS:
        t = time.time()
        E.simulate_game = lambda *a, _w=w, **k: orig(*a, **{**k, "wmodel": _w})
        df, gl = B.run(n=1200, seed=11, weeks=WEEKS)
        res[name] = players(df, gl)
        if "lines" not in res:
            res["lines"] = lines(gl)
        p = res[name]
        print(f"{name:>7}: all log loss {p['all']['logloss']:.4f} brier {p['all']['brier']:.4f} | " + " | ".join(f"{k} ll {p[k]['logloss']:.4f}" + (f" mae {p[k]['mae_yards']:.1f}" if 'mae_yards' in p[k] else '') for k in ("pass", "rec", "rush", "atd")) + f"   [{time.time() - t:.0f}s]", flush=True)
    E.simulate_game = orig
    json.dump(res, open("output/matchup_backtest.json", "w"), indent=1)
    print(json.dumps(res["lines"], indent=1))
