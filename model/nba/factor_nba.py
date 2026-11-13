"""Factor study for the NBA sim: a change is kept only if it improves forecasts on games it was not tuned on.

Run: ../venv/bin/python factor_nba.py
Tunes on Jan 14 - Jun 2026 (every 3rd game date), confirms on Oct 20, 2026 onward. Scores, on DraftKings main lines only:
log loss of the raw sim (lower is better), for all lines and for players whose team had a regular out; plus the error of the mean projection.
"""
import json
import sys
import time
import numpy as np
import pandas as pd
import backtest_nba as B
import nba_model as M

CONFIGS = {"base (off)": {}, "half": {s: .5 for s in ("reb", "ast", "fg3", "stl", "blk", "tov")}, "full": {s: 1.0 for s in ("reb", "ast", "fg3", "stl", "blk", "tov")},
           "reb+ast full": {"reb": 1.0, "ast": 1.0}}
WINDOWS = {"tune": ("2026-01-14", "2026-06-30", 3), "confirm": ("2026-10-20", "2099-01-01", 1)}
NP = ["reb", "ast", "ra", "fg3"]  # markets the factor touches most (points are already conserved)


def ll(p, y):
    p = np.clip(p, 1e-3, 1 - 1e-3)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def score(b):
    b = b[(b.kind == "line") & (b.actual != b.line)].copy()
    y = (b.actual > b.line).astype(int).values
    b["y"] = y
    out = dict(n=len(b), ll=ll(b.p_over.values, y), dk=ll(b.dk_p.values, y))
    a = b[b.vac_min >= 20]
    k = b[b.market.isin(NP)]
    ka = k[k.vac_min >= 20]
    out.update(n_aff=len(a), ll_aff=ll(a.p_over.values, a.y.values) if len(a) else None, n_np=len(k), ll_np=ll(k.p_over.values, k.y.values),
               n_np_aff=len(ka), ll_np_aff=ll(ka.p_over.values, ka.y.values) if len(ka) else None, dk_np_aff=ll(ka.dk_p.values, ka.y.values) if len(ka) else None,
               mae_np_aff=float((ka["mean"] - ka.actual).abs().mean()) if len(ka) else None, bias_np_aff=float((ka["mean"] - ka.actual).mean()) if len(ka) else None)
    return out


if __name__ == "__main__":
    d = M.load()
    res = {}
    for w, (a, z, ev) in WINDOWS.items():
        for name, lam in CONFIGS.items():
            t = time.time()
            b = B.run(d, a, z, ev, 1500, dict(M.PARAMS, vac=lam), quiet=True)
            r = score(b)
            res[f"{w}|{name}"] = r
            print(f"{w:>7} {name:<13} lines {r['n']:>6}  ll {r['ll']:.4f} (DK {r['dk']:.4f})  touched stats: all {r['ll_np']:.4f} | regular out: n {r['n_np_aff']:>5} ll {r['ll_np_aff']:.4f} (DK {r['dk_np_aff']:.4f}) "
                  f"mean err {r['mae_np_aff']:.3f} bias {r['bias_np_aff']:+.3f}   [{time.time() - t:.0f}s]", flush=True)
    json.dump(res, open(f"{B.HERE}/output/factor_vac.json", "w"), indent=1)
