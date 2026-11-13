"""Fit the final probability from the backtest rows (output/bt_rows.csv.gz) and check it on games it never saw.

Fits, logistic on log-odds:
  alt:         p = sigmoid(a + b*logit(sim))                      used for the sim's lower "N+" thresholds (no DraftKings price)
  model:       the raw sim, for median lines DraftKings has not posted (a fit hurt out of sample)
  blend:       p = sigmoid(a + b*logit(sim) + c*logit(DraftKings)) used for priced lines (over/under de-vigged; milestones vig in)
Fit on 2025-26 (games before Oct 1, 2026); tested on 2026-27 so the track-record numbers are out of sample.
Writes output/calibration.json (used by run_nba.py) and output/bt_summary.json (shown on the hub's Track record page).
"""
import json
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = f"{HERE}/output"
SPLIT = "2026-10-01"
BANDS = [(.9, 1.01, "90%+"), (.8, .9, "80–90%"), (.7, .8, "70–80%"), (.6, .7, "60–70%"), (.5, .6, "50–60%")]


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-3, 1 - 1e-3)
    return np.log(p / (1 - p))


def fit(X, y, l2=1e-3):
    X = np.c_[np.ones(len(X)), X]
    w = np.zeros(X.shape[1])
    w[1] = 1.0
    for _ in range(50):
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (p - y) + l2 * w
        H = (X * (p * (1 - p))[:, None]).T @ X + l2 * np.eye(len(w))
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def pred(w, X):
    return 1 / (1 + np.exp(-(np.c_[np.ones(len(X)), X] @ w)))


def ll(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def bands(p, y, side_only=True):
    """Grade the favored side: probability max(p, 1-p), hit if that side came in."""
    q = np.maximum(p, 1 - p)
    h = np.where(p >= .5, y, 1 - y)
    out = []
    for lo, hi, lab in BANDS:
        m = (q >= lo) & (q < hi)
        out.append(dict(band=lab, n=int(m.sum()), hits=int(h[m].sum()), said=round(float(q[m].mean()), 4) if m.any() else None))
    return out


def main():
    b = pd.read_csv(f"{OUT}/bt_rows.csv.gz")
    b = b[b.actual != b.line]
    b["y"] = (b.actual > b.line).astype(int)
    tr, te = b[b.date < SPLIT], b[b.date >= SPLIT]
    amin = {"pts": 10, "reb": 4, "ast": 3, "fg3": 2, "pr": 12, "pa": 12, "ra": 6, "pra": 15}  # same cut as run_nba.ALT_MIN
    un = lambda x: x[(x.kind == "alt") & (x.proj_min >= 18) & (x.line + .5 >= x.market.map(amin))]  # alt N+ lines the hub shows: sim only, own fit
    wm = fit(logit(un(tr).p_over)[:, None], un(tr).y.values)
    # median "model" lines stay on the raw sim: a fit on 2025-26 made them worse on 2026-27 (log loss .670 vs .656)
    md = te[te.kind == "model"]
    bmod = bands(md.p_over.values, md.y.values)
    pr_tr, pr_te = tr[tr.dk_p.notna()], te[te.dk_p.notna()]
    Xb = lambda x: np.c_[logit(x.p_over), logit(x.dk_p), (x.kind == "milestone").astype(float) * logit(x.dk_p)]
    wb = fit(Xb(pr_tr), pr_tr.y.values)
    res = dict(alt=dict(a=float(wm[0]), b=float(wm[1])), model=dict(a=0.0, b=1.0), blend=dict(a=float(wb[0]), b=float(wb[1]), c=float(wb[2]), cm=float(wb[3])), split=SPLIT,
               n_train=int(len(tr)), n_test=int(len(te)), n_priced_train=int(len(pr_tr)), n_priced_test=int(len(pr_te)))
    print(f"train {len(tr)} rows ({len(pr_tr)} priced), test {len(te)} rows ({len(pr_te)} priced)")
    print("model-only fit", np.round(wm, 3), " blend fit [a, sim, DK, DK x milestone]", np.round(wb, 3))
    tu = un(te)
    yt = tu.y.values
    print(f"test, alt lines: raw sim {ll(tu.p_over, yt):.4f}  calibrated sim {ll(pred(wm, logit(tu.p_over)[:, None]), yt):.4f}")
    y2 = pr_te.y.values
    pb = pred(wb, Xb(pr_te))
    pdk = np.where(pr_te.kind == "milestone", pr_te.dk_p * .95, pr_te.dk_p)  # rough vig removal, for comparison only
    print(f"test, DraftKings-priced: calibrated sim {ll(pred(wm, logit(pr_te.p_over)[:, None]), y2):.4f}  DraftKings {ll(pdk, y2):.4f}  blend {ll(pb, y2):.4f}")
    bt = bands(pb, y2)
    for x in bt:
        print(f"  {x['band']:>7}: said {x['said']}  hit {x['hits']}/{x['n']} = {x['hits'] / max(x['n'], 1):.3f}")
    bm = bands(pred(wm, logit(tu.p_over)[:, None]), yt)
    print("alt lines (sim only, calibrated):")
    for x in bm:
        print(f"  {x['band']:>7}: said {x['said']}  hit {x['hits']}/{x['n']} = {x['hits'] / max(x['n'], 1):.3f}")
    res["bt"] = dict(priced=bt, alt=bm, model=bmod, from_=te.date.min(), to=te.date.max(), games=int(te.game_id.nunique()), ll_blend=ll(pb, y2), ll_dk=ll(pdk, y2))
    json.dump(res, open(f"{OUT}/calibration.json", "w"), indent=1)
    json.dump(res["bt"], open(f"{OUT}/bt_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
