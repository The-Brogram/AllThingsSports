"""Live track record across every reviewed slate: python checkpoint.py (or slate.py checkpoint).

Answers the two questions the 4-6 week checkpoint is for:
  1. Are live forecasts calibrated? 70% calls should hit about 70%, most of all in the Most likely tier.
  2. Do our picks beat the closing line? Without prop prices this is the only test of the Value tier.
"""
import glob
import os
import numpy as np
import pandas as pd
import odds
from config import ROOT

tr = pd.read_csv(os.path.join(ROOT, "output", "track_record.csv"), low_memory=False)
tr["date"] = tr.date.astype(str)
print(f"{tr.date.nunique()} slates reviewed ({', '.join(sorted(tr.date.unique()))}), {len(tr)} scored forecasts.\n")

base = tr.groupby(["market", "threshold"]).hit.transform("mean")  # hindsight base rate, so a hard benchmark
print("Accuracy by market (Brier; lower is better):")
print(tr.assign(b=(tr.cal_p - tr.hit) ** 2, bb=(base - tr.hit) ** 2).groupby("market").agg(n=("hit", "size"), model=("b", "mean"), base_rate=("bb", "mean")).round(4), "\n")

band = pd.cut(tr.cal_p, [0, .3, .5, .7, .85, 1], labels=["<30%", "30-50%", "50-70%", "70-85%", "85%+"], include_lowest=True)
print("Calibration by band (pred should match hit):")
print(tr.groupby(band, observed=True).agg(n=("hit", "size"), pred=("cal_p", "mean"), hit=("hit", "mean")).round(3), "\n")

t1 = tr[tr.cal_p >= .7]
if len(t1):
    # one forecast per player and market at his highest 70%+ threshold, as the Picks tab shows it
    top = t1.sort_values("threshold").groupby(["date", "pid", "market"]).tail(1)
    se = np.sqrt(top.hit.mean() * (1 - top.hit.mean()) / len(top))
    print(f"Most likely tier (highest 70%+ line per player): {len(top)} calls, predicted {top.cal_p.mean():.1%}, hit {top.hit.mean():.1%} (±{1.64 * se:.1%}, 90%).\n")

frames = [odds.clv(os.path.basename(f)[6:14]) for f in sorted(glob.glob(os.path.join(ROOT, "output", "picks_????????.csv")))]
frames = [f for f in frames if f is not None]
if frames:
    c = pd.concat(frames)
    print("Closing line value, first-run picks (positive = market moved toward us):")
    print(c.groupby(["kind", "flagged"]).clv.agg(n="size", mean="mean", share_positive=lambda x: (x > 0).mean()).round(3))
    v = c[(c.kind == "prop") & ~c.flagged]
    if len(v) >= 30:
        print(f"\nValue-tier props: mean CLV {v.clv.mean():+.2f} on {len(v)} picks, {(v.clv > 0).mean():.0%} moved our way. "
              + ("That supports trusting the Value tier." if v.clv.mean() > 0 and (v.clv > 0).mean() > .52 else "That does not yet support trusting the Value tier."))
    else:
        print(f"\nOnly {len(v)} unflagged prop picks have closing lines; keep logging before judging the Value tier.")
else:
    print("No closing lines yet. Odds snapshots must be taken before kickoff (slate.py odds) for closing line value.")
