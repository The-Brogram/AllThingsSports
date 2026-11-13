"""Grade a finished week's player forecasts. Run: python fetch_cfb.py && python parse_cfb.py && python review_cfb.py [week]
Default: every week that has forecasts (output/predictions_w<N>.csv), final games, and no review yet.
Writes output/review_w<N>.json, which the season hub's college Track record page reads. Game picks are graded by hub_cfb.py."""
import glob
import json
import os
import re
import sys
import numpy as np
import pandas as pd
import cfb_model as C

FORCE = "--force" in sys.argv
KEY = {"rec": "rec_y", "rush": "rush_y", "pass": "pass_y"}


def review(wk, d):
    f = pd.read_csv(f"output/predictions_w{wk}.csv")
    g = d["games"]
    gw = g[(g.season == 2026) & (g.week == wk) & (g.stype == 2)]
    lab = gw.away + "@" + gw.home
    if not set(f.game) <= set(lab) and not FORCE:
        return None  # wait until every forecast game is final
    f = f[f.game.isin(set(lab))].copy()
    if f.empty:
        return None
    p = d["players"][d["players"].game_id.isin(gw.game_id)]
    box = p.groupby("pid").agg(rec_y=("rec_y", "sum"), rush_y=("rush_y", "sum"), pass_y=("pass_y", "sum"), td=("rush_td", "sum"), rtd=("rec_td", "sum"))
    box["atd"] = box.td + box.rtd
    f["act"] = [box.at[i, KEY.get(m, "atd")] if i in box.index else 0.0 for i, m in zip(f.pid, f.market)]
    f["absent"] = ~f.pid.isin(box.index)
    f["y"] = (f.act >= f.threshold).astype(int)
    mk = []
    for m, x in f.groupby("market"):
        b = x.groupby("threshold").y.transform("mean")
        pr = x.cal_p.clip(.001, .999)
        one = x.drop_duplicates(["pid", "market"])
        mk.append(dict(market=m, n=len(x), brier=round(float(((pr - x.y) ** 2).mean()), 4), base=round(float(((b - x.y) ** 2).mean()), 4),
                       mae=round(float((one.proj - one.act).abs().mean()), 1) if m != "atd" else None, absent=round(float(one.absent.mean()), 3)))
    f["bin"] = (f.cal_p * 10).clip(0, 9).astype(int)
    bins = [dict(bin=int(k), n=int(len(x)), pred=round(float(x.cal_p.mean()), 3), hit=round(float(x.y.mean()), 3)) for k, x in f.groupby("bin")]
    hi = f[f.cal_p >= .7]
    return dict(week=wk, games=int(f.game.nunique()), forecasts=len(f), markets=mk, bins=bins, hi70=dict(n=int(len(hi)), hit=round(float(hi.y.mean()), 3) if len(hi) else None, pred=round(float(hi.cal_p.mean()), 3) if len(hi) else None))


def main():
    args = [a for a in sys.argv[1:] if a.isdigit()]
    weeks = [int(a) for a in args] or sorted(int(re.search(r"_w(\d+)", x).group(1)) for x in glob.glob("output/predictions_w*.csv"))
    d = C.load()
    g = d["games"]
    for wk in weeks:
        out = f"output/review_w{wk}.json"
        if not args and os.path.exists(out):
            continue
        done = g[(g.season == 2026) & (g.week == wk) & (g.stype == 2)]
        if done.empty:
            print(f"Week {wk}: no finished games yet")
            continue
        r = review(wk, d)
        if r is None:
            print(f"Week {wk}: waiting on games that are not final (use --force to grade what is done)")
            continue
        json.dump(r, open(out, "w"), indent=1)
        print(f"Week {wk}: {r['games']} games, {r['forecasts']} forecasts; 70%+ hit {r['hi70']['hit']} on {r['hi70']['n']}")
        print(pd.DataFrame(r["markets"]).to_string(index=False))


if __name__ == "__main__":
    main()
