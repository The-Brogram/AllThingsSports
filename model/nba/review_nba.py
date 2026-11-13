"""Grade a night's forecasts against the box scores (run the next morning, after fetch_nba.py and parse_nba.py).

Run: ../venv/bin/python review_nba.py [--date 2026-11-10] [--force]
With no --date it grades every past forecast night that has no review yet, so a missed morning is caught up on the next run.
Writes output/review_<date>.json: game picks right/wrong, and every player line graded on the side we favored
(milestones "N+" only have an over side). Players who did not play are voided, as books do.
"""
import argparse
import glob
import json
import os
import pickle
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import nba_model as M

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = f"{HERE}/output"
BANDS = [(.9, 1.01, "90%+"), (.8, .9, "80–90%"), (.7, .8, "70–80%"), (.6, .7, "60–70%"), (.5, .6, "50–60%")]


def picks(pr):
    """One graded side per row: the side the model favors; milestone rows only when the over is favored."""
    x = pr.copy()
    x["side"] = np.where(x.p_over >= .5, "over", "under")
    x["p"] = np.where(x.side == "over", x.p_over, 1 - x.p_over)
    return x[~((x.kind == "milestone") & (x.side == "under"))]


def grade(date, d, force=False):
    """Grade one night. Returns True when a review file was written."""
    f = f"{OUT}/predictions_{date}.csv"
    if not os.path.exists(f):
        print("no forecasts for", date)
        return False
    pr = pd.read_csv(f, dtype={"pid": str, "game_id": str})
    meta = pickle.load(open(f"{OUT}/nba_{date}.pkl", "rb"))
    g = d["games"].set_index("game_id")
    gids = [x["game_id"] for x in meta["games"]]
    if not all(g.loc[i, "completed"] for i in gids if i in g.index) and not force:
        print(f"{date}: not all games final yet")
        return False
    box = d["players"][d["players"].game_id.isin(gids)].set_index(["game_id", "pid"])
    games = []
    for x in meta["games"]:
        r = g.loc[x["game_id"]]
        if not r.completed:
            continue
        m = r.hs - r["as"]
        fav = x["home"] if x["p_home"] >= .5 else x["away"]
        games.append(dict(game_id=x["game_id"], home=x["home"], away=x["away"], hs=int(r.hs), as_=int(r["as"]), pick=fav, p=max(x["p_home"], 1 - x["p_home"]),
                          right=bool((m > 0) == (fav == x["home"])), margin_q=x["margin_q"], in_50=bool(x["margin_q"][1] <= m <= x["margin_q"][3])))
    rows = []
    for r in picks(pr).itertuples():
        k = (r.game_id, r.pid)
        if k not in box.index or box.loc[k, "min"] <= 0:
            continue
        b = box.loc[k]
        act = float(sum(b[s] for s in M.COMBO.get(r.market, (r.market,)))) if r.market != "dd" else float(sum(b[s] >= 10 for s in ("pts", "reb", "ast")) >= 2)
        if act == r.line:
            continue  # push
        hit = (act > r.line) == (r.side == "over")
        rows.append(dict(game_id=r.game_id, pid=r.pid, name=r.name, team=r.team, market=r.market, kind=r.kind, line=r.line, side=r.side, p=round(float(r.p), 4),
                         actual=act, hit=bool(hit), dk=None if pd.isna(r.dk_over) else round(float(r.dk_over if r.side == "over" else 1 - r.dk_over), 4)))
    bt = pd.DataFrame(rows)
    bands = []
    for lo, hi, lab in BANDS:
        z = bt[(bt.p >= lo) & (bt.p < hi)] if len(bt) else bt
        bands.append(dict(band=lab, n=int(len(z)), hits=int(z.hit.sum()) if len(z) else 0, said=round(float(z.p.mean()), 4) if len(z) else None))
    out = dict(date=date, graded=datetime.now(timezone.utc).isoformat(timespec="seconds"), games=games, bands=bands, n=len(bt),
               lines=bt.sort_values("p", ascending=False).astype(object).where(bt.notna(), None).to_dict("records") if len(bt) else [])
    json.dump(out, open(f"{OUT}/review_{date}.json", "w"), default=float, allow_nan=False)
    gr = sum(x["right"] for x in games)
    print(f"{date}: winners {gr}/{len(games)}; " + "; ".join(f"{b['band']} {b['hits']}/{b['n']}" for b in bands if b["n"]))
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None, help="grade this night only; default: every forecast night that has no review yet")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    d = M.load()
    if a.date:
        grade(a.date, d, a.force)
        return
    today = f"{datetime.now(timezone.utc) - timedelta(hours=5):%Y-%m-%d}"
    todo = sorted(x for x in (os.path.basename(f)[12:22] for f in glob.glob(f"{OUT}/predictions_*.csv")) if x < today and not os.path.exists(f"{OUT}/review_{x}.json"))
    if not todo:
        print("no forecasts waiting to be graded")
    for x in todo:
        grade(x, d, a.force)
    try:  # results at real prices and closing line value, for every logged night that is final
        import edge_nba
        edge_nba.grade()
    except Exception as ex:  # noqa: BLE001
        print("price grading skipped:", repr(ex))


if __name__ == "__main__":
    main()
