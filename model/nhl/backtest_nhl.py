"""Backtest the NHL simulation on every regular-season game since 2021-22 with a clean closing line.

For each game:
- fit the simulation to the closing no-vig moneyline and total (as a live run would);
- compare what the simulation implies with what happened, and with the market's own puck-line price where one exists.

Run: ../venv/bin/python backtest_nhl.py [--train-only] [--n 3000]
Writes output/backtest.json (the numbers the checkpoint and the hub's method notes quote).
"""
import argparse
import csv
import gzip
import json
import os
import numpy as np
import nhl_model as M
# empty-net goals per game, NHL stats API team realtime (fetched 2026-11-20); ESPN season = end year
EN_RATE = {2022: 0.363, 2023: 0.334, 2024: 0.341, 2025: 0.399, 2026: 0.387, 2027: 0.349}
CLUBS = {"ANA", "ARI", "BOS", "BUF", "CAR", "CBJ", "CGY", "CHI", "COL", "DAL", "DET", "EDM", "FLA", "LA", "MIN", "MTL", "NJ", "NSH", "NYI",
         "NYR", "OTT", "PHI", "PIT", "SEA", "SJ", "STL", "TB", "TOR", "UTA", "UTAH", "VAN", "VGK", "WPG", "WSH"}

HERE = os.path.dirname(os.path.abspath(__file__))


def rows():
    out = []
    for r in csv.DictReader(gzip.open(f"{HERE}/data/history.csv.gz", "rt")):
        if r["stype"] != "2" or not r["hml"] or not r["ou"] or not r["oo"] or not r["uo"]:
            continue
        if r["home"] not in CLUBS or r["away"] not in CLUBS:   # All-Star and 4 Nations games are filed as regular season
            continue
        r["season"] = int(r["season"])
        out.append(r)
    return out


def ll(p, y):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def run(rs, n, p=M.P):
    res = []
    for i, r in enumerate(rs):
        ph = M.novig(r["hml"], r["aml"])
        line = float(r["ou"])
        po = M.novig(r["oo"], r["uo"])
        d = M.Draws(n, 1000 + i)
        mh, ma = M.fit(ph, line, po, d, p=p)
        s = M.sim(mh, ma, d, p=p)
        q = M.summary(s, total=line)
        hs, as_, hreg, areg = int(r["hs"]), int(r["as"]), int(r["hreg"]), int(r["areg"])
        fav_home = ph >= 0.5
        e = dict(season=r["season"], ph=ph, pot=q["pot"], pso=q["pso"], ot=int(hreg == areg), so=int(r["detail"] == "Final/SO"),
                 rh=q["rh"], yrh=int(hreg > areg), ra=q["ra"], yra=int(areg > hreg),
                 pfav=q["home_m15"] if fav_home else q["away_m15"], yfav=int((hs - as_ if fav_home else as_ - hs) >= 2),
                 pdog=1 - (q["home_m15"] if fav_home else q["away_m15"]),
                 xg=q["xh"] + q["xa"], goals=hreg + areg + (hs + as_ - hreg - areg if r["detail"] != "Final/SO" else 0),
                 en=q["en"], po=q["po"], yo=(hs + as_ > line) * 1 + (hs + as_ == line) * -1, pomkt=po)
        # the market's own puck-line price (home spread -1.5 or +1.5), where coherent
        if r["spread"] and r["hpl"] and r["apl"] and abs(abs(float(r["spread"])) - 1.5) < 1e-6:
            sp = float(r["spread"])
            nv_home = M.novig(r["hpl"], r["apl"])
            if 1.0 <= M.implied(r["hpl"]) + M.implied(r["apl"]) <= 1.10:
                p_home_cover = q["home_m15"] if sp < 0 else 1 - q["away_m15"]
                e.update(pl_model=p_home_cover, pl_mkt=nv_home, pl_y=int((hs - as_) + sp > 0))
        res.append(e)
    return res


def report(res, label):
    def mean(k, rr):
        return float(np.mean([x[k] for x in rr]))
    out = dict(label=label, games=len(res))
    out["ot"] = dict(model=mean("pot", res), actual=mean("ot", res))
    out["shootout"] = dict(model=mean("pso", res), actual=mean("so", res))
    out["reg_home_win"] = dict(model=mean("rh", res), actual=mean("yrh", res))
    out["fav_minus_1_5"] = dict(model=mean("pfav", res), actual=mean("yfav", res))
    out["goals_reg_ot"] = dict(model=mean("xg", res), actual=mean("goals", res))
    seasons = sorted({x["season"] for x in res})
    out["empty_net_per_game"] = dict(model=mean("en", res), actual=float(np.mean([EN_RATE.get(x["season"], np.nan) for x in res])),
                                     note="actual from the NHL stats API (team realtime emptyNetGoals), season averages")
    out["so_share_of_ot"] = dict(model=mean("pso", res) / max(mean("pot", res), 1e-9), actual=mean("so", res) / max(mean("ot", res), 1e-9))
    out["by_season"] = [dict(season=s_, games=len(rr), ot_model=mean("pot", rr), ot=mean("ot", rr), so_model=mean("pso", rr), so=mean("so", rr),
                             fav_model=mean("pfav", rr), fav=mean("yfav", rr)) for s_ in seasons for rr in [[x for x in res if x["season"] == s_]]]
    pl = [x for x in res if "pl_model" in x]
    if pl:
        a = ll([x["pl_model"] for x in pl], [x["pl_y"] for x in pl])
        b = ll([x["pl_mkt"] for x in pl], [x["pl_y"] for x in pl])
        dif = a - b
        rng = np.random.default_rng(0)
        bs = [dif[rng.integers(0, len(dif), len(dif))].mean() for _ in range(400)]
        out["puck_line"] = dict(games=len(pl), model_logloss=float(a.mean()), market_logloss=float(b.mean()),
                                diff=float(dif.mean()), diff_range=[float(np.percentile(bs, 5)), float(np.percentile(bs, 95))],
                                model_mean=mean("pl_model", pl), market_mean=mean("pl_mkt", pl), actual=mean("pl_y", pl))
    # OT calibration by band
    bands = []
    for lo, hi in ((0, .2), (.2, .22), (.22, .24), (.24, 1)):
        rr = [x for x in res if lo <= x["pot"] < hi]
        if rr:
            bands.append(dict(band=f"{lo:.2f}-{hi:.2f}", games=len(rr), model=mean("pot", rr), actual=mean("ot", rr)))
    out["ot_bands"] = bands
    a = ll([x["pot"] for x in res], [x["ot"] for x in res]).mean()
    b = ll([out["ot"]["actual"]] * len(res), [x["ot"] for x in res]).mean()
    out["ot_logloss"] = dict(model=float(a), flat_rate=float(b))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--train-only", action="store_true")
    a = ap.parse_args()
    rs = rows()
    train = [r for r in rs if r["season"] <= 2024]
    test = [r for r in rs if r["season"] >= 2025]
    out = dict(params=M.P, train=report(run(train, a.n), "2021-22 to 2023-24 (tuning)"))
    if not a.train_only:
        out["test"] = report(run(test, a.n), "2024-25 to Nov 19 2026 (out of sample)")
    os.makedirs(f"{HERE}/output", exist_ok=True)
    json.dump(out, open(f"{HERE}/output/backtest.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
