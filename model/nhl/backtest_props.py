"""Test the props model at real DraftKings prices (ESPN kept NHL props from May 2026).

For every game with props: simulate it from its closing moneyline and total, price each prop from pre-game player history only, and
compare with what happened and with DraftKings' own price.
- Two-way props (assists, points, shots, saves): the market chance is the no-vig price (the over is listed first; checked by the skeptic).
- Anytime goal is one-way: the market chance is the implied chance divided by DraftKings' margin factor, which is fit on the tuning
  period only.
- Tuning period: before 2026-11-01 (the 2026 playoffs and October). Test: November 2026, never used for fitting.
- The blend weight (model vs market, on the log-odds scale) is fit on the tuning period and judged on the test period.

Run: ../venv/bin/python backtest_props.py   -> output/backtest_props.json
"""
import csv
import gzip
import json
import os
from collections import defaultdict
import numpy as np
import nhl_model as M
import nhl_props as NP

HERE = os.path.dirname(os.path.abspath(__file__))
SPLIT = "2026-11-01"


def lg(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def ll(p, y):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def rows():
    hist = {r["id"]: r for r in csv.DictReader(gzip.open(f"{HERE}/data/history.csv.gz", "rt"))}
    box = defaultdict(dict)
    meta = {}
    for r in csv.DictReader(gzip.open(f"{HERE}/data/players.csv.gz", "rt")):
        box[r["gid"]][r["aid"]] = r
        meta[r["gid"]] = r
    props = defaultdict(list)
    for r in csv.DictReader(gzip.open(f"{HERE}/data/props_hist.csv.gz", "rt")):
        props[r["gid"]].append(r)
    return hist, box, meta, props


def main():
    hist, box, meta, props = rows()
    H = NP.History()
    out = []
    skipped = defaultdict(int)
    for gi, (gid, pr) in enumerate(sorted(props.items(), key=lambda x: x[1][0]["date"])):
        h = hist.get(gid)
        if not h or not h["hml"] or not h["ou"] or not h["oo"]:
            skipped["no closing line"] += 1
            continue
        g = meta[gid]
        date, home, away = g["date"], g["home"], g["away"]
        ph, po, line = M.novig(h["hml"], h["aml"]), M.novig(h["oo"], h["uo"]), float(h["ou"])
        d = M.Draws(6000, int(gid) % 100000)
        playoff = g["stype"] == "3"
        mh, ma = M.fit(ph, line, po, d, playoff)
        s = M.sim(mh, ma, d, playoff)
        rng = np.random.default_rng(int(gid) % 100000 + 7)
        teams = {home: NP.History.team(H, home, date), away: NP.History.team(H, away, date)}
        pairs = defaultdict(list)
        for r in sorted(pr, key=lambda x: int(x["seq"])):
            pairs[(r["aid"], r["mk"], r["line"])].append(r)
        cache = {}
        for (aid, mk, ln), it in pairs.items():
            b = box[gid].get(aid)
            if not b:
                skipped["player did not play (void)"] += 1
                continue
            tm = b["tm"]
            side = "h" if tm == home else "a"
            T, O = teams.get(tm), teams.get(away if side == "h" else home)
            if not T or not O:
                skipped["team history short"] += 1
                continue
            if mk == "saves":
                if b["pos"] != "G":
                    skipped["saves: not the starter"] += 1
                    continue
                v = NP.saves_draws(rng, s, side, O["sf"], T["sa"])
                actual = int(b["sv"] or 0)
            else:
                if aid not in cache:
                    f = H.skater(aid, date)
                    if not f:
                        skipped["no player history"] += 1
                        continue
                    cache[aid] = NP.skater_draws(rng, s, side, f, T["gf"])
                v = cache[aid][mk]
                actual = int(b[{"goal": "g", "ast": "a", "pts": None, "sog": "sog"}[mk]] if mk != "pts" else int(b["g"]) + int(b["a"]))
            L = 0.5 if mk == "goal" else float(ln)
            if mk != "goal" and actual == L:
                continue
            y = int(actual > L)
            pm = float((v > L).mean())
            o = it[0]["price"]
            if mk == "goal":
                out.append(dict(gid=gid, date=date, mk=mk, pm=pm, imp=M.implied(o), y=y, price=o))
            elif len(it) >= 2 and 1.0 <= M.implied(o) + M.implied(it[1]["price"]) <= 1.12:
                out.append(dict(gid=gid, date=date, mk=mk, pm=pm, nv=M.novig(o, it[1]["price"]), y=y, price=o, under=it[1]["price"]))
            else:
                skipped["incoherent or one-sided pair"] += 1
        if gi % 50 == 0:
            print(gi, date, len(out), flush=True)
    res = dict(split=SPLIT, rows=len(out), skipped=dict(skipped), markets={})
    for mk in ("goal", "ast", "pts", "sog", "saves"):
        R = [r for r in out if r["mk"] == mk]
        tr = [r for r in R if r["date"] < SPLIT]
        te = [r for r in R if r["date"] >= SPLIT]
        if len(tr) < 50 or len(te) < 30:
            res["markets"][mk] = dict(train=len(tr), test=len(te), note="too few to judge")
            continue
        def mkt(rr, c=1.0):
            return np.array([r["nv"] if "nv" in r else min(r["imp"] / c, .99) for r in rr])
        c = 1.0
        if mk == "goal":   # DraftKings' margin on anytime goal, fit on the tuning period: market chance = implied / c
            ytr = np.array([r["y"] for r in tr])
            cs = np.linspace(1.0, 1.6, 61)
            c = float(cs[np.argmin([ll(mkt(tr, x), ytr).mean() for x in cs])])
        ws = np.linspace(0, 1, 21)
        ytr, yte = np.array([r["y"] for r in tr]), np.array([r["y"] for r in te])
        def blend(rr, w):
            return 1 / (1 + np.exp(-(w * lg([r["pm"] for r in rr]) + (1 - w) * lg(mkt(rr, c)))))
        w = float(ws[np.argmin([ll(blend(tr, x), ytr).mean() for x in ws])])
        def stats(rr, y):
            a, m, b = ll([r["pm"] for r in rr], y), ll(mkt(rr, c), y), ll(blend(rr, w), y)
            dif = b - m
            rg = np.random.default_rng(0)
            bs = [dif[rg.integers(0, len(dif), len(dif))].mean() for _ in range(400)]
            return dict(n=len(rr), hit=float(y.mean()), model_mean=float(np.mean([r["pm"] for r in rr])), market_mean=float(mkt(rr, c).mean()),
                        ll_model=float(a.mean()), ll_market=float(m.mean()), ll_blend=float(b.mean()),
                        blend_minus_market=float(dif.mean()), range=[float(np.percentile(bs, 5)), float(np.percentile(bs, 95))])
        # calibration of the model alone, by band (test period)
        bands = []
        pm_te = np.array([r["pm"] for r in te])
        for lo, hi in ((0, .1), (.1, .2), (.2, .3), (.3, .4), (.4, .5), (.5, .6), (.6, .7), (.7, 1.01)):
            k = (pm_te >= lo) & (pm_te < hi)
            if k.sum() >= 15:
                bands.append(dict(band=f"{lo:.1f}-{min(hi, 1):.1f}", n=int(k.sum()), model=float(pm_te[k].mean()), actual=float(yte[k].mean())))
        # at the price: back the model's side when the blend is 3+ points above the market chance (two-way: either side)
        bets = []
        for r, pb in zip(te, blend(te, w)):
            mc = r["nv"] if "nv" in r else min(r["imp"] / c, .99)
            if pb - mc >= .03:
                bets.append(profit(r["price"], r["y"]))
            elif "nv" in r and mc - pb >= .03:
                bets.append(profit(r["under"], 1 - r["y"]))
        res["markets"][mk] = dict(margin_factor=c if mk == "goal" else None, blend_weight=w, train=stats(tr, ytr), test=stats(te, yte), test_bands=bands,
                                  test_bets=dict(n=len(bets), roi=float(np.mean(bets)) if bets else None))
    os.makedirs(f"{HERE}/output", exist_ok=True)
    json.dump(res, open(f"{HERE}/output/backtest_props.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


def profit(price, hit):
    p = float(str(price).replace("+", ""))
    return (p / 100 if p > 0 else 100 / -p) if hit else -1.0


if __name__ == "__main__":
    main()
