"""Odds log for closing line value. Every snapshot appends DraftKings lines (via ESPN) to output/odds_log.csv.
The last snapshot before kickoff is the closing line. Run python odds.py for a quick refresh between full runs."""
import glob
import json
import os
import re
from datetime import datetime, timezone
import pandas as pd
from config import DATE, ESPN, ROOT

D = os.path.join(ROOT, "data")
LOG = os.path.join(ROOT, "output", "odds_log.csv")
TM = {"WSH": "WAS", "LAR": "LA"}


def num(x):
    try:
        return float(str(x).lstrip("ou").replace("+", ""))
    except (TypeError, ValueError):
        return None


def rows(now):
    out = []
    for ev in json.load(open(f"{D}/{ESPN}"))["events"]:
        e, c = ev["id"], ev["competitions"][0]
        side = {x["homeAway"]: TM.get(x["team"]["abbreviation"], x["team"]["abbreviation"]) for x in c["competitors"]}
        base = dict(snap=now, date=DATE, event=e, game=f"{side['away']}@{side['home']}", kick=ev["date"])
        s = json.load(open(f"{D}/espn/sum_{e}.json"))
        pc = next((p for p in s.get("pickcenter", []) if p.get("provider", {}).get("id") == "100"), None)
        if pc:
            for mk, key, sides in (("moneyline", "moneyline", ("home", "away")), ("spread", "pointSpread", ("home", "away")), ("total", "total", ("over", "under"))):
                for sd in sides:
                    q = (pc.get(key) or {}).get(sd) or {}
                    cl, op = q.get("close") or {}, q.get("open") or {}
                    out.append(dict(base, market=mk, side=side.get(sd, sd), athlete=None, line=num(cl.get("line")), price=num(cl.get("odds")), open_line=num(op.get("line")), open_price=num(op.get("odds"))))
        for f in glob.glob(f"{D}/espn/pp/{e}_*.json"):
            for r in json.load(open(f)).get("items", []):
                if "athlete" not in r:
                    continue
                m = re.search(r"athletes/(\d+)", r["athlete"].get("$ref") or "")
                if not m:  # ESPN sometimes lists a prop with no player link (seen Nov 15, 2026): skip it rather than lose the whole snapshot
                    continue
                aid = int(m.group(1))
                out.append(dict(base, market=r["type"]["name"], side="", athlete=aid, line=num(r.get("current", {}).get("target", {}).get("value")), price=None,
                                open_line=num(r.get("open", {}).get("target", {}).get("value")), open_price=None, updated=r.get("lastUpdated")))
    return out


def snapshot():
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
    df = pd.DataFrame(rows(now))
    df.to_csv(LOG, mode="a", header=not os.path.exists(LOG), index=False)
    print(f"odds snapshot {now}: {len(df)} lines logged to {LOG}")
    return df


def closing(log=None):
    """Last line seen before each game's kickoff, per market, side and player."""
    x = pd.read_csv(log or LOG, low_memory=False)
    x = x[pd.to_datetime(x.snap, utc=True) < pd.to_datetime(x.kick, utc=True)]
    return x[x.snap == x.groupby("event").snap.transform("max")]


def clv(date):
    """Closing line value: did the line move toward each pick between our run and kickoff? Positive is good."""
    f = os.path.join(os.path.dirname(LOG), f"picks_{date}.csv")
    if not (os.path.exists(f) and os.path.exists(LOG)):
        return None
    pk, cl = pd.read_csv(f), closing()
    cl = cl[cl.date.astype(str) == str(date)]
    out = []
    for r in pk.itertuples():
        if r.kind == "prop":
            c = cl[(cl.event == r.event) & (cl.market == r.market) & (cl.athlete == r.athlete)].line.median()
            if pd.notna(c):
                out.append(dict(kind="prop", flagged=r.flagged, pick=f"{r.player} {r.side} {r.line:g}", close=c, clv=(c - r.line) if r.side == "over" else (r.line - c)))
            continue
        g = cl[cl.event == r.event]
        m = r.market.split()
        if m[0] in ("Over", "Under"):
            c = g[(g.market == "total") & (g.side == m[0].lower())].line.median()
            v = (c - float(m[1])) if m[0] == "Over" else (float(m[1]) - c)
        elif m[-1] == "moneyline":
            c = g[(g.market == "moneyline") & (g.side == m[0])].price.median()
            ip = lambda o: abs(o) / (abs(o) + 100) if o < 0 else 100 / (o + 100)
            v = ip(c) - ip(r.price)  # closing implied chance above what we paid
        else:
            c = g[(g.market == "spread") & (g.side == m[0])].line.median()
            v = float(m[1]) - c
        if pd.notna(c):
            out.append(dict(kind="game", flagged=r.flagged, pick=r.market, close=c, clv=v))
    return pd.DataFrame(out).assign(date=date) if out else None



if __name__ == "__main__":
    import fetch_data
    fetch_data.fetch_espn()
    snapshot()
