"""Phase 15 postgame review. After fetch_data.py picks up final stats: python review.py [week] [YYYYMMDD]
Scores closing line value (picks_<date>.csv vs odds_log.csv) and output/predictions_<date>.csv, and appends to output/track_record.csv. Runs on any game with results."""
import os
import sys
import numpy as np
import pandas as pd
import engine as E

import config
week = int(sys.argv[1]) if len(sys.argv) > 1 else config.WEEK
date = sys.argv[2] if len(sys.argv) > 2 else config.DATE
print(f"Review of {date} (Week {week})\n")


def clv(date):
    import odds
    o = odds.clv(date)
    if o is None:
        print("No picks, odds log or matching closing lines for this slate, so closing line value is skipped.\n")
        return
    print("Closing line value (points; moneyline in implied probability). Positive means the market moved toward the pick:")
    print(o.groupby(["kind", "flagged"]).clv.agg(["size", "mean", lambda x: (x > 0).mean()]).rename(columns={"<lambda_0>": "share_positive"}).round(3), "\n")


clv(date)
p = pd.read_csv(f"output/predictions_{date}.csv")
s = pd.read_csv(f"{E.DATA}/stats_player_week_2026.csv", low_memory=False)
s = s[s.week == week].set_index("player_id")
if s.empty:
    sys.exit(f"No Week {week} results yet. Run fetch_data.py after the games.")
col = {"rec": "receiving_yards", "rush": "rushing_yards", "pass": "passing_yards"}
def actual(r):
    if r.pid not in s.index:
        return np.nan
    a = s.loc[r.pid]
    a = a.iloc[0] if isinstance(a, pd.DataFrame) else a
    return a.rushing_tds + a.receiving_tds if r.market == "atd" else a[col[r.market]]
p["actual"] = [actual(r) for r in p.itertuples()]
played = p.dropna(subset=["actual"]).copy()
if played.empty:
    sys.exit(f"No forecast players have Week {week} results yet. Sunday's games must finish and post first.")
played["hit"] = (played.actual >= played.threshold).astype(int)
played["bin"] = (played.cal_p * 10).clip(0, 9).astype(int)
print(f"Scored {len(played)} of {len(p)} forecasts. Unscored players did not record a stat line: check inactives before blaming the model.\n")
print("Brier by market (lower is better; 0.25 is a coin flip):")
print(played.groupby("market").apply(lambda g: pd.Series(dict(n=len(g), brier=((g.cal_p - g.hit) ** 2).mean(), log_loss=-(g.hit * np.log(g.cal_p.clip(.01, .99)) + (1 - g.hit) * np.log(1 - g.cal_p.clip(.01, .99))).mean())), include_groups=False).round(4))
y = played[played.market != "atd"].drop_duplicates(["pid", "market"])
print("\nYardage MAE:\n", (y.proj - y.actual).abs().groupby(y.market).mean().round(1))
print("\nCalibration (predicted vs actual):\n", played.groupby("bin").agg(n=("hit", "size"), pred=("cal_p", "mean"), hit=("hit", "mean")).round(3))
y = y.assign(err=y.actual - y.proj)
print("\nLargest misses, for the opportunity / efficiency / script / injury review:")
print(y.reindex(y.err.abs().sort_values(ascending=False).index).head(15)[["game", "player", "market", "proj", "actual", "err"]].round(1).to_string(index=False))
tr = "output/track_record.csv"  # rerunning a review replaces that slate's rows instead of duplicating them
old = pd.read_csv(tr, low_memory=False) if os.path.exists(tr) else pd.DataFrame()
if len(old):
    old = old[~((old.week == week) & (old.date.astype(str) == str(date)))]
pd.concat([old, played.assign(week=week, date=date)], ignore_index=True).to_csv(tr, index=False)
