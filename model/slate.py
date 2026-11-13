"""One command per step of the week. Dates are YYYYMMDD; season and week come from the schedule.

  python slate.py next                 upcoming slates and what to run when
  python slate.py run [date]           refresh data, simulate, build the report (default: next slate not yet played).
                                       Run again after inactives post; the first run's picks are kept for closing line value.
  python slate.py odds [date]          line snapshot for closing line value (default: today). Run just before kickoff.
  python slate.py review [date ...]    score forecasts and closing line value (default: every finished slate not yet reviewed
                                       whose player stats have posted, so Monday scores Thu+Sun and Tuesday scores Monday night)
  python slate.py checkpoint           all reviewed slates so far: calibration by band and closing line value
  python slate.py hub                  rebuild the season hub (output/hub): every game, side by side, track record.
                                       run and review rebuild it too. With "refresh" it pulls the latest schedule and lines first.

News for one slate (QB notes, held backfields) goes in slate.json with that slate's date.
"""
import os
import subprocess
import sys
import time
from datetime import datetime
import config

PY = sys.executable


def step(args, date=None):
    env = {**os.environ, **({"NFL_DATE": date} if date else {})}
    t = time.time()
    print(f"-> {' '.join(args)}" + (f" [{date}]" if date else ""), flush=True)
    if subprocess.run([PY, *args], cwd=config.ROOT, env=env).returncode:
        sys.exit(f"{args[0]} failed, so later steps were not run.")
    print(f"   done in {time.time() - t:.0f}s", flush=True)


def slates():
    g = config.schedule()
    g = g[g.season == g.season.max()]
    return g.groupby("date").agg(week=("week", "first"), n=("home_team", "size"), first=("gametime", "min"), done=("result", lambda x: x.notna().all())).reset_index()


def today():
    return datetime.now().strftime("%Y%m%d")


def next_slate():
    s = slates()
    s = s[(s.date >= today()) & ~s.done]
    return s.date.iloc[0] if len(s) else today()


def last_week():
    s = slates()
    done = s[s.done & (s.date <= today())]
    if done.empty:
        return []
    wk = done.week.iloc[-1]
    return s[(s.week == wk) & s.done].date.tolist()


def unreviewed():
    """Finished slates with saved forecasts that are not in the track record yet: (ready, waiting for player stats)."""
    import pandas as pd
    g = config.schedule()
    g = g[(g.season == g.season.max()) & (g.date <= today())]
    tr = f"{config.ROOT}/output/track_record.csv"
    seen = set(pd.read_csv(tr, usecols=["date"]).date.astype(str)) if os.path.exists(tr) else set()
    st = pd.read_csv(f"{config.ROOT}/data/stats_player_week_{g.season.max()}.csv", usecols=["week", "team"], low_memory=False)
    ready, waiting = [], []
    for d, x in g.groupby("date"):
        if d in seen or not x.result.notna().all() or not os.path.exists(f"{config.ROOT}/output/predictions_{d}.csv"):
            continue
        teams = set(x.home_team) | set(x.away_team)
        posted = set(st[st.week == x.week.iloc[0]].team)
        (ready if teams <= posted else waiting).append(d)
    return ready, waiting


def main():
    cmd, args = (sys.argv[1] if len(sys.argv) > 1 else ""), sys.argv[2:]
    if cmd in ("run", "odds", "review", "hub"):
        try:  # GitHub also runs this program: start from its results when it ran more recently than this computer
            import publish_site
            print("sync:", publish_site.pull())
        except Exception as e:  # noqa: BLE001  offline: carry on with what is here
            print("sync skipped:", repr(e))
    if cmd == "next":
        s = slates()
        s = s[(s.date >= today())].head(6)
        for r in s.itertuples():
            d = datetime.strptime(r.date, "%Y%m%d")
            print(f"{d:%a %b %-d}  Week {r.week}  {r.n} game{'s' if r.n > 1 else ''}, first kickoff {r.first} ET   python slate.py run {r.date}")
        print("\nLast finished week to review:", ", ".join(last_week()) or "none")
    elif cmd == "run":
        d = args[0] if args else next_slate()
        step(["fetch_data.py"], d)
        step(["run_week.py"], d)
        step(["report.py"], d)
        step(["season.py"])
        print(f"Report: output/nfl_{d}_report.html\nHub: output/hub/index.html")
    elif cmd == "odds":
        step(["odds.py"], args[0] if args else today())
    elif cmd == "review":
        step(["fetch_data.py", "results"])  # final stats and results first, so the last finished week is current
        if args:
            dates = [d for d in args if os.path.exists(f"{config.ROOT}/output/predictions_{d}.csv")]
        else:
            dates, waiting = unreviewed()
            if waiting:
                print("Final scores are in but player stats have not posted yet for:", ", ".join(waiting), "(rerun review later)")
        if not dates:
            sys.exit("No finished slate with saved forecasts to review.")
        print("Reviewing:", ", ".join(dates))
        for d in dates:
            step(["review.py"], d)
        step(["season.py"])
    elif cmd == "hub":
        if args[:1] == ["refresh"]:
            step(["fetch_data.py", "results"])  # nflverse schedule, results and look-ahead lines
        step(["season.py"])
    elif cmd == "checkpoint":
        step(["checkpoint.py"])
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
