"""NFL and college runs on GitHub (no computer needed). Lives at the root of the site repository; the program is under model/.

The "football" workflow calls this with the schedule line that fired, and the table below turns that into a job:
  nfl-review  score finished slates, running totals      nfl-hub    late review if any, refresh schedule and lines
  nfl-today   run today's slate if there is a game        nfl-next   run the next slate (Friday: the Sunday games)
  nfl-odds    log a line snapshot for closing line value  hub        rebuild the hub only
  cfb-run     fetch, simulate and write the week's college Game Books      cfb-review   grade the college week
Run by hand: python3 cloud_model.py "" <job>. A push that adds model/run_once.txt (one job per line) runs those jobs once.
Every run writes model/logs/football.log and model/output/cloud_model.json, which the workflow commits with the results.
What it cannot do: the lineup and quarterback news checks. slate.json and cfb/qb.json come from the desktop copy.
"""
import glob
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone

SITE = os.path.dirname(os.path.abspath(__file__))
M = os.path.join(SITE, "model")
CRON = {"15 14 * * 1,2": "nfl-review", "15 15 * * 3": "nfl-hub", "55 13 * * 1": "nfl-today", "30 0 * * 5,2": "nfl-today", "20 15 * * 5": "nfl-next", "30 13 * * 0": "nfl-today",
        "45 16 * * 0": "nfl-today", "45 20 * * 0": "nfl-odds", "5 1 * * 1": "nfl-odds", "15 16 * * 4": "cfb-run", "10 14 * * 6": "cfb-run", "10 15 * * 0": "cfb-review"}
LOG = []
LAST = [""]


def say(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True)
    LOG.append(line)


def step(name, cmd, sub=""):
    t = datetime.now()
    r = subprocess.run([sys.executable, *cmd], cwd=os.path.join(M, sub), capture_output=True, text=True)
    LAST[0] = r.stdout
    tail = (r.stdout + ("\n" + r.stderr if r.returncode else "")).strip().splitlines()[-12:]
    say(f"[{name}] {'ok' if r.returncode == 0 else 'FAILED'} in {(datetime.now() - t).seconds}s")
    for x in tail:
        say("   ", x[:300])
    return r.returncode == 0


def game_today():
    r = subprocess.run([sys.executable, "-c", "import slate,sys; s=slate.slates(); sys.exit(0 if (s.date==slate.today()).any() else 3)"], cwd=M)
    return r.returncode == 0


def cfb_guard():
    """A reason not to trust the college download; empty when all is well."""
    m = re.search(r"(\d+) games on file", LAST[0])
    have = sum(os.path.getsize(f) > 2000 for f in glob.glob(f"{M}/cfb/data/games/*.json"))
    if not m or have < 0.98 * int(m.group(1)):
        return f"only {have} of {m.group(1) if m else '?'} college game files downloaded: stopping before the stats are rebuilt from a partial set"
    return ""


def cfb_sim_safe():
    """run_cfb keeps a started game's pre-kickoff forecast only if this week's cache is on hand. If the week was already simulated
    (forecasts saved) but the cache is gone, a rerun would re-forecast games after they started, so the sim is skipped."""
    r = subprocess.run([sys.executable, "-c", "import run_cfb;print(run_cfb.week())"], cwd=os.path.join(M, "cfb"), capture_output=True, text=True)
    wk = r.stdout.strip().splitlines()[-1] if r.returncode == 0 and r.stdout.strip() else ""
    if wk and os.path.exists(f"{M}/cfb/output/predictions_w{wk}.csv") and not os.path.exists(f"{M}/cfb/data/cfb_slate_w{wk}.pkl"):
        say(f"week {wk} already has saved forecasts but its simulation cache is not here: sim skipped so games that have started keep their original forecasts")
        return False
    return True


def prune():
    """Simulation caches are large: keep the last 16 days of NFL slates and the two newest college weeks."""
    cut = f"{datetime.now() - timedelta(days=16):%Y%m%d}"
    for f in glob.glob(f"{M}/data/slate_*.pkl"):
        if re.search(r"slate_(\d{8})", f).group(1) < cut:
            os.remove(f)
    wk = sorted(glob.glob(f"{M}/cfb/data/cfb_slate_w*.pkl"), key=lambda f: int(re.search(r"_w(\d+)", f).group(1)))
    for f in wk[:-2]:
        os.remove(f)


def job(mode):
    """Returns (did every needed step work, was the hub rebuilt)."""
    nfl = lambda *a: step(" ".join(a), ["slate.py", *a])
    fresh = lambda: step("nflverse refresh", ["fetch_data.py", "results"])
    if mode == "nfl-review":
        if not nfl("review"):
            say("nothing to review (or stats not posted): refreshing the hub instead")
            if not nfl("hub", "refresh"):
                return False, False
        nfl("checkpoint")
        return True, True
    if mode == "nfl-hub":
        nfl("review")
        return (ok := nfl("hub", "refresh")), ok
    if mode == "nfl-today":
        fresh()
        if not game_today():
            say("no NFL game today: nothing to run")
            return True, False
        return (ok := step("run today", ["slate.py", "run", f"{datetime.now():%Y%m%d}"])), ok
    if mode == "nfl-next":
        fresh()
        return (ok := nfl("run")), ok
    if mode == "nfl-odds":
        return nfl("odds"), False
    if mode in ("cfb-run", "cfb-review"):
        fresh()
        seq = ["fetch_cfb.py", "parse_cfb.py"] + (["weather_cfb.py", "run_cfb.py", "report_cfb.py"] if mode == "cfb-run" else ["review_cfb.py"])
        ok = True
        for s in seq:
            if s == "run_cfb.py" and not cfb_sim_safe():
                break
            good = step(s, [s], "cfb")
            if s == "fetch_cfb.py" and good and (why := cfb_guard()):
                say(why)
                return False, False
            if not good and s in ("fetch_cfb.py", "parse_cfb.py", "run_cfb.py", "report_cfb.py"):
                ok = False
                break  # review_cfb.py says "waiting on games" with a failure code: that is normal, the hub still rebuilds
        built = step("season.py", ["season.py"])
        return ok and built, built
    fresh()
    return (ok := step("season.py", ["season.py"])), ok


def main():
    sched, asked = (sys.argv[1:] + ["", ""])[:2]
    modes = [asked or CRON.get(sched) or "hub"]
    once = os.path.join(M, "run_once.txt")
    if not asked and not sched and os.path.exists(once):
        modes = [x.strip() for x in open(once).read().split() if x.strip()] or modes
    if os.path.exists(once):
        os.remove(once)
    say(f"football run {datetime.now():%a %b %d %Y %I:%M %p} ET: {', '.join(modes)}" + (f" (schedule {sched})" if sched else ""))
    bad, built = [], False
    for m in modes:
        say(f"== {m}")
        try:
            ok, b = job(m)
        except Exception as e:  # noqa: BLE001  record it and still save what finished
            say("crashed:", repr(e))
            ok, b = False, False
        built = built or b
        if not ok:
            bad.append(m)
    if built:
        sys.path.insert(0, M)
        import publish_site
        hub = os.path.join(M, "output", "hub")
        try:  # with no simulation cache on hand the Fantasy page builds empty: keep the one already on the site
            empty = not json.load(open(os.path.join(hub, "fantasy.json"))).get("slates")
        except (OSError, ValueError):
            empty = True
        publish_site.copy_hub(hub, SITE, {"fantasy.json"} if empty else ())
        say("hub copied to the site")
    prune()
    os.makedirs(os.path.join(M, "logs"), exist_ok=True)
    open(os.path.join(M, "logs", "football.log"), "w").write("\n".join(LOG) + "\n")
    json.dump(dict(ran=f"{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ}", modes=modes, failed=bad), open(os.path.join(M, "output", "cloud_model.json"), "w"))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
