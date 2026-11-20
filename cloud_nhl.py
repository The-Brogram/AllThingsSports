"""NHL runs on GitHub (no computer needed): grade finished nights, sim today's games, rebuild the NHL hub file.

Lives at the root of the public site repository next to model/nhl. The "nhl" workflow runs it:
- in the morning (mode "morning"): grade last night, then sim every game with a line, using the Expected goalies;
- every 30 minutes from late morning to late evening (mode "pregame"): re-sim any game starting in the next 30-110 minutes with the
  starters as confirmed by then, and unlock goalie saves props for a Confirmed starter. Nothing to do means nothing is rebuilt.
Run by hand: python3 cloud_nhl.py morning|pregame
"""
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

SITE = os.path.dirname(os.path.abspath(__file__))
NHL = os.path.join(SITE, "model", "nhl")


def step(name, cmd):
    r = subprocess.run([sys.executable, *cmd], cwd=NHL, capture_output=True, text=True)
    tail = (r.stdout + r.stderr).strip().splitlines()[-4:]
    print(f"[{name}] {'ok' if r.returncode == 0 else 'FAILED'}", *tail, sep="\n  ")
    return r


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] else "morning"
    bad = []
    if mode == "morning":
        if step("grade", ["review_nhl.py"]).returncode:
            bad.append("grade")
    r = step("sim", ["run_nhl.py", "--mode", mode])
    if r.returncode:
        bad.append("sim")
    if mode == "pregame" and "nothing to run" in r.stdout:
        return  # no game in the window: leave the page and the stamp alone
    if step("hub", ["hub_nhl.py", SITE]).returncode:
        bad.append("hub")
    json.dump(dict(ran=f"{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ}", mode=mode, failed=bad),
              open(os.path.join(NHL, "output", "cloud_stamp.json"), "w"))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
