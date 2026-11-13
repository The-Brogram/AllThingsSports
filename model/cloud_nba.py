"""NBA runs on GitHub (no computer needed): refresh data, grade finished nights, sim today's slate, rebuild the NBA hub files.

Lives at the root of the public site repository next to model/nba (the NBA code, data and outputs). The "nba" workflow runs it
three times a day and commits what changed. Run by hand: python3 cloud_nba.py
A failed step is reported and the rest still run, so one bad download never leaves the page without a rebuild.
"""
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

SITE = os.path.dirname(os.path.abspath(__file__))
NBA = os.path.join(SITE, "model", "nba")
STEPS = [("fetch", ["fetch_nba.py"]), ("parse", ["parse_nba.py"]), ("shots", ["shots_nba.py"]), ("grade", ["review_nba.py"]),
         ("sim", ["run_nba.py"]), ("hub", ["hub_nba.py", SITE])]


def main():
    bad = []
    for name, cmd in STEPS:
        r = subprocess.run([sys.executable, *cmd], cwd=NBA, capture_output=True, text=True)
        tail = (r.stdout + r.stderr).strip().splitlines()[-4:]
        print(f"[{name}] {'ok' if r.returncode == 0 else 'FAILED'}", *tail, sep="\n  ")
        if r.returncode:
            bad.append(name)
    json.dump(dict(ran=f"{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ}", failed=bad),
              open(os.path.join(NBA, "output", "cloud_stamp.json"), "w"))
    sys.exit(1 if "hub" in bad or "sim" in bad else 0)


if __name__ == "__main__":
    main()
