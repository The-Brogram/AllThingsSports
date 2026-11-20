"""Copy the built hub into the public site repository (../site) and push it to GitHub, where GitHub Pages serves it.

Run after season.py: venv/bin/python publish_site.py [--no-push]

Since Nov 13, 2026 the site repository also holds the whole program under model/ so GitHub can run it without this computer
(cloud_model.py for NFL and college, cloud_nba.py for the NBA; the workflows below). How the two copies stay in step:
- Code and settings (the .py files, hub_template.html, slate.json, cfb/qb.json) are edited here and copied up on every publish.
- Outputs and data follow "newest run wins": pull() brings GitHub's copy down when GitHub ran last, and a publish sends this
  computer's copy up when a run here is newer. The NBA is tracked separately from NFL + college.
- Large files that every run re-downloads or rebuilds (this season's nflverse files, simulation caches, raw college game files)
  are never stored in the repository; GitHub fetches them each run and keeps the caches between runs.
"""
import glob
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
HUB = os.path.join(ROOT, "output", "hub")
SITE = os.environ.get("HUB_SITE") or os.path.join(os.path.dirname(ROOT), "site")
MODEL = os.path.join(SITE, "model")
# The NBA code and data are public in the site repository (Josh's choice, Nov 13, 2026). The NFL and college code, data and
# pick records go there only when this is True.
FOOTBALL_PUBLIC = True  # Josh approved making them public on Nov 13, 2026
HEAD = ('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">'
        '<style>:root{color-scheme:light;box-sizing:border-box;padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}'
        'body{margin:0;padding:0}img{max-width:100%}[hidden]{display:none!important}</style></head><body>\n')
# this season's nflverse files: fetch_data.py re-downloads them on every run, so they are not stored in the repository
VOLATILE = ["stats_player_week_2026.csv", "play_by_play_2026.csv.gz", "snap_counts_2026.csv", "injuries_2026.csv", "ftn_charting_2026.csv", "ngs_receiving.csv.gz",
            "ngs_passing.csv.gz", "ngs_rushing.csv.gz", "depth_charts_2026.csv", "roster_2026.csv", "advstats_week_pass_2026.csv", "advstats_week_rec_2026.csv",
            "advstats_week_rush_2026.csv", "advstats_week_def_2026.csv", "roster_weekly_2026.csv"]
SKIP = ["__pycache__", "*.pyc", "*.part", "*.tmp", "*.log", ".DS_Store", "cloud_*.json"]
# (folder, extra names to leave out) for NFL + college outputs and data, and for the NBA
STATE = {"model": [("output", ["hub", "fs", "fantasy_bt_rows.pkl", "fantasy_bt_draws.npy", "factor_study.html"]),
                   ("data", ["slate_*.pkl"] + VOLATILE),
                   ("cfb/output", []),
                   ("cfb/data", ["*.pkl", "games"])],
         "nba": [("nba/output", ["bt_rows.csv.gz"]), ("nba/data", [])],
         "nhl": [("nhl/output", ["backtest_log.txt"]), ("nhl/data", ["raw"])]}
MARK = {"model": ["output/predictions_*.csv", "output/track_record.csv", "output/odds_log.csv", "cfb/output/predictions_w*.csv", "cfb/output/review_w*.json"],
        "nba": ["nba/output/predictions_*.csv", "nba/output/review_*.json"],
        "nhl": ["nhl/output/slate_*.json", "nhl/output/review_*.json"]}
STAMP = {"model": "output/cloud_model.json", "nba": "nba/output/cloud_stamp.json", "nhl": "nhl/output/cloud_stamp.json"}
NHL_HUB = ("nhl.json",)
NBA_HUB = ("nba.json", "nbap.json", "nbl.json", "nblh.json", "nbm.json", "nbw.json", "heads.json")
IGNORE = "\n".join(["__pycache__/", "*.pyc", "*.part", "*.tmp", ".DS_Store", "model/output/hub/", "model/data/slate_*.pkl", "model/cfb/data/*.pkl", "model/cfb/data/games/", "model/nhl/data/raw/"]
                   + ["model/data/" + v for v in VOLATILE]) + "\n"

SCORES_FLOW = """name: scores
on:
  schedule:
    # game hours only, every 15 minutes (times are UTC; Eastern is 5 hours behind in winter, 4 in summer)
    - cron: "*/15 0-6 * * *"      # every night 7 PM to 2 AM ET: NBA, weeknight football
    - cron: "*/15 17-23 * * 6,0"  # Saturday and Sunday noon to 7 PM ET: college and NFL afternoons
    - cron: "5 15,20 * * 1-5"     # weekday catch-up at 10 AM and 3 PM ET (holiday or early games)
  workflow_dispatch:
permissions:
  contents: write
concurrency:
  group: scores
  cancel-in-progress: false
jobs:
  scores:
    runs-on: ubuntu-latest
    timeout-minutes: 5
    steps:
      - uses: actions/checkout@v4
      - run: python3 cloud_scores.py .
      - run: |
          if [ -n "$(git status --porcelain scores.json reviews.json)" ]; then
            git config user.name "scores-bot"
            git config user.email "scores-bot@users.noreply.github.com"
            git add scores.json reviews.json
            git commit -m "scores"
            for i in 1 2 3; do git pull --rebase -q && git push -q && break; sleep 5; done
          fi
"""
SETUP = """      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.14"
      - run: pip install -q numpy==2.5.4 pandas==3.1.0 pillow==13.0.0
"""
SAVE = """      - if: always()
        run: |
          git config user.name "%s"
          git config user.email "%s@users.noreply.github.com"
          git add -A
          if ! git diff --cached --quiet; then
            git commit -q -m "%s"
            for i in 1 2 3; do git pull --rebase -q && git push -q && break; sleep 5; done
          fi
"""
NBA_FLOW = """name: nba
on:
  schedule:
    - cron: "45 14 * * *"   # 9:45 AM ET: grade last night, build today's slate
    - cron: "30 22 * * *"   # 5:30 PM ET: evening sim with later lines and injuries
    - cron: "15 2 * * *"    # 9:15 PM ET: late games rerun
  workflow_dispatch:
permissions:
  contents: write
concurrency:
  group: model
  cancel-in-progress: false
jobs:
  nba:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    env:
      TZ: America/New_York
    steps:
""" + SETUP + "      - run: python cloud_nba.py\n" + SAVE % ("nba-bot", "nba-bot", "nba run")
NHL_FLOW = """name: nhl
on:
  schedule:
    # times are UTC (Eastern is 5 hours behind in winter, 4 in summer)
    - cron: "0 15 * * *"            # 10 AM ET: grade last night, sim every game with a line (Expected goalies)
    - cron: "15,45 16-23 * * *"     # 11:15 AM to 6:45 PM ET, every 30 minutes: re-sim games starting in the next 30-110 minutes
    - cron: "15,45 0-3 * * *"       # 7:15 PM to 10:45 PM ET
  workflow_dispatch:
    inputs:
      mode:
        description: "morning or pregame"
        default: morning
permissions:
  contents: write
concurrency:
  group: nhl
  cancel-in-progress: false
jobs:
  nhl:
    runs-on: ubuntu-latest
    timeout-minutes: 20
    env:
      TZ: America/New_York
    steps:
""" + SETUP + """      - run: python cloud_nhl.py "${{ github.event.schedule == '0 15 * * *' && 'morning' || (github.event.inputs.mode || 'pregame') }}"
""" + SAVE % ("nhl-bot", "nhl-bot", "nhl run")
MODEL_FLOW = """name: football
on:
  schedule:
    # NFL (times are UTC; the comment gives Eastern winter time)
    - cron: "15 14 * * 1,2"   # Mon, Tue 9:15 AM: review finished slates
    - cron: "15 15 * * 3"     # Wed 10:15 AM: hub refresh
    - cron: "55 13 * * 1"     # Mon 8:55 AM: first build of tonight's Monday game (ready by about 9)
    - cron: "30 0 * * 5,2"    # Thu, Mon 7:30 PM: tonight's game (Monday: rerun with inactives, news and line moves)
    - cron: "20 15 * * 5"     # Fri 10:20 AM: first run of the Sunday slate
    - cron: "30 13 * * 0"     # Sun 8:30 AM: Sunday slate again
    - cron: "45 16 * * 0"     # Sun 11:45 AM: after inactives
    - cron: "45 20 * * 0"     # Sun 3:45 PM: line snapshot
    - cron: "5 1 * * 1"       # Sun 8:05 PM: line snapshot
    # college
    - cron: "15 16 * * 4"     # Thu 11:15 AM: first sims of the week
    - cron: "10 14 * * 6"     # Sat 9:10 AM: rerun
    - cron: "10 15 * * 0"     # Sun 10:10 AM: grade the week
  workflow_dispatch:
    inputs:
      mode:
        description: "nfl-review, nfl-hub, nfl-today, nfl-next, nfl-odds, cfb-run, cfb-review or hub"
        default: hub
  push:
    paths: [".github/workflows/football.yml", "model/run_once.txt"]
permissions:
  contents: write
concurrency:
  group: model
  cancel-in-progress: false
jobs:
  football:
    runs-on: ubuntu-latest
    timeout-minutes: 90
    env:
      TZ: America/New_York
    steps:
""" + SETUP + """      - uses: actions/cache@v4
        with:
          path: |
            model/data/slate_*.pkl
            model/cfb/data/*.pkl
            model/cfb/data/games
          key: sims-${{ github.run_id }}
          restore-keys: sims-
      - run: python cloud_model.py "${{ github.event.schedule }}" "${{ github.event.inputs.mode }}"
""" + SAVE % ("model-bot", "model-bot", "football run")


def git(*a):
    return subprocess.run(["git", "-C", SITE, *a], capture_output=True, text=True)


def newest(base, which):
    f = [x for p in MARK[which] for x in glob.glob(os.path.join(base, p))]
    return max(map(os.path.getmtime, f), default=0)


def cloud_ran(which):
    f = os.path.join(MODEL, STAMP[which])
    if not os.path.exists(f):
        return 0
    return datetime.strptime(json.load(open(f))["ran"], "%Y-%m-%dT%H:%M:%S%z").timestamp()


def mirror(src, dst, skip=()):
    """Copy a file or folder, only the files that are missing or differ in size or time, leaving out anything matching SKIP or skip."""
    if os.path.isfile(src):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        s = os.stat(src)
        if not os.path.exists(dst) or os.path.getsize(dst) != s.st_size or abs(os.path.getmtime(dst) - s.st_mtime) > 1:
            shutil.copy2(src, dst)
        return
    ign = shutil.ignore_patterns(*SKIP, *skip)
    for root, dirs, files in os.walk(src):
        drop = ign(root, dirs + files) if root == src else shutil.ignore_patterns(*SKIP)(root, dirs + files)
        dirs[:] = [d for d in dirs if d not in drop]
        for f in files:
            if f not in drop:
                mirror(os.path.join(root, f), os.path.join(dst, os.path.relpath(root, src), f))


def pull():
    """Bring GitHub's outputs and data down to this project for each part GitHub ran more recently than this computer."""
    if not os.path.isdir(os.path.join(SITE, ".git")) or not git("remote").stdout.strip():
        return "no site repository"
    r = git("pull", "--rebase", "-q")
    if r.returncode:
        return "pull failed: " + r.stderr.strip()[:200]
    said = []
    for which, name in (("model", "NFL + college"), ("nba", "NBA"), ("nhl", "NHL")):
        ran, mine = cloud_ran(which), newest(ROOT, which)
        if not ran:
            said.append(f"{name}: GitHub has not run yet")
        elif abs(ran - mine) < 1:
            said.append(f"{name}: in sync")
        elif ran < mine:
            said.append(f"{name}: this computer is newer, kept")
        else:
            for d, skip in STATE[which]:
                if os.path.isdir(os.path.join(MODEL, d)):
                    mirror(os.path.join(MODEL, d), os.path.join(ROOT, d), skip)
            for p in MARK[which]:  # stamp as "as of the GitHub run": in sync now, and a later run here counts as newer
                for f in glob.glob(os.path.join(ROOT, p)):
                    os.utime(f, (ran, ran))
            said.append(f"{name}: synced from GitHub")
    return "; ".join(said)


pull_nba = pull  # older name, still imported by season.py copies


def push_model():
    """Code and settings always go up. Outputs and data go up for each part where this computer ran more recently (or GitHub never has)."""
    os.makedirs(MODEL, exist_ok=True)
    for sub in ("", "cfb", "nba", "nhl") if FOOTBALL_PUBLIC else ("nba", "nhl"):
        for f in glob.glob(os.path.join(ROOT, sub, "*.py")):
            mirror(f, os.path.join(MODEL, sub, os.path.basename(f)))
    for f in ("hub_template.html", "slate.json", "cfb/qb.json", "ucl/model.json", "README.md", "SCHEDULE.md"):
        if FOOTBALL_PUBLIC and os.path.exists(os.path.join(ROOT, f)):
            mirror(os.path.join(ROOT, f), os.path.join(MODEL, f))
    newer = {}
    for which in STATE:
        if which == "model" and not FOOTBALL_PUBLIC:
            newer[which] = True  # GitHub does not run football yet: this computer's pages are the ones to publish
            continue
        newer[which] = newest(ROOT, which) >= cloud_ran(which) - 1  # in sync counts: pages built here from the same data carry any newer layout
        if newer[which]:
            for d, skip in STATE[which]:
                mirror(os.path.join(ROOT, d), os.path.join(MODEL, d), skip)
    return newer


def copy_hub(hub, site, skip=()):
    """Put a built hub folder at the site root: the page (wrapped as a full HTML document), its data files and the Game Books."""
    for name in os.listdir(hub):
        src = os.path.join(hub, name)
        if name in ("files.json", "scores.json", "reviews.json") or name.startswith("_") or name in skip:
            continue
        if os.path.isdir(src):
            shutil.copytree(src, os.path.join(site, name), dirs_exist_ok=True)
        elif name == "index.html":
            open(os.path.join(site, name), "w").write(HEAD + open(src).read() + "\n</body></html>\n")
        else:
            shutil.copy2(src, os.path.join(site, name))


def main():
    os.makedirs(os.path.join(SITE, ".github", "workflows"), exist_ok=True)
    if not os.path.isdir(os.path.join(SITE, ".git")):
        subprocess.run(["git", "init", "-q", "-b", "main", SITE], check=True)
    print("site:", pull())
    newer = push_model()
    # where GitHub's run is the newer one, the page it built is newer too: leave those files alone
    keep = set()
    if not newer["nba"]:
        keep |= {n for n in NBA_HUB if os.path.exists(os.path.join(SITE, n))}
    if not newer["nhl"]:
        keep |= {n for n in NHL_HUB if os.path.exists(os.path.join(SITE, n))}
    copy_hub(HUB, SITE, keep)
    for f in ("cloud_scores.py", "cloud_nba.py", "cloud_nhl.py") + (("cloud_model.py",) if FOOTBALL_PUBLIC else ()):
        shutil.copy2(os.path.join(ROOT, f), os.path.join(SITE, f))
    for name, text in (("scores.yml", SCORES_FLOW), ("nba.yml", NBA_FLOW), ("nhl.yml", NHL_FLOW)) + ((("football.yml", MODEL_FLOW),) if FOOTBALL_PUBLIC else ()):
        open(os.path.join(SITE, ".github", "workflows", name), "w").write(text)
    open(os.path.join(SITE, ".nojekyll"), "w").write("")
    open(os.path.join(SITE, ".gitignore"), "w").write(IGNORE)
    if not os.path.exists(os.path.join(SITE, "scores.json")):
        subprocess.run([sys.executable, os.path.join(ROOT, "cloud_scores.py"), SITE], check=False)
    git("add", "-A")
    if git("status", "--porcelain").stdout.strip():
        git("commit", "-q", "-m", "hub update")
        print("site: committed")
    else:
        print("site: no change")
    if "--no-push" in sys.argv or not git("remote").stdout.strip():
        print("site: not pushed (no GitHub remote yet)" if "--no-push" not in sys.argv else "site: push skipped")
        return
    git("pull", "--rebase", "-q")
    r = git("push", "-q", "origin", "main")
    print("site: pushed" if r.returncode == 0 else "site: push failed: " + r.stderr.strip()[:300])


if __name__ == "__main__":
    main()
