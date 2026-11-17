# Weekly cadence (Eastern time)

Scheduled tasks in the Claude app run these; each runs only while the app is open (a missed run starts on next launch).
The app adds a few minutes of delay to recurring tasks.

| Task | Runs at (with the app's delay) | Command |
|---|---|---|
| Tuesday review | Tue 9:13 AM | `slate.py review`, then `slate.py checkpoint`, republish hub |
| Wednesday hub refresh | Wed 10:14 AM | `slate.py hub refresh`, republish hub |
| Thursday TNF run | Thu 7:35 PM | `slate.py run <today>` if there is a game |
| Friday Sunday-slate run | Fri 10:14 AM | news check, `slate.py run <sunday>` |
| Sunday rerun after inactives | Sun 11:55 AM | news check, `slate.py run <today>` |
| Sunday late-game lines | Sun 3:55 PM | `slate.py odds` |
| Sunday night lines | Sun 8:13 PM | `slate.py odds` |
| Monday MNF run | Mon 7:31 PM | `slate.py run <today>` |

Every run and review task also republishes the season hub to the `hub` URL in `output/artifacts.json`.
| Five-week checkpoint (once) | Tue Dec 8, 9:45 AM | `slate.py checkpoint` after the Week 13 review |

All tasks exist in the app (Scheduled section) and were saved in Auto mode. Edit them there; the prompts below are copies.

## Keeping stats current

Stats, averages and ranks are rebuilt from fresh downloads after every round of games:

| Sport | When | What refreshes |
|---|---|---|
| NFL | Mon 9:24 AM, Tue 9:13 AM, Wed 10:14 AM | `fetch_data.py results` re-downloads nflverse player stats, play-by-play, snap counts, injuries, depth charts, rosters and weekly rosters, then `season.py` rebuilds Player lookup, WR/CB, Team vs position, Fantasy |
| CFB | Sun 10:13 AM (plus Thu and Sat runs) | `fetch_cfb.py` + `parse_cfb.py` pull every finished game's box score and plays; `season.py` rebuilds cfb.json (Team vs position, weeks) and cfbl.json (Season leaders, defense vs position) |
| NBA | Every day 9:48 AM (`nba-morning-grade`), 5:42 PM (`nba-evening-sim`), 9:25 PM (`nba-late-resim`) | `fetch_nba.py` (finals) + `parse_nba.py` + `shots_nba.py` + `review_nba.py`; `season.py` rebuilds Season leaders, matchup ranks, shot zones and the Track record. The morning and evening runs both do a full fetch and grade every forecast night that has no review yet, so one missed run is caught up by the next. |

Every `season.py` run ends with three `Data freshness` lines (OK / STALE); the tasks report any STALE line. A sport is STALE when a game that started long enough ago has no final score on file (NBA 8 hours, NFL 36 hours, college 60 hours), when player stats trail the finished games, or when a finished slate with saved forecasts was never graded.

**Auto mode matters.** A scheduled task only runs unattended if it was saved in Auto mode (the create result says so). The first three NBA tasks (Nov 10) were not, so every run paused at its first command and none ever finished; they were replaced on Nov 12. After creating any task, check the result says "saved in Auto mode", and check its first run finished.
NFL downloads now replace a file only when the download succeeds, so a failed or partial download keeps the last good copy.

## Task prompts

Each prompt is self-contained because every run starts fresh.

### Tuesday review (cron `0 9 * * 2`)
You run the weekly postgame review (Phase 15) for Josh's NFL prediction program at /Users/joshuacruz/Desktop/TheBrogram/program. Read README.md and PROMPT.md there first.
1. `cd /Users/joshuacruz/Desktop/TheBrogram/program && venv/bin/python slate.py review`. It refreshes nflverse results and scores every slate of the last finished week that has saved forecasts. If a slate has no results yet (Monday-night stats post some time Tuesday), say so and score the rest.
2. `venv/bin/python slate.py checkpoint` for running totals.
3. Report briefly in plain language: per slate, Brier vs base rate by market, calibration (did 70% calls hit about 70%?), and the Most likely tier's hit rate; the biggest misses sorted into volume, efficiency, game script, injury/inactive, or variance (check inactives first); closing line value where available; and any systematic problem, as a suggestion only.
Rules: never invent stats; separate facts from estimates; never say guaranteed. Do not change model code (changes go through factor_study.py and need Josh's OK) and do not commit. If a step fails, report the error output.

### Thursday TNF run (cron `30 19 * * 4`) and Monday MNF run (cron `30 19 * * 1`)
You run tonight's NFL game for Josh's prediction program at /Users/joshuacruz/Desktop/TheBrogram/program. Read README.md and PROMPT.md there first.
1. Get today's date (`date +%Y%m%d`) and check `venv/bin/python slate.py next` lists a game today; if not, stop.
2. Check confirmed inactives and QB or backfield news (ESPN or NFL.com). Add confirmed changes to slate.json under "slates" → today's date: "notes" by team (a key ending in "_" only shows the note) and "hold_rushing" for backfields with changing roles. Keep the JSON valid.
3. `venv/bin/python slate.py run <today>`.
4. Publish output/nfl_<today>_report.html with the Artifact tool if available, reusing the URL for today in output/artifacts.json if present (record it there); else give the file path.
5. Report: link, projected score and win chance, the Most likely tier, any Value picks with minimum price, any Traps, and questionable players.
Rules: never invent stats, odds or injuries; never say guaranteed. Do not change model code or commit. Report errors as they are.

### Sunday rerun after inactives (cron `45 11 * * 0`)
You rerun today's Sunday NFL slate after inactives for Josh's prediction program at /Users/joshuacruz/Desktop/TheBrogram/program. Read README.md and PROMPT.md there first.
1. Get today's date (`date +%Y%m%d`) and check `venv/bin/python slate.py next` lists games today; if not, stop.
2. Check confirmed inactives and late QB or backfield news. Add confirmed changes to slate.json under "slates" → today's date. Keep the JSON valid.
3. `venv/bin/python slate.py run <today>`. The first run's picks stay for closing line value; this writes picks_<date>_final.csv and logs a pre-kickoff odds snapshot.
4. Republish output/nfl_<today>_report.html, reusing the URL in output/artifacts.json for today if present; else publish and record it, or give the file path.
5. Report what changed since Friday (inactives, QB changes, line moves) and the updated Most likely picks for the 1 PM games, with picks to drop called out first. Skip a London game that has already started.
Rules as above.

### Sunday late-game lines (cron `45 15 * * 0`)
Run `cd /Users/joshuacruz/Desktop/TheBrogram/program && venv/bin/python slate.py odds` to log a line snapshot before the late-afternoon kickoffs. Report only the line it prints, or the error. Do not change any files.

### CLV checkpoint (one time, Tue Dec 8 2026 9:30 AM ET)
At /Users/joshuacruz/Desktop/TheBrogram/program run `venv/bin/python slate.py checkpoint` after the Tuesday review. Tell Josh whether, after about five live weeks, (1) the Most likely tier hits near its predicted rate and (2) first-run picks beat closing lines. If closing line value is positive and most picks moved his way, the Value tier has earned some trust and a paid odds feed with prop prices is worth considering; if not, keep using the Most likely tier and treat Value as leads only.

## The agent team (Nov 13, 2026)

Four agent definitions live in `~/.claude/agents/` and are called by the scheduled jobs (each job's SKILL.md ends with a "Team" block):
- `sports-news-desk` runs the lineup and injury check before every NFL, college and NBA simulation.
- `sports-data-checker` runs after each rebuild; a job does not publish if it answers "Do not publish".
- `sports-grader` does the grading in the review jobs and writes the report in a fixed order.
- `sports-skeptic` must rule on any claimed edge or improvement before a job reports it (Holds, Lead or Fails).
If an agent is unavailable in a run, the job does that part itself and says so.

## Public page and live scores (Nov 13, 2026)

The hub is also served at https://the-brogram.github.io/AllThingsSports/ from the public repository `The-Brogram/AllThingsSports` (local folder `../site`).
- `publish_site.py` copies the built hub there and pushes it; every scheduled job runs it after its rebuild. Only the built page, its data files and the Game Books go public, never model code or bet logs.
- On GitHub, the `scores` workflow runs `cloud_scores.py` every 10 minutes (no computer needed) and saves `scores.json` when a score changes. The hub page reads it on load and shows live scores and finals its own data does not have yet.
- Stats, grades and forecasts are still rebuilt by the jobs on this Mac. The push uses the GitHub token stored in the Mac keychain; when it expires, `publish_site.py` prints "push failed".

## NBA runs on GitHub (since Nov 13, 2026)

The NBA no longer needs this computer. The public site repository (`../site`, The-Brogram/AllThingsSports) holds the NBA code, data and
outputs under `model/nba`, and its "nba" workflow runs `cloud_nba.py` at 9:45 AM, 5:30 PM and 9:15 PM ET (fetch, parse, shots, grade, sim,
rebuild the NBA hub files, commit). Rules:

- NBA code is edited here; `publish_site.py` copies `nba/*.py` up on every push.
- Data and outputs follow "newest run wins": `season.py` and `publish_site.py` pull GitHub's copy down when GitHub ran last, and push this
  computer's copy up when a run here is newer. `model/nba/output/cloud_stamp.json` records GitHub's last run and any failed step.
- The NBA code and data are public in that repository. NFL and college code are not there.
- GitHub cannot publish to the Claude artifact link; that copy updates only when a job here runs.
- The three desktop NBA jobs stay on as a fallback until GitHub's schedule has fired on its own for a week.

## NFL and college on GitHub (switched on Nov 13, 2026)

`cloud_model.py` and the "football" workflow (written by `publish_site.py`) run the NFL and college jobs on GitHub on the same timetable
as the desktop jobs. Josh approved making the code, data and pick records public (`FOOTBALL_PUBLIC = True`). First run on GitHub:
Sunday slate in under 6 minutes, all 1,686 college game files downloaded in 21 seconds. Each run commits `model/logs/football.log`.
To run a job by hand without a GitHub login: put job names (one per line) in `site/model/run_once.txt` and publish.
Known limits: no lineup or quarterback news check on GitHub (slate.json and cfb/qb.json come from this computer), and the college sim
is skipped for a week already forecast when its simulation cache is not on GitHub (so started games keep their original forecasts).
The score checker runs only in game hours, every 15 minutes (nightly 7 PM to 2 AM ET, weekend afternoons, two weekday catch-ups).

NFL post-game reviews: `nfl_reviews.py` builds `nflrev.json` for every finished game (kept in `output/nfl_reviews.json`, fetched once each).

Monday night game (since Nov 16, 2026): GitHub builds the Game Book Monday at about 9 AM ET (cron `55 13 * * 1`) and again at 7:30 PM with
inactives, news and line moves. The morning run is the first run, so its picks are the ones closing line value is measured from.
