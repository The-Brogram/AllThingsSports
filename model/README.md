# NFL model

Season and week come from the schedule, so a new week needs no setup. The analysis prompt is `PROMPT.md`.

    python3 -m venv venv && venv/bin/pip install pandas numpy   # once
    venv/bin/python slate.py next             # upcoming slates and their dates
    venv/bin/python slate.py run [date]       # refresh data, 20,000 sims per game, build the report (default: next slate)
    venv/bin/python slate.py odds [date]      # line snapshot just before kickoff, for closing line value
    venv/bin/python slate.py review [date]    # after the games (default: every slate of the last finished week)
    venv/bin/python slate.py checkpoint       # live calibration and closing line value across all reviewed slates
    venv/bin/python slate.py hub [refresh]    # rebuild the season hub; "refresh" pulls the latest schedule and lines first

Weekly cadence (ET): Tuesday review; Thursday run before TNF; Friday first Sunday run; Sunday rerun after inactives
plus odds snapshots before the late and night games; Monday run before MNF. After 4-6 weeks, `checkpoint` decides
whether the Value tier has earned trust.

Outputs land in `output/` by date: `nfl_<date>_report.html`, `predictions_<date>.csv`, `picks_<date>.csv` (first run, used
for closing line value; reruns write `picks_<date>_final.csv`). `odds_log.csv` keeps every line snapshot, and
`track_record.csv` keeps every scored forecast (rerunning a review replaces that slate's rows).
`backtest.py` refits calibration; `factor_study.py` tests factors and model changes (see its docstring).
Downloaded data and simulation caches live in `data/`, which git ignores.

Season hub (`season.py` -> `output/hub/`): one page for every game of the season, with week tabs, a side-by-side panel per
game, a compare-any-two-teams view and the live track record. Each game shows the best number available: the final score,
the Game Book (`report.py` saves `summary_<date>.json` for it), 90% market / 10% ratings once nflverse posts a line, or an
early ratings-only projection. `run` and `review` rebuild it. Publish `output/hub/index.html` with the files map in
`output/hub/files.json` to the `hub` URL in `output/artifacts.json`, so the link never changes. The page template is
`hub_template.html`.

Late news goes in `slate.json` under `slates` → that date: `notes` by team (a team key flags all its props; a key
ending in `_` only shows the note) and `hold_rushing` for backfields whose roles are changing. When the depth chart lags the news, `qb` {team: gsis_id}
names the starter and `out` {team: [gsis_id]} removes players the injury feed missed. Confirm starting QBs
and players returning from injury before each run.

## WR / CB matchups (hub sidebar → NFL → WR / CB matchups)
`venv/bin/python wrcb.py` builds output/hub/wrcb.json for the coming week (season.py does it automatically).
`venv/bin/python wrcb_backtest.py` re-tests it (needs data/advstats_week_def_2025.csv and data/roster_2025.csv, already downloaded).
Free data only; columns marked EST on the page are estimates. The backtest found no predictive edge, so it is context.

## Player Lookup (hub sidebar → NFL → Player lookup)

`players.py` writes `output/hub/players.json`: every QB, RB, WR and TE game from 2022 through the current 2026 week
(regular season and playoffs), about 1.8 MB. `season.py` rebuilds it on every hub run and lists it in `files.json`,
so the scheduled republish ships it automatically. The hub loads it only when the Player Lookup page opens.
Page: search, 2026 leaders, per-player tiles, season-by-season table, game-by-game chart with a "line to test",
splits against every opponent, and a game log filtered by season, game type, opponent and venue.
To add a sport later, add a `.sport` block to the sidebar in `hub_template.html`.

## Fantasy (hub sidebar → NFL → Fantasy)
fantasy.py scores every simulated stat line from the latest week's slate caches (data/slate_<date>.pkl) into fantasy points, so each player gets a full range: average, floor (10th pct), median, ceiling (90th pct), boom and bust chances, plus 400 linked draws per player for the start/sit tool (same draw index = same simulated game). Scoring is chosen on the page (Standard / Half / PPR, 4- or 6-pt pass TD); fumbles and two-pointers are not simulated. season.py builds output/hub/fantasy.json on every run.
fantasy_backtest.py re-simulates 2025 wk 5-18 and 2026 wk 4-8 walk-forward: the projection beats each player's 8-game average at every position (avg miss 4.72 vs 5.00 PPR pts), the 10-90 band holds 83%, and start/sit chances are calibrated. Raw boom chances ran low and bust chances high, so both get a logistic recalibration (fit 2025, checked on 2026, refit on all; stored in output/fantasy_backtest.json, applied by fantasy.py). Rerun the summary without re-simulating: `venv/bin/python fantasy_backtest.py summary`.

## O-Line / D-Line tab (every Game Book)
lines.py builds each Game Book's O-Line / D-Line tab: both pass-protection matchups side by side. It shows pressure, sack and QB-hit rates allowed vs created (PFR charting + play-by-play, shrunk toward league with 150 dropbacks), last-4-game pressure, blitz rates, the QB's time to throw (NGS), the depth-chart starting five with injury status and next man up, and the defense's top pass rushers. Expected pressure combines the two sides on the odds scale; ±2 points vs league average is the "Line favored" / "Rush favored" call. lines_backtest.py (walk-forward, 562 team-games): the matchup predicts actual pressure (r 0.20, better than either side alone) and sacks, but not QB or receiving yards beyond the model (r −0.04 / −0.03), so it is labeled context for yardage picks.

## Five-season study and what it changed (Nov 2026)

`five_season_study.py`, `five_season_round2.py` and `matchup_backtest.py` test whether a team-vs-team model can pick winners or beat the spread (walk-forward, tuned on 2021-24, confirmed on 2025-26). Findings: the sportsbook favorite is the ceiling on every game (66.5% winners; team models 63.5-64.6%); no team-stat model beat the spread; scoring and quarterback changes are the factors that matter; schedule adjustment, recent form, the running matchup and rest/weather add nothing measurable. The simulation therefore stays centered on the market.

`spread_rules.py` (called by `season.py` on every hub build) applies the two things worth tracking:
- **Confident winners** on the NFL Weeks page: each favorite of 3+ with the 2021-25 win rate of favorites that size (3-3.5: 63%, 4-6.5: 68%, 7-9.5: 74%, 10+: 88%; all 3+: 70.6%).
- **Cash Center** (hub sidebar, top): a personal desk. Pick NFL, College Football or NBA and see that sport's confident winners: every favorite of 3+ on the current slate with the straight-up win rate of favorites that size (`spread_rules.other_conf` builds the college and NBA tables from their games files), our model's win chance beside it, and an "Add to my card" button. The card is a notepad kept in the viewer's browser (localStorage), not a bet slip. Routes: `#cash`, `#cash-nfl`, `#cash-cfb`, `#cash-nba`.
- **Spread rules, tracked live** on the NFL Track record: division underdogs, underdogs getting 7+, teams starting a different quarterback than usual, and every underdog as a baseline. Picks are logged before kickoff in `output/spread_picks.csv` (first line seen is kept; games that have kicked off are never logged) and graded at that line and price. These are leads, not proven edges: break-even at -110 is 52.4%.

## College football (cfb/)

Weekly sims for every game with an AP Top 25 team or an SEC, ACC, Big Ten or Big 12 team, shown in the hub's College Football dropdown (Weeks, Track record, Team vs position).

```
cd cfb
../venv/bin/python fetch_cfb.py      # ESPN week slate (data/sb.json), results, AP poll, Covers injuries; --week N for another week
../venv/bin/python parse_cfb.py
../venv/bin/python weather_cfb.py    # NWS kickoff forecast per outdoor stadium -> output/weather_w<N>.json
../venv/bin/python run_cfb.py        # 20,000 sims per game -> data/cfb_slate_w<N>.pkl, output/predictions_w<N>.csv
../venv/bin/python qb_check.py       # flags QBs to confirm by hand; starters and notes go in qb.json under the week
../venv/bin/python report_cfb.py     # Game Books -> output/cfb_w<N>_report.html, output/summary_w<N>.json
../venv/bin/python review_cfb.py     # after the week: grades player forecasts -> output/review_w<N>.json
cd .. && venv/bin/python season.py   # hub_cfb.py builds output/hub/cfb.json and copies the week Game Books
```
A rerun keeps the pre-kickoff forecast for games already under way. Scheduled: Thursday 11 AM first sims, Saturday 9 AM rerun with news, Sunday 10 AM review.

## NBA (nba/)
Every game every night, hub sidebar → NBA (Tonight, Track record, Season leaders). Season leaders reads output/hub/nbl.json (this season + last, regular season only; All-Star games excluded) and nblh.json (photos). Player profiles also show matchup sections from output/hub/nbm.json (hub_nba.matchups): defense vs G/F/C, shot zones from ESPN play-by-play (nba/shots_nba.py, run after parse_nba.py), and similar players vs the opponent. Data is ESPN only (NBA.com blocks scripts): box scores, DraftKings spread/total in pickcenter, DraftKings player props (core API, 25 per page), and the injury feed.

Pipeline, from `nba/`:
1. `../venv/bin/python fetch_nba.py --today` (or no flag to backfill every date since 2025-10-21), then `../venv/bin/python parse_nba.py` → data/games.csv, players.csv.gz, props.csv.gz.
2. `../venv/bin/python run_nba.py` → output/predictions_<date>.csv and nba_<date>.pkl. 10,000 sims per game.
3. `cd .. && venv/bin/python season.py` builds output/hub/nba.json; republish index.html + nba.json.
4. Next morning: fetch + parse, then `../venv/bin/python review_nba.py` → output/review_<date>.json (shown on Track record).

Model (`nba_model.py`): margin and total come from the DraftKings spread/total plus last season's residuals, so team points are market-centered. Minutes blend each player's recent minutes with the league's minutes for his rank in tonight's rotation (fixes crowded early-season rosters), then starters lose minutes in blowouts. Per-minute rates are recency weighted, lightly shrunk to position averages, adjusted for opponent and pace; points are a split of the simulated team points, so a teammate's absence raises everyone else's share. Negative-binomial noise per stat, fit on DraftKings lines.

Probabilities: `backtest_nba.py` (walk-forward, only earlier games) → `calibrate_nba.py` fits on 2025-26 and tests on 2026-27. DraftKings lines use a blend of sim and DraftKings price; "model lines" (lower N+ thresholds, lines DraftKings has not posted) use the sim alone. Injuries: ESPN only says Out / Day-To-Day, so the blurb is read ("ruled out", "doubtful", "questionable", "probable"); a blurb is ignored once the team has played since it was written. Past-game props exist on ESPN for only ~260 games (mostly April 2026 onward), and its DraftKings feed lists about one milestone per player.

### Proof layer: results at real prices (nba/edge_nba.py)

Hit rate is not profit, so the NBA side is also graded at DraftKings' prices.
- `edge_nba.py log` (called by run_nba.py) records every priced side at the price on offer; the first price seen is kept in `output/picks/<date>.csv.gz`.
- `edge_nba.py grade` (called by review_nba.py) grades each final night at those prices and compares them with the closing prices (closing line value) -> `output/edge_live.json`.
- `edge_nba.py backtest` joins the walk-forward backtest to real prices on games after the blend's fit date -> `output/edge.json`: return per bet with 80% ranges from resampling whole games, claimed value vs realised, and segments. Rerun it after `backtest_nba.py` and `calibrate_nba.py`.
- `factor_nba.py` is the factor study: a model change is tuned on Jan-Jun 2026 and kept only if it also helps on Oct 20, 2026 onward.
- First run (Nov 12, 2026, 154 unseen games): no proven edge. The model's side on every over/under returned -2.0% per bet (blind betting -6.4%); DraftKings' no-vig probability is as sharp as ours (log loss .6861 vs .6866); "N+" lines lose about 13% per bet.
- Data trap found by this test: ESPN files another market (first-quarter rebounds, by the look of it) under "Rebounds Milestones". `parse_nba.drop_wrong_market` rejects any N+ price that is impossible given the same player's main line. Before the fix those rows made the blend look better than DraftKings.
