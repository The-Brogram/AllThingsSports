"""Offensive line vs defensive line: can this line protect its quarterback against this pass rush?

Free data only:
  - pressures, hurries, QB hits, sacks per game: PFR advanced passing (nflverse pfr_advstats), per QB, summed per team
  - dropbacks and sacks: nflverse play-by-play (regular season)
  - time to throw: NFL Next Gen Stats (nflverse ngs_passing), weighted by attempts
  - blitz rate and pass rushers per dropback: FTN charting (2026)
  - starters: nflverse depth chart; status: this week's injury report
Pressure rates are shrunk toward the league average (K dropbacks of league-average play) so a few games do not swing them.
The matchup's expected pressure combines the two rates on the odds scale: odds(exp) = odds(line allows) x odds(rush gets) / odds(league).
Individual tackle-vs-edge win rates are paid charting data and are not included.
"""
import os
import numpy as np
import pandas as pd
from engine import DATA

ROOT = os.path.dirname(os.path.abspath(__file__))
K = 150  # dropbacks of league-average pressure added to each team's rate
OL = ["LT", "LG", "C", "RG", "RT"]
_CACHE = {}


def odds(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return p / (1 - p)


def combine(ol, dl, lg):
    o = odds(ol) * odds(dl) / odds(lg)
    return o / (1 + o)


def team_games(season):
    """One row per team-game, offense view: dropbacks, pressures, hits, hurries, sacks allowed, attempts, opponent."""
    if season in _CACHE:
        return _CACHE[season]
    p = pd.read_csv(f"{DATA}/play_by_play_{season}.csv.gz", low_memory=False,
                    usecols=["game_id", "season_type", "week", "posteam", "defteam", "qb_dropback", "sack", "pass_attempt"])
    p = p[(p.season_type == "REG") & (p.qb_dropback == 1)]
    db = p.groupby(["game_id", "week", "posteam", "defteam"]).agg(db=("qb_dropback", "size"), sk=("sack", "sum")).reset_index()
    a = pd.read_csv(f"{DATA}/advstats_week_pass_{season}.csv")
    a = a[a.game_type == "REG"].groupby(["game_id", "team"]).agg(pr=("times_pressured", "sum"), hit=("times_hit", "sum"),
                                                                    hur=("times_hurried", "sum"), blz=("times_blitzed", "sum")).reset_index()
    x = db.merge(a, left_on=["game_id", "posteam"], right_on=["game_id", "team"], how="inner").drop(columns="team")
    x = x.rename(columns={"posteam": "off", "defteam": "dfn"}).sort_values("week")
    _CACHE[season] = x
    return x


def rates(rows, lg):
    """Shrunk pressure rate plus raw sack and hit rates for a set of team-games."""
    db = rows.db.sum()
    return dict(g=len(rows), db=int(db), pr=(rows.pr.sum() + K * lg["pr"]) / (db + K), pr_raw=rows.pr.sum() / max(db, 1),
                sk=rows.sk.sum() / max(db, 1), hit=rows.hit.sum() / max(db, 1), blz=rows.blz.sum() / max(db, 1))


def league(x):
    return dict(pr=x.pr.sum() / x.db.sum(), sk=x.sk.sum() / x.db.sum(), hit=x.hit.sum() / x.db.sum(), blz=x.blz.sum() / x.db.sum())


def profile(season, thru, off, dfn, last=4):
    """Rates through week `thru` for `off`'s line and `dfn`'s rush, season and last `last` games, plus the matchup."""
    x = team_games(season)
    x = x[x.week <= thru]
    lg = league(x)
    o, d = x[x.off == off], x[x.dfn == dfn]
    out = dict(lg=lg)
    for key, oo, dd in (("season", o, d), ("last", o.tail(last), d.tail(last))):
        ro, rd = rates(oo, lg), rates(dd, lg)
        out[key] = dict(ol=ro, dl=rd, exp=float(combine(ro["pr"], rd["pr"], lg["pr"])))
    return out


def ranks(season, thru):
    """League ranks, 1 = best: protection (lowest pressure / sack / hit allowed) and pass rush (highest generated)."""
    x = team_games(season)
    x = x[x.week <= thru]
    lg = league(x)
    o = {t: rates(g, lg) for t, g in x.groupby("off")}
    d = {t: rates(g, lg) for t, g in x.groupby("dfn")}
    rk = lambda m, k, hi: {t: i + 1 for i, (t, _) in enumerate(sorted(m.items(), key=lambda kv: -kv[1][k] if hi else kv[1][k]))}
    return dict(ol={k: rk(o, k, False) for k in ("pr", "sk", "hit")}, dl={k: rk(d, k, True) for k in ("pr", "sk", "hit")}, blz_ol=rk(o, "blz", True))


def ttt(season, team, qb_id=None):
    """Average time to throw (NGS), weighted by attempts; for one QB if given, else the team."""
    n = pd.read_csv(f"{DATA}/ngs_passing.csv.gz")
    n = n[(n.season == season) & (n.season_type == "REG") & (n.week > 0) & (n.team_abbr == team)]
    if qb_id is not None and (n.player_gsis_id == qb_id).any():
        n = n[n.player_gsis_id == qb_id]
    return float((n.avg_time_to_throw * n.attempts).sum() / n.attempts.sum()) if len(n) and n.attempts.sum() else np.nan


def lg_ttt(season):
    n = pd.read_csv(f"{DATA}/ngs_passing.csv.gz")
    n = n[(n.season == season) & (n.season_type == "REG") & (n.week > 0)]
    return float((n.avg_time_to_throw * n.attempts).sum() / n.attempts.sum())


def qb_pressure(season, team, qb_name):
    """This QB's own pressure-to-sack rate and pressure rate with this team (PFR names are 'F.Last' style or full)."""
    a = pd.read_csv(f"{DATA}/advstats_week_pass_{season}.csv")
    a = a[(a.team == team) & (a.game_type == "REG")]
    last = qb_name.split()[-1]
    q = a[a.pfr_player_name.str.contains(last, na=False)]
    return dict(g=len(q), pr=int(q.times_pressured.sum()), sk=int(q.times_sacked.sum()))


def rushers(season, team, n=5):
    a = pd.read_csv(f"{DATA}/advstats_week_def_{season}.csv")
    a = a[(a.team == team) & (a.game_type == "REG")]
    g = a.groupby("pfr_player_name").agg(pr=("def_pressures", "sum"), sk=("def_sacks", "sum"), hit=("def_times_hitqb", "sum"),
                                          hur=("def_times_hurried", "sum"), g=("week", "nunique"), last=("week", "max")).reset_index()
    tot = a.def_pressures.sum()
    g["share"] = g.pr / max(tot, 1)
    return g.sort_values("pr", ascending=False).head(n).to_dict("records")


# ---------------- Game Book tab ----------------
EDGE = 0.02  # expected pressure this far above / below league average tips the call (about the backtest's thirds)


def verdict(exp, lg):
    d = exp - lg
    return ("Rush favored", "bad") if d >= EDGE else (("Line favored", "good") if d <= -EDGE else ("Even", "mute"))


def _bt():
    f = os.path.join(ROOT, "output", "lines_backtest.json")
    return __import__("json").load(open(f)) if os.path.exists(f) else None


def tab(c, gg, s, a, h, week, ui, season=2026):
    """The Lines tab: both pass-protection matchups side by side, starters, pass rushers, and what the backtest says."""
    mrow, better, T, CTX, PRED, e, pc, n, qb_change = (ui[k] for k in ("mrow", "better", "T", "CTX", "PRED", "e", "pc", "n", "qb_change"))
    R = ranks(season, week)
    inj = {t: {x["name"]: x for x in s["inj"].get(t, [])} for t in (a, h)}
    lgt = lg_ttt(season)
    st = lambda t, nm: (inj[t].get(nm) or {}).get("status")
    chip = lambda v: f' <span class="tag {"warn" if v in ("Out", "Injured Reserve", "Doubtful") else "mute"}">{e("IR" if v == "Injured Reserve" else v)}</span>' if v and v != "Active" else ""
    out, summary = [], []
    for off, dfn in ((a, h), (h, a)):
        P = profile(season, week, off, dfn)
        S, Lr, lg = P["season"], P["last"], P["lg"]
        lab, cls = verdict(S["exp"], lg["pr"])
        q = gg.qb(off)
        qn = q["name"] if q else "the starting QB"
        qt = ttt(season, off, q.get("player_id") if q else None)
        ol = c.oline(off, gg.c_status)
        for y in ol:
            if y["status"] in (None, "Active") and st(off, y["name"]):
                y["status"] = st(off, y["name"])
        miss = [y for y in ol if y["status"] in ("Out", "Injured Reserve", "Doubtful")]
        qq = [y for y in ol if y["status"] == "Questionable"]
        ru = rushers(season, dfn)
        summary.append((off, dfn, S["exp"], lab, cls, len(miss)))
        head = (f'<div class="lvh"><div><small>{off} pass protection vs {dfn} pass rush</small><b>{pc(S["exp"])} <span>expected pressure on {e(qn)}\'s dropbacks</span></b>'
                f'<span class="note">League average {pc(lg["pr"])} · last 4 games of each: {pc(Lr["exp"])}</span></div><span class="tag {cls} lvt">{lab}</span></div>')
        warn = ""
        if off in qb_change:
            warn += (f"<p class='warn'>{e(off)} is starting {e(qn)} in place of the usual starter. Its protection numbers come mostly from games with a different "
                     f"quarterback, and pressure depends partly on how long the QB holds the ball, so read them with care.</p>")
        if miss:
            warn += f"<p class='warn'>{e(off)} line is missing {len(miss)} listed starter{'s' if len(miss) > 1 else ''}: " + ", ".join(f"{y['pos']} {e(y['name'])} ({e(y['status'])}; next up {e(y['backup'])})" for y in miss) + ". The season numbers were mostly built with them on the field.</p>"
        hd = (f'<div class="mcols"><div class="l"><b>{off} offensive line</b><small>what {off} allows · rank 1 = best protection</small></div>'
              f'<div class="lab">2026 · {S["ol"]["g"]} and {S["dl"]["g"]} games</div><div class="r"><b>{dfn} defensive front</b><small>what {dfn} creates · rank 1 = most pressure</small></div></div>')
        rws = [hd]
        for k, labl in (("pr", "Pressure rate<small>per dropback, shrunk to league</small>"), ("sk", "Sack rate"), ("hit", "QB hit rate")):
            lr, rr = R["ol"][k].get(off), R["dl"][k].get(dfn)
            rws.append(mrow(labl, pc(S["ol"][k], 1), pc(S["dl"][k], 1), lr, rr, better(lr, rr)))
        rws.append(mrow("Last 4 games<small>pressure rate</small>", pc(Lr["ol"]["pr"], 1), pc(Lr["dl"]["pr"], 1)))
        rws.append(mrow(f"Blitz rate<small>{off} faced · {dfn} sends</small>", pc(S["ol"]["blz"], 1), pc(S["dl"]["blz"], 1)))
        rws.append(mrow(f"Time to throw<small>{e(qn)} · league {lgt:.2f}s</small>", f"{qt:.2f}s" if qt == qt else "–", ""))
        olt = T(["Spot", "Expected starter", "Status", "Snap % last 3", "Next up"],
                [[y["pos"], e(y["name"]) + chip(y["status"]), e(y["status"] or "Active"),
                  pc((y["snap3"] if y["snap3"] and y["snap3"] <= 1 else (y["snap3"] or 0) / 100)), e(y["backup"])] for y in ol])
        rut = T(["Pass rusher", "Games", "Pressures", "Per game", "Sacks", "QB hits", "Share of team pressure"],
                [[e(x["pfr_player_name"]) + chip(st(dfn, x["pfr_player_name"])), x["g"], int(x["pr"]), n(x["pr"] / max(x["g"], 1), 1), n(x["sk"], 1), int(x["hit"]), pc(x["share"])] for x in ru])
        sim = gg.side(off)
        rec = sorted([p for p in gg.pl if p["team"] == off and p["pos"] in ("WR", "TE")], key=lambda p: -p["rec_y"].mean())[:3]
        pj = (f"<p class='note'>Model's {off} passing game for this matchup: {e(qn)} {n(q['pass_y'].mean(), 0) if q else '–'} pass yards, {n(sim['sacks'].mean(), 1)} sacks; "
              + ", ".join(f"{e(p['name'])} {n(p['rec_y'].mean(), 0)} rec yds" for p in rec)
              + ". These already include each defense's sack rate; the pressure matchup itself is not added (see the backtest below).</p>") if q else ""
        out.append(f'<section class="mblock"><h4>{off} protection vs {dfn} rush {CTX}</h4>{head}{warn}{"".join(rws)}'
                   f'<div class="pcols"><div class="pcol"><h5>{off} starting line</h5>{olt}{"<p class=note>Questionable: " + ", ".join(e(y["name"]) for y in qq) + ". Check inactives.</p>" if qq else ""}</div>'
                   f'<div class="pcol"><h5>{dfn} top pass rushers</h5>{rut}</div></div>{pj}</section>')
    top = "".join(f'<div class="lsum"><small>{o} line vs {d} rush</small><b>{pc(x)}</b><span class="tag {cl}">{lab}</span>{f"<em>{m} OL starter{"s" if m > 1 else ""} out</em>" if m else ""}</div>'
                  for o, d, x, lab, cl, m in summary)
    b = _bt()
    btxt = ""
    if b:
        th, ty, pr, ps = b["thirds"], b["thirds_yards"], b["pressure"], b["passing"]
        btxt = (f'<section class="mblock"><h4>Does the line matchup predict? <span class="kind pred">Pressure: yes</span> <span class="kind ctx">Passing yards: no edge</span></h4>'
                f'<p class="note" style="margin:0 0 8px">Rebuilt game by game from data before each game, {b["n"]} team-games ({e(b["seasons"])}).</p>'
                + T(["Matchup call", "Team-games", "Expected pressure", "Actual pressure", "Actual sack rate", "QB yds vs model", "Receiving yds vs model"],
                    [[x["group"], x["n"], pc(x["exp"], 1), pc(x["act"], 1), pc(x["sack"], 1), f'{y["qb_res"]:+.1f}', f'{y["rec_res"]:+.1f}'] for x, y in zip(th, ty)])
                + f'<p class="note">The matchup does forecast pressure: correlation {pr["r_matchup"]:.2f} with the pressure that happened (90% range {pr["ci_matchup"][0]:.2f} to {pr["ci_matchup"][1]:.2f}), '
                f'better than the line alone ({pr["r_line_only"]:.2f}) or the rush alone ({pr["r_rush_only"]:.2f}). Rush-favored games had more sacks. '
                f'But it did not predict passing yards beyond the model: correlation with QB yards minus projection {ps["r_qb"]:+.2f} (90% range {ps["ci_qb"][0]:+.2f} to {ps["ci_qb"][1]:+.2f}), '
                f'with receiving yards {ps["r_rec"]:+.2f}. (Every row beat the model on receiving yards by a similar amount, a general lean in those backtest weeks; the gap between rows is what would show a line effect, and it is small.) Use it to judge sacks, hurried throws and whether a QB has time, not to move a yardage pick on its own.</p></section>')
    intro = (f'<section class="mblock"><h4>Line matchup at a glance {CTX}</h4><div class="lsums">{top}</div>'
             f'<p class="note">Expected pressure combines what each offensive line allows with what the opposing front creates, compared with the league average. '
             f'“Line favored” and “Rush favored” mean at least {int(EDGE * 100)} points below or above league average. Pressures come from PFR charting (hurries, QB hits and sacks), shrunk toward league average early in the season. '
             f'Individual tackle-vs-edge win rates are paid data and are not included.</p></section>')
    return intro + "".join(out) + btxt
