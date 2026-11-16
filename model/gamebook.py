"""Game Book sections laid out like a sportsbook game page: Overview, Odds, Special Teams, Splits and Lineups.

Everything here is read from data the pipeline already pulls (games.csv, 2026 play-by-play, the depth chart,
ESPN's DraftKings open/close numbers and our own odds snapshots). Nothing is estimated except where labeled.
"""
import json
import math
import os
import numpy as np
import html
import pandas as pd
import engine as E
from config import OUT_IDS, QB_START

ODDS_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "odds_log.csv")


def _e(x):
    import html
    return html.escape(str(x))


def _n(x, d=1):
    return "–" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{d}f}"


def _ml(o):
    return "–" if o is None or (isinstance(o, float) and not math.isfinite(o)) else f"{o:+.0f}"


def _sp(x):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "–"
    return "PK" if x == 0 else f"{x:+g}"


def novig(ml_a, ml_h):
    """Market win chance with the bookmaker margin removed."""
    p = lambda o: abs(o) / (abs(o) + 100) if o < 0 else 100 / (o + 100)
    a, h = p(ml_a), p(ml_h)
    return a / (a + h), h / (a + h)


def _num(s):
    try:
        return float(str(s).lstrip("ou").replace("+", ""))
    except (TypeError, ValueError):
        return None


# ---------- odds ----------

def market_history(g):
    """Open (ESPN/DraftKings), each of our snapshots, and now. Lines are from the away team's side."""
    a, h = g["away"], g["home"]
    pts = []
    f = f"{E.DATA}/espn/sum_{g['id']}.json"
    if os.path.exists(f):
        pcen = (json.load(open(f)).get("pickcenter") or [{}])[0]
        sp, to, ml = pcen.get("pointSpread", {}), pcen.get("total", {}), pcen.get("moneyline", {})
        op = dict(label="Open", spread=_num(sp.get("away", {}).get("open", {}).get("line")), total=_num(to.get("over", {}).get("open", {}).get("line")),
                  ml_a=_num(ml.get("away", {}).get("open", {}).get("odds")), ml_h=_num(ml.get("home", {}).get("open", {}).get("odds")))
        if any(v is not None for k, v in op.items() if k != "label"):
            pts.append(op)
    if os.path.exists(ODDS_LOG):
        L = pd.read_csv(ODDS_LOG, usecols=["snap", "event", "market", "side", "line", "price"], dtype={"event": str})
        L = L[(L.event == str(g["id"])) & L.market.isin(["moneyline", "spread", "total"])]
        for snap, x in L.groupby("snap", sort=True):
            pick = lambda m, s, col: next(iter(x[(x.market == m) & (x.side == s)][col].tolist()), None)
            pts.append(dict(label=pd.Timestamp(snap).tz_convert("America/New_York").strftime("%a %-I:%M %p"), spread=pick("spread", a, "line"), total=pick("total", "over", "line"),
                            ml_a=pick("moneyline", a, "price"), ml_h=pick("moneyline", h, "price")))
    pts.append(dict(label="Now", spread=g["spread_home"], total=g["total"], ml_a=g.get("ml_a"), ml_h=g.get("ml_h")))
    return pts


def _spark(vals, labels, fmt, color, gid, key):
    """Small step chart of a line over time. Labels sit under each point; the value sits above it."""
    ok = [(i, v) for i, v in enumerate(vals) if v is not None and math.isfinite(v)]
    if len(ok) < 2:
        return "<p class='note'>Not enough snapshots yet to chart a move.</p>"
    W, H, px, py = 640, 170, 46, 34
    lo, hi = min(v for _, v in ok), max(v for _, v in ok)
    if hi == lo:
        lo, hi = lo - 1, hi + 1
    X = lambda i: px + i * (W - 2 * px) / max(len(vals) - 1, 1)
    Y = lambda v: py + (hi - v) / (hi - lo) * (H - 2 * py - 10)
    d = "M" + " L".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in ok)
    dots = "".join(f'<circle cx="{X(i):.1f}" cy="{Y(v):.1f}" r="4.5" fill="{color}"/><text x="{X(i):.1f}" y="{Y(v) - 11:.1f}" text-anchor="middle" class="sv">{_e(fmt(v))}</text>' for i, v in ok)
    labs = "".join(f'<text x="{X(i):.1f}" y="{H - 6}" text-anchor="middle" class="sl">{_e(lab)}</text>' for i, lab in enumerate(labels))
    grid = f'<line x1="{px}" x2="{W - px}" y1="{H - 24}" y2="{H - 24}" stroke="var(--line2)"/>'
    return f'<svg class="spark" viewBox="0 0 {W} {H}" role="img" aria-label="{_e(key)} movement">{grid}<path d="{d}" fill="none" stroke="{color}" stroke-width="2.5"/>{dots}{labs}</svg>'


def odds_movement(g, gid):
    a, h = g["away"], g["home"]
    P = market_history(g)
    labels = [p["label"] for p in P]
    views = [("sp", "Spread", _spark([p["spread"] for p in P], labels, _sp, "var(--acc)", gid, "Spread"), f"{a} spread"),
             ("ou", "O/U", _spark([p["total"] for p in P], labels, lambda v: f"{v:g}", "var(--fg)", gid, "Total"), "Game total"),
             ("ml", "Moneyline", _spark([p["ml_a"] for p in P], labels, _ml, "var(--acc2)", gid, "Moneyline"), f"{a} moneyline")]
    btn = "".join(f'<button type="button" class="seg" data-om="{k}">{lab}</button>' for k, lab, _, _ in views)
    panes = "".join(f'<div data-omv="{k}"{"" if k == "sp" else " hidden"}><p class="sub">{_e(cap)}</p>{svg}</div>' for k, _, svg, cap in views)
    rows = "".join(f"<tr><td>{_e(p['label'])}</td><td>{_sp(p['spread'])}</td><td>{_n(p['total'], 1) if p['total'] is not None else '–'}</td><td>{_ml(p['ml_a'])}</td><td>{_ml(p['ml_h'])}</td></tr>" for p in P)
    table = f'<div class="tw"><table><thead><tr><th>When</th><th>{a} spread</th><th>Total</th><th>{a} ML</th><th>{h} ML</th></tr></thead><tbody>{rows}</tbody></table></div>'
    return f'<div class="om"><div class="segs" role="group" aria-label="Market">{btn}</div>{panes}</div>', table, P


# ---------- team results from games.csv ----------

def _team_games(G, t, seasons=None, done=True):
    x = G[(G.home_team == t) | (G.away_team == t)]
    if seasons is not None:
        x = x[x.season.isin(seasons)]
    if done:
        x = x[x.result.notna()]
    x = x.sort_values(["season", "week"])
    rows = []
    for r in x.itertuples():
        home = r.home_team == t
        pf, pa = (r.home_score, r.away_score) if home else (r.away_score, r.home_score)
        # spread_line is the home team's expected margin; a team covers when its margin beats its own expected margin
        exp = (r.spread_line if home else -r.spread_line) if pd.notna(r.spread_line) else None
        m = pf - pa if pd.notna(pf) else None
        ats = None if exp is None or m is None else ("W" if m > exp else "L" if m < exp else "P")
        ou = None if pd.isna(r.total_line) or m is None else ("O" if pf + pa > r.total_line else "U" if pf + pa < r.total_line else "P")
        rows.append(dict(season=r.season, week=r.week, gt=r.game_type, date=r.gameday, home=home, opp=r.away_team if home else r.home_team, pf=pf, pa=pa,
                         su=None if m is None else ("W" if m > 0 else "L" if m < 0 else "T"), ats=ats, ou=ou, fav=None if exp is None else exp > 0, exp=exp, tl=r.total_line))
    return rows


def _rec(rows, k):
    w = sum(r[k] in ("W", "O") for r in rows)
    l = sum(r[k] in ("L", "U") for r in rows)
    t = sum(r[k] in ("P", "T") for r in rows)
    return f"{w}-{l}" + (f"-{t}" if t else "")


def sos(c, rt, season=2026):
    """Strength of schedule from the model's adjusted ratings (offense + defense, points vs average). Rank 1 = toughest."""
    g = c.g26
    st = lambda o: rt["off"].get(o, 0) + rt["def"].get(o, 0)
    teams = sorted(set(g.home_team) | set(g.away_team))
    out = {}
    for t in teams:
        x = g[(g.home_team == t) | (g.away_team == t)]
        opp = lambda y: [a if h == t else h for h, a in zip(y.home_team, y.away_team)]
        played, left = opp(x[x.result.notna()]), opp(x[x.result.isna()])
        out[t] = dict(played=float(np.mean([st(o) for o in played])) if played else float("nan"), left=float(np.mean([st(o) for o in left])) if left else float("nan"), n_left=len(left))
    for k in ("played", "left"):
        s = pd.Series({t: v[k] for t, v in out.items()}).rank(ascending=False)
        for t in out:
            out[t][k + "_rk"] = None if math.isnan(out[t][k]) else int(s[t])
    return out


# ---------- team logos (ESPN PNGs cached at 64px in data/logos, embedded once per report) ----------
LOGO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "logos")
NICK = {"ARI": "Cardinals", "ATL": "Falcons", "BAL": "Ravens", "BUF": "Bills", "CAR": "Panthers", "CHI": "Bears", "CIN": "Bengals", "CLE": "Browns", "DAL": "Cowboys", "DEN": "Broncos",
        "DET": "Lions", "GB": "Packers", "HOU": "Texans", "IND": "Colts", "JAX": "Jaguars", "KC": "Chiefs", "LA": "Rams", "LAC": "Chargers", "LV": "Raiders", "MIA": "Dolphins",
        "MIN": "Vikings", "NE": "Patriots", "NO": "Saints", "NYG": "Giants", "NYJ": "Jets", "PHI": "Eagles", "PIT": "Steelers", "SEA": "Seahawks", "SF": "49ers", "TB": "Buccaneers",
        "TEN": "Titans", "WAS": "Commanders"}


def logo_css():
    """One CSS rule per team logo; missing logos fall back to the abbreviation (tl() prints it)."""
    import base64
    out = [".tl{display:inline-block;width:var(--ls,26px);height:var(--ls,26px);flex:none;background:center/contain no-repeat;vertical-align:middle}"]
    for t in NICK:
        f = os.path.join(LOGO_DIR, f"{t}.png")
        if os.path.exists(f):
            out.append(f".tl-{t}{{background-image:url(data:image/png;base64,{base64.b64encode(open(f, 'rb').read()).decode()})}}")
    return "".join(out)


def tl(t, size=None):
    st = f' style="--ls:{size}px"' if size else ""
    return f'<span class="tl tl-{t}" role="img" aria-label="{t}"{st}></span>'


SOS_CSS = """<style>
.sosg{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,320px),1fr));gap:14px;margin-top:6px;align-items:start}
.sosc{border:1px solid var(--line);border-radius:12px;padding:14px;display:grid;gap:10px;min-width:0;background:var(--bg)}
.sosh{display:flex;align-items:center;gap:10px}.sosh b{font:800 17px var(--d);font-stretch:105%;text-transform:uppercase}.sosh small{color:var(--mut);font-size:12.5px;display:block}
.sosr{display:grid;grid-template-columns:96px minmax(0,1fr) auto;gap:10px;align-items:center;font-size:13px}.sosr>span:first-child{color:var(--mut)}
.sosbar{height:10px;border-radius:5px;background:var(--line);overflow:hidden}.sosbar i{display:block;height:100%;border-radius:5px}
.sosk{font:700 13px var(--m);min-width:92px;text-align:right}.sosk small{display:block;font:500 10.5px var(--m);color:var(--mut)}
.t-hard{background:var(--bad)}.t-mid{background:#e3a008}.t-easy{background:var(--good)}
.c-hard{color:var(--bad)}.c-mid{color:#a86f00}.c-easy{color:var(--good)}
.sosc details{border-top:1px solid var(--line);padding-top:8px}.sosc summary{cursor:pointer;font:700 12px var(--m);text-transform:uppercase;letter-spacing:.06em;color:var(--fg);list-style:none;display:flex;justify-content:space-between}
.sosc summary::after{content:"+";color:var(--acc)}.sosc details[open] summary::after{content:"–"}.sosc summary::-webkit-details-marker{display:none}
.sost{width:100%;border-collapse:collapse;font-size:13px;margin-top:6px}.sost th{font:600 10px var(--m);text-transform:uppercase;letter-spacing:.06em;color:var(--mut);text-align:left;padding:4px}
.sost td{padding:5px 4px;border-top:1px solid var(--line);vertical-align:middle}.sost td.n{text-align:right;font-family:var(--m);white-space:nowrap}.sost th.n{text-align:right}
.sost .op{display:flex;align-items:center;gap:7px}.rkp{display:inline-block;min-width:34px;text-align:center;font:700 11.5px var(--m);border-radius:6px;padding:2px 4px;border:1px solid currentColor}
@media(max-width:520px){.sosr{grid-template-columns:78px minmax(0,1fr) auto}.sost .hid{display:none}}
</style>"""


def sos_cards(c, rt, a, h):
    """Strength of schedule cards: logo, a green-to-red bar with the league rank (1 = toughest), and dropdowns of
    opponents played / still to play with each opponent's offense and defense rank from the adjusted ratings."""
    SO = sos(c, rt)
    g = c.g26
    teams = sorted(set(g.home_team) | set(g.away_team))
    N = len(teams)
    off = pd.Series({t: rt["off"].get(t, 0) for t in teams}).rank(ascending=False, method="min").astype(int)
    de = pd.Series({t: rt["def"].get(t, 0) for t in teams}).rank(ascending=False, method="min").astype(int)
    tone = lambda r: "hard" if r <= round(N / 3) else ("mid" if r <= round(2 * N / 3) else "easy")
    word = lambda r: {"hard": "tough", "mid": "average", "easy": "soft"}[tone(r)]
    def bar(lab, v, r, n):
        if r is None:
            return f'<div class="sosr"><span>{lab}</span><div class="sosbar"></div><span class="sosk">–</span></div>'
        return (f'<div class="sosr"><span>{lab}<br><small>{n} games</small></span><div class="sosbar"><i class="t-{tone(r)}" style="width:{(N + 1 - r) / N * 100:.0f}%"></i></div>'
                f'<span class="sosk c-{tone(r)}">#{r} of {N}<small>{word(r)} · {v:+.1f}</small></span></div>')
    rk = lambda r: f'<span class="rkp c-{tone(r)}">#{r}</span>'
    cards = []
    for t in (a, h):
        x = g[(g.home_team == t) | (g.away_team == t)].sort_values("week")
        played, left = x[x.result.notna()], x[x.result.isna()]
        def rows(y, done):
            out = []
            for r in y.itertuples():
                hm = r.home_team == t
                o = r.away_team if hm else r.home_team
                if done:
                    us, th = (r.home_score, r.away_score) if hm else (r.away_score, r.home_score)
                    res = f"{'W' if us > th else 'L' if us < th else 'T'} {int(us)}-{int(th)}"
                else:
                    res = pd.Timestamp(r.gameday).strftime("%b %-d")
                out.append(f'<tr><td class="n">{int(r.week)}</td><td><span class="op">{tl(o, 22)}{"vs" if hm else "@"} {NICK.get(o, o)}</span></td><td class="n hid">{res}</td>'
                           f'<td class="n">{rk(int(off[o]))}</td><td class="n">{rk(int(de[o]))}</td></tr>')
            return "".join(out)
        head = '<tr><th class="n">Wk</th><th>Opponent</th><th class="n hid">{}</th><th class="n">Off</th><th class="n">Def</th></tr>'
        tab = lambda y, done, lab: (f'<details><summary>{lab} ({len(y)})</summary><table class="sost"><thead>{head.format("Result" if done else "Date")}</thead><tbody>{rows(y, done)}</tbody></table></details>'
                                    if len(y) else "")
        cards.append(f'<div class="sosc"><div class="sosh">{tl(t, 40)}<div><b>{NICK.get(t, t)}</b><small>{t} · 2026 schedule</small></div></div>'
                     f'{bar("Played so far", SO[t]["played"], SO[t]["played_rk"], len(played))}{bar("Still to play", SO[t]["left"], SO[t]["left_rk"], SO[t]["n_left"])}'
                     f'{tab(played, True, "Who they played")}{tab(left, False, "Still to play")}</div>')
    return SOS_CSS + f'<div class="sosg">{"".join(cards)}</div>'


def team_bars(G, a, h, mrow, better):
    """2026 results side by side, each with a league rank so the bars mean the same thing as the matchup tab."""
    teams = sorted(set(G[G.season == 2026].home_team))
    stat = {}
    for t in teams:
        r = _team_games(G, t, [2026])
        r = [x for x in r if x["gt"] == "REG"]
        gp = len(r) or 1
        pf, pa = sum(x["pf"] for x in r), sum(x["pa"] for x in r)
        w = sum(x["su"] == "W" for x in r) + .5 * sum(x["su"] == "T" for x in r)
        stat[t] = dict(wpct=w / gp, ppg=pf / gp, papg=pa / gp, diff=(pf - pa) / gp, wins=w)
    rk = {k: pd.Series({t: v[k] for t, v in stat.items()}).rank(ascending=k == "papg", method="min") for k in ("wpct", "ppg", "papg", "diff", "wins")}
    M = [("Win %", "wpct", lambda v: f"{v * 100:.0f}%"), ("Points per game", "ppg", lambda v: f"{v:.1f}"), ("Points allowed / game", "papg", lambda v: f"{v:.1f}"),
         ("Point differential / game", "diff", lambda v: f"{v:+.1f}"), ("Wins", "wins", lambda v: f"{v:g}")]
    return [mrow(lab, f(stat[a][k]), f(stat[h][k]), int(rk[k][a]), int(rk[k][h]), better(int(rk[k][a]), int(rk[k][h]))) for lab, k, f in M]


_BOX = {}


def _box(season):
    """One season's player lines (regular season and playoffs), read once per run. None when that season's file is not on hand."""
    if season not in _BOX:
        import os
        f = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", f"stats_player_week_{season}.csv")
        try:
            d = pd.read_csv(f, low_memory=False)
            ic = "passing_interceptions" if "passing_interceptions" in d else "interceptions"
            d = d.rename(columns={ic: "ints"})
            for c in ("completions", "attempts", "passing_yards", "passing_tds", "ints", "carries", "rushing_yards", "rushing_tds", "receptions", "targets", "receiving_yards", "receiving_tds"):
                d[c] = pd.to_numeric(d.get(c, 0), errors="coerce").fillna(0)
            _BOX[season] = d
        except OSError:
            _BOX[season] = None
    return _BOX[season]


def top_performers(r):
    """Each team's passer, two leading rushers and three leading receivers from one past game, as popup HTML ('' if no box score)."""
    d = _box(int(r.season))
    if d is None:
        return ""
    d = d[(d.week == r.week) & d.team.isin([r.away_team, r.home_team])]
    if not len(d):
        return ""
    i = lambda v: int(round(v))
    td = lambda v: f", {i(v)} TD" if v else ""
    cols = []
    for t in (r.away_team, r.home_team):
        x = d[d.team == t]
        rows = []
        for _, q in x[x.attempts > 0].sort_values("attempts", ascending=False).head(1).iterrows():
            rows.append(("Passing", q.player_display_name, f"{i(q.completions)}/{i(q.attempts)}, {i(q.passing_yards)} yds{td(q.passing_tds)}" + (f", {i(q.ints)} INT" if q.ints else "")))
        for _, q in x[x.carries > 0].sort_values("rushing_yards", ascending=False).head(2).iterrows():
            rows.append(("Rushing", q.player_display_name, f"{i(q.carries)} car, {i(q.rushing_yards)} yds{td(q.rushing_tds)}"))
        for _, q in x[x.targets > 0].sort_values("receiving_yards", ascending=False).head(3).iterrows():
            rows.append(("Receiving", q.player_display_name, f"{i(q.receptions)}/{i(q.targets)} tgt, {i(q.receiving_yards)} yds{td(q.receiving_tds)}"))
        body = "".join(f"<tr><td>{c}</td><td><b>{html.escape(str(n))}</b></td><td>{html.escape(l)}</td></tr>" for c, n, l in rows) or "<tr><td colspan='3'>No player lines on file.</td></tr>"
        cols.append(f"<div class='pcol'><h5>{t}</h5><div class='tw'><table><thead><tr><th>Category</th><th>Player</th><th>Line</th></tr></thead><tbody>{body}</tbody></table></div></div>")
    return f"<div class='pcols'>{''.join(cols)}</div>"


def head_to_head(G, a, h, k=6, gid=""):
    x = G[(((G.home_team == a) & (G.away_team == h)) | ((G.home_team == h) & (G.away_team == a))) & G.result.notna()].sort_values(["season", "week"]).tail(k)
    if not len(x):
        return "<p class='note'>These teams have not met since 1999.</p>", ""
    cells, wins = [], {a: 0, h: 0}
    for r in reversed(list(x.itertuples())):
        win = r.home_team if r.result > 0 else r.away_team if r.result < 0 else None
        if win:
            wins[win] += 1
        cover = None
        if pd.notna(r.spread_line):
            d = r.result - r.spread_line
            cover = r.home_team if d > 0 else r.away_team if d < 0 else "Push"
        ou = "–" if pd.isna(r.total_line) else ("Over" if r.total > r.total_line else "Under" if r.total < r.total_line else "Push")
        wk = f"{'Wk ' + str(r.week) if r.game_type == 'REG' else r.game_type}"
        top = top_performers(r)
        did = f"h2-{gid}-{r.season}-{r.week}"
        inner = (f'<small>{r.season} · {wk}</small><div class="h2s"><span class="{"w" if win == r.away_team else ""}">{r.away_team} {int(r.away_score)}</span>'
                 f'<span class="{"w" if win == r.home_team else ""}">@ {r.home_team} {int(r.home_score)}</span></div>'
                 f'<small>Cover: {cover or "–"} · {ou}{"" if pd.isna(r.total_line) else " " + format(r.total_line, "g")}</small>')
        if top:  # the card opens a popup with that game's top performers
            marg = abs(int(r.home_score) - int(r.away_score))
            cells.append(f'<button class="h2c h2b" data-dlg="{did}" title="Top performers from this game">{inner}<small class="h2m">Top performers &rarr;</small></button>'
                         f'<dialog class="h2d" id="{did}"><button class="h2x" data-close aria-label="Close">&#10005;</button><small>{r.season} · {wk} · {r.gameday}</small>'
                         f'<h4>{r.away_team} {int(r.away_score)} at {r.home_team} {int(r.home_score)}</h4><p class="note">{(win + " won by " + str(marg)) if win else "Tie"}. Cover: {cover or "–"} · {ou}{"" if pd.isna(r.total_line) else " " + format(r.total_line, "g")}.</p>'
                         f'{top}<p class="note">Each team\'s passer, two leading rushers and three leading receivers, from nflverse player stats.</p></dialog>')
        else:
            cells.append(f'<div class="h2c">{inner}</div>')
    summ = f"{a} {wins[a]} – {wins[h]} {h} in the last {len(x)} meetings"
    return f'<div class="h2">{"".join(cells)}</div>', summ


def last10(G, t):
    r = [x for x in _team_games(G, t, [2025, 2026])][-10:]
    if not r:
        return "<p class='note'>No games yet.</p>"
    gp = len(r)
    pf, pa = sum(x["pf"] for x in r) / gp, sum(x["pa"] for x in r) / gp
    head = (f'<div class="l10s"><div><small>Record</small><b>{_rec(r, "su")}</b></div><div><small>PF / G</small><b>{pf:.1f}</b></div><div><small>PA / G</small><b>{pa:.1f}</b></div>'
            f'<div><small>Total / G</small><b>{pf + pa:.1f}</b></div><div><small>ATS</small><b>{_rec(r, "ats")}</b></div><div><small>O/U</small><b>{_rec(r, "ou")}</b></div></div>')
    chips = "".join(f'<span class="res {x["su"].lower()}" title="{x["season"]} Wk {x["week"]} {"vs" if x["home"] else "@"} {x["opp"]} {int(x["pf"])}-{int(x["pa"])}">{x["su"]}</span>' for x in r)
    rows = "".join(f'<tr><td>{x["season"]} {"Wk " + str(x["week"]) if x["gt"] == "REG" else x["gt"]}</td><td>{"vs" if x["home"] else "@"} {x["opp"]}</td><td class="{"good" if x["su"] == "W" else "bad" if x["su"] == "L" else ""}">{x["su"]} {int(x["pf"])}-{int(x["pa"])}</td>'
                   f'<td>{_sp(x["exp"] and -x["exp"]) if x["exp"] is not None else "–"}</td><td>{x["ats"] or "–"}</td><td>{_n(x["tl"], 1) if pd.notna(x["tl"]) else "–"}</td><td>{x["ou"] or "–"}</td></tr>' for x in reversed(r))
    tab = f'<details><summary>Game by game</summary><div class="tw"><table><thead><tr><th>Game</th><th>Opp</th><th>Result</th><th>Line</th><th>ATS</th><th>Total</th><th>O/U</th></tr></thead><tbody>{rows}</tbody></table></div></details>'
    return f'<div class="chips">{chips}</div>{head}{tab}'


def splits_table(G, t):
    r = [x for x in _team_games(G, t, [2026]) if x["gt"] == "REG"]
    groups = [("All games", r), ("Home", [x for x in r if x["home"]]), ("Away", [x for x in r if not x["home"]]),
              ("As favorite", [x for x in r if x["fav"]]), ("As underdog", [x for x in r if x["fav"] is False]), ("Last 3", r[-3:])]
    body = "".join(f"<tr><td>{lab}</td><td>{len(x)}</td><td>{_rec(x, 'su')}</td><td>{_rec(x, 'ats')}</td><td>{_rec(x, 'ou')}</td>"
                   f"<td>{_n(np.mean([y['pf'] for y in x]), 1) if x else '–'}</td><td>{_n(np.mean([y['pa'] for y in x]), 1) if x else '–'}</td>"
                   f"<td>{_n(np.mean([y['pf'] - y['pa'] - y['exp'] for y in x if y['exp'] is not None]), 1) if any(y['exp'] is not None for y in x) else '–'}</td></tr>" for lab, x in groups)
    return f'<div class="tw"><table><thead><tr><th>{t} 2026</th><th>G</th><th>Straight up</th><th>ATS</th><th>O/U</th><th>PF</th><th>PA</th><th>Avg cover margin</th></tr></thead><tbody>{body}</tbody></table></div>'


# ---------- special teams ----------

def _st_team(p, t):
    """Kicking, punting and return numbers for one team from 2026 regular-season play-by-play."""
    other = lambda r: r.away_team if r.home_team == t else r.home_team
    mine = p[(p.home_team == t) | (p.away_team == t)]
    games = mine.game_id.nunique() or 1
    fg = mine[(mine.field_goal_attempt == 1) & (mine.posteam == t)]
    band = lambda lo, hi: fg[(fg.kick_distance >= lo) & (fg.kick_distance <= hi)]
    fgb = {lab: (int((b.field_goal_result == "made").sum()), len(b)) for lab, b in (("<40", band(0, 39)), ("40-49", band(40, 49)), ("50+", band(50, 99)))}
    xp = mine[(mine.extra_point_attempt == 1) & (mine.posteam == t)]
    pu = mine[(mine.punt_attempt == 1) & (mine.posteam == t)]
    net = pu.kick_distance.fillna(0) - pu.return_yards.fillna(0) - 20 * pu.touchback.fillna(0)
    ko = mine[(mine.kickoff_attempt == 1)]
    ko_ret = ko[(ko.return_team == t) & (ko.touchback != 1) & ko.return_yards.notna()]
    ko_cov = ko[(ko.return_team.notna()) & (ko.return_team != t) & (ko.touchback != 1) & ko.return_yards.notna()]
    pr = mine[(mine.punt_attempt == 1) & (mine.return_team == t) & (mine.return_yards.notna()) & (mine.punt_fair_catch != 1) & (mine.touchback != 1)]
    pr_cov = pu[(pu.return_yards.notna()) & (pu.punt_fair_catch != 1) & (pu.touchback != 1)]
    opp_fg = mine[(mine.field_goal_attempt == 1) & (mine.posteam != t)]
    kick = fg.kicker_player_name.dropna().mode()
    punter = pu.punter_player_name.dropna().mode()
    return dict(games=games, fg=(int((fg.field_goal_result == "made").sum()), len(fg)), fgb=fgb, xp=(int((xp.extra_point_result == "good").sum()), len(xp)),
                fg_long=int(fg[fg.field_goal_result == "made"].kick_distance.max()) if (fg.field_goal_result == "made").any() else None,
                punts=len(pu), gross=float(pu.kick_distance.mean()) if len(pu) else float("nan"), net=float(net.mean()) if len(pu) else float("nan"),
                in20=int(pu.punt_inside_twenty.fillna(0).sum()), kr=float(ko_ret.return_yards.mean()) if len(ko_ret) else float("nan"), kr_n=len(ko_ret),
                pr=float(pr.return_yards.mean()) if len(pr) else float("nan"), pr_n=len(pr),
                kr_allowed=float(ko_cov.return_yards.mean()) if len(ko_cov) else float("nan"), pr_allowed=float(pr_cov.return_yards.mean()) if len(pr_cov) else float("nan"),
                opp_fg=(int((opp_fg.field_goal_result == "made").sum()), len(opp_fg)), st_td=int(((mine.return_touchdown == 1) & (mine.td_team == t) & (mine.special == 1)).sum()) if "special" in mine else 0,
                kicker=kick.iloc[0] if len(kick) else "–", punter=punter.iloc[0] if len(punter) else "–")


def special_teams(c, a, h, mrow, better):
    p = c.p26
    if "kickoff_attempt" not in p:
        return "<p class='note'>Special-teams columns are missing from the play-by-play file.</p>"
    teams = sorted(p.posteam.dropna().unique())
    S = {t: _st_team(p, t) for t in teams}
    pct = lambda x: x[0] / x[1] if x[1] else float("nan")
    metric = {"fg": lambda s: pct(s["fg"]), "xp": lambda s: pct(s["xp"]), "net": lambda s: s["net"], "gross": lambda s: s["gross"], "kr": lambda s: s["kr"], "pr": lambda s: s["pr"],
              "kr_allowed": lambda s: s["kr_allowed"], "pr_allowed": lambda s: s["pr_allowed"], "fg50": lambda s: pct(s["fgb"]["50+"])}
    low = {"kr_allowed", "pr_allowed"}
    rk = {k: pd.Series({t: f(S[t]) for t in teams}).rank(ascending=k in low, method="min") for k, f in metric.items()}
    R = lambda k, t: None if pd.isna(rk[k].get(t)) else int(rk[k][t])
    row = lambda lab, k, fmt: mrow(lab, fmt(S[a], k), fmt(S[h], k), R(k, a), R(k, h), better(R(k, a), R(k, h)))
    frac = lambda key: (lambda s, k: f"{s[key][0]}/{s[key][1]} · {pct(s[key]) * 100:.0f}%" if s[key][1] else "–")
    fb = lambda band: (lambda s, k: f"{s['fgb'][band][0]}/{s['fgb'][band][1]}" + (f" · {pct(s['fgb'][band]) * 100:.0f}%" if s["fgb"][band][1] else ""))
    yd = lambda key: (lambda s, k: _n(s[key], 1))
    kick = [row("Field goals", "fg", frac("fg")), mrow("FG under 40 yds", fb("<40")(S[a], 0), fb("<40")(S[h], 0)), mrow("FG 40-49 yds", fb("40-49")(S[a], 0), fb("40-49")(S[h], 0)),
            row("FG 50+ yds", "fg50", fb("50+")), mrow("Longest made", str(S[a]["fg_long"] or "–"), str(S[h]["fg_long"] or "–")), row("Extra points", "xp", frac("xp"))]
    punt = [mrow("Punts", str(S[a]["punts"]), str(S[h]["punts"])), row("Gross punt avg", "gross", yd("gross")), row("Net punt avg", "net", yd("net")),
            mrow("Punts inside the 20", str(S[a]["in20"]), str(S[h]["in20"]))]
    ret = [row("Kick return avg", "kr", yd("kr")), row("Punt return avg", "pr", yd("pr")), row("Kick return avg allowed", "kr_allowed", yd("kr_allowed")), row("Punt return avg allowed", "pr_allowed", yd("pr_allowed")),
           mrow("Opponent FGs", f"{S[a]['opp_fg'][0]}/{S[a]['opp_fg'][1]}", f"{S[h]['opp_fg'][0]}/{S[h]['opp_fg'][1]}")]
    who = f'<div class="mcols"><div class="l"><b>{a}</b><small>K {_e(S[a]["kicker"])} · P {_e(S[a]["punter"])}</small></div><div class="lab">2026 season<br># = league rank, 1 is best</div><div class="r"><b>{h}</b><small>K {_e(S[h]["kicker"])} · P {_e(S[h]["punter"])}</small></div></div>'
    return who, kick, punt, ret


# ---------- lineups ----------

STP = ["PK", "P", "LS", "KR", "PR"]
OUT_ST = ("Out", "Injured Reserve", "Doubtful")


def lineup(c, t, inj):
    """Depth-chart starters with this week's injury status. In each depth-chart slot the first player who is not out starts;
    the slate's QB override (slate.json) wins over a depth chart that has not caught up."""
    dc = c.dc[c.dc.team == t]
    status = {x["name"]: x["status"] for x in inj}
    out_ids = set(OUT_IDS.get(t, []))
    st_of = lambda nm, pid: "Out" if pid in out_ids else status.get(nm)
    out = lambda nm, pid: st_of(nm, pid) in OUT_ST

    def chip(nm, pid):
        st = st_of(nm, pid)
        if not st:
            return ""
        cls = "o" if st in OUT_ST else "q"
        return f' <span class="ij {cls}">{_e({"Injured Reserve": "IR", "Questionable": "Q", "Doubtful": "D", "Out": "Out"}.get(st, st))}</span>'

    ORDER = {p: i for i, p in enumerate(["QB", "RB", "FB", "WR", "TE", "LT", "LG", "C", "RG", "RT"])}

    def unit(rows, keep=None, dedup=True):
        items, seen = [], set()
        groups = sorted(rows.groupby("pos_slot", sort=True), key=lambda kv: (ORDER.get(kv[1].pos_abb.iloc[0], 50), kv[0]))
        for slot, x in groups:
            if keep is not None and x.pos_abb.iloc[0] not in keep:
                continue
            x = x.sort_values("pos_rank")
            ok = [r for r in x.itertuples() if not out(r.player_name, r.gsis_id) and not (dedup and r.gsis_id in seen)]
            skipped = bool(ok) and any(out(q.player_name, q.gsis_id) for q in x.itertuples() if q.pos_rank < ok[0].pos_rank)
            r = ok[0] if ok else x.iloc[0]
            pid, nm, pos = (r.gsis_id, r.player_name, r.pos_abb) if ok else (r["gsis_id"], r["player_name"], r["pos_abb"])
            note = ""
            if pos == "QB" and t in QB_START:
                pid = QB_START[t]
                nm = next((q.player_name for q in c.dc.itertuples() if q.gsis_id == pid), nm)
                note = " <span class='ij n'>Starting (news)</span>"
            elif skipped:
                note = " <span class='ij n'>Next up</span>"
            elif not ok:
                note = " <span class='note'>· no healthy backup listed</span>"
            seen.add(pid)
            items.append(f"<li><span class='pos'>{pos}</span><span>{_e(nm)}{chip(nm, pid)}{note}</span></li>")
        return "".join(items)
    qgrp = dc[dc.pos_abb == "QB"].pos_grp.mode()
    off = dc[dc.pos_grp == qgrp.iloc[0]] if len(qgrp) else dc.iloc[:0]
    dgrp = dc[dc.pos_abb.isin(["LCB", "RCB", "FS", "SS"])].pos_grp.mode()
    de = dc[dc.pos_grp == dgrp.iloc[0]] if len(dgrp) else dc.iloc[:0]
    stt = dc[dc.pos_grp == "Special Teams"]
    dt = pd.Timestamp(dc.dt.max()).strftime("%b %-d") if len(dc) else "–"
    return (f'<div class="lu"><h5>{t}</h5><p class="sub">Depth chart of {dt}</p>'
            f'<h6>Offense{" · " + _e(qgrp.iloc[0]) if len(qgrp) else ""}</h6><ul>{unit(off)}</ul>'
            f'<h6>Defense{" · " + _e(dgrp.iloc[0]) if len(dgrp) else ""}</h6><ul>{unit(de)}</ul><h6>Special teams</h6><ul>{unit(stt, set(STP), dedup=False)}</ul></div>')


# ---------- touchdowns ----------

K_SHRINK = 40  # red-zone plays of league-average prior blended into each defense's rate, so one fluky game cannot make a matchup


def _rz_plays(p):
    x = p[(p.yardline_100 <= 20) & p.play_type.isin(["pass", "run"]) & (p.qb_kneel != 1) & (p.qb_spike != 1) & (p.two_point_attempt != 1)]
    return x.assign(ptd=(x.pass_touchdown == 1) & (x.td_team == x.posteam), rtd=(x.rush_touchdown == 1) & (x.td_team == x.posteam))


def rz_units(c):
    """Per-team red-zone offense and defense for 2026 (inside the 20), plus league rates and ranks."""
    p = c.p26
    rz = _rz_plays(p)
    games = pd.concat([p[["game_id", "posteam"]].rename(columns={"posteam": "t"}), p[["game_id", "defteam"]].rename(columns={"defteam": "t"})]).dropna().drop_duplicates().groupby("t").size()
    drv = p[p.posteam.notna() & p.fixed_drive.notna()].groupby(["game_id", "posteam", "defteam", "fixed_drive"]).agg(minyl=("yardline_100", "min"), res=("fixed_drive_result", "last")).reset_index()
    drv = drv[drv.minyl <= 20]
    out = {}
    for t in sorted(games.index):
        for side, col, dcol in (("off", "posteam", "posteam"), ("def", "defteam", "defteam")):
            x = rz[rz[col] == t]
            d = drv[drv[dcol] == t]
            pa, ru = x[x.pass_attempt == 1], x[x.rush_attempt == 1]
            out.setdefault(t, {})[side] = dict(g=int(games[t]), trips=len(d) / games[t], td_pct=float((d.res == "Touchdown").mean()) if len(d) else float("nan"), n_trips=len(d),
                                               pass_rate=float(len(pa) / max(len(x), 1)), pa=len(pa), ru=len(ru), ptd=int(pa.ptd.sum()), rtd=int(ru.rtd.sum()),
                                               ptd_rate=float(pa.ptd.mean()) if len(pa) else float("nan"), rtd_rate=float(ru.rtd.mean()) if len(ru) else float("nan"),
                                               pa_g=len(pa) / games[t], ru_g=len(ru) / games[t])
    lg = dict(ptd=float(rz[rz.pass_attempt == 1].ptd.mean()), rtd=float(rz[rz.rush_attempt == 1].rtd.mean()), td_pct=float((drv.res == "Touchdown").mean()))
    for t, v in out.items():  # shrunk defensive rates feed the matchup score
        dd = v["def"]
        dd["ptd_s"] = (dd["ptd"] + K_SHRINK * lg["ptd"]) / (dd["pa"] + K_SHRINK)
        dd["rtd_s"] = (dd["rtd"] + K_SHRINK * lg["rtd"]) / (dd["ru"] + K_SHRINK)
    rank = {}
    for side in ("off", "def"):
        for k in ("trips", "td_pct", "ptd_rate", "rtd_rate", "pass_rate"):
            s = pd.Series({t: out[t][side][k] for t in out})
            rank[(side, k)] = s.rank(ascending=(side == "def" and k != "pass_rate"), method="min")
    return out, lg, rank


def rz_players(c, t):
    """Red-zone usage and 2026 touchdowns for players whose latest 2026 team is t."""
    rz = _rz_plays(c.p26[c.p26.posteam == t])
    pa, ru = rz[rz.pass_attempt == 1], rz[rz.rush_attempt == 1]
    i10 = rz[rz.yardline_100 <= 10]
    tg = pa.groupby("receiver_player_id").size()
    tg10 = i10[i10.pass_attempt == 1].groupby("receiver_player_id").size()
    car = ru.groupby("rusher_player_id").size()
    car10 = i10[i10.rush_attempt == 1].groupby("rusher_player_id").size()
    st = c.d["stats"]
    st = st[(st.season == 2026) & (st.season_type == "REG")].sort_values("week")
    cur = st.groupby("player_id").tail(1)
    ids = set(cur[cur.team == t].player_id)
    tot = st[st.player_id.isin(ids)].groupby("player_id").agg(name=("name", "last"), pos=("position", "last"), rush_td=("rushing_tds", "sum"), rec_td=("receiving_tds", "sum"), g=("week", "nunique"))
    rows = []
    for pid, r in tot.iterrows():
        d = dict(pid=pid, name=r["name"], pos=r["pos"], g=int(r["g"]), rush_td=int(r.rush_td), rec_td=int(r.rec_td),
                 tg=int(tg.get(pid, 0)), tg10=int(tg10.get(pid, 0)), car=int(car.get(pid, 0)), car10=int(car10.get(pid, 0)))
        d["tg_sh"] = d["tg"] / max(len(pa), 1)
        d["car_sh"] = d["car"] / max(len(ru), 1)
        if d["tg"] + d["car"] + d["rush_td"] + d["rec_td"] > 0:
            rows.append(d)
    return sorted(rows, key=lambda d: -(d["tg"] + d["car"] + 3 * (d["rush_td"] + d["rec_td"]))), dict(pa=len(pa), ru=len(ru), pa10=int((i10.pass_attempt == 1).sum()), ru10=int((i10.rush_attempt == 1).sum()))


def _pct(x, d=0):
    return "–" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{100 * x:.{d}f}%"


def touchdowns(c, a, h, pl, inj, mrow, better, T, CTX, PRED):
    """The Touchdowns tab: player red-zone usage and TDs, each red-zone offense against the other defense, and a top 5."""
    U, lg, RK = rz_units(c)
    rk = lambda side, k, t: None if pd.isna(RK[(side, k)].get(t)) else int(RK[(side, k)][t])
    model = {p["player_id"]: p for p in pl}
    status = {t: {x["name"]: x["status"] for x in inj.get(t, [])} for t in (a, h)}
    out_ids = {t: set(OUT_IDS.get(t, [])) for t in (a, h)}
    gone = lambda t, nm, pid: pid in out_ids[t] or status[t].get(nm) in ("Out", "Injured Reserve", "Doubtful")
    sec = lambda title, kind, body, note="": f'<section class="mblock"><h4>{title} {kind}</h4>{body}{f"<p class=note>{note}</p>" if note else ""}</section>'
    html, cand = "", []

    # 1. each red-zone offense against the opposing red-zone defense
    def matchup(o, d):
        O, D = U[o]["off"], U[d]["def"]
        head = (f'<div class="mcols"><div class="l"><b>{o} offense</b><small>what {o} does inside the 20</small></div><div class="lab">2026 red zone<br># = league rank, 1 is best</div>'
                f'<div class="r"><b>{d} defense</b><small>what {d} allows inside the 20</small></div></div>')
        R = lambda k: (rk("off", k, o), rk("def", k, d))
        row = lambda lab, k, f: mrow(lab, f(O[k]), f(D[k]), *R(k), better(*R(k)))
        rows = [head, row("Red-zone trips / game", "trips", lambda v: f"{v:.1f}"), row("Trips ending in a TD", "td_pct", _pct),
                mrow("Pass rate inside the 20", _pct(O["pass_rate"]), _pct(D["pass_rate"])),
                row("TD rate on red-zone passes", "ptd_rate", lambda v: _pct(v, 1)), row("TD rate on red-zone runs", "rtd_rate", lambda v: _pct(v, 1)),
                mrow("Passing / rushing TDs inside the 20", f"{O['ptd']} / {O['rtd']}", f"{D['ptd']} / {D['rtd']}")]
        lean = []
        if D["ptd_s"] > lg["ptd"] * 1.12:
            lean.append(f"{d} is soft on red-zone passes ({_pct(D['ptd_rate'], 1)} TD rate, #{rk('def', 'ptd_rate', d)}; league {_pct(lg['ptd'], 1)}), which favors {o}'s pass catchers.")
        elif D["ptd_s"] < lg["ptd"] * .88:
            lean.append(f"{d} shuts down red-zone passes ({_pct(D['ptd_rate'], 1)}, #{rk('def', 'ptd_rate', d)}; league {_pct(lg['ptd'], 1)}).")
        if D["rtd_s"] > lg["rtd"] * 1.12:
            lean.append(f"{d} gives up red-zone rushing TDs ({_pct(D['rtd_rate'], 1)} per carry, #{rk('def', 'rtd_rate', d)}; league {_pct(lg['rtd'], 1)}), which favors {o}'s runners.")
        elif D["rtd_s"] < lg["rtd"] * .88:
            lean.append(f"{d} stuffs red-zone runs ({_pct(D['rtd_rate'], 1)} per carry, #{rk('def', 'rtd_rate', d)}; league {_pct(lg['rtd'], 1)}).")
        lean = "".join(f"<p class='lean'>{_e(x)}</p>" for x in lean) or "<p class='lean'>Neither red-zone split for this defense is far from league average once its sample is shrunk.</p>"
        return f'<section class="mblock"><h4>When {o} reaches the red zone {CTX}</h4>{"".join(rows)}{lean}</section>'
    html += matchup(a, h) + matchup(h, a)

    # 2. players: red-zone share, TDs, model chance, and the matchup lift
    for t, opp in ((a, h), (h, a)):
        P, tot = rz_players(c, t)
        D = U[opp]["def"]
        body = []
        for d in P[:14]:
            m = model.get(d["pid"])
            out = gone(t, d["name"], d["pid"])
            # expected red-zone TDs per game from this player's share against this defense vs against a league-average one
            pa_g, ru_g = U[t]["off"]["pa_g"], U[t]["off"]["ru_g"]
            neutral = pa_g * d["tg_sh"] * lg["ptd"] + ru_g * d["car_sh"] * lg["rtd"]
            vs = pa_g * d["tg_sh"] * D["ptd_s"] + ru_g * d["car_sh"] * D["rtd_s"]
            lift = vs / neutral if neutral > 0 else 1.0
            lift = float(np.clip(lift, .7, 1.4))
            st = "Out" if out else status[t].get(d["name"], "")
            tagc = f" <span class='ij {'o' if out else 'q'}'>{_e({'Questionable': 'Q', 'Injured Reserve': 'IR'}.get(st, st))}</span>" if st else ""
            body.append([f"{_e(d['name'])}{tagc}", d["pos"], f"{d['tg']} · <b>{_pct(d['tg_sh'])}</b>", f"{d['tg10']}", f"{d['car']} · <b>{_pct(d['car_sh'])}</b>", f"{d['car10']}",
                         str(d["rec_td"]), str(d["rush_td"]), f"<b>{d['rec_td'] + d['rush_td']}</b>", _pct(m["atd"]) if m and not out else "–",
                         f"<span class='{'good' if lift >= 1.08 else 'bad' if lift <= .92 else ''}'>{lift:.2f}×</span>" if neutral > 0 else "–"])
            if m and not out and m["atd"] > .04:
                pr = pa_g * d["tg_sh"] * lg["ptd"]
                main = "rec" if pr >= neutral / 2 else "rush"
                why = (f"{_pct(d['tg_sh'])} of {t}'s red-zone targets ({d['tg']} of {tot['pa']}); {opp} allows a TD on {_pct(D['ptd_rate'], 1)} of red-zone passes (#{rk('def', 'ptd_rate', opp)} of 32)"
                       if main == "rec" else
                       f"{_pct(d['car_sh'])} of {t}'s red-zone carries ({d['car']} of {tot['ru']}); {opp} allows a TD on {_pct(D['rtd_rate'], 1)} of red-zone runs (#{rk('def', 'rtd_rate', opp)} of 32)")
                adj = 1 - (1 - m["atd"]) ** lift
                cand.append(dict(name=d["name"], team=t, pos=d["pos"], p=m["atd"], adj=adj, lift=lift, why=why, td=d["rec_td"] + d["rush_td"], route=main))
        cap = (f"2026 regular season, players whose latest game was for {t}. RZ = inside the opponent's 20. {t} ran {tot['pa']} red-zone passes and {tot['ru']} red-zone runs. "
               f"Model TD % is the calibrated anytime chance from the simulation. Matchup is how much {opp}'s red-zone defense moves this player's expected red-zone TDs against a league-average defense.")
        html += sec(f"{t} red-zone usage and touchdowns", CTX, T(["Player", "Pos", "RZ targets · share", "Inside 10", "RZ carries · share", "Inside 10", "Rec TD", "Rush TD", "Total TD", "Model TD %", f"Matchup vs {opp}"], body), cap)

    # 3. top 5: ranked by the model chance after the matchup lift, so a heavy red-zone target against a leaky pass defense can pass a running back
    top = sorted(cand, key=lambda x: -x["adj"])[:5]
    cards = "".join(f"<li><span class='rk'>{i + 1}</span><div><b>{_e(x['name'])}</b> <small>{x['team']} {x['pos']} · {x['td']} TD this season</small><p>{_e(x['why'])}.</p></div>"
                    f"<div class='tdp'><b>{_pct(x['adj'])}</b><small>with matchup</small><span>model {_pct(x['p'])} · {x['lift']:.2f}×</span></div></li>" for i, x in enumerate(top))
    top_html = (f'<section class="mblock tdtop"><h4>Top 5 touchdown candidates {CTX}</h4><ol class="tdl">{cards}</ol>'
                f'<p class="note">Ranked by the calibrated model chance after the red-zone matchup is applied, so a receiver who draws a big share of red-zone targets against a defense that leaks red-zone passing TDs can rank above a running back. '
                f'The model chance is calibrated. The matchup adjustment was backtested on 5,922 past forecasts (2025 and the 2026 holdout) and did not improve them: players it marked down still scored at or above the model\'s rate, and its top 5 hit no more often than the model\'s. Read it as context, not as an edge. Defensive rates are blended with {K_SHRINK} plays of league average to keep small samples from swinging it. '
                f'Players listed Out, Doubtful or IR are excluded.</p></section>')
    return top_html + html
