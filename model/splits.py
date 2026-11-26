"""Game Book "<Team> Splits" tabs: one per team.

Each tab has four parts:
  1. WR/CB matchups for this game (output/hub/wrcb.json, built by wrcb.py), in both directions.
  2. Run defense: 2026 season, last 5, 2025 season, or any single 2026 game. A seven-gap map of yards per carry allowed,
     compared with where the opponent's offense runs.
  3. Pass defense: the same views. A depth-by-side map of yards per attempt allowed, compared with where the opponent throws.
  4. Unit by unit: this team's offense vs the opponent's defense, and its defense vs the opponent's offense, grouped.
Everything is descriptive (context), not a model input. Left / right is always the offense's view, so a defense's
"left end" is where an offense runs to its own left.
"""
import html
import json
import math
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
WRCB = os.path.join(ROOT, "output", "hub", "wrcb.json")
RUNZ = [("LE", "Left end"), ("LT", "Left tackle"), ("LG", "Left guard"), ("MID", "Middle"), ("RG", "Right guard"), ("RT", "Right tackle"), ("RE", "Right end")]
DEPTH = [("Deep 20+", "Deep 20+"), ("Inter 10-19", "10-19"), ("Short 0-9", "0-9")]
SIDES = ["left", "middle", "right"]
MIN_ZONE = 8  # fewer plays than this in a cell and the colour stays neutral


def e(x):
    return html.escape(str(x))


def _ok(x):
    return x is not None and not (isinstance(x, float) and not math.isfinite(x))


def n(x, d=1):
    return f"{x:.{d}f}" if _ok(x) else "–"


def pc(x, d=0):
    return f"{100 * x:.{d}f}%" if _ok(x) else "–"


def sg(x, d=2):
    return f"{x:+.{d}f}" if _ok(x) else "–"


# ---------- play selection ----------

def rushes(p):
    return p[(p.rush_attempt == 1) & (p.qb_scramble != 1) & (p.play_type == "run")]


def passes(p):
    return p[(p.pass_attempt == 1) & (p.sack != 1) & (p.play_type == "pass")]


def dropbacks(p):
    return p[(p.qb_dropback == 1) & (p.play_type.isin(["pass", "run"]))]


def run_zone(p):
    loc, gap = p.run_location, p.run_gap
    side = loc.map({"left": "L", "right": "R"})
    g = gap.map({"end": "E", "tackle": "T", "guard": "G"})
    z = side + g
    return z.where(loc != "middle", "MID")


def pass_depth(p):
    return pd.cut(p.air_yards, [-99, 9.5, 19.5, 99], labels=["Short 0-9", "Inter 10-19", "Deep 20+"])


# ---------- one window of defensive plays ----------

def rush_stats(p, games):
    r = rushes(p)
    g = max(games, 1)
    return dict(g=games, car=len(r) / g, yds=r.yards_gained.sum() / g, ypc=r.yards_gained.mean() if len(r) else float("nan"),
                succ=r.success.mean() if len(r) else float("nan"), expl=(r.yards_gained >= 12).mean() if len(r) else float("nan"),
                td=int(r.rush_touchdown.sum()), epa=r.epa.mean() if len(r) else float("nan"),
                stuff=(r.yards_gained <= 0).mean() if len(r) else float("nan"))


def pass_stats(p, games):
    a, db = passes(p), dropbacks(p)
    g = max(games, 1)
    py = a.passing_yards.fillna(0).sum() if "passing_yards" in a else a.yards_gained.sum()
    return dict(g=games, att=len(a) / g, yds=py / g, ypa=py / len(a) if len(a) else float("nan"),
                cmp=a.complete_pass.mean() if len(a) else float("nan"), sacks=db.sack.sum() / g, ints=int(a.interception.sum()),
                td=int(a.pass_touchdown.sum()), epa=db.epa.mean() if len(db) else float("nan"),
                expl=(a.yards_gained >= 20).mean() if len(a) else float("nan"), succ=db.success.mean() if len(db) else float("nan"))


def run_map(r):
    """Carries and yards per carry by gap. r: rush plays."""
    z = r.assign(z=run_zone(r)).dropna(subset=["z"])
    tot = len(z) or 1
    out = {}
    for k, _ in RUNZ:
        x = z[z.z == k]
        out[k] = dict(n=len(x), share=len(x) / tot, ypc=x.yards_gained.mean() if len(x) else float("nan"),
                      succ=x.success.mean() if len(x) else float("nan"))
    return out


def pass_map(a):
    x = a[a.air_yards.notna() & a.pass_location.isin(SIDES)].assign(dp=lambda d: pass_depth(d))
    tot = len(x) or 1
    out = {}
    for dk, _ in DEPTH:
        for s in SIDES:
            y = x[(x.dp == dk) & (x.pass_location == s)]
            out[(dk, s)] = dict(n=len(y), share=len(y) / tot, ypa=y.yards_gained.mean() if len(y) else float("nan"),
                                cmp=y.complete_pass.mean() if len(y) else float("nan"), epa=y.epa.mean() if len(y) else float("nan"))
    return out


# ---------- ranks across the league for a full season ----------

RUSH_RANK = {"yds": True, "ypc": True, "succ": True, "expl": True, "epa": True, "stuff": False}
PASS_RANK = {"yds": True, "ypa": True, "cmp": True, "epa": True, "expl": True, "succ": True, "sacks": False}


def league(p):
    """Per-defense season numbers, their ranks (1 = stingiest) and league averages for one season of play-by-play."""
    teams = sorted(p.defteam.dropna().unique())
    gp = {t: p[(p.defteam == t) | (p.posteam == t)].game_id.nunique() for t in teams}
    R = {t: rush_stats(p[p.defteam == t], gp[t]) for t in teams}
    P = {t: pass_stats(p[p.defteam == t], gp[t]) for t in teams}
    rank = lambda D, k, low: pd.Series({t: D[t][k] for t in teams}).rank(ascending=low, method="min")
    rr = {k: rank(R, k, low) for k, low in RUSH_RANK.items()}
    pr = {k: rank(P, k, low) for k, low in PASS_RANK.items()}
    games = sum(gp.values()) / 2 or 1
    lr, lp = rush_stats(p, int(games * 2)), pass_stats(p, int(games * 2))
    return dict(R=R, P=P, rr=rr, pr=pr, lr=lr, lp=lp, rmap=run_map(rushes(p)), pmap=pass_map(passes(p)), gp=gp)


# ---------- visuals ----------

def _heat(diff, scale, good_low=True, n_=99):
    """Background for a map cell: red where the defense gives up more than the league, green where it gives up less."""
    if not _ok(diff) or n_ < MIN_ZONE:
        return "background:var(--pan2)"
    v = max(-1.0, min(1.0, diff / scale)) * (1 if good_low else -1)
    a = 0.10 + 0.42 * abs(v)
    col = "242,92,95" if v > 0 else "47,210,122"
    return f"background:rgba({col},{a:.2f})"


def run_strip(dm, lg, om, opp, show_off=True):
    """Seven gaps across the line of scrimmage. Top row: what this defense allows; bottom row: where the opponent runs."""
    cells = []
    for k, lab in RUNZ:
        d, L = dm[k], lg[k]
        diff = d["ypc"] - L["ypc"] if _ok(d["ypc"]) and _ok(L["ypc"]) else float("nan")
        thin = " thin" if d["n"] < MIN_ZONE else ""
        cells.append(f'<div class="rz{thin}" style="{_heat(diff, 2.0, True, d["n"])}" title="{e(lab)}: {d["n"]} carries, {n(d["ypc"])} YPC allowed (league {n(L["ypc"])})">'
                     f'<small>{k}</small><b>{n(d["ypc"])}</b><span>{d["n"]} car</span><em>{sg(diff, 1)}</em></div>')
    line = '<div class="rline">' + "".join(f"<i>{x}</i>" for x in ("", "T", "G", "C", "G", "T", "")) + "</div>"
    off = ""
    if show_off and om:
        mx = max(v["share"] for v in om.values()) or 1
        off = '<div class="rstrip off">' + "".join(
            f'<div class="ro" title="{e(lab)}: {pc(om[k]["share"])} of {opp} carries, {n(om[k]["ypc"])} YPC">'
            f'<span class="sh"><span style="height:{om[k]["share"] / mx * 100:.0f}%"></span></span><b>{pc(om[k]["share"])}</b><small>{n(om[k]["ypc"])}</small></div>'
            for k, lab in RUNZ) + "</div>"
        off = f'<p class="maplab">Where {e(opp)} runs <small>share of its carries · its YPC there · 2026</small></p>' + off
    return (f'<p class="maplab">Yards per carry allowed by gap <small>± vs league in that gap</small></p>'
            f'<div class="rstrip">{"".join(cells)}</div>{line}{off}')


def pass_grid(dm, lg, om, opp, show_off=True):
    def cell(dk, s):
        d, L = dm[(dk, s)], lg[(dk, s)]
        diff = d["ypa"] - L["ypa"] if _ok(d["ypa"]) and _ok(L["ypa"]) else float("nan")
        return (f'<div class="pz{" thin" if d["n"] < MIN_ZONE else ""}" style="{_heat(diff, 3.0, True, d["n"])}" title="{e(dk)} {s}: {d["n"]} attempts, {n(d["ypa"])} yds/att, {pc(d["cmp"])} complete (league {n(L["ypa"])})">'
                f'<b>{n(d["ypa"])}</b><span>{pc(d["cmp"])} · {d["n"]} att</span><em>{sg(diff, 1)}</em></div>')

    def ocell(dk, s):
        o = om[(dk, s)]
        a = min(1, o["share"] / 0.25)
        return (f'<div class="pz" style="background:rgba(107,140,255,{0.08 + 0.5 * a:.2f})" title="{e(dk)} {s}: {pc(o["share"])} of {opp} attempts, {n(o["ypa"])} yds/att">'
                f'<b>{pc(o["share"])}</b><span>{n(o["ypa"])} yds/att</span></div>')
    head = '<div class="pgh"><span></span>' + "".join(f"<small>{s.title()}</small>" for s in SIDES) + "</div>"
    rows = lambda f: "".join(f'<div class="pgr"><small>{lab}</small>{"".join(f(dk, s) for s in SIDES)}</div>' for dk, lab in DEPTH)
    d = f'<div class="pgrid">{head}{rows(cell)}<div class="los">line of scrimmage</div></div>'
    o = f'<div class="pgrid">{head}{rows(ocell)}<div class="los">line of scrimmage</div></div>' if show_off and om else ""
    left = f'<div><p class="maplab">Yards per attempt allowed <small>completion % · attempts · ± vs league</small></p>{d}</div>'
    right = f'<div><p class="maplab">Where {e(opp)} throws <small>share of its attempts · its yds/att · 2026</small></p>{o}</div>' if o else ""
    return f'<div class="pmaps">{left}{right}</div>'


def tiles(items):
    return '<div class="stiles">' + "".join(
        f'<div class="st{(" " + cls) if cls else ""}"><small>{lab}</small><b>{val}</b><span>{sub}</span></div>' for lab, val, sub, cls in items) + "</div>"


def _rk(r):
    return f"#{int(r)} of 32" if _ok(r) else ""


def _cls(r):
    return "" if not _ok(r) else ("good" if r <= 8 else "bad" if r >= 25 else "")


# ---------- the run and pass defense blocks ----------

def _games(p, t):
    g = p[(p.home_team == t) | (p.away_team == t)].drop_duplicates("game_id").sort_values("week")
    return [dict(gid=r.game_id, week=int(r.week), opp=r.away_team if r.home_team == t else r.home_team, home=r.home_team == t) for r in g.itertuples()]


def _window_label(x):
    return f"Week {x['week']} {'vs' if x['home'] else '@'} {x['opp']}"


def _rush_tiles(s, rk, lg, season):
    R = (lambda k: _rk(rk[k][TEAM]) if season else f"league {fmt_lg(k, lg)}")
    C = (lambda k: _cls(rk[k][TEAM]) if season else "")
    return tiles([("Rush yds / game", n(s["yds"], 0), R("yds"), C("yds")), ("Yards per carry", n(s["ypc"], 2), R("ypc"), C("ypc")),
                  ("Carries faced / game", n(s["car"], 1), f"league {fmt_lg('car', lg)}", ""), ("Success rate allowed", pc(s["succ"], 1), R("succ"), C("succ")),
                  ("Explosive runs (12+)", pc(s["expl"], 1), R("expl"), C("expl")), ("Stuffed at or behind line", pc(s["stuff"], 1), R("stuff"), C("stuff")),
                  ("EPA per rush", sg(s["epa"], 3), R("epa"), C("epa")), ("Rush TDs allowed", str(s["td"]), f"{s['g']} game{'s' if s['g'] != 1 else ''}", "")])


def _pass_tiles(s, rk, lg, season):
    R = (lambda k: _rk(rk[k][TEAM]) if season else f"league {fmt_lg(k, lg)}")
    C = (lambda k: _cls(rk[k][TEAM]) if season else "")
    return tiles([("Pass yds / game", n(s["yds"], 0), R("yds"), C("yds")), ("Yards per attempt", n(s["ypa"], 2), R("ypa"), C("ypa")),
                  ("Completion % allowed", pc(s["cmp"], 1), R("cmp"), C("cmp")), ("Attempts faced / game", n(s["att"], 1), f"league {fmt_lg('att', lg)}", ""),
                  ("Sacks / game", n(s["sacks"], 1), R("sacks"), C("sacks")), ("EPA per dropback", sg(s["epa"], 3), R("epa"), C("epa")),
                  ("Explosive passes (20+)", pc(s["expl"], 1), R("expl"), C("expl")), ("TD / INT allowed", f"{s['td']} / {s['ints']}", f"{s['g']} game{'s' if s['g'] != 1 else ''}", "")])


TEAM = None  # set per call so the tile helpers can read the team's rank


def fmt_lg(k, lg):
    v = lg.get(k)
    return {"yds": n(v, 0), "ypc": n(v, 2), "ypa": n(v, 2), "car": n(v, 1), "att": n(v, 1), "sacks": n(v, 1), "epa": sg(v, 3)}.get(k, pc(v, 1))


def _read_run(t, opp, dm, lg, om):
    """Two plain sentences: where the opponent runs most and how this defense does there; this defense's softest gap."""
    out = []
    if om:
        k = max(om, key=lambda z: om[z]["share"])
        lab = dict(RUNZ)[k].lower()
        d = dm[k]
        if d["n"] >= MIN_ZONE and _ok(d["ypc"]):
            word = "more" if d["ypc"] > lg[k]["ypc"] else "less"
            out.append(f"{opp}'s most-used gap is the {lab} ({pc(om[k]['share'])} of its carries, {n(om[k]['ypc'])} YPC). {t} gives up {n(d['ypc'])} YPC there, {word} than the league's {n(lg[k]['ypc'])}.")
    soft = [(dm[k]["ypc"] - lg[k]["ypc"], k) for k, _ in RUNZ if dm[k]["n"] >= MIN_ZONE and _ok(dm[k]["ypc"])]
    if soft:
        dv, k = max(soft)
        if dv > 0.4:
            o = f" {opp} runs there on {pc(om[k]['share'])} of its carries." if om else ""
            out.append(f"{t}'s softest gap is the {dict(RUNZ)[k].lower()}: {n(dm[k]['ypc'])} YPC on {dm[k]['n']} carries (league {n(lg[k]['ypc'])}).{o}")
    return "".join(f"<li>{e(x)}</li>" for x in out)


def _read_pass(t, opp, dm, lg, om):
    out = []
    if om:
        k = max(om, key=lambda z: om[z]["share"])
        d = dm[k]
        if d["n"] >= MIN_ZONE:
            out.append(f"{opp} throws {k[0].split()[0].lower()} {k[1]} most ({pc(om[k]['share'])} of attempts). {t} allows {n(d['ypa'])} yds/att there (league {n(lg[k]['ypa'])}).")
        deep = sum(om[(dk, s)]["share"] for dk, s in om if dk == "Deep 20+")
        dd = [dm[("Deep 20+", s)] for s in SIDES]
        dn = sum(x["n"] for x in dd)
        if dn >= MIN_ZONE:
            ypa = sum((x["ypa"] if _ok(x["ypa"]) else 0) * x["n"] for x in dd) / dn
            lgd = [lg[("Deep 20+", s)] for s in SIDES]
            lypa = sum((x["ypa"] if _ok(x["ypa"]) else 0) * x["n"] for x in lgd) / max(1, sum(x["n"] for x in lgd))
            out.append(f"Deep shots: {opp} throws 20+ yards downfield on {pc(deep)} of attempts; {t} allows {n(ypa)} yds/att on deep throws (league {n(lypa)}).")
    soft = [(dm[k]["ypa"] - lg[k]["ypa"], k) for k in dm if dm[k]["n"] >= MIN_ZONE and _ok(dm[k]["ypa"]) and _ok(lg[k]["ypa"])]
    if soft:
        dv, k = max(soft)
        if dv > 1.0:
            out.append(f"{t}'s softest zone is {k[0].lower()} to the {k[1]}: {n(dm[k]['ypa'])} yds/att on {dm[k]['n']} attempts (league {n(lg[k]['ypa'])}).")
    return "".join(f"<li>{e(x)}</li>" for x in out)


def defense_blocks(c, t, opp, gid, L26, L25):
    """Run-defense and pass-defense blocks for team t, each with one view selector."""
    global TEAM
    TEAM = t
    p26, p25 = c.p26, c.p25
    games = _games(p26, t)
    d26 = p26[p26.defteam == t]
    opp_r, opp_p = rushes(p26[p26.posteam == opp]), passes(p26[p26.posteam == opp])
    om_r, om_p = run_map(opp_r), pass_map(opp_p)
    views = [("s26", "2026 season", d26, len(games), True, L26), ("l5", "Last 5 games", d26[d26.game_id.isin([g["gid"] for g in games[-5:]])], min(5, len(games)), False, L26)]
    if t in L25["R"]:
        views.append(("s25", "2025 season", p25[p25.defteam == t], L25["gp"][t], True, L25))
    gviews = [(f"g{g['week']}", _window_label(g), d26[d26.game_id == g["gid"]], 1, False, L26) for g in reversed(games)]

    def select(kind):
        main = "".join(f'<option value="{kind}{k}">{lab}</option>' for k, lab, *_ in views)
        per = "".join(f'<option value="{kind}{k}">{lab}</option>' for k, lab, *_ in gviews)
        return (f'<label class="win" for="{gid}-{t}-{kind}">View <select id="{gid}-{t}-{kind}" data-win>{main}'
                f'<optgroup label="One 2026 game">{per}</optgroup></select></label>')

    rb, pb = [], []
    for i, (k, lab, dp, g, season, L) in enumerate(views + gviews):
        hid = "" if i == 0 else " hidden"
        rk_r, rk_p = (L["rr"], L["pr"]) if season and t in L["R"] else ({}, {})
        rs, ps = rush_stats(dp, g), pass_stats(dp, g)
        dm_r, dm_p = run_map(rushes(dp)), pass_map(passes(dp))
        cur = k in ("s26", "l5")  # the opponent comparison is only shown next to current-season views
        cap = f'<p class="vcap">{e(lab)}{"" if season else " · no league rank for a short sample; tiles compare with the league average"}</p>'
        rb.append(f'<div data-w="r{k}"{hid}>{cap}{_rush_tiles(rs, rk_r, L["lr"], bool(rk_r))}{run_strip(dm_r, L["rmap"], om_r if cur else None, opp, cur)}'
                  + (f'<ul class="read">{_read_run(t, opp, dm_r, L["rmap"], om_r)}</ul>' if cur else "") + "</div>")
        pb.append(f'<div data-w="p{k}"{hid}>{cap}{_pass_tiles(ps, rk_p, L["lp"], bool(rk_p))}{pass_grid(dm_p, L["pmap"], om_p if cur else None, opp, cur)}'
                  + (f'<ul class="read">{_read_pass(t, opp, dm_p, L["pmap"], om_p)}</ul>' if cur else "") + "</div>")
    note_r = ("Regular season, designed runs only (QB scrambles and kneels left out). Gaps come from play-by-play run direction: "
              "the end is outside the tackle, and middle runs have no gap. Left and right are the offense's view. "
              f"Cells with fewer than {MIN_ZONE} carries stay grey. Red: this defense gives up more than the league in that gap; green: less. "
              "Rank 1 is the stingiest defense.")
    note_p = ("Regular season. Throw zones use air yards (where the ball was caught or landed) and the offense's left / middle / right; "
              f"sacks and throwaways without a target side are left out of the map. Cells with fewer than {MIN_ZONE} attempts stay grey. "
              "Blue shading on the right shows how often the opponent throws there. Rank 1 is the stingiest defense.")
    run = f'<section class="mblock"><h4>{e(t)} against the rush <span class="kind ctx">Context</span>{select("r")}</h4>{"".join(rb)}<p class="note">{note_r}</p></section>'
    pas = f'<section class="mblock"><h4>{e(t)} against the pass <span class="kind ctx">Context</span>{select("p")}</h4>{"".join(pb)}<p class="note">{note_p}</p></section>'
    return run, pas


# ---------- WR / CB ----------

def _wrcb_game(week, a, h):
    try:
        d = json.load(open(WRCB))
    except (OSError, ValueError):
        return None, None
    g = next((x for x in d.get("games", []) if x["id"] == f"2026_{week}_{a}_{h}"), None)
    return g, d


def _sidebar(v):
    return ('<span class="lmr">' + "".join(f'<i style="width:{(x or 0) * 100:.0f}%" title="{lab} {pc(x)}">{lab}</i>' for lab, x in zip("LMR", v)) + "</span>")


def _wr_rows(blk, edge):
    out = []
    for w in blk["wr"]:
        adv = w.get("adv", "Even")
        sc = w.get("score") or 0
        pos = max(-60, min(60, sc)) / 60 * 50
        bar = (f'<span class="dv"><span class="{"pw" if sc > 0 else "pc"}" style="{"left:50%" if sc > 0 else f"left:{50 + pos:.0f}%"};width:{abs(pos):.0f}%"></span></span>')
        mix = ", ".join(f"{e(nm)} {pc(sh)}" for nm, sh in (w.get("mix") or [])[:3])
        inj = f' <span class="ij">{e(w["inj"])}</span>' if w.get("inj") else ""
        out.append(f'<div class="wcr"><div class="wn"><b>{e(w["name"])}</b><small>{e(w["pos"])} · {w.get("tgt", 0)} tgt · {pc(w.get("route"))} routes</small>{inj}</div>'
                   f'<span class="adv a{e(adv)}">{"Receiver edge" if adv == "WR" else "Corner edge" if adv == "CB" else "Even"}</span>'
                   f'<div class="wsc"><span class="pct">WR <b>{w.get("rp", "–")}</b></span>{bar}<span class="pct">CB <b>{w.get("cp", "–")}</b></span><em>{sc:+d}</em></div>'
                   f'<div class="wmx"><small>Most likely across from</small> {mix or "–"}</div>'
                   f'<div class="wsd"><small>Target side</small>{_sidebar(w.get("tsplit") or [0, 0, 0])}</div></div>')
    return "".join(out)


def _cb_rows(blk):
    rows = []
    for c in blk["cb"]:
        inj = f' <span class="ij">{e(c["inj"])}</span>' if c.get("inj") else ""
        lk = c.get("lock") or 0
        rows.append(f'<div class="cbr"><div class="wn"><b>{e(c["name"])}</b><small>{e(c.get("role", ""))} · {pc(c.get("snap"))} snaps</small>{inj}</div>'
                    f'<div class="lock"><span class="bar"><span style="width:{lk}%"></span></span><b>{lk}</b><small>lockdown pctl</small></div>'
                    f'<div class="cbs"><span>{c.get("rec", 0)}/{c.get("tga", 0)}</span><span>{c.get("yds", 0)} yds</span><span>{c.get("td", 0)} TD</span></div></div>')
    return "".join(rows)


def wrcb_block(week, a, h, t, opp):
    g, d = _wrcb_game(week, a, h)
    if not g:
        return f'<section class="mblock"><h4>{e(t)} WR / CB matchups <span class="kind ctx">Context</span></h4><p class="note">No WR/CB matchup data for this game yet.</p></section>'
    off = g["away"] if g["away"]["off"] == t else g["home"]
    de = g["home"] if off is g["away"] else g["away"]
    bt = d.get("backtest", {})
    tabs = (f'<label class="win" for="wc-{a}{h}-{t}">Show <select id="wc-{a}{h}-{t}" data-win>'
            f'<option value="wo{t}">{e(t)} receivers vs {e(opp)} corners</option><option value="wd{t}">{e(opp)} receivers vs {e(t)} corners</option></select></label>')
    body = (f'<div data-w="wo{t}"><div class="wcg">{_wr_rows(off, d.get("edge", 20))}</div><p class="maplab">{e(opp)} corners</p><div class="cbg">{_cb_rows(off)}</div></div>'
            f'<div data-w="wd{t}" hidden><div class="wcg">{_wr_rows(de, d.get("edge", 20))}</div><p class="maplab">{e(t)} corners</p><div class="cbg">{_cb_rows(de)}</div></div>')
    note = (f"2026 through week {d.get('through', '?')}. WR is the receiver's fantasy points per route percentile; CB is the percentile of how little the corners he is "
            f"most likely to see allow per coverage snap. The bar leans green for the receiver, red for the corner; {d.get('edge', 20)}+ either way is called an edge. "
            "Routes and alignment are estimates from free data (see the WR / CB page). "
            f"In the backtest the score did not predict receiver output (r {bt.get('r_2025', '–')} in 2025, {bt.get('r_2026', '–')} in 2026), so this is context, not a pick.")
    return f'<section class="mblock"><h4>{e(t)} WR / CB matchups <span class="kind ctx">Context</span>{tabs}</h4>{body}<p class="note">{e(note)}</p></section>'


# ---------- unit by unit ----------

GROUPS = [("Scoring and overall", [("Points per game", "ppg", 1), ("Yards per game", "ypg", 0), ("Yards per play", "ypp", 2), ("EPA per play", "epa", 3), ("Success rate", "success", -1)]),
          ("Passing", [("Pass yards per game", "pass_y", 0), ("Yards per attempt", "ypa", 2), ("Pass EPA per play", "pass_epa", 3), ("Explosive pass rate (20+)", "expl_pass", -1), ("Sacks per game", "sacks", 1)]),
          ("Rushing", [("Rush yards per game", "rush_y", 0), ("Yards per carry", "ypc", 2), ("Rush EPA per play", "rush_epa", 3), ("Explosive run rate (12+)", "expl_rush", -1)]),
          ("Situational", [("3rd-down rate", "third", -1), ("Red-zone TD rate", "rz_td", -1)]),
          ("Mistakes", [("Turnovers per game", "to", 2), ("Penalties per game", "penalties", 1)])]


def unit_blocks(c, R, t, opp, H):
    """H: report helpers (mrow, side_head, better, fmt, SUBLAB)."""
    def block(lt, ls, rt, rs, title):
        L, Rr = c.team_side(lt, ls), c.team_side(rt, rs)
        views = []
        for wk, wlab in (("Season", "2026 season"), ("Last 5", "Last 5 games")):
            rows = [H["side_head"](lt, ls, rt, rs, "2026 season<br># = league rank, 1 is best" if wk == "Season" else "Last 5 games<br>no rank for a short sample")]
            for gname, ms in GROUPS:
                rows.append(f'<p class="ugh">{gname}</p>')
                for lab, k, d in ms:
                    lv, rv = L[wk].get(k), Rr[wk].get(k)
                    lr, rr = (R[ls][k].get(lt), R[rs][k].get(rt)) if wk == "Season" else (None, None)
                    label = lab + (f"<small>{H['SUBLAB'][k]}</small>" if k in H["SUBLAB"] else "")
                    rows.append(H["mrow"](label, H["fmt"](lv, d), H["fmt"](rv, d), lr, rr, H["better"](lr, rr)))
            views.append((wk, "".join(rows)))
        key = f"u{lt}{ls}{rt}"
        sel = (f'<label class="win" for="{key}">View <select id="{key}" data-win><option value="{key}s">2026 season</option><option value="{key}5">Last 5 games</option></select></label>')
        body = f'<div data-w="{key}s">{views[0][1]}</div><div data-w="{key}5" hidden>{views[1][1]}</div>'
        return f'<section class="mblock"><h4>{title} <span class="kind ctx">Context</span>{sel}</h4>{body}<p class="note">Green is the better-ranked unit in each row. Season form, not a projection, and not adjusted for opponents faced.</p></section>'
    return block(t, "off", opp, "def", f"{e(t)} offense vs {e(opp)} defense") + block(t, "def", opp, "off", f"{e(t)} defense vs {e(opp)} offense")


def team_tab(c, R, gid, week, a, h, t, opp, L26, L25, H, extra=""):
    run, pas = defense_blocks(c, t, opp, gid, L26, L25)
    return wrcb_block(week, a, h, t, opp) + run + pas + unit_blocks(c, R, t, opp, H) + extra


CSS = """
.vcap{font:600 11px var(--m);text-transform:uppercase;letter-spacing:.08em;color:var(--dim);margin:2px 0 8px}
.stiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(132px,1fr));gap:8px;margin-bottom:14px}
.stiles .st{background:var(--bg);border:1px solid var(--line);border-radius:10px;padding:9px 12px;min-width:0}
.stiles small{display:block;font:600 10px var(--m);text-transform:uppercase;letter-spacing:.07em;color:var(--dim)}
.stiles b{display:block;font:700 20px/1.25 var(--m);font-variant-numeric:tabular-nums}
.stiles span{font:500 11.5px var(--m);color:var(--mut)}
.stiles .good{border-color:rgba(47,210,122,.45)}.stiles .good b{color:var(--good)}
.stiles .bad{border-color:rgba(242,92,95,.45)}.stiles .bad b{color:var(--bad)}
.maplab{font:600 13px var(--b);margin:14px 0 6px;display:flex;flex-wrap:wrap;gap:4px 10px;align-items:baseline}.maplab small{font:500 11.5px var(--m);color:var(--dim)}
.rstrip{display:grid;grid-template-columns:repeat(7,minmax(0,1fr));gap:4px}
.rz{border-radius:8px;padding:8px 2px;text-align:center;display:grid;gap:1px;border:1px solid var(--line);min-width:0}
.rz small{font:600 10px var(--m);color:var(--mut);letter-spacing:.05em}.rz b{font:700 17px/1.2 var(--m);font-variant-numeric:tabular-nums}
.rz span{font:500 10.5px var(--m);color:var(--mut)}.rz em{font:600 10.5px var(--m);font-style:normal;color:var(--fg);opacity:.85}
.rz.thin b,.pz.thin b{color:var(--dim)}
.rline{display:grid;grid-template-columns:repeat(7,minmax(0,1fr));gap:4px;margin-top:4px;border-top:2px dashed var(--line2);padding-top:4px}
.rline i{font:700 10px var(--m);font-style:normal;color:var(--dim);text-align:right;transform:translateX(50%)}
.rline i:first-child{visibility:hidden}
.rstrip.off .ro{display:grid;justify-items:center;gap:2px;min-width:0}
.ro .sh{display:flex;align-items:flex-end;height:44px;width:60%;background:var(--bg);border-radius:4px;overflow:hidden;border:1px solid var(--line)}
.ro .sh span{display:block;width:100%;background:rgba(107,140,255,.75)}
.ro b{font:700 12.5px var(--m)}.ro small{font:500 10.5px var(--m);color:var(--mut)}
.pmaps{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,300px),1fr));gap:8px 20px}
.pgrid{display:grid;gap:4px}
.pgh,.pgr{display:grid;grid-template-columns:52px repeat(3,minmax(0,1fr));gap:4px;align-items:stretch}
.pgh small{font:600 10px var(--m);text-transform:uppercase;letter-spacing:.06em;color:var(--dim);text-align:center}
.pgr>small{font:600 10.5px var(--m);color:var(--mut);align-self:center}
.pz{border-radius:8px;padding:9px 4px;text-align:center;display:grid;gap:1px;border:1px solid var(--line);min-width:0}
.pz b{font:700 17px/1.2 var(--m);font-variant-numeric:tabular-nums}.pz span{font:500 10.5px var(--m);color:var(--mut)}.pz em{font:600 10.5px var(--m);font-style:normal;opacity:.85}
.los{margin-left:56px;border-top:2px dashed var(--line2);padding-top:3px;font:600 9.5px var(--m);text-transform:uppercase;letter-spacing:.1em;color:var(--dim);text-align:center}
ul.read{margin:12px 0 4px;padding-left:18px;font-size:14px}ul.read li{margin-block:4px}ul.read:empty{display:none}
.wcg{display:grid;gap:8px}
.wcr{display:grid;grid-template-columns:minmax(0,1.3fr) auto minmax(0,1.4fr);grid-template-areas:"n a s" "m m d";gap:6px 14px;align-items:center;background:var(--bg);border:1px solid var(--line);border-radius:12px;padding:10px 14px}
.wcr .wn{grid-area:n;min-width:0}.wcr .adv{grid-area:a}.wcr .wsc{grid-area:s}.wcr .wmx{grid-area:m;min-width:0}.wcr .wsd{grid-area:d}
.wn b{font:600 15px var(--b)}.wn small{display:block;font:500 11.5px var(--m);color:var(--mut)}
.ij{font:700 10px var(--m);border-radius:5px;padding:1px 5px;background:rgba(168,111,10,.16);color:#e3b341}
.adv{font:700 10.5px var(--m);text-transform:uppercase;letter-spacing:.06em;border-radius:999px;padding:3px 10px;border:1px solid var(--line2);color:var(--mut);white-space:nowrap}
.adv.aWR{background:var(--grn-d);border-color:rgba(47,210,122,.45);color:var(--good)}.adv.aCB{background:var(--red-d);border-color:rgba(242,92,95,.45);color:var(--bad)}
.wsc{display:grid;grid-template-columns:auto minmax(60px,1fr) auto auto;gap:8px;align-items:center;font:500 11px var(--m);color:var(--mut)}.wsc b{color:var(--fg)}.wsc em{font-style:normal;font-weight:700;color:var(--fg);min-width:30px;text-align:right}
.dv{position:relative;height:8px;border-radius:4px;background:var(--line)}.dv::after{content:"";position:absolute;left:50%;top:-3px;bottom:-3px;width:1px;background:var(--line2)}
.dv span{position:absolute;top:0;bottom:0;border-radius:4px}.dv .pw{background:var(--good)}.dv .pc{background:var(--bad)}
.wmx{font-size:13px;color:var(--fg)}.wmx small,.wsd small{font:600 10px var(--m);text-transform:uppercase;letter-spacing:.06em;color:var(--dim);margin-right:6px}
.wsd{display:flex;align-items:center;gap:6px}
.lmr{display:flex;width:120px;height:16px;border-radius:5px;overflow:hidden;border:1px solid var(--line)}
.lmr i{display:grid;place-items:center;font:700 9px var(--m);font-style:normal;color:var(--bg);background:rgba(107,140,255,.75);overflow:hidden}.lmr i:nth-child(2){background:rgba(107,140,255,.5)}
.cbg{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,220px),1fr));gap:8px}
.cbr{background:var(--bg);border:1px solid var(--line);border-radius:12px;padding:10px 12px;display:grid;gap:6px;min-width:0}
.lock{display:grid;grid-template-columns:1fr auto auto;gap:6px;align-items:center}.lock b{font:700 14px var(--m)}.lock small{font:500 10.5px var(--m);color:var(--dim)}
.lock .bar{height:6px;background:var(--line);border-radius:3px;overflow:hidden;display:block}.lock .bar span{display:block;height:100%;background:var(--bad);opacity:.8;border-radius:3px}
.cbs{display:flex;gap:12px;font:500 12px var(--m);color:var(--mut)}
p.ugh{font:700 11px var(--m);text-transform:uppercase;letter-spacing:.1em;color:var(--good);margin:16px 0 2px;padding-bottom:4px;border-bottom:1px solid var(--line2)}
@media (max-width:640px){
 .wcr{grid-template-columns:minmax(0,1fr) auto;grid-template-areas:"n a" "s s" "m m" "d d"}
 .rz{padding:6px 1px}.rz b{font-size:14px}.rz span,.rz em{font-size:9.5px}
 .pgh,.pgr{grid-template-columns:40px repeat(3,minmax(0,1fr))}.los{margin-left:44px}.pz b{font-size:15px}
}
"""
