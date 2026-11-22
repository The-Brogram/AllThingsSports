"""Team logos for the NBA, NHL, college football and the Champions League, from ESPN's public team lists.

Each logo is downloaded once (ESPN's dark-background version when it has one), shrunk to 64 px and kept in
data/logos/<sport>/<team key>.png, where the key is the one the hub already uses (NBA/NHL/college abbreviations,
ESPN ids for Champions League clubs). build() copies them into the hub's logos/<sport>/ folder.
The NFL logos stay where they were (data/logos/*.png). A team with no logo just shows its name.
"""
import glob
import io
import json
import os
import shutil
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "data", "logos")
LEAGUES = {"nba": "basketball/nba", "nhl": "hockey/nhl", "cfb": "football/college-football", "ucl": "soccer/uefa.champions"}
HUBJSON = {"nba": "nba.json", "nhl": "nhl.json", "cfb": "cfb.json", "ucl": "ucl.json"}
SIZE = 64


def get(url):
    """curl, like hub_ucl.py: Python's own certificate store is missing on this computer."""
    r = subprocess.run(["curl", "-sf", "--compressed", "--max-time", "25", url], capture_output=True, timeout=40)
    if r.returncode or not r.stdout:
        raise OSError(f"curl failed ({r.returncode}) for {url}")
    return r.stdout


def espn_teams(sport):
    url = f"https://site.api.espn.com/apis/site/v2/sports/{LEAGUES[sport]}/teams?limit=1000" + ("&groups=80" if sport == "cfb" else "")
    d = json.loads(get(url))
    return [x["team"] for x in d["sports"][0]["leagues"][0]["teams"]]


def logo_url(team):
    logos = team.get("logos") or []
    for want in (("full", "dark"), ("default",)):
        for lg in logos:
            if all(w in (lg.get("rel") or []) for w in want):
                return lg.get("href")
    return logos[0].get("href") if logos else None


def match(sport, ours, espn):
    """Our team key -> ESPN team. College matches on the full name first (abbreviations clash, e.g. OSU)."""
    byid = {e["id"]: e for e in espn}
    byab = {e.get("abbreviation"): e for e in espn}
    byname = {e.get("displayName"): e for e in espn}
    out = {}
    for k, v in ours.items():
        if sport == "ucl":
            e = byid.get(k)
        elif sport == "cfb":
            e = byname.get(v.get("name")) or byab.get(k)
        else:
            e = byab.get(k) or byname.get(v.get("name"))
        if e:
            out[k] = e
    return out


def shrink(raw):
    from PIL import Image
    im = Image.open(io.BytesIO(raw)).convert("RGBA")
    bb = im.getchannel("A").getbbox()
    if bb:
        im = im.crop(bb)
    w, h = im.size
    s = max(w, h)
    sq = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    sq.paste(im, ((s - w) // 2, (s - h) // 2))
    buf = io.BytesIO()
    sq.resize((SIZE, SIZE), Image.LANCZOS).save(buf, "PNG", optimize=True)
    return buf.getvalue()


def fetch(hub):
    """Download any logo not already kept. Returns {sport: count newly saved}."""
    new = {}
    for sport, f in HUBJSON.items():
        path = os.path.join(hub, f)
        if not os.path.exists(path):
            continue
        ours = json.load(open(path)).get("teams") or {}
        os.makedirs(os.path.join(SRC, sport), exist_ok=True)
        todo = [k for k in ours if k and k != "TBD" and not os.path.exists(os.path.join(SRC, sport, k + ".png"))]
        if not todo:
            continue
        try:
            m = match(sport, {k: ours[k] for k in todo}, espn_teams(sport))
        except Exception as e:  # noqa: BLE001  ESPN down: try again next build
            print(f"Logos {sport}: team list failed", repr(e))
            continue
        n = 0
        for k, e in m.items():
            url = logo_url(e)
            if not url:
                continue
            try:
                open(os.path.join(SRC, sport, k + ".png"), "wb").write(shrink(get(url)))
                n += 1
            except Exception as ex:  # noqa: BLE001
                print(f"Logos {sport} {k}: failed", repr(ex))
        new[sport] = n
    return new


def build(hub, files):
    """Fetch anything missing, then copy every kept logo into the hub (logos/<sport>/KEY.png)."""
    new = fetch(hub)
    counts = {}
    for sport in HUBJSON:
        srcs = glob.glob(os.path.join(SRC, sport, "*.png"))
        if not srcs:
            continue
        os.makedirs(os.path.join(hub, "logos", sport), exist_ok=True)
        for s in srcs:
            name = os.path.basename(s)
            dst = os.path.join(hub, "logos", sport, name)
            shutil.copy2(s, dst)
            files[f"logos/{sport}/{name}"] = dst
        counts[sport] = len(srcs)
    print("Team logos:", ", ".join(f"{k} {v}" + (f" (+{new[k]} new)" if new.get(k) else "") for k, v in counts.items()))
