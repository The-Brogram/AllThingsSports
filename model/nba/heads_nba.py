"""Player headshots for the hub: download ESPN's headshot once per player, shrink it, and cache it under data/heads/.
The hub cannot load images from other sites, so build() returns {pid: data URI} for the players it needs, written to heads.json.
Players with no ESPN headshot (404) are remembered in data/heads/missing.json and get initials in the hub.
"""
import base64
import io
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
DIR = f"{HERE}/data/heads"
URL = "https://a.espncdn.com/i/headshots/nba/players/full/{}.png"
W = 168  # px wide, ~2x the 84px card photo


def fetch(pid):
    f = f"{DIR}/{pid}.webp"
    if os.path.exists(f):
        return True
    r = subprocess.run(["curl", "-s", "-m", "30", "-w", "%{http_code}", "-o", "-", URL.format(pid)], capture_output=True)
    code, body = r.stdout[-3:], r.stdout[:-3]
    if code != b"200" or not body:
        return False
    try:
        im = Image.open(io.BytesIO(body)).convert("RGBA")
    except OSError:
        return False
    box = im.getbbox()  # trim empty transparent border
    if box:
        im = im.crop(box)
    im.thumbnail((W, W * 2))
    im.save(f, "WEBP", quality=72, method=6)  # keeps transparency, so it sits on any card color
    return True


def build(pids, out):
    os.makedirs(DIR, exist_ok=True)
    mf = f"{DIR}/missing.json"
    missing = set(json.load(open(mf))) if os.path.exists(mf) else set()
    todo = [p for p in pids if p not in missing and not os.path.exists(f"{DIR}/{p}.webp")]
    with ThreadPoolExecutor(8) as ex:
        ok = dict(zip(todo, ex.map(fetch, todo)))
    missing |= {p for p, v in ok.items() if not v}
    json.dump(sorted(missing), open(mf, "w"))
    heads = {}
    for p in pids:
        f = f"{DIR}/{p}.webp"
        if os.path.exists(f):
            heads[p] = "data:image/webp;base64," + base64.b64encode(open(f, "rb").read()).decode()
    json.dump(heads, open(out, "w"), separators=(",", ":"))
    return len(heads), len(todo)
