"""Champions League pages for the hub: ucl/ucl.json, copied into the hub as ucl.json.

The Champions League model lives outside this folder (TheBrogram/soccer) and runs only on this computer.
Each matchday page it exports (soccer/web/matchups_<date>.json) is gathered here, with final scores from
ESPN once a match is over. The result is kept in ucl/ucl.json inside the program, so GitHub's runs (which
have no soccer folder) still publish the last copy built here instead of an empty one.
"""
import glob
import json
import os
import re
import subprocess
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
SOCCER = os.path.join(os.path.dirname(ROOT), "soccer", "web")
KEEP = os.path.join(ROOT, "ucl", "ucl.json")
ESPN = "https://site.api.espn.com/apis/site/v2/sports/soccer/uefa.champions/scoreboard?dates={}"


def finals(day, have):
    """Final scores for one date from ESPN, keyed by event id. Earlier finals are reused, never refetched."""
    out = {k: v for k, v in have.items() if v.get("final")}
    try:
        raw = subprocess.run(["curl", "-s", "--compressed", "--max-time", "20", ESPN.format(day.replace("-", ""))],
                             capture_output=True, text=True, timeout=30).stdout
        for e in json.loads(raw).get("events", []):
            if e["id"] in out:
                continue
            st = e["status"]["type"]
            c = {t["homeAway"]: t for t in e["competitions"][0]["competitors"]}
            out[e["id"]] = {"final": bool(st.get("completed")), "state": st.get("shortDetail", ""),
                            "hs": int(c["home"].get("score") or 0), "as": int(c["away"].get("score") or 0)}
    except Exception as ex:  # noqa: BLE001  results are an extra; the matchday pages still build
        print("UCL results skipped for", day, repr(ex))
    return out


def build(hub):
    old = json.load(open(KEEP)) if os.path.exists(KEEP) else {"days": []}
    src = sorted(glob.glob(os.path.join(SOCCER, "matchups_*.json")))
    if src:  # this computer: rebuild from the soccer model's own exports
        prev = {d["day"]: d for d in old.get("days", [])}
        days = []
        for f in src:
            day = re.search(r"matchups_(\d{4}-\d{2}-\d{2})\.json$", f)
            if not day:
                continue
            day = day.group(1)
            page = json.load(open(f))
            if isinstance(page, list):  # the first export was a bare list of matches, with no label or pick
                page = {"date": datetime.strptime(day, "%Y-%m-%d").strftime("%A %-d %B %Y"), "pick": None, "matches": page}
            have = (prev.get(day) or {}).get("res", {})
            res = finals(day, have) if day <= datetime.now().strftime("%Y-%m-%d") else have
            days.append({"day": day, "label": page["date"], "pick": page.get("pick"), "matches": page["matches"],
                         "res": {k: v for k, v in res.items() if any(m["id"] == k for m in page["matches"])}})
        data = {"built": datetime.now().strftime("%Y-%m-%d %H:%M"), "days": days}
        os.makedirs(os.path.dirname(KEEP), exist_ok=True)
        json.dump(data, open(KEEP, "w"), separators=(",", ":"))
    elif not old["days"]:
        return {}
    out = os.path.join(hub, "ucl.json")
    with open(KEEP) as a, open(out, "w") as b:
        b.write(a.read())
    return {"ucl.json": out}


if __name__ == "__main__":
    print(build(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "output", "hub")))
