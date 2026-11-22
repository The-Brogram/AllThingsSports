"""Picks Desk: Josh's private page for typing in the parlay and picks of the day.

It runs only on this computer (http://127.0.0.1:8771, never reachable from outside), so there is nothing public to protect.
- Save writes picks/picks.json and copies it into the built hub (output/hub/picks.json).
- Save and publish also runs publish_site.py, which pushes the hub to the public site (GitHub keychain, no token handled here).
The Claude hub link picks the change up the next time a job (or Claude) publishes it.

Run: venv/bin/python picks_desk.py      (opens the page in your browser; Ctrl+C to stop)
"""
import json
import os
import shutil
import subprocess
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "picks", "picks.json")
HUB = os.path.join(ROOT, "output", "hub", "picks.json")
PAGE = os.path.join(ROOT, "picks_desk.html")
PORT = 8771
SPORTS = {"nfl", "cfb", "nba", "nhl", "ucl", "mlb", "other"}
RESULTS = {"open", "won", "lost", "push", "void"}


def text(v, n=200):
    return str(v or "").strip()[:n]


def num(v):
    try:
        f = float(v)
        return f if f == f else None
    except (TypeError, ValueError):
        return None


def odds(o, fmt):
    """Keep odds as typed, but only if they make sense: American at least 100 in size, a multiplier above 1."""
    v = num(str(o or "").lower().replace("x", "").strip())
    if v is None:
        return ""
    if fmt == "x":
        return f"{v:g}" if v > 1 else ""
    return f"{v:+g}" if abs(v) >= 100 else ""


def leg(x):
    fmt = "x" if x.get("fmt") == "x" else "us"
    return dict(bet=text(x.get("bet")), sport=x.get("sport") if x.get("sport") in SPORTS else "other", odds=odds(x.get("odds"), fmt), fmt=fmt,
                result=x.get("result") if x.get("result") in RESULTS else "open")


def pick(x):
    out = leg(x)
    u, c = num(x.get("units")), num(x.get("chance"))
    out.update(units=u if u and u > 0 else 1, chance=c if c and 0 < c < 100 else None, note=text(x.get("note"), 300))
    return out


def clean(d):
    days = []
    for day in d.get("days") or []:
        date = text(day.get("date"), 10)
        if len(date) != 10:
            continue
        p = day.get("parlay") or {}
        legs = [leg(x) for x in p.get("legs") or [] if text(x.get("bet"))]
        picks = [pick(x) for x in day.get("picks") or [] if text(x.get("bet"))]
        if not legs and not picks:
            continue
        out = dict(date=date, picks=picks)
        if legs:
            par = pick(p)
            par.pop("bet"), par.pop("sport")
            par["legs"] = legs
            out["parlay"] = par
        days.append(out)
    days.sort(key=lambda x: x["date"], reverse=True)
    return dict(locked=bool(d.get("locked")), days=days)


def save(d):
    d = clean(d)
    os.makedirs(os.path.dirname(SRC), exist_ok=True)
    json.dump(d, open(SRC, "w"), indent=1)
    if os.path.isdir(os.path.dirname(HUB)):
        shutil.copy2(SRC, HUB)
    return d


class H(BaseHTTPRequestHandler):
    def send(self, code, body, kind="application/json"):
        b = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", kind + "; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            return self.send(200, open(PAGE).read(), "text/html")
        if self.path.startswith("/picks.json"):
            return self.send(200, open(SRC).read() if os.path.exists(SRC) else '{"locked": true, "days": []}')
        self.send(404, '{"error": "not found"}')

    def do_POST(self):
        if self.headers.get("Origin") not in (None, f"http://127.0.0.1:{PORT}", f"http://localhost:{PORT}"):
            return self.send(403, '{"error": "only this computer can save picks"}')
        try:
            d = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        except ValueError:
            return self.send(400, '{"error": "bad data"}')
        if self.path == "/save":
            return self.send(200, json.dumps(dict(ok=True, data=save(d))))
        if self.path == "/publish":
            data = save(d)
            r = subprocess.run([sys.executable, os.path.join(ROOT, "publish_site.py")], cwd=ROOT, capture_output=True, text=True,
                               env=dict(os.environ, GIT_TERMINAL_PROMPT="0"), timeout=600)
            tail = "\n".join((r.stdout + r.stderr).strip().splitlines()[-4:])
            return self.send(200, json.dumps(dict(ok=r.returncode == 0, data=data, log=tail)))
        self.send(404, '{"error": "not found"}')

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), H)
    url = f"http://127.0.0.1:{PORT}/"
    print("Picks Desk:", url, "(Ctrl+C to stop)")
    if "--no-open" not in sys.argv:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
