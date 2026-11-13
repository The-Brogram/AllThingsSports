"""List the quarterback the model picked for every team on this week's slate and flag the ones to check by hand.
Run after run_cfb.py: python qb_check.py
Flagged (**): the modeled QB has under 40% of the team's 2026 attempts (a recent change), or a QB on that team is on the injury list
and has not played since being listed. Confirmed starters and notes go in qb.json under the week, then rerun run_cfb.py and report_cfb.py."""
import pickle
import run_cfb as R

wk = R.week()
c = pickle.load(open(R.cache(wk), "rb"))
hd = R.hand(wk)
n = 0
for s in c["res"]:
    g = s["game"]
    for side, tm in (("a", g["away"]), ("h", g["home"])):
        df = s["df" + side]
        q = df[df.pos == "QB"]
        qb = q.name.iat[0] if len(q) else None
        f = s["qb_flag"].get(tm)
        qinj = [x for x in s["inj"].get(tm, []) if x["pos"] == "QB" and not x.get("stale")]
        mark = (f is not None or qinj) and tm not in hd.get("note", {})
        n += bool(mark)
        print(("** " if mark else "   ") + f"{tm:5} {qb}" + (f"  share of attempts {f:.0%}" if f is not None else "") +
              (f"  QB injury list: {[(x['name'], x['status']) for x in qinj]}" if qinj else "") + ("  [checked: " + hd["note"][tm][:60] + "...]" if tm in hd.get("note", {}) else ""))
print(f"\nWeek {wk}: {n} teams flagged for a hand check.")
