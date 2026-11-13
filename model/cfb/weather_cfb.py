"""National Weather Service hourly forecast at each outdoor stadium for kickoff. Run: python weather_cfb.py
Stadium coordinates come from the venue's city (Open-Meteo geocoding, cached in data/venues.json). Indoor venues are skipped.
Writes output/weather_w<week>.json, keyed by home team."""
import json
import os
import subprocess
import urllib.parse
import pandas as pd
import cfb_model as C
import run_cfb as R

VEN = f"{C.D}/venues.json"
# city geocoding misses or lands far from the stadium for a few campuses
FIX = {"NAVY": (38.9845, -76.5079), "ARMY": (41.3876, -73.9637), "AF": (38.9969, -104.8436), "PSU": (40.8122, -77.8561), "CLEM": (34.6788, -82.8432),
       "BC": (42.3350, -71.1664), "PITT": (40.4468, -80.0158), "MIA": (25.9580, -80.2389), "ND": (41.6985, -86.2339), "VT": (37.2200, -80.4180)}
STATES = {"AL": "Alabama", "AZ": "Arizona", "AR": "Arkansas", "CA": "California", "CO": "Colorado", "CT": "Connecticut", "FL": "Florida", "GA": "Georgia", "IA": "Iowa", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "KS": "Kansas",
          "KY": "Kentucky", "LA": "Louisiana", "MA": "Massachusetts", "MD": "Maryland", "MI": "Michigan", "MN": "Minnesota", "MO": "Missouri", "MS": "Mississippi", "NC": "North Carolina", "NE": "Nebraska", "NJ": "New Jersey",
          "NM": "New Mexico", "NV": "Nevada", "NY": "New York", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania", "SC": "South Carolina", "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VA": "Virginia",
          "WA": "Washington", "WI": "Wisconsin", "WV": "West Virginia", "HI": "Hawaii", "WY": "Wyoming", "DC": "District of Columbia"}


def get(u):
    r = subprocess.run(["curl", "-s", "-m", "20", "-H", "User-Agent: cfb-model", u], capture_output=True, text=True).stdout
    return json.loads(r)


def coords(g, ven):
    if g["home"] in FIX and not g["neutral"]:
        return FIX[g["home"]]
    key = f"{g['city']}, {g['state']}"
    if key not in ven:
        st = STATES.get(g["state"], g["state"])
        res = get("https://geocoding-api.open-meteo.com/v1/search?count=10&country=US&name=" + urllib.parse.quote(g["city"])).get("results", [])
        res = [x for x in res if x.get("admin1") == st] or res
        ven[key] = (res[0]["latitude"], res[0]["longitude"]) if res else None
    return ven[key]


def main():
    ven = json.load(open(VEN)) if os.path.exists(VEN) else {}
    wk = R.week()
    out = {}
    for g in R.slate()[0]:
        if g["indoor"]:
            out[g["home"]] = dict(indoor=True)
            continue
        try:
            la, lo = coords(g, ven)
            hrs = get(get(f"https://api.weather.gov/points/{la:.4f},{lo:.4f}")["properties"]["forecastHourly"])["properties"]["periods"]
            k = pd.Timestamp(g["kick"])
            h = min(hrs, key=lambda p: abs(pd.Timestamp(p["startTime"]) - k))
            h2 = min(hrs, key=lambda p: abs(pd.Timestamp(p["startTime"]) - k - pd.Timedelta(hours=3)))
            if abs(pd.Timestamp(h["startTime"]) - k) > pd.Timedelta(hours=2):
                raise ValueError("kickoff is past the hourly forecast window")
            out[g["home"]] = dict(temp=h["temperature"], wind=h["windSpeed"], dir=h["windDirection"], rain=h.get("probabilityOfPrecipitation", {}).get("value"), short=h["shortForecast"], temp_end=h2["temperature"], wind_end=h2["windSpeed"], rain_end=h2.get("probabilityOfPrecipitation", {}).get("value"))
        except (KeyError, ValueError, TypeError) as e:
            out[g["home"]] = dict(error=repr(e)[:80])
    json.dump(ven, open(VEN, "w"), indent=0)
    json.dump(out, open(f"output/weather_w{wk}.json", "w"), indent=1)
    ok = sum("temp" in v for v in out.values())
    print(f"Week {wk}: {ok} forecasts, {sum(bool(v.get('indoor')) for v in out.values())} indoor, {sum('error' in v for v in out.values())} unavailable")


if __name__ == "__main__":
    main()
