"""One page per sport, so each has its own web address, title and description for search engines and sharing.

Each page is the hub itself opened on that sport (window.HUBSTART), with <base href="../"> so it reads the same
data files as the main page. They are complete HTML documents (the main index.html is wrapped by publish_site.py
instead). Also writes sitemap.xml and robots.txt for the public site.
"""
import html
import os

SITE_URL = "https://the-brogram.github.io/AllThingsSports/"
PAGES = [
    ("nfl", "nfl", "NFL predictions and Game Books",
     "Win chances, projected scores and player lines for every NFL game this week, from thousands of simulated games, graded after every game."),
    ("college-football", "cfb", "College football predictions",
     "Win chances and Game Books for every Top 25, SEC, ACC, Big Ten and Big 12 game this week, graded after every game."),
    ("nba", "nba", "NBA predictions tonight",
     "Win chances for every NBA game tonight from 10,000 simulations, plus player lines next to DraftKings' prices."),
    ("nhl", "nhl", "NHL predictions today",
     "Who wins every NHL game today, how it splits between regulation, overtime and a shootout, and goalie news."),
    ("champions-league", "ucl-md", "Champions League predictions",
     "Chances for every Champions League match from expected goals, with the table, results and scouting notes."),
    ("fantasy", "fantasy", "NFL fantasy football rankings",
     "Weekly NFL fantasy rankings and start or sit help from the same simulations behind our NFL predictions."),
]
VIEWPORT = '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">'
RESET = ('<style>:root{color-scheme:light;box-sizing:border-box;padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}'
         'body{margin:0;padding:0}img{max-width:100%}[hidden]{display:none!important}</style>')


def build(hub):
    page = open(os.path.join(hub, "index.html")).read()
    t0, d0 = page.index("<title>"), page.index('<meta name="description"')
    head_end = page.index(">", d0) + 1
    rest = page[head_end:]
    files = {}
    for folder, start, title, desc in PAGES:
        t, d, url = html.escape(title + " · AllThingsSports"), html.escape(desc), SITE_URL + folder + "/"
        head = (f'<!doctype html><html lang="en"><head><meta charset="utf-8">{VIEWPORT}<base href="../"><title>{t}</title>'
                f'<meta name="description" content="{d}"><link rel="canonical" href="{url}">'
                f'<meta property="og:title" content="{t}"><meta property="og:description" content="{d}"><meta property="og:url" content="{url}">'
                f'<meta property="og:type" content="website">{RESET}<script>window.HUBSTART="{start}"</script>')
        body = page[:t0].replace('<meta charset="utf-8">', "") + rest
        out = os.path.join(hub, folder, "index.html")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        open(out, "w").write(head + "</head><body>\n" + body + "\n</body></html>\n")
        files[folder + "/index.html"] = out
    urls = [SITE_URL] + [SITE_URL + f + "/" for f, *_ in PAGES]
    open(os.path.join(hub, "sitemap.xml"), "w").write('<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
                                                     + "".join(f"  <url><loc>{u}</loc></url>\n" for u in urls) + "</urlset>\n")
    open(os.path.join(hub, "robots.txt"), "w").write(f"User-agent: *\nAllow: /\nSitemap: {SITE_URL}sitemap.xml\n")
    return files


if __name__ == "__main__":
    print(build(os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "hub")))
