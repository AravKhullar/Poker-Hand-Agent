"""Fetch Wikipedia strategy articles into data/corpus/.

Calls Wikipedia's action API directly with requests, rather than the
`wikipedia` PyPI package — that package is unmaintained and no longer sends
a proper User-Agent, which makes Wikipedia's API reject the request (you'll
see it fail with a JSONDecodeError because the response isn't JSON at all,
it's an error page). Wikipedia's API etiquette requires a descriptive
User-Agent identifying the client; this script sends one.

Usage:
    uv run --group scripts python scripts/fetch_wiki.py
"""

import os

import requests

OUTPUT_DIR = "data/corpus"
API_URL = "https://en.wikipedia.org/w/api.php"

# Wikipedia asks that automated clients identify themselves with a contact
# point. Fill in your email or GitHub URL — this is standard API etiquette,
# not optional flavor text.
USER_AGENT = "PokerHandReviewAgent/0.1 (personal portfolio project; contact: <your email or github>)"

PAGES = [
    "Poker strategy",
    "Pot odds",
    "Poker probability",
    "Betting in poker",
    "Texas hold 'em",
    "Pot (poker)",
]


def slugify(title: str) -> str:
    return (
        title.lower()
        .replace(" ", "_")
        .replace("'", "")
        .replace("(", "")
        .replace(")", "")
        + ".txt"
    )


def fetch_page(title: str) -> None:
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": 1,
        "redirects": 1,       # follow redirect pages (e.g. alternate titles)
        "titles": title,
        "format": "json",
    }
    resp = requests.get(
        API_URL, params=params, headers={"User-Agent": USER_AGENT}, timeout=15
    )
    resp.raise_for_status()
    data = resp.json()

    pages = data.get("query", {}).get("pages", {})
    page = next(iter(pages.values()), None)

    if page is None or "missing" in page:
        print(f"  SKIP '{title}': page not found")
        return

    extract = page.get("extract", "")
    if not extract or len(extract) < 200:
        print(f"  WARNING '{title}': extract looks too short ({len(extract)} chars)")

    real_title = page.get("title", title)
    url = f"https://en.wikipedia.org/wiki/{real_title.replace(' ', '_')}"

    path = os.path.join(OUTPUT_DIR, slugify(title))
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {real_title}\nSource: {url}\n\n{extract}")
    print(f"  saved {path} ({len(extract)} chars)")


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Fetching {len(PAGES)} pages into {OUTPUT_DIR}/")
    for title in PAGES:
        try:
            fetch_page(title)
        except requests.RequestException as e:
            print(f"  FAILED '{title}': {e}")
    print("Done.")


if __name__ == "__main__":
    main()