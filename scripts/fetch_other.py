"""Fetch a small, explicit list of free strategy-blog articles into
data/corpus/.

Targeted fetcher, not a crawler: only requests the exact URLs listed below,
never follows links, waits between requests.

Content extraction strategy (v2 — fixed after a real bug on gtowizard.com):
  1. Hard-exclude noise by TAG NAME first: <nav>, <footer>, <header>,
     <aside> are structurally chrome on essentially every site, regardless
     of what CSS classes they carry. This alone strips sidebars, site
     headers/footers, and promo/cookie-banner <aside> elements.
  2. <body> and <html> are NEVER treated as noise, even if they carry
     classes that happen to match a noise keyword. A previous version of
     this script checked keywords as a raw substring against every
     ancestor's class list, including <body> — and gtowizard.com's <body>
     has classes like "tag-cash tag-mtt", where "tag" is a substring. That
     flagged <body> as noise, and since <body> is an ancestor of every
     paragraph on the page, it silently zeroed out the entire extraction.
     Lesson: never let a keyword check propagate through the document root.
  3. Beyond tag names, a short list of class/id keywords (sidebar, related,
     comment, cookie, promo, etc.) catches remaining noise containers that
     aren't structural HTML5 tags.
  4. Score remaining candidate containers by how much real paragraph text
     they hold, keep the highest-scoring one.
  5. Strip Ghost-CMS shortcode markers like {{width: 75%}} or
     {{color: green}}...{{/color}} — these are client-side-only styling
     hints that leak into the raw HTML as literal text on Ghost-powered
     blogs (gtowizard.com is one). Left in, they'd pollute the corpus with
     junk tokens.

Before adding a URL: confirm the page is freely readable (no paywall/login).

Usage:
    uv run --group scripts python scripts/fetch_strategy_blogs.py
"""

import os
import re
import time

import requests
from bs4 import BeautifulSoup, Tag

OUTPUT_DIR = "data/corpus"
REQUEST_DELAY_SECONDS = 2.0
USER_AGENT = "Mozilla/5.0 (personal research project; contact: <your email>)"

URLS = [
    "https://blog.gtowizard.com/playing-under-10bb-part-1-cev/",
    "https://blog.gtowizard.com/playing-under-10bb-part-2-icm/",
    "https://blog.gtowizard.com/the-four-point-checklist-for-calling-turn-barrels/",
    "https://blog.gtowizard.com/3-way-how-to-exploit-a-calling-station/",
    "https://blog.gtowizard.com/exploiting-profiles-episode-ii-the-nit/",
    "https://blog.gtowizard.com/exploiting-profiles-episode-i-the-calling-station/",
    "https://blog.gtowizard.com/probing-out-of-position-in-3-way-pots/",
    "https://blog.gtowizard.com/mississippi-straddles-building-a-cbetting-strategy-vs-btn/",
    "https://blog.gtowizard.com/the-10-most-important-concepts-for-new-poker-players/",
    "https://blog.gtowizard.com/principles-of-gto/",
    "https://blog.gtowizard.com/how-to-become-a-gto-wizard/",
    "https://blog.gtowizard.com/poker-concepts-youre-using-wrong-and-how-theyre-hurting-your-winrate/",
    "https://blog.gtowizard.com/poker-strategies-tournaments-vs-cash-games/",
]

# Structural tags that are essentially always chrome, not article content.
NOISE_TAGS = {"nav", "footer", "header", "aside", "form"}

# Class/id substrings for noise containers that aren't structural HTML5
# tags. Keep these SPECIFIC — a short/generic word (like the old "tag") can
# accidentally match something on <body> or another root-level element and
# silently wipe out the whole extraction, since the noise check propagates
# through every ancestor of every paragraph.
NOISE_KEYWORDS = [
    "sidebar", "share", "social", "related", "recommend", "comment",
    "newsletter", "subscribe", "advert", "promo", "cookie", "breadcrumb",
    "widget", "post-tags", "toc-container", "logo-dropdown", "search-bar",
]

# Ghost CMS shortcode markers, e.g. {{width: 75%}}, {{/width}},
# {{question-mark}}, {{color: green}}. Strip entirely.
GHOST_SHORTCODE_RE = re.compile(r"\{\{.*?\}\}")


def is_noise(tag: Tag) -> bool:
    if tag.name in NOISE_TAGS:
        return True
    if tag.name in ("body", "html", "[document]"):
        return False  # never let root-level classes poison every descendant
    identifiers = " ".join(tag.get("class", [])) + " " + (tag.get("id") or "")
    identifiers = identifiers.lower()
    return any(kw in identifiers for kw in NOISE_KEYWORDS)


def has_noise_ancestor(tag: Tag) -> bool:
    return any(is_noise(anc) for anc in tag.parents if isinstance(anc, Tag))


def score_element(tag: Tag) -> int:
    total = 0
    for p in tag.find_all(["p", "li"]):
        if has_noise_ancestor(p):
            continue
        total += len(p.get_text(strip=True))
    return total


def clean_text(text: str) -> str:
    text = GHOST_SHORTCODE_RE.sub("", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def extract_main_content(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")

    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()

    candidates = [
        c for c in soup.find_all(["article", "main", "div", "section"])
        if not is_noise(c)
    ]

    if not candidates:
        paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
        return "\n\n".join(clean_text(p) for p in paragraphs if p.strip())

    best = max(candidates, key=score_element)

    paragraphs = []
    for p in best.find_all(["p", "li"]):
        if has_noise_ancestor(p):
            continue
        text = clean_text(p.get_text(strip=True))
        if text:
            paragraphs.append(text)

    return "\n\n".join(paragraphs)


def slugify(url: str) -> str:
    tail = url.rstrip("/").split("/")[-1]
    tail = re.sub(r"[^a-zA-Z0-9_-]", "_", tail) or "page"
    return tail[:60] + ".txt"


def fetch_url(url: str) -> None:
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=15)
    resp.raise_for_status()

    print(f"  status={resp.status_code} html_len={len(resp.text)} for {url}")

    text = extract_main_content(resp.text)
    if len(text) < 300:
        debug_path = os.path.join(OUTPUT_DIR, slugify(url).replace(".txt", "_RAW.html"))
        with open(debug_path, "w", encoding="utf-8") as f:
            f.write(resp.text)
        print(
            f"  WARNING: only extracted {len(text)} chars from {url}. "
            f"Saved raw HTML to {debug_path} for inspection."
        )
        return

    path = os.path.join(OUTPUT_DIR, slugify(url))
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Source: {url}\n\n{text}")
    print(f"  saved {path} ({len(text)} chars)")


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Fetching {len(URLS)} article(s) into {OUTPUT_DIR}/")
    for i, url in enumerate(URLS):
        try:
            fetch_url(url)
        except requests.RequestException as e:
            print(f"  FAILED {url}: {e}")
        if i < len(URLS) - 1:
            time.sleep(REQUEST_DELAY_SECONDS)
    print("Done. Review each file — trim any leftover noise by hand.")


if __name__ == "__main__":
    main()