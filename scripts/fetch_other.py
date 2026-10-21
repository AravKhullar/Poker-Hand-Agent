"""Fetch a curated poker research corpus into data/corpus/.

This is a targeted fetcher, NOT a crawler.

Sources are explicitly defined below and grouped by topic. The fetcher
supports both HTML articles and PDF research papers.

HTML extraction is intentionally tolerant because GTO Wizard uses
Ghost-CMS-style markup and embeds literal shortcode markers such as:

    {{width: 75%}}
    {{color: green}}
    {{/color}}
    {{grid: 2}}

The extractor:
  1. Removes scripts/styles/media and obvious structural chrome.
  2. Tries JSON-LD articleBody when available.
  3. Prefers semantic article/main/articleBody containers.
  4. Falls back to scoring content containers by real text density.
  5. Preserves headings and paragraph/list structure.
  6. Removes Ghost shortcode markers.
  7. Never silently discards a page because extraction is short.
     Problematic pages are saved and reported.

PDFs are extracted with pypdf.

Usage:

    uv run --group scripts python scripts/fetch_other.py

Output:

    data/corpus/
        01_fundamentals/
        02_preflop/
        03_postflop/
        04_exploitative/
        05_solver_theory/
        06_llm_poker/
        07_evaluation/
        08_game_theory/
"""

from __future__ import annotations

import json
import re
import time
import traceback
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup, Tag
from pypdf import PdfReader

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path("data/corpus")

REQUEST_DELAY_SECONDS = 2.0
REQUEST_TIMEOUT_SECONDS = 30
MAX_RETRIES = 3

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/150.0 Safari/537.36 "
    "(personal research project)"
)


# ---------------------------------------------------------------------------
# Source definition
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Source:
    category: str
    url: str


SOURCES = [
    # -----------------------------------------------------------------------
    # 01 - Fundamentals
    # -----------------------------------------------------------------------

    Source(
        "01_fundamentals",
        "https://blog.gtowizard.com/what-is-gto-in-poker/",
    ),
    Source(
        "01_fundamentals",
        "https://blog.gtowizard.com/what-does-gto-aim-to-achieve/",
    ),
    Source(
        "01_fundamentals",
        "https://blog.gtowizard.com/what-is-equity-in-poker/",
    ),
    Source(
        "01_fundamentals",
        "https://blog.gtowizard.com/what-is-expected-value-in-poker/",
    ),
    Source(
        "01_fundamentals",
        "https://blog.gtowizard.com/equity-realization/",
    ),
    Source(
        "01_fundamentals",
        "https://blog.gtowizard.com/the-10-most-important-concepts-for-new-poker-players/",
    ),

    # -----------------------------------------------------------------------
    # 02 - Preflop
    # -----------------------------------------------------------------------

    Source(
        "02_preflop",
        "https://blog.gtowizard.com/preflop-range-morphology/",
    ),
    Source(
        "02_preflop",
        "https://blog.gtowizard.com/range-morphology/",
    ),

    # -----------------------------------------------------------------------
    # 03 - Postflop
    # -----------------------------------------------------------------------

    Source(
        "03_postflop",
        "https://blog.gtowizard.com/the-mechanics-of-c-bet-sizing/",
    ),
    Source(
        "03_postflop",
        "https://blog.gtowizard.com/principles-of-turn-strategy/",
    ),
    Source(
        "03_postflop",
        "https://blog.gtowizard.com/how-to-analyze-turn-textures-in-poker/",
    ),

    # -----------------------------------------------------------------------
    # 04 - Exploitative
    # -----------------------------------------------------------------------

    Source(
        "04_exploitative",
        "https://blog.gtowizard.com/the-five-imbalances-of-exploitative-poker/",
    ),
    Source(
        "04_exploitative",
        "https://blog.gtowizard.com/introducing-nodelocking/",
    ),
    Source(
        "04_exploitative",
        "https://blog.gtowizard.com/the-limits-of-nodelocking/",
    ),

    # -----------------------------------------------------------------------
    # 05 - Solver Theory
    # -----------------------------------------------------------------------

    Source(
        "05_solver_theory",
        "https://blog.gtowizard.com/how-solvers-work/",
    ),
    Source(
        "05_solver_theory",
        "https://blog.gtowizard.com/introducing-gto-wizard-ai/",
    ),

    # -----------------------------------------------------------------------
    # 06 - LLM Poker
    # -----------------------------------------------------------------------

    Source(
        "06_llm_poker",
        "https://arxiv.org/pdf/2401.06781",
    ),
    Source(
        "06_llm_poker",
        "https://arxiv.org/pdf/2605.30094",
    ),
    Source(
        "06_llm_poker",
        "https://arxiv.org/pdf/2602.00528",
    ),

    # -----------------------------------------------------------------------
    # 07 - Evaluation
    # -----------------------------------------------------------------------

    Source(
        "07_evaluation",
        "https://arxiv.org/pdf/2603.23660",
    ),
    Source(
        "07_evaluation",
        "https://arxiv.org/pdf/1701.01724",
    ),
    Source(
        "07_evaluation",
        "https://arxiv.org/pdf/2201.11580",
    ),

    # -----------------------------------------------------------------------
    # 08 - Game Theory
    # -----------------------------------------------------------------------

    Source(
        "08_game_theory",
        "https://martin.zinkevich.org/publications/regretpoker.pdf",
    ),
    Source(
        "08_game_theory",
        "https://proceedings.neurips.cc/paper_files/paper/2009/file/00411460f7c92d2124a67ea0f4cb5f85-Paper.pdf",
    ),
    Source(
        "08_game_theory",
        "https://arxiv.org/pdf/1407.5042",
    ),
]


# ---------------------------------------------------------------------------
# HTML noise handling
# ---------------------------------------------------------------------------

# Only tags that are structurally almost always chrome.
STRUCTURAL_NOISE_TAGS = {
    "nav",
    "footer",
    "header",
    "aside",
    "form",
    "script",
    "style",
    "noscript",
    "svg",
    "template",
}


# Keep this list specific.
# DO NOT use generic terms like "tag", "content", "page", etc.
NOISE_KEYWORDS = {
    "sidebar",
    "side-bar",
    "social-share",
    "share-buttons",
    "share-button",
    "related-posts",
    "related-articles",
    "recommended",
    "recommendations",
    "comments",
    "comment-section",
    "newsletter",
    "subscribe",
    "advertisement",
    "advert",
    "cookie",
    "breadcrumb",
    "site-header",
    "site-footer",
    "mobile-menu",
    "menu-drawer",
    "popup",
    "modal",
    "promo",
    "promotion",
}


# Ghost / GTO Wizard shortcode syntax.
#
# Examples:
#   {{width: 75%}}
#   {{/width}}
#   {{color: green}}
#   {{center}}
#   {{/center}}
#   {{grid: 2fr 1fr}}
#
GHOST_SHORTCODE_RE = re.compile(r"\{\{.*?\}\}", flags=re.DOTALL)


# Collapse repeated whitespace without destroying paragraph structure.
WHITESPACE_RE = re.compile(r"[ \t]+")


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Clean extracted text while preserving useful content."""

    text = GHOST_SHORTCODE_RE.sub("", text)

    # Remove common invisible Unicode junk.
    text = text.replace("\xa0", " ")
    text = text.replace("\u200b", "")
    text = text.replace("\ufeff", "")

    text = WHITESPACE_RE.sub(" ", text)

    return text.strip()


def normalize_heading(text: str) -> str:
    text = clean_text(text)
    return text


def slugify_url(url: str) -> str:
    """Create a stable filename from a URL."""

    parsed = urlparse(url)

    path = parsed.path.rstrip("/")
    tail = path.split("/")[-1]

    if not tail:
        tail = parsed.netloc

    tail = re.sub(r"\.pdf$", "", tail, flags=re.IGNORECASE)
    tail = re.sub(r"[^a-zA-Z0-9_-]+", "_", tail)
    tail = re.sub(r"_+", "_", tail).strip("_")

    return (tail[:100] or "document") + ".txt"


def is_structural_noise(tag: Tag) -> bool:
    return tag.name.lower() in STRUCTURAL_NOISE_TAGS


def has_noise_class(tag: Tag | None) -> bool:
    """Check classes and IDs on the current element only."""

    if tag is None:
        return False

    attrs = getattr(tag, "attrs", None)

    if not isinstance(attrs, dict):
        return False

    classes = attrs.get("class") or []

    if isinstance(classes, str):
        classes = [classes]

    identifiers = [
        str(value)
        for value in classes
        if value
    ]

    tag_id = attrs.get("id")

    if tag_id:
        identifiers.append(str(tag_id))

    identifier_string = " ".join(identifiers).lower()

    return any(
        keyword in identifier_string
        for keyword in NOISE_KEYWORDS
    )


def is_noise(tag: Tag | None) -> bool:
    """Return True when a tag is clearly page chrome."""

    if tag is None:
        return False

    name = getattr(tag, "name", None)

    if name is None:
        return False

    if name in {"html", "body", "[document]"}:
        return False

    if name in STRUCTURAL_NOISE_TAGS:
        return True

    return has_noise_class(tag)

def remove_noise(soup: BeautifulSoup) -> None:
    """Remove obvious page chrome safely.

    Important:
    Remove only top-level noise containers. Do not iterate over every
    descendant and decompose parents while descendant references are
    still being processed.
    """

    noise_tags = []

    for tag in soup.find_all(True):
        if is_noise(tag):
            noise_tags.append(tag)

    # Remove outermost noise containers first.
    #
    # If a <nav> contains a <div class="sidebar">, removing the nav
    # already removes the sidebar. We don't want to decompose both.
    outermost = []

    for tag in noise_tags:
        has_noise_parent = any(
            parent in noise_tags
            for parent in tag.parents
            if isinstance(parent, Tag)
        )

        if not has_noise_parent:
            outermost.append(tag)

    for tag in outermost:
        # Use extract() rather than decompose().
        #
        # extract() removes the node from the tree without destroying
        # the Tag object's internal attributes.
        tag.extract()


# ---------------------------------------------------------------------------
# JSON-LD extraction
# ---------------------------------------------------------------------------

def extract_json_ld_article_body(soup: BeautifulSoup) -> str | None:
    """
    Some CMS pages expose a clean articleBody through JSON-LD.

    If available, this is usually much cleaner than scraping rendered HTML.
    """

    scripts = soup.find_all(
        "script",
        attrs={"type": re.compile(r"application/ld\+json", re.I)},
    )

    for script in scripts:
        raw = script.string or script.get_text()

        if not raw.strip():
            continue

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue

        objects = data if isinstance(data, list) else [data]

        for obj in objects:
            if not isinstance(obj, dict):
                continue

            article_body = obj.get("articleBody")

            if isinstance(article_body, str) and len(article_body.strip()) > 500:
                return clean_text(article_body)

    return None


# ---------------------------------------------------------------------------
# HTML extraction
# ---------------------------------------------------------------------------

def extract_blocks(container: Tag | None) -> list[str]:
    """Extract useful article blocks while preserving headings."""

    if container is None:
        return []

    blocks: list[str] = []

    for element in container.find_all(
        ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "blockquote"]
    ):
        if element is None:
            continue

        text = clean_text(
            element.get_text(" ", strip=True)
        )

        if not text:
            continue

        if len(text) < 2:
            continue

        tag_name = element.name.lower()

        if tag_name.startswith("h"):
            blocks.append(
                f"\n## {normalize_heading(text)}\n"
            )

        elif tag_name == "li":
            blocks.append(f"- {text}")

        elif tag_name == "blockquote":
            blocks.append(f"> {text}")

        else:
            blocks.append(text)

    return blocks


def score_container(container: Tag | None) -> float:
    """
    Score a possible article container.

    We reward:
      - lots of paragraph text
      - multiple paragraphs
      - headings
      - article/main semantics

    We penalize:
      - huge link-heavy containers
      - obvious UI density
    """

    if container is None:
        return 0.0

    paragraphs = container.find_all("p")
    headings = container.find_all(
        ["h1", "h2", "h3", "h4"]
    )
    links = container.find_all("a")

    paragraph_text = sum(
        len(clean_text(p.get_text(" ", strip=True)))
        for p in paragraphs
        if p is not None
    )

    heading_text = sum(
        len(clean_text(h.get_text(" ", strip=True)))
        for h in headings
        if h is not None
    )

    link_text = sum(
        len(clean_text(a.get_text(" ", strip=True)))
        for a in links
        if a is not None
    )

    if paragraph_text < 200:
        return 0.0

    link_ratio = link_text / max(paragraph_text, 1)

    score = (
        paragraph_text
        + heading_text * 0.5
        + len(paragraphs) * 100
        + len(headings) * 50
    )

    if link_ratio > 1.5:
        score *= 0.25
    elif link_ratio > 0.8:
        score *= 0.6

    name = getattr(container, "name", None)

    if name == "article":
        score *= 1.5

    if name == "main":
        score *= 1.3

    itemprop = container.get("itemprop")

    if itemprop == "articleBody":
        score *= 1.5

    return score


def extract_html_content(html: str) -> tuple[str, str]:
    """
    Return (title, content).
    """

    soup = BeautifulSoup(html, "lxml")

    # Grab title before removing anything.
    title = ""

    title_tag = soup.find("title")
    if title_tag:
        title = clean_text(title_tag.get_text(" ", strip=True))

    h1 = soup.find("h1")
    if h1:
        h1_text = clean_text(h1.get_text(" ", strip=True))
        if h1_text:
            title = h1_text

    # First choice: JSON-LD article body.
    json_ld_body = extract_json_ld_article_body(soup)

    if json_ld_body:
        return title, json_ld_body

    # Remove things that can never be useful article text.
    for tag in soup.find_all(
        ["script", "style", "noscript", "svg", "template"]
    ):
        tag.decompose()

    # Remove structural chrome.
    remove_noise(soup)

    # Semantic candidates first.
    candidates: list[Tag] = []

    selectors = [
        '[itemprop="articleBody"]',
        "article",
        "main",
        '[role="main"]',
    ]

    for selector in selectors:
        for tag in soup.select(selector):
            if isinstance(tag, Tag):
                candidates.append(tag)

    # Then use broader containers as fallback.
    candidates.extend(
        soup.find_all(["div", "section"])
    )

    # Deduplicate candidates by object identity.
    unique_candidates = []
    seen_ids = set()

    for candidate in candidates:
        if id(candidate) not in seen_ids:
            seen_ids.add(id(candidate))
            unique_candidates.append(candidate)

    scored = [
        (score_container(candidate), candidate)
        for candidate in unique_candidates
    ]

    scored.sort(key=lambda x: x[0], reverse=True)

    # Prefer the best-scoring content container.
    if scored and scored[0][0] > 0:
        best = scored[0][1]
        blocks = extract_blocks(best)

        if blocks:
            return title, "\n\n".join(blocks)

    # Final fallback: body.
    body = soup.body

    if body:
        blocks = extract_blocks(body)

        if blocks:
            return title, "\n\n".join(blocks)

    return title, ""


# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------

def extract_pdf_content(data: bytes) -> tuple[str, str]:
    """Extract text from a PDF."""

    reader = PdfReader(BytesIO(data))

    pages: list[str] = []

    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            print(f"    WARNING: PDF page {page_number} failed: {exc}")
            continue

        text = clean_text(text)

        if text:
            pages.append(
                f"[Page {page_number}]\n{text}"
            )

    content = "\n\n".join(pages)

    # PDF metadata title if available.
    title = ""

    try:
        metadata = reader.metadata

        if metadata:
            title = clean_text(
                metadata.title or ""
            )
    except Exception:
        pass

    return title, content


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def fetch(url: str) -> requests.Response:
    """Fetch with retries for transient failures."""

    headers = {
        "User-Agent": USER_AGENT,
        "Accept": (
            "text/html,application/xhtml+xml,"
            "application/xml;q=0.9,"
            "application/pdf;q=0.8,*/*;q=0.7"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(
                url,
                headers=headers,
                timeout=REQUEST_TIMEOUT_SECONDS,
                allow_redirects=True,
            )

            if response.status_code in {429, 500, 502, 503, 504}:
                raise requests.HTTPError(
                    f"temporary HTTP {response.status_code}"
                )

            response.raise_for_status()
            return response

        except requests.RequestException as exc:
            last_error = exc

            if attempt == MAX_RETRIES:
                break

            sleep_time = REQUEST_DELAY_SECONDS * attempt

            print(
                f"    retry {attempt}/{MAX_RETRIES - 1} "
                f"in {sleep_time:.1f}s: {exc}"
            )

            time.sleep(sleep_time)

    raise last_error or RuntimeError("request failed")


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def write_document(
    source: Source,
    title: str,
    content: str,
    content_type: str,
) -> Path:
    category_dir = OUTPUT_DIR / source.category
    category_dir.mkdir(parents=True, exist_ok=True)

    filename = slugify_url(source.url)

    path = category_dir / filename

    metadata = (
        f"Source: {source.url}\n"
        f"Category: {source.category}\n"
        f"Title: {title}\n"
        f"Content-Type: {content_type}\n"
        f"\n"
        f"{content.strip()}\n"
    )

    path.write_text(
        metadata,
        encoding="utf-8",
    )

    return path


# ---------------------------------------------------------------------------
# Main fetch logic
# ---------------------------------------------------------------------------

def fetch_source(source: Source) -> None:
    print(f"\n[{source.category}]")
    print(f"  {source.url}")

    response = fetch(source.url)

    content_type = response.headers.get(
        "Content-Type",
        "",
    ).lower()

    is_pdf = (
        "application/pdf" in content_type
        or source.url.lower().endswith(".pdf")
        or response.content[:4] == b"%PDF"
    )

    print(
        f"  status={response.status_code} "
        f"content_type={content_type or 'unknown'} "
        f"bytes={len(response.content)}"
    )

    if is_pdf:
        title, content = extract_pdf_content(response.content)
        actual_type = "pdf"
    else:
        title, content = extract_html_content(
            response.text
        )
        actual_type = "html"

    # Use URL slug as title fallback.
    if not title:
        title = slugify_url(source.url).replace(".txt", "")

    content_length = len(content.strip())

    if content_length < 200:
        print(
            f"  WARNING: only extracted {content_length} characters"
        )
        print(
            "  The document will STILL be saved so you can inspect it."
        )

    path = write_document(
        source=source,
        title=title,
        content=content,
        content_type=actual_type,
    )

    print(
        f"  saved={path} "
        f"title={title!r} "
        f"chars={content_length}"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Fetching {len(SOURCES)} curated poker documents "
        f"into {OUTPUT_DIR}/"
    )

    failures = []

    for index, source in enumerate(SOURCES):
        try:
            fetch_source(source)

        except Exception as exc:
            print(
                f"  FAILED: {source.url}\n"
                f"  ERROR: {exc}"
            )

            traceback.print_exc()

            failures.append(source.url)

        if index < len(SOURCES) - 1:
            time.sleep(REQUEST_DELAY_SECONDS)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)

    print(f"Documents attempted: {len(SOURCES)}")
    print(f"Failures: {len(failures)}")

    if failures:
        print("\nFailed URLs:")
        for url in failures:
            print(f"  - {url}")

    print(
        f"\nCorpus location: {OUTPUT_DIR.resolve()}"
    )


if __name__ == "__main__":
    main()