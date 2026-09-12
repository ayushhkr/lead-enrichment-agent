"""
Step 2: Context pre-processing & token optimization.

Strips scripts, styles, SVGs, and nav/footer boilerplate from raw HTML,
then reduces it to clean text so we never feed a raw HTML tree to the LLM.
"""
from __future__ import annotations

import re
from typing import List

from bs4 import BeautifulSoup

from agent.config import settings
from agent.crawler import PageResult

# Tags that carry no useful extraction signal for this use case.
STRIP_TAGS = ["script", "style", "svg", "noscript", "iframe", "link", "meta", "form", "button"]
# Common boilerplate containers worth dropping when present.
STRIP_SELECTORS = ["nav", "[role='navigation']", ".cookie", "[class*='cookie-banner']", "[role='dialog']"]


def html_to_clean_text(html: str, max_chars: int = None) -> str:
    """Convert raw HTML into compact, LLM-friendly plain text."""
    soup = BeautifulSoup(html, "html.parser")

    for tag_name in STRIP_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    for selector in STRIP_SELECTORS:
        for tag in soup.select(selector):
            tag.decompose()

    # Keep link destinations that can carry contact or professional-profile data.
    for anchor in soup.find_all("a", href=True):
        href = anchor["href"].strip()
        label = anchor.get_text(" ", strip=True)
        if href.startswith("mailto:") or "linkedin.com" in href.lower():
            anchor.replace_with(f"{label} {href}".strip())

    text = soup.get_text(separator="\n")

    # Collapse excess whitespace/blank lines left behind by stripped tags.
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    limit = max_chars or settings.max_chars_per_page
    return cleaned[:limit]


def build_context_for_llm(page_results: List[PageResult]) -> str:
    """
    Combine cleaned text from all successfully-fetched pages of a domain
    into one bounded context block, labeled by source URL so the LLM
    can still reason about where each fact came from.
    """
    blocks: List[str] = []
    running_total = 0

    for result in page_results:
        if result.status != "ok" or not result.html:
            continue

        cleaned = html_to_clean_text(result.html)
        if not cleaned:
            continue

        block = f"### SOURCE: {result.url}\n{cleaned}"
        if running_total + len(block) > settings.max_chars_total:
            remaining = settings.max_chars_total - running_total
            if remaining <= 200:
                break
            block = block[:remaining]

        blocks.append(block)
        running_total += len(block)

        if running_total >= settings.max_chars_total:
            break

    return "\n\n".join(blocks)
