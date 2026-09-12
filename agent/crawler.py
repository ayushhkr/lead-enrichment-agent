"""
Step 1: Automated browsing & content retrieval.

Uses Playwright (headless Chromium) to fetch a domain's homepage plus a
small set of likely-relevant subpages, rendering JS where needed. Every
page fetch is isolated so that one broken URL never aborts the whole run.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional
from urllib.parse import urljoin, urlparse, urlunparse

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

from agent.config import settings

logger = logging.getLogger(__name__)

HIGH_PRIORITY = ("about", "about-us", "company", "team", "leadership", "founder", "founders", "people", "contact", "contact-us")
MEDIUM_PRIORITY = ("pricing", "customers", "solutions", "products", "enterprise", "partners")
LOW_PRIORITY = ("careers", "blog", "docs", "resources")


@dataclass
class PageResult:
    url: str
    status: str  # "ok" | "timeout" | "error" | "not_found"
    html: Optional[str] = None
    error: Optional[str] = None


def _normalize_domain(domain: str) -> str:
    domain = domain.strip().rstrip("/")
    if not domain.startswith("http"):
        domain = f"https://{domain}"
    return domain


def _normalized_url(url: str) -> str:
    """Remove fragments while retaining meaningful paths and query strings."""
    parts = urlparse(url)
    return urlunparse((parts.scheme, parts.netloc.lower(), parts.path or "/", "", parts.query, ""))


def _is_internal(url: str, base_url: str) -> bool:
    host = urlparse(url).hostname or ""
    base_host = urlparse(base_url).hostname or ""
    return host == base_host or host.removeprefix("www.") == base_host.removeprefix("www.")


def _rank_links(links: List[dict], base_url: str) -> List[str]:
    """Return a small, deterministic set of useful same-site links."""
    ranked = []
    seen = {_normalized_url(base_url)}
    for link in links:
        href = link.get("href", "")
        if not href or href.startswith(("mailto:", "tel:", "javascript:")):
            continue
        url = _normalized_url(urljoin(base_url, href))
        if not _is_internal(url, base_url) or url in seen:
            continue
        haystack = f"{urlparse(url).path} {link.get('text', '')}".lower()
        score = 0
        if any(keyword in haystack for keyword in HIGH_PRIORITY):
            score = 3
        elif any(keyword in haystack for keyword in MEDIUM_PRIORITY):
            score = 2
        elif any(keyword in haystack for keyword in LOW_PRIORITY):
            score = 1
        if score:
            seen.add(url)
            ranked.append((score, url))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [url for _, url in ranked]


def crawl_domain(domain: str) -> List[PageResult]:
    """
    Fetch the homepage + candidate subpages for a single domain.
    Never raises — every failure is captured as a PageResult with an
    error status so the caller can proceed to the next domain/page.
    """
    base_url = _normalize_domain(domain)
    results: List[PageResult] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=settings.request_headers["User-Agent"],
            java_script_enabled=True,
        )

        def fetch(url: str) -> Optional[List[dict]]:
            page = context.new_page()
            try:
                response = page.goto(
                    url,
                    wait_until=settings.nav_wait_until,
                    timeout=settings.page_timeout_ms,
                )
                if response is None:
                    results.append(PageResult(url=url, status="error", error="No response"))
                    return None

                if response.status == 404:
                    results.append(PageResult(url=url, status="not_found"))
                    return None

                if response.status >= 400:
                    results.append(
                        PageResult(url=url, status="error", error=f"HTTP {response.status}")
                    )
                    return None

                # Give lazy-loaded / JS content a brief moment to settle.
                page.wait_for_timeout(500)
                html = page.content()
                results.append(PageResult(url=url, status="ok", html=html))
                return page.eval_on_selector_all(
                    "a[href]", "anchors => anchors.map(a => ({href: a.href, text: (a.innerText || a.textContent || '').trim()}))"
                )

            except PlaywrightTimeout:
                results.append(PageResult(url=url, status="timeout", error="Navigation timed out"))
            except Exception as exc:  # noqa: BLE001 - deliberately broad: crawler must never crash the run
                logger.warning("Failed to fetch %s: %s", url, exc)
                results.append(PageResult(url=url, status="error", error=str(exc)))
            finally:
                page.close()
            return None

        homepage_links = fetch(base_url) or []
        discovered = _rank_links(homepage_links, base_url)
        selected = discovered[: max(0, settings.max_relevant_pages - 1)]

        # Only supplement sparse discovery; never crawl the whole hardcoded list.
        if len(selected) < settings.min_discovered_pages:
            for path in settings.subpages:
                fallback = _normalized_url(urljoin(base_url, path))
                if fallback != _normalized_url(base_url) and fallback not in selected:
                    selected.append(fallback)
                if len(selected) >= settings.max_relevant_pages - 1:
                    break

        logger.info("Discovered %d relevant homepage links; crawling %d subpages", len(discovered), len(selected))
        for url in selected:
            fetch(url)

        context.close()
        browser.close()

    return results
