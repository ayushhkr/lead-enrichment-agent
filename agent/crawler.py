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

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

from agent.config import settings

logger = logging.getLogger(__name__)


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

        for path in settings.subpages:
            url = f"{base_url}{path}"
            page = context.new_page()
            try:
                response = page.goto(
                    url,
                    wait_until=settings.nav_wait_until,
                    timeout=settings.page_timeout_ms,
                )
                if response is None:
                    results.append(PageResult(url=url, status="error", error="No response"))
                    continue

                if response.status == 404:
                    results.append(PageResult(url=url, status="not_found"))
                    continue

                if response.status >= 400:
                    results.append(
                        PageResult(url=url, status="error", error=f"HTTP {response.status}")
                    )
                    continue

                # Give lazy-loaded / JS content a brief moment to settle.
                page.wait_for_timeout(500)
                html = page.content()
                results.append(PageResult(url=url, status="ok", html=html))

            except PlaywrightTimeout:
                results.append(PageResult(url=url, status="timeout", error="Navigation timed out"))
            except Exception as exc:  # noqa: BLE001 - deliberately broad: crawler must never crash the run
                logger.warning("Failed to fetch %s: %s", url, exc)
                results.append(PageResult(url=url, status="error", error=str(exc)))
            finally:
                page.close()

        context.close()
        browser.close()

    return results
