"""
Central configuration for the lead enrichment agent.
Reads all tunables from environment variables so nothing is hardcoded.
"""
import os
from dataclasses import dataclass, field
from typing import List

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    # --- LLM provider ---
    # "anthropic" (Claude), "openai", or "groq" (any OpenAI-compatible endpoint)
    llm_provider: str = os.getenv("LLM_PROVIDER", "anthropic")
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")

    anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    # --- Crawling ---
    subpages: List[str] = field(default_factory=lambda: [
        "", "/about", "/about-us", "/company", "/team", "/leadership",
        "/contact", "/contact-us", "/pricing",
    ])
    page_timeout_ms: int = int(os.getenv("PAGE_TIMEOUT_MS", "15000"))
    nav_wait_until: str = os.getenv("NAV_WAIT_UNTIL", "domcontentloaded")
    max_chars_per_page: int = int(os.getenv("MAX_CHARS_PER_PAGE", "6000"))
    max_chars_total: int = int(os.getenv("MAX_CHARS_TOTAL", "18000"))
    request_headers: dict = field(default_factory=lambda: {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        )
    })

    # --- Resilience ---
    max_retries: int = int(os.getenv("MAX_RETRIES", "2"))
    retry_backoff_seconds: float = float(os.getenv("RETRY_BACKOFF_SECONDS", "1.5"))

    # --- Output ---
    output_dir: str = os.getenv("OUTPUT_DIR", "output")


settings = Settings()
