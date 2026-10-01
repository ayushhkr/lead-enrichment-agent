"""
Autonomous Lead Enrichment Agent — entry point.

Usage:
    python main.py postman.com supabase.com vapi.ai
    python main.py --domains-file domains.txt
    python main.py                     # falls back to the 3 sample domains

Pipeline per domain (Step 1-4 of the assignment):
    crawl_domain()          -> raw HTML per page, isolated failures
    build_context_for_llm() -> cleaned, token-bounded text
    extract_company_intel() -> validated structured JSON via LLM tool calling

Every domain is wrapped in its own try/except so one bad site never stops
the batch. Results are written to output/output.json and output/output.csv,
plus output/cost_report.json when cost tracking is enabled.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List

from agent.config import settings
from agent.crawler import crawl_domain
from agent.cleaner import build_context_for_llm
from agent.extractor import extract_company_intel, CostTracker
from agent.schema import CompanyIntel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("lead_agent")

DEFAULT_DOMAINS = ["postman.com", "supabase.com", "vapi.ai"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autonomous Lead Enrichment Agent")
    parser.add_argument("domains", nargs="*", help="Company domains, e.g. postman.com")
    parser.add_argument("--domains-file", type=str, default=None,
                        help="Path to a text file with one domain per line")
    return parser.parse_args()


def resolve_domains(args: argparse.Namespace) -> List[str]:
    if args.domains_file:
        path = Path(args.domains_file)
        return [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if args.domains:
        return args.domains
    logger.info("No domains provided — using default sample set: %s", DEFAULT_DOMAINS)
    return DEFAULT_DOMAINS


def process_domain(domain: str, cost_tracker: CostTracker) -> CompanyIntel:
    """Run the full crawl -> clean -> extract chain for one domain, with retries."""
    last_error = None

    for attempt in range(1, settings.max_retries + 2):  # e.g. max_retries=2 -> 3 attempts
        try:
            logger.info("[%s] Crawling (attempt %d)...", domain, attempt)
            page_results = crawl_domain(domain)

            ok_pages = [p for p in page_results if p.status == "ok"]
            logger.info("[%s] Fetched %d/%d pages successfully", domain, len(ok_pages), len(page_results))

            if not ok_pages:
                raise RuntimeError("No pages could be fetched (site may be blocking bots or down)")

            context = build_context_for_llm(page_results)

            logger.info("[%s] Extracting structured intel via %s...", domain, settings.llm_provider)
            intel = extract_company_intel(domain, context, cost_tracker)
            intel.sources_used = [p.url for p in ok_pages]
            return intel

        except Exception as exc:  # noqa: BLE001 - top-level guard per assignment's Step 4
            last_error = exc
            logger.warning("[%s] Attempt %d failed: %s", domain, attempt, exc)
            if attempt <= settings.max_retries:
                time.sleep(settings.retry_backoff_seconds * attempt)

    logger.error("[%s] All attempts failed. Recording empty/low-confidence result. Last error: %s",
                domain, last_error)
    return CompanyIntel(
        domain=domain,
        company_overview="",
        target_audience=[],
        contact_points=[],
        key_team_members=[],
        data_confidence_score=0.0,
        sources_used=[],
    )


def write_outputs(results: List[CompanyIntel], cost_tracker: CostTracker) -> None:
    out_dir = Path(settings.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON output
    json_path = out_dir / "output.json"
    json_path.write_text(
        json.dumps([r.model_dump() for r in results], indent=2), encoding="utf-8"
    )
    logger.info("Wrote %s", json_path)

    # CSV output 
    csv_path = out_dir / "output.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "domain", "company_overview", "target_audience", "contact_points",
            "key_team_members", "data_confidence_score", "sources_used",
        ])
        for r in results:
            writer.writerow([
                r.domain,
                r.company_overview,
                "; ".join(r.target_audience),
                "; ".join(r.contact_points),
                "; ".join(f"{m.name} ({m.title or 'n/a'}) {m.linkedin_url or ''}".strip()
                        for m in r.key_team_members),
                r.data_confidence_score,
                "; ".join(r.sources_used),
            ])
    logger.info("Wrote %s", csv_path)

    # Bonus: cost report
    if cost_tracker.records:
        cost_path = out_dir / "cost_report.json"
        cost_path.write_text(
            json.dumps(
                {"total_estimated_cost_usd": cost_tracker.total_cost(), "records": cost_tracker.as_list()},
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info("Wrote %s (total est. cost: $%.6f)", cost_path, cost_tracker.total_cost())


def main() -> None:
    args = parse_args()
    domains = resolve_domains(args)

    provider_key_map = {
        "anthropic": settings.anthropic_api_key,
        "openai": settings.openai_api_key,
        "groq": settings.groq_api_key,
    }
    if not provider_key_map.get(settings.llm_provider):
        logger.error(
            "No API key set for LLM_PROVIDER=%s. Set the relevant key in your .env file.",
            settings.llm_provider,
        )
        sys.exit(1)

    logger.info("Starting run for %d domain(s) using provider=%s", len(domains), settings.llm_provider)

    cost_tracker = CostTracker()
    results: List[CompanyIntel] = []

    for domain in domains:
        intel = process_domain(domain, cost_tracker)
        results.append(intel)
        logger.info("[%s] Done. Confidence=%.2f", domain, intel.data_confidence_score)

    write_outputs(results, cost_tracker)
    logger.info("Run complete. %d/%d domains yielded non-empty data.",
                sum(1 for r in results if r.data_confidence_score > 0), len(results))


if __name__ == "__main__":
    main()
