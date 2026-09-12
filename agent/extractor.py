"""
Step 3: LLM extraction with strict structured outputs (tool calling).

Supports three interchangeable providers so graders/reviewers can run this
with whatever key they have on hand: Anthropic (Claude), OpenAI, or Groq
(OpenAI-compatible). All three are forced into the same Pydantic schema.
"""
from __future__ import annotations

import json
import logging
from typing import Optional, Tuple

from agent.config import settings
from agent.schema import CompanyIntel, COMPANY_INTEL_JSON_SCHEMA

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a B2B research analyst. You will be given cleaned text scraped "
    "from a company's public website (homepage + subpages). Extract only "
    "what is explicitly present or strongly implied in the text — never "
    "invent names, emails, or facts. If a field cannot be found, leave it "
    "empty (empty string / empty list) rather than guessing. "
    "Call the `record_company_intel` tool exactly once with your findings."
)

TOOL_NAME = "record_company_intel"


def _build_user_prompt(domain: str, context: str) -> str:
    return (
        f"Company domain: {domain}\n\n"
        f"Scraped website content:\n{context if context else '[No content could be retrieved for this domain.]'}"
    )


class CostTracker:
    """Accumulates token usage / estimated spend across all domains."""

    def __init__(self):
        self.records = []

    def add(self, domain: str, provider: str, model: str, input_tokens: int, output_tokens: int, cost_usd: float):
        self.records.append({
            "domain": domain,
            "provider": provider,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "estimated_cost_usd": round(cost_usd, 6),
        })

    def total_cost(self) -> float:
        return round(sum(r["estimated_cost_usd"] for r in self.records), 6)

    def as_list(self):
        return self.records


# Rough per-1K-token pricing for cost estimation (USD). Update as needed.
_PRICING = {
    "anthropic": {"input": 0.003, "output": 0.015},   # Claude Sonnet-class
    "openai": {"input": 0.00015, "output": 0.0006},   # gpt-4o-mini-class
    "groq": {"input": 0.0, "output": 0.0},            # varies by model; many are free/cheap
}


def _estimate_cost(provider: str, input_tokens: int, output_tokens: int) -> float:
    rates = _PRICING.get(provider, {"input": 0.0, "output": 0.0})
    return (input_tokens / 1000) * rates["input"] + (output_tokens / 1000) * rates["output"]


def _extract_with_anthropic(domain: str, context: str) -> Tuple[Optional[dict], int, int]:
    import anthropic

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    tool = {
        "name": TOOL_NAME,
        "description": "Record structured company intelligence extracted from website content.",
        "input_schema": COMPANY_INTEL_JSON_SCHEMA,
    }

    response = client.messages.create(
        model=settings.anthropic_model,
        max_tokens=1500,
        system=SYSTEM_PROMPT,
        tools=[tool],
        tool_choice={"type": "tool", "name": TOOL_NAME},
        messages=[{"role": "user", "content": _build_user_prompt(domain, context)}],
    )

    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens

    for block in response.content:
        if block.type == "tool_use" and block.name == TOOL_NAME:
            return block.input, input_tokens, output_tokens

    return None, input_tokens, output_tokens


def _extract_with_openai_compatible(domain: str, context: str, provider: str) -> Tuple[Optional[dict], int, int]:
    """Works for both OpenAI and Groq (Groq exposes an OpenAI-compatible API)."""
    from openai import OpenAI

    if provider == "openai":
        client = OpenAI(api_key=settings.openai_api_key)
        model = settings.openai_model
    else:
        client = OpenAI(api_key=settings.groq_api_key, base_url="https://api.groq.com/openai/v1")
        model = settings.groq_model

    tool = {
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "description": "Record structured company intelligence extracted from website content.",
            "parameters": COMPANY_INTEL_JSON_SCHEMA,
        },
    }

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(domain, context)},
        ],
        tools=[tool],
        tool_choice={"type": "function", "function": {"name": TOOL_NAME}},
    )

    usage = response.usage
    input_tokens = usage.prompt_tokens if usage else 0
    output_tokens = usage.completion_tokens if usage else 0

    message = response.choices[0].message
    if message.tool_calls:
        args_str = message.tool_calls[0].function.arguments
        try:
            return json.loads(args_str), input_tokens, output_tokens
        except json.JSONDecodeError:
            logger.error("Failed to parse tool call arguments for %s", domain)
            return None, input_tokens, output_tokens

    return None, input_tokens, output_tokens


def extract_company_intel(
    domain: str, context: str, cost_tracker: Optional[CostTracker] = None
) -> CompanyIntel:
    """
    Run structured extraction for one domain. Never raises on LLM/parsing
    failure — returns a low-confidence CompanyIntel stub instead so the
    pipeline can continue to the next domain.
    """
    provider = settings.llm_provider

    try:
        if provider == "anthropic":
            raw, in_tok, out_tok = _extract_with_anthropic(domain, context)
        elif provider in ("openai", "groq"):
            raw, in_tok, out_tok = _extract_with_openai_compatible(domain, context, provider)
        else:
            raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")

        if cost_tracker is not None:
            model_used = getattr(settings, f"{provider}_model")
            cost = _estimate_cost(provider, in_tok, out_tok)
            cost_tracker.add(domain, provider, model_used, in_tok, out_tok, cost)

        if raw is None:
            raise ValueError("Model did not return a tool call")

        raw.setdefault("domain", domain)
        return CompanyIntel.model_validate(raw)

    except Exception as exc:  # noqa: BLE001 - extraction must never crash the run
        logger.error("Extraction failed for %s: %s", domain, exc)
        return CompanyIntel(
            domain=domain,
            company_overview="",
            target_audience="",
            contact_points=[],
            key_team_members=[],
            data_confidence_score=0.0,
            sources_used=[],
        )
