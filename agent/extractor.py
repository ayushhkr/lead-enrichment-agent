"""
Step 3: LLM extraction with strict structured outputs (tool calling).

Supports three interchangeable providers so graders/reviewers can run this
with whatever key they have on hand: Anthropic (Claude), OpenAI, or Groq
(OpenAI-compatible). All three are forced into the same Pydantic schema.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Optional, Tuple

from agent.config import settings
from agent.schema import CompanyIntel, COMPANY_INTEL_JSON_SCHEMA

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a B2B research analyst. You will be given cleaned text scraped "
    "from a company's public website (homepage + subpages). Extract only "
    "what is explicitly present or strongly implied in the text — never "
    "invent names, emails, or facts. target_audience must be a concise list of "
    "three to six genuine ICP/customer groups: buyer/user roles, teams, or "
    "organization types. Combine closely related categories. Do not copy every "
    "marketing keyword, use case, beginner segment, or standalone industry "
    "mentioned on the site. Only include an industry when it is explicitly "
    "presented as a customer group. "
    "If a field cannot be found, leave it "
    "empty (empty string / empty list) rather than guessing. "
    "Call the `record_company_intel` tool exactly once with your findings."
)

TOOL_NAME = "record_company_intel"
JSON_FALLBACK_PROMPT = SYSTEM_PROMPT.replace(
    "Call the `record_company_intel` tool exactly once with your findings.",
    "Return only one valid JSON object that conforms to the supplied JSON schema.",
) + (
    " For every extracted team member, preserve their source-supported role/title "
    "in `title`; use null only when no title is present in the scraped context. "
    "Preserve LinkedIn URLs when present, and do not invent titles, URLs, or contacts."
)
MAX_TARGET_AUDIENCES = 6
LOW_SIGNAL_AUDIENCE = re.compile(r"\b(beginner|student|hobbyist|hackathon|vibe)\b", re.IGNORECASE)
AUDIENCE_GROUP_MARKER = re.compile(
    r"\b(agenc(?:y|ies)|business(?:es)?|builders?|coders?|compan(?:y|ies)|"
    r"customers?|developers?|engineers?|enterprises?|founders?|marketers?|"
    r"organizations?|professionals?|providers?|saas|startups?|teams?|users?)\b",
    re.IGNORECASE,
)


def _normalized_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.casefold())


def _merge_titles(existing_title: Optional[str], new_title: Optional[str]) -> Optional[str]:
    """Combine distinct, explicitly extracted titles without inventing a role."""
    titles = []
    seen = set()
    for title in (existing_title, new_title):
        for part in (title or "").split(";"):
            cleaned = part.strip()
            key = re.sub(r"\s+", " ", cleaned).casefold()
            if cleaned and key not in seen:
                seen.add(key)
                titles.append(cleaned)
    return "; ".join(titles) or None


def _post_process_intel(intel: CompanyIntel) -> CompanyIntel:
    """Apply conservative, deterministic quality guards to validated model output."""
    audiences = []
    seen_audiences = set()
    for audience in intel.target_audience:
        label = audience.strip()
        key = label.casefold()
        if (
            not label
            or key in seen_audiences
            or LOW_SIGNAL_AUDIENCE.search(label)
            or not AUDIENCE_GROUP_MARKER.search(label)
        ):
            continue
        seen_audiences.add(key)
        audiences.append(label)
        if len(audiences) == MAX_TARGET_AUDIENCES:
            break
    intel.target_audience = audiences

    people_by_name = {}
    for member in intel.key_team_members:
        key = _normalized_name(member.name)
        if not key:
            continue
        existing = people_by_name.get(key)
        if existing is None:
            people_by_name[key] = member
            continue
        existing.title = _merge_titles(existing.title, member.title)
        if not existing.linkedin_url and member.linkedin_url:
            existing.linkedin_url = member.linkedin_url
    intel.key_team_members = list(people_by_name.values())
    return intel


def _completeness_score(intel: CompanyIntel) -> float:
    """Simple evidence-based confidence independent of model self-assessment."""
    fields_present = [
        bool(intel.company_overview.strip()),
        bool(intel.target_audience),
        bool(intel.contact_points),
        bool(intel.key_team_members),
        any(member.linkedin_url for member in intel.key_team_members),
    ]
    return sum(fields_present) / len(fields_present)


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


def _usage_tokens(response) -> Tuple[int, int]:
    usage = response.usage
    return (
        usage.prompt_tokens if usage else 0,
        usage.completion_tokens if usage else 0,
    )


def _is_groq_tool_use_failure(exc: Exception) -> bool:
    """Identify Groq's HTTP 400 failures caused by forced tool use."""
    status_code = getattr(exc, "status_code", None)
    if status_code is not None and status_code != 400:
        return False
    error_text = " ".join(
        str(value)
        for value in (str(exc), getattr(exc, "code", ""), getattr(exc, "body", ""))
    ).casefold()
    return any(
        marker in error_text
        for marker in (
            "tool_use_failed",
            "failed to parse tool call arguments as json",
            "model did not call a tool",
            "tool choice is required",
        )
    )


def _extract_with_groq_json_fallback(client, model: str, domain: str, context: str) -> Tuple[Optional[dict], int, int]:
    """One JSON-mode retry for Groq when forced tool use is rejected."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    f"{JSON_FALLBACK_PROMPT}\n\nJSON schema:\n"
                    f"{json.dumps(COMPANY_INTEL_JSON_SCHEMA)}"
                ),
            },
            {"role": "user", "content": _build_user_prompt(domain, context)},
        ],
        response_format={"type": "json_object"},
    )
    input_tokens, output_tokens = _usage_tokens(response)
    content = response.choices[0].message.content
    if not content:
        return None, input_tokens, output_tokens
    try:
        parsed = json.loads(content)
        return (parsed if isinstance(parsed, dict) else None), input_tokens, output_tokens
    except json.JSONDecodeError:
        logger.error("Failed to parse Groq JSON fallback for %s", domain)
        return None, input_tokens, output_tokens


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

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(domain, context)},
            ],
            tools=[tool],
            tool_choice={"type": "function", "function": {"name": TOOL_NAME}},
        )
    except Exception as exc:
        if provider == "groq" and _is_groq_tool_use_failure(exc):
            logger.warning("Groq rejected required tool use for %s; retrying once in JSON mode", domain)
            return _extract_with_groq_json_fallback(client, model, domain, context)
        raise

    input_tokens, output_tokens = _usage_tokens(response)

    message = response.choices[0].message
    if message.tool_calls:
        args_str = message.tool_calls[0].function.arguments
        try:
            return json.loads(args_str), input_tokens, output_tokens
        except json.JSONDecodeError:
            logger.error("Failed to parse tool call arguments for %s", domain)
            if provider == "groq":
                logger.warning("Groq returned malformed tool arguments for %s; retrying once in JSON mode", domain)
                return _extract_with_groq_json_fallback(client, model, domain, context)
            return None, input_tokens, output_tokens

    if provider == "groq":
        logger.warning("Groq did not return the required tool call for %s; retrying once in JSON mode", domain)
        return _extract_with_groq_json_fallback(client, model, domain, context)
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
        intel = _post_process_intel(CompanyIntel.model_validate(raw))
        intel.data_confidence_score = _completeness_score(intel)
        return intel

    except Exception as exc:  # noqa: BLE001 - extraction must never crash the run
        logger.error("Extraction failed for %s: %s", domain, exc)
        return CompanyIntel(
            domain=domain,
            company_overview="",
            target_audience=[],
            contact_points=[],
            key_team_members=[],
            data_confidence_score=0.0,
            sources_used=[],
        )
