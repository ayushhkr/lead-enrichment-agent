# Autonomous Lead Enrichment Agent

A Python pipeline that takes company domains as input, crawls their public
web presence with a headless browser, cleans the content down to
LLM-friendly text, and extracts structured company intelligence (overview,
ICP, contacts, leadership, confidence score) using strict LLM tool calling.

## Architecture

```
main.py                # CLI entry point / orchestrator
agent/
  config.py             # env-driven settings
  crawler.py            # Step 1: Playwright-based crawling
  cleaner.py             # Step 2: HTML -> clean, token-bounded text
  schema.py               # Pydantic output contract
  extractor.py             # Step 3: LLM structured extraction (+ cost tracking)
domains.txt                # sample input (the 3 test domains)
output/                      # generated output.json / output.csv (gitignored)
```

Each domain runs through: `crawl_domain()` -> `build_context_for_llm()` ->
`extract_company_intel()`, all inside a per-domain try/except with retries,
so a single broken site (404, bot block, timeout) can never crash the batch
(Step 4 of the assignment).

## Setup

1. **Clone and create a virtual environment**
   ```bash
   git clone <this-repo>
   cd lead-enrichment-agent
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   playwright install chromium   # downloads the headless browser binary
   ```

3. **Configure environment variables**
   ```bash
   cp .env.example .env
   ```
   Open `.env` and set:
   - `LLM_PROVIDER` — `anthropic`, `openai`, or `groq`
   - The matching API key (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or `GROQ_API_KEY`)

## Running it

```bash
# Run on the 3 assignment test domains (default if no args given)
python main.py

# Or explicitly
python main.py postman.com supabase.com vapi.ai

# Or from a file, one domain per line
python main.py --domains-file domains.txt
```

Output is written to:
- `output/output.json` — full structured results
- `output/output.csv` — flattened, spreadsheet-friendly version
- `output/cost_report.json` — per-domain token usage and estimated USD cost

## How each requirement is met

| Requirement | Where |
|---|---|
| Headless browser automation, JS rendering | `agent/crawler.py` (Playwright, `domcontentloaded` wait) |
| Subpage discovery | `agent/crawler.py` ranks rendered homepage links; configured paths are a sparse-discovery fallback |
| No raw HTML to the LLM | `agent/cleaner.py` strips noise while retaining footer contact/LinkedIn data, then truncates text |
| Structured output (Pydantic + tool calling) | `agent/schema.py` + `agent/extractor.py` |
| Confidence score | `CompanyIntel.data_confidence_score`, deterministic completeness across five extracted evidence types |
| Graceful fallback (404s, timeouts, bot blocks) | `PageResult.status` in crawler + per-domain retry loop in `main.py` |
| Cost tracking (bonus) | `CostTracker` in `agent/extractor.py` → `output/cost_report.json` |

## Notes / design choices

- **Provider-agnostic extraction**: the same Pydantic schema is enforced via
  Anthropic tool calling or OpenAI/Groq function calling, so the grader can
  run this with whichever API key they have.
- **Token budget**: `MAX_CHARS_PER_PAGE` and `MAX_CHARS_TOTAL` (in `.env`)
  cap how much text reaches the LLM per page and per domain, keeping cost
  and latency predictable even on content-heavy sites.
- **Retries**: each domain gets `MAX_RETRIES` extra attempts with linear
  backoff before it's recorded as a zero-confidence result — the run itself
  never halts.

## Known limitations / next steps

- LinkedIn URLs for leadership are only pulled from what's linked directly
  on the company's own site. Wiring up a search API (SerpAPI/Tavily) as
  described in the assignment's bonus section would let the agent look up
  founders whose LinkedIn isn't linked from the site itself.
- Sites requiring login or aggressive bot-detection (Cloudflare challenge
  pages, etc.) will return an `error`/`timeout` status for that page rather
  than being bypassed — by design, this project doesn't attempt to defeat
  bot protections.
