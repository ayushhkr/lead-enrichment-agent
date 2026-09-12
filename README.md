# Autonomous Lead Enrichment Agent

An AI-powered lead enrichment agent that takes company domains as input, autonomously discovers relevant pages from their public websites, extracts useful company intelligence, and produces validated structured output.

Built as a take-home assignment for the **AI Engineer Intern** role at SoftwareBrio.

---

## Overview

The agent accepts a list of company domains and uses **Playwright** to browse their websites dynamically.

Instead of crawling an entire website or relying only on hardcoded URLs, the agent:

1. Opens the company homepage.
2. Discovers internal links from the rendered page.
3. Ranks links based on their relevance to company enrichment.
4. Crawls a bounded number of relevant pages.
5. Cleans and reduces the extracted webpage content.
6. Sends the relevant context to an LLM.
7. Extracts structured company intelligence.
8. Validates the result using Pydantic.
9. Calculates a deterministic data-confidence score.
10. Produces JSON and CSV output.

The system is designed to continue operating when individual pages, API requests, or LLM tool calls fail.

---

## Features

- Dynamic homepage link discovery
- Relevance-based internal link ranking
- Bounded website crawling
- JavaScript-rendered page support through Playwright
- HTML/content cleaning before LLM processing
- LLM-based company intelligence extraction
- Pydantic schema validation
- Target audience / ICP extraction
- Public contact-point extraction
- Leadership and team-member extraction
- LinkedIn URL extraction when available
- Duplicate team-member handling
- Deterministic data-confidence scoring
- Retry and backoff handling for transient failures
- JSON fallback for LLM tool-call failures
- JSON and CSV output
- Token and estimated API-cost tracking
- Per-domain failure isolation

---

## Architecture

```text
                    Company Domain
                          │
                          ▼
                 ┌─────────────────┐
                 │    Playwright   │
                 │     Crawler     │
                 └────────┬────────┘
                          │
                          ▼
                    Load Homepage
                          │
                          ▼
                Discover Internal Links
                          │
                          ▼
                 Rank Relevant Links
                          │
                          ▼
              Select Bounded Page Set
                          │
                          ▼
                Crawl Relevant Pages
                          │
                          ▼
                  Clean Page Content
                          │
                          ▼
                 Build LLM Context
                          │
                          ▼
                 ┌─────────────────┐
                 │       LLM       │
                 │ Groq / Provider │
                 └────────┬────────┘
                          │
             ┌────────────┴────────────┐
             │                         │
       Tool-call success        Tool-call failure
             │                         │
             │                    JSON fallback
             │                         │
             └────────────┬────────────┘
                          ▼
                 Pydantic Validation
                          │
                          ▼
               Data Confidence Scoring
                          │
                          ▼
                 JSON / CSV Output
