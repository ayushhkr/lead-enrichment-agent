# Autonomous Lead Enrichment Agent

An AI agent that takes a list of company domains, browses their public websites, and returns validated, structured company intelligence as JSON and CSV.

Give it `postman.com`; get back what the company does, who it sells to, who leads it, how to contact it, and a confidence score for how complete the data is.


---
<img width="996" height="736" alt="image" src="https://github.com/user-attachments/assets/98536a00-168f-42f8-a7ad-53e74ac9cf37" />

## What it produces

For each domain, the agent extracts:

- **Company summary**: what the company does
- **Target audience / ICP**: who the product is for
- **Leadership and team members**: names, titles, and LinkedIn URLs when available (deduplicated)
- **Public contact points**: emails, phone numbers, contact pages
- **Data-confidence score**: a deterministic score based on how many fields were found
- **Run metadata**: tokens used and estimated API cost

See [`sample_output.json`](sample_output.json) for a real example.

```json
```


## How it works

Instead of crawling an entire site or hardcoding URLs like `/about`, the agent decides where to look based on what each site actually links to.

```
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
```

1. **Discover**: load the homepage in a real browser (Playwright, so JavaScript-rendered sites work) and collect internal links.
2. **Rank**: score links by relevance to company enrichment (about, team, contact, product pages, etc.).
3. **Crawl**: visit only a bounded number of the top-ranked pages.
4. **Clean**: strip scripts, navigation, and boilerplate so the LLM sees signal, not markup.
5. **Extract**: send the cleaned context to the LLM to extract structured fields.
6. **Validate**: enforce the output schema with Pydantic.
7. **Score**: compute a deterministic confidence score from the validated result.
8. **Export**: write JSON and CSV.

---

## Design decisions

- **Bounded crawling.** Each domain has a page limit, which keeps cost, latency, and token usage predictable.
- **Link ranking over hardcoded paths.** Sites structure themselves differently. Ranking discovered links works on sites where `/about` doesn't exist.
- **Content cleaning before the LLM.** Sending less, cleaner text lowers token cost and reduces hallucination risk.
- **Schema validation with Pydantic.** LLM output is untrusted input. Invalid or malformed results are caught instead of silently written to the output.
- **Deterministic confidence score.** The score is computed from the validated data (which fields were found), not self-reported by the LLM, so it is reproducible and can't be inflated by the model.
- **JSON fallback for tool-call failures.** If structured tool-calling fails, the agent falls back to plain JSON extraction instead of dropping the domain.
- **Failure isolation.** A failed page, API call, or domain doesn't stop the batch. Transient errors are retried with backoff.
- **Cost tracking.** Token usage and estimated API cost are recorded per run.

---

## Quick start

**Requirements:** Python 3.10+ and an LLM API key.

```bash
git clone https://github.com/ayushhkr/lead-enrichment-agent.git
cd lead-enrichment-agent

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
playwright install chromium
```

Configure your API key:

```bash
cp .env.example .env
# then edit .env and add your key
```

Add the domains you want to enrich to `domains.txt`, one per line:

```
example.com
another-company.io
```

Run:

```bash
python main.py
```

Results are written as JSON and CSV. 

---

## Project structure

```
lead-enrichment-agent/
├── agent/               # crawling, link ranking, LLM extraction, validation, scoring
├── main.py              # entry point: reads domains.txt, runs the pipeline, writes output
├── domains.txt          # input list of company domains
├── sample_output.json   # example output
├── requirements.txt
└── .env.example         # environment variable template
```

<!-- TODO: list the main files inside agent/ with one line each -->

---

## Limitations

- Works on public pages only; information not published on a company's site won't be found.
- Extraction quality depends on the LLM and on how much useful content the bounded page set contains. That is what the confidence score is meant to reflect.
- Sites with aggressive bot protection or login walls may return little or no content.
- Please use responsibly and respect each site's terms of service.

---

## Roadmap

- Evaluation set: measure extraction accuracy against hand-labeled domains
- Unit tests for link ranking, content cleaning, and confidence scoring
- Concurrent crawling across domains
- Support for additional LLM providers
- Optional CRM export (CSV is supported today)

---

## Output

The agent generates structured lead enrichment results and exports the
processed data for further analysis or downstream workflows.
The agent returns structured lead enrichment data in JSON and CSV formats.

## Error Handling

The lead enrichment pipeline handles failures from external services gracefully.
API errors, missing lead information, and incomplete enrichment results should
be reported clearly without stopping the processing of other leads.

## Tech stack

Python · Playwright · Pydantic · LLM API (Groq or compatible provider)
