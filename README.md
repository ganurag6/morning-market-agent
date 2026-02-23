# Morning Market Brief Agent

Generate a daily "Morning Market Brief" into local files. If API keys are missing, the pipeline runs in mock mode and still writes realistic sample outputs.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Copy `.env.example` to `.env` and add your keys:

```
PERPLEXITY_API_KEY=...
OPENAI_API_KEY=...
```

## Run

```bash
python -m agent.run --date 2026-02-14 --tz America/Chicago --watchlist "AAPL,MSFT,NVDA" --out ./out
```

Outputs:
- `out/YYYY-MM-DD/research.json`
- `out/YYYY-MM-DD/brief.md`

To force mock mode (no external calls):

```bash
python -m agent.run --date 2026-02-14 --tz America/Chicago --watchlist "AAPL,MSFT,NVDA" --out ./out --mock
```

## Notes

- Perplexity /search provides source material for the research pack.
- OpenAI turns the search results into strict research JSON, then generates the markdown brief without adding facts or trade instructions.
- Headlines are deduped by title and source URL.
