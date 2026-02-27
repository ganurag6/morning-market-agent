from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, Iterable

import httpx


logger = logging.getLogger(__name__)


class MissingAPIKeyError(RuntimeError):
    pass


class OpenAIClient:
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 5,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise MissingAPIKeyError(
                "OPENAI_API_KEY is missing. Set it in the environment or .env."
            )
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o")
        self.timeout = timeout or 90.0
        self.max_retries = max_retries
        self.base_url = "https://api.openai.com/v1/chat/completions"

    def generate_brief(
        self,
        *,
        research: Dict[str, Any],
        date: str,
        tz: str,
        watchlist: Iterable[str],
    ) -> str:
        watchlist_text = ", ".join([w for w in watchlist if w]) or "None"
        prompt = (
            "Write a concise, professional Morning Market Brief in markdown. "
            "Use ONLY the research JSON provided. Do not add facts not in the JSON. "
            "Do not include trade instructions or recommendations. "
            "If a detail is missing, say 'Not specified'.\n\n"
            "The brief MUST include a section titled '## SPY Tomorrow: Scenario Map (Not a recommendation)' "
            "that summarizes 2-3 conditional scenarios derived strictly from the research JSON's "
            "market_state, events, and weekly_context sections. "
            "Each scenario must be written as an if/then market condition "
            "(e.g., reaction to yields, whether price is inside or outside the expected move, "
            "or holding/losing key levels). "
            "Do NOT include trade instructions in the scenarios.\n\n"
            f"Date: {date} ({tz})\n"
            f"Watchlist: {watchlist_text}\n\n"
            "Research JSON:\n"
            f"{json.dumps(research, indent=2)}\n"
        )

        payload = {
            "model": self.model,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": "You are a careful financial writer."},
                {"role": "user", "content": prompt},
            ],
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        response = self._post_with_retries(self.base_url, payload, headers)
        data = response.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("Unexpected OpenAI response shape.") from exc
        return content.strip()

    def generate_research_from_search(
        self,
        *,
        search_bundle: Dict[str, Any],
        date: str,
        tz: str,
        watchlist: Iterable[str],
        as_of: str,
    ) -> Dict[str, Any]:
        from datetime import datetime as _dt, timedelta
        watchlist_text = ", ".join([w for w in watchlist if w]) or "None"
        date_obj = _dt.strptime(date, "%Y-%m-%d").date()
        day_of_week = date_obj.strftime("%A")
        # Find next trading day (skip weekends; holidays are noted in prompt)
        _next = date_obj + timedelta(days=1)
        while _next.weekday() >= 5:  # 5=Sat, 6=Sun
            _next += timedelta(days=1)
        next_trading_day = _next.strftime("%Y-%m-%d (%A)")
        schema = """
{
  "as_of": string,
  "earnings": [ {"company":string,"ticker":string,"date":string,"time_hint":string|null,"notes":string,"sources":[string]} ],
  "events": [ {"event":string,"date_time_local":string|null,"region":string,"why_it_matters":string,"sources":[string]} ],
  "headlines": [ {"title":string,"topic":string,"tickers":[string], "impact":"high"|"medium"|"low", "one_line_take":string, "sources":[string]} ],
  "weekly_context": {
    "themes":[ {"theme":string,"evidence":[string],"sources":[string]} ],
    "market_moves":[ {"asset":string,"move":string,"window":string,"sources":[string]} ]
  },
  "market_state": {
    "spy": {"last":string,"prev_ohlc":{"o":string,"h":string,"l":string,"c":string},"premarket_change":string,"sources":[string]},
    "futures": {"es_change":string,"overnight_high":string,"overnight_low":string,"notes":string,"sources":[string]},
    "volatility": {"vix":string,"vix_change":string,"vvix":string,"term_structure":string,"expected_move_1d":string,"expected_move_7d":string,"sources":[string]},
    "rates_fx_oil": {"us10y":string,"us10y_change":string,"dxy":string,"dxy_change":string,"wti":string,"wti_change":string,"sources":[string]},
    "breadth": {"adv_dec":string,"notes":string,"leaders":string,"laggards":string,"sources":[string]},
    "options_positioning": {"put_wall":string,"call_wall":string,"gamma_flip":string,"zero_dte_notes":string,"sources":[string]}
  }
}
""".strip()
        prompt = (
            "You are given search results from Perplexity /search. "
            "Use ONLY the information in those results to build a Morning Market Brief research pack. "
            "Do not add facts that are not supported by the search results. "
            "Return ONLY valid JSON that matches the schema below. No markdown, no code fences.\n\n"
            f"As-of value (must use exactly): {as_of}\n"
            f"Date: {date} ({day_of_week}) ({tz})\n"
            f"Next trading day: {next_trading_day}\n"
            f"Watchlist: {watchlist_text}\n\n"
            "Requirements:\n"
            "- Earnings and events for the next 7 calendar days starting from the NEXT TRADING DAY.\n"
            "- CRITICAL DATE RULES for earnings and events:\n"
            f"  - Today is {day_of_week} {date}. If today is a weekend or holiday, there are NO events or earnings today.\n"
            "  - US markets are CLOSED on Saturdays, Sundays, and federal holidays (Presidents' Day = 3rd Monday of Feb, etc.).\n"
            "  - If a source lists an event for 'this week' without a specific date, set date_time_local to null. NEVER assign it to today if today is a weekend/holiday.\n"
            "  - Only use dates that are explicitly stated in the source data.\n"
            "- Macro/events in the next 7 calendar days (central banks, CPI/PPI, PMI, jobs, auctions, GDP, PCE).\n"
            "- 12 to 25 headlines from the last ~18 hours across broad market, sectors, and the watchlist.\n"
            "- Weekly context covering last 5 trading days (themes and market moves).\n"
            "- Every item must include at least one source URL string from the search results.\n"
            "- If a detail is missing, use 'Not specified' in that field or return an empty list.\n\n"
            "market_state extraction rules (CRITICAL — read carefully):\n"
            "- spy.last: Most recent SPY ETF price (not ES futures). Look for '$SPX', 'SPY', or 'S&P 500' closing/last price.\n"
            "- spy.prev_ohlc: Prior session open/high/low/close for SPY specifically. Use the ETF price, not the index or futures.\n"
            "- spy.premarket_change: Premarket % change for SPY if available.\n"
            "- futures.es_change: The E-mini S&P 500 (ES) futures % change or point change. Look for 'ESH' contract or 'E-mini S&P' settlement.\n"
            "- futures.overnight_high / overnight_low: The ES futures overnight session high and low. These are futures prices (e.g. 6000+), NOT ETF prices.\n"
            "- volatility.vix: The CBOE VIX index level (a number typically between 10 and 80).\n"
            "- volatility.vix_change: VIX point or % change from prior close.\n"
            "- volatility.vvix: CBOE VVIX level if mentioned.\n"
            "- volatility.term_structure: 'contango' if front-month VIX < second-month, 'backwardation' if front > second, or 'Not specified'.\n"
            "- volatility.expected_move_1d / expected_move_7d: SPY implied move from options if stated. Do NOT calculate this yourself.\n"
            "- rates_fx_oil.us10y: 10-year US Treasury yield (e.g. '4.05%'). Look for 'T-note yield', '10-year yield', 'TNX'.\n"
            "- rates_fx_oil.us10y_change: Yield change in bps or % from prior session.\n"
            "- rates_fx_oil.dxy: US Dollar Index level. Look for 'DXY', 'dollar index', 'USD index'.\n"
            "- rates_fx_oil.wti: WTI crude oil price. Look for 'WTI', 'crude oil', 'CL' futures.\n"
            "- breadth.adv_dec: NYSE or Nasdaq advance/decline ratio or count.\n"
            "- breadth.leaders / laggards: Best and worst performing sectors.\n"
            "- options_positioning: ONLY populate put_wall, call_wall, gamma_flip if explicit strike levels are stated in the sources. Do NOT infer or estimate. Use 'Not specified' otherwise.\n"
            "- Prefer primary sources: CBOE, CME, Barchart, Treasury.gov, Fed, BLS, BEA, Reuters, Bloomberg, FT, WSJ.\n"
            "- If sources disagree on a number, provide a range and note 'mixed reports'.\n"
            "- Do NOT confuse ES futures levels (e.g. 6851) with SPY ETF prices (e.g. 601). They are different instruments.\n\n"
            f"Schema:\n{schema}\n\n"
            "Search results JSON:\n"
            f"{json.dumps(search_bundle, indent=2)}\n"
        )

        payload = {
            "model": self.model,
            "temperature": 0.2,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a precise financial research assistant.",
                },
                {"role": "user", "content": prompt},
            ],
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        response = self._post_with_retries(self.base_url, payload, headers)
        data = response.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("Unexpected OpenAI response shape.") from exc

        json_text = _strip_code_fences(content)
        try:
            parsed = json.loads(json_text)
        except json.JSONDecodeError as exc:
            raise ValueError("OpenAI research response was not valid JSON.") from exc
        return _normalize_keys(parsed)

    def generate_trade_recs(
        self,
        *,
        snapshot: Dict[str, Any],
        signals: list[Dict[str, Any]],
        research: Dict[str, Any],
        options_chains: Dict[str, Any],
        sympathy_map: Dict[str, Any],
        max_allocation: float = 5000.0,
        watchlist: Iterable[str] = (),
        watchlist_scans: list[Dict[str, Any]] | None = None,
        earnings_timing: Dict[str, str] | None = None,
    ) -> Dict[str, Any]:
        """Generate specific trade recommendations from rule signals and market data."""
        watchlist_text = ", ".join([w for w in watchlist if w]) or "SPY"

        triggered = [s for s in signals if s.get("triggered")]
        long_count = sum(1 for s in triggered if s.get("direction") == "long")
        short_count = sum(1 for s in triggered if s.get("direction") == "short")

        events_today = []
        for ev in research.get("events", []):
            dt_str = ev.get("date_time_local", "") or ""
            if research.get("date", "") in dt_str or "today" in dt_str.lower():
                events_today.append(ev)

        earnings_upcoming = research.get("earnings", [])[:5]
        earnings_timing = earnings_timing or {}

        schema = """{
  "recommendations": [
    {
      "ticker": "SPY",
      "direction": "call" | "put",
      "strike": 685.0,
      "expiry": "2026-02-28",
      "entry_timing": "At open" | "Wait for 9:45 AM CT pullback" | ...,
      "allocation_dollars": 1500,
      "max_loss_dollars": 750,
      "stop_loss_pct": 50,
      "take_profit_pct": 100,
      "triggered_rules": ["R4", "R8"],
      "reasoning": "...",
      "confidence": "high" | "medium" | "low"
    }
  ],
  "sympathy_plays": [
    {
      "primary_ticker": "NVDA",
      "primary_catalyst": "Earnings after close",
      "sympathy_ticker": "AMD",
      "beta": 1.01,
      "direction": "call" | "put",
      "entry_timing": "NEXT DAY ONLY" | "Same day" | null,
      "reasoning": "..."
    }
  ]
}"""

        # Build watchlist scan section
        scan_text = ""
        if watchlist_scans:
            scan_text = (
                f"\nWatchlist Technical Scans ({len(watchlist_scans)} tickers):\n"
                f"{json.dumps(watchlist_scans, indent=2)}\n\n"
                "Use the watchlist scans to find individual stock opportunities. "
                "Tickers with signals like 'deeply_oversold', 'oversold', '5d_selloff', "
                "'well_below_sma', or 'high_volume' are good candidates for mean-reversion "
                "call trades. Tickers with 'overbought' or 'momentum_breakout' may be "
                "put candidates or avoid.\n"
            )

        # Build per-ticker signal section (R15/R16 signals)
        per_ticker_signals = [
            s for s in signals
            if s.get("triggered") and s.get("target_ticker")
        ]
        per_ticker_text = ""
        if per_ticker_signals:
            per_ticker_text = (
                "\n*** PER-TICKER EVENT SIGNALS (HIGH PRIORITY) ***\n"
                f"{json.dumps(per_ticker_signals, indent=2)}\n\n"
                "IMPORTANT: These per-ticker signals represent event-driven patterns "
                "(sell-the-news, mega-cap earnings vol). They should directly influence "
                "your recommendations for the specific tickers listed.\n"
            )

        prompt = (
            "You are a quantitative trading assistant. Generate specific trade "
            "recommendations based ONLY on the provided signals, market data, and "
            "available options contracts. Do not add information not in the input.\n\n"
            f"Date: {research.get('date', 'unknown')}\n"
            f"Watchlist: {watchlist_text}\n\n"
            f"Market Snapshot:\n{json.dumps(snapshot, indent=2)}\n\n"
            f"Active Signals ({len(triggered)} triggered, {long_count} long, {short_count} short):\n"
            f"{json.dumps(triggered, indent=2)}\n\n"
            f"Today's Events:\n{json.dumps(events_today, indent=2)}\n\n"
            f"Upcoming Earnings (next 5):\n{json.dumps(earnings_upcoming, indent=2)}\n\n"
            f"Sympathy Map:\n{json.dumps(sympathy_map, indent=2)}\n\n"
            f"Available Options Chains:\n{json.dumps(options_chains, indent=2)}\n\n"
            f"{scan_text}"
            f"{per_ticker_text}"
            f"Earnings Timing Tags:\n{json.dumps(earnings_timing, indent=2)}\n\n"
            f"Max Total Allocation: ${max_allocation:.0f}\n\n"
            "Rules for generating recommendations:\n"
            "1. ALWAYS include at least 1 SPY trade (call or put based on macro signals).\n"
            "2. Generate 5 to 7 total trade recommendations (including exactly 1 hedge trade) — diversify across SPY and individual stocks.\n"
            "3. Use triggered macro signals (R1-R14) for SPY directional trades.\n"
            "4. Use watchlist scan data for individual stock picks — oversold stocks with below-SMA "
            "signals are good call candidates; overbought stocks are put candidates.\n"
            "5. Use earnings catalysts and sympathy map for event-driven plays.\n"
            "6. Select strikes: MUST be within 0-1% OTM. NEVER pick strikes >1.5% OTM. "
            "Prefer the preferred_call_strike or preferred_put_strike from the options chain data.\n"
            "7. Select expiry: nearest Friday at least 2 days out from the available options.\n"
            "8. Allocation per trade: $300-$500 (low confidence), $500-$1500 (medium), $1500-$2500 (high).\n"
            "9. Stop loss: 50% of premium paid.\n"
            "10. Take profit: 100-200% of premium paid.\n"
            "11. If 3+ bullish signals, increase allocation. If mixed (long+short), reduce sizes.\n"
            "12. MANDATORY hedge trade: include exactly one hedge trade in the opposite direction "
            "(e.g. SPY put if mostly bullish). Allocate 15-20% of total allocation to it. Non-negotiable.\n"
            "13. Include entry timing: 'at open' for gap plays, '9:45 AM CT' for pullback entries, "
            "'after earnings' for catalyst plays.\n"
            "14. For sympathy plays, only include if a clear catalyst exists (earnings, major event).\n"
            "15. If no signals are triggered at all, return at least 2 low-confidence watchlist-based trades.\n"
            "16. Each recommendation MUST have a unique reasoning — explain why THIS specific trade, not generic.\n"
            "17. Strikes MUST exist in the provided options chain data. Do not invent strikes.\n"
            "18. Max 30% allocation per individual trade. No single trade should dominate.\n"
            "19. Deeply oversold stocks (extreme_oversold or deeply_oversold signals) with upcoming earnings = HIGH confidence plays.\n"
            "20. After-close earnings → sympathy plays MUST be tagged entry_timing='NEXT DAY ONLY'. "
            "Before-open earnings → 'Same day'. Use the Earnings Timing Tags data.\n"
            "21. For each trade, explain the strike choice vs ATM in the reasoning field.\n"
            "22. SELL-THE-NEWS (R15): If a per-ticker signal shows sell-the-news pattern, "
            "generate a PUT recommendation on that specific ticker. The stock rallied into "
            "earnings and is now fading — classic post-earnings fade. Use ATM or slightly OTM put.\n"
            "23. MEGA-CAP VOL EVENT (R16): If a per-ticker signal shows muted mega-cap reaction, "
            "reduce long exposure on that ticker. If a mega-cap has upcoming earnings after close, "
            "use smaller sizing on that ticker's positions (binary event risk).\n"
            "24. ALL-BULLISH CONTRARIAN (R17): If flagged, acknowledge one-directional positioning "
            "risk. The hedge allocation is enforced to 20% in post-processing — do NOT reduce it.\n\n"
            "Return ONLY valid JSON matching this schema (no markdown, no code fences):\n"
            f"{schema}\n"
        )

        payload = {
            "model": self.model,
            "temperature": 0.2,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a quantitative trading assistant. "
                        "Generate precise, actionable trade recommendations "
                        "based strictly on the provided data and signals."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        response = self._post_with_retries(self.base_url, payload, headers)
        data = response.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("Unexpected OpenAI response shape.") from exc

        json_text = _strip_code_fences(content)
        try:
            parsed = json.loads(json_text)
        except json.JSONDecodeError as exc:
            raise ValueError("OpenAI trade recs response was not valid JSON.") from exc
        return _normalize_keys(parsed)

    def _post_with_retries(
        self, url: str, payload: Dict[str, Any], headers: Dict[str, str]
    ) -> httpx.Response:
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = httpx.post(
                    url, json=payload, headers=headers, timeout=self.timeout
                )
                if response.status_code in {429} or response.status_code >= 500:
                    raise httpx.HTTPStatusError(
                        f"HTTP {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                response.raise_for_status()
                return response
            except (httpx.RequestError, httpx.HTTPStatusError) as exc:
                last_error = exc
                if attempt == self.max_retries:
                    break
                backoff = 2 ** attempt * 5
                logger.warning(
                    "OpenAI request failed (attempt %s/%s). Retrying in %ss.",
                    attempt,
                    self.max_retries,
                    backoff,
                )
                time.sleep(backoff)
        raise RuntimeError("OpenAI request failed after retries.") from last_error


def _normalize_keys(obj: Any) -> Any:
    """Fix common LLM key typos like dots instead of underscores."""
    _aliases = {"0dte_notes": "zero_dte_notes"}
    if isinstance(obj, dict):
        fixed = {}
        for k, v in obj.items():
            new_key = k.replace(".", "_").replace("-", "_").replace(" ", "_")
            new_key = _aliases.get(new_key, new_key)
            fixed[new_key] = _normalize_keys(v)
        return fixed
    if isinstance(obj, list):
        return [_normalize_keys(item) for item in obj]
    return obj


def _strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        lines = [line for line in cleaned.splitlines() if line.strip()]
        if lines and lines[0].lower().startswith("json"):
            lines = lines[1:]
        cleaned = "\n".join(lines)
    return cleaned.strip()
