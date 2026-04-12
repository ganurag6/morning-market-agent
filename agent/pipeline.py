from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import date as date_type, datetime, time
from pathlib import Path
from typing import Any, Dict, Iterable, List
from zoneinfo import ZoneInfo

from .openai_client import MissingAPIKeyError as OpenAIMissingAPIKeyError
from .openai_client import OpenAIClient
from .perplexity_client import MissingAPIKeyError as PerplexityMissingAPIKeyError
from .perplexity_client import PerplexityClient
from .schema import HeadlineItem, ResearchPack


logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    date: str
    out_dir: Path
    research_path: Path
    brief_path: Path
    mock_mode: bool


def run_pipeline(
    *,
    date: str | None,
    tz: str,
    watchlist: Iterable[str],
    out_dir: str | Path,
    mock_mode: bool | None = None,
) -> PipelineResult:
    tzinfo = ZoneInfo(tz)
    target_date = _resolve_date(date, tzinfo)
    date_str = target_date.strftime("%Y-%m-%d")

    output_dir = Path(out_dir) / date_str
    output_dir.mkdir(parents=True, exist_ok=True)

    missing_keys = _missing_api_keys()
    if mock_mode is None:
        mock_mode = bool(missing_keys)

    if mock_mode:
        if missing_keys:
            logger.error(
                "Missing API keys: %s. Running in mock mode with no external calls.",
                ", ".join(missing_keys),
            )
        else:
            logger.info("Mock mode enabled.")
        research_pack = ResearchPack.model_validate(
            _build_mock_research(date_str=date_str, tz=tz, watchlist=watchlist)
        )
        research_pack.headlines = dedupe_headlines(research_pack.headlines)
        research = research_pack.model_dump()
        brief_md = _build_mock_brief(research, date_str, tz, watchlist)
    else:
        if missing_keys:
            raise RuntimeError(
                "Missing API keys: "
                + ", ".join(missing_keys)
                + ". Set them or run in mock mode."
            )
        try:
            search_bundle = _collect_search_bundle(
                date_str=date_str, tz=tz, watchlist=watchlist
            )
        except PerplexityMissingAPIKeyError as exc:
            raise RuntimeError(str(exc)) from exc

        try:
            openai_client = OpenAIClient()
            research_raw = openai_client.generate_research_from_search(
                search_bundle=search_bundle,
                date=date_str,
                tz=tz,
                watchlist=watchlist,
                as_of=search_bundle["as_of"],
            )
            research_raw["as_of"] = search_bundle["as_of"]
        except OpenAIMissingAPIKeyError as exc:
            raise RuntimeError(str(exc)) from exc

        research_pack = ResearchPack.model_validate(research_raw)
        research_pack.headlines = dedupe_headlines(research_pack.headlines)
        research = research_pack.model_dump()

        brief_md = openai_client.generate_brief(
            research=research, date=date_str, tz=tz, watchlist=watchlist
        )

    research_path = output_dir / "research.json"
    brief_path = output_dir / "brief.md"

    _write_json(research_path, research)
    _write_text(brief_path, brief_md)

    return PipelineResult(
        date=date_str,
        out_dir=output_dir,
        research_path=research_path,
        brief_path=brief_path,
        mock_mode=mock_mode,
    )


def _resolve_date(date_value: str | None, tzinfo: ZoneInfo) -> date_type:
    if date_value:
        try:
            return datetime.strptime(date_value, "%Y-%m-%d").date()
        except ValueError as exc:
            raise ValueError("--date must be in YYYY-MM-DD format.") from exc
    return datetime.now(tzinfo).date()


def _collect_search_bundle(
    *, date_str: str, tz: str, watchlist: Iterable[str]
) -> Dict[str, Any]:
    tzinfo = ZoneInfo(tz)
    as_of = datetime.now(tzinfo).isoformat(timespec="seconds")
    watchlist_list = [w for w in watchlist if w]
    watchlist_text = ", ".join(watchlist_list) if watchlist_list else "S&P 500"

    queries = [
        {
            "name": "earnings",
            "query": f"earnings calendar {date_str} next 7 days {watchlist_text}",
            "recency": "week",
        },
        {
            "name": "macro_events",
            "query": f"US economic calendar week of {date_str} specific dates CPI PPI PMI jobs FOMC central bank speeches Treasury auctions GDP",
            "recency": "week",
        },
        {
            "name": "headlines_market",
            "query": "stock market headlines last 18 hours sectors futures rates dollar oil",
            "recency": "day",
        },
        {
            "name": "headlines_watchlist",
            "query": f"{watchlist_text} news last 18 hours",
            "recency": "day",
        },
        {
            "name": "weekly_context",
            "query": "market weekly recap last 5 trading days stocks bonds dollar oil",
            "recency": "week",
        },
        {
            "name": "futures_levels",
            "query": f"S&P 500 E-mini ES futures price today settlement overnight high low SPY closing price {date_str}",
            "recency": "week",
        },
        {
            "name": "volatility",
            "query": f"CBOE VIX index closing level today VIX futures term structure contango backwardation {date_str}",
            "recency": "week",
        },
        {
            "name": "options_positioning",
            "query": "SPY options largest open interest strikes put call volume gamma exposure 0DTE options flow",
            "recency": "week",
        },
        {
            "name": "rates_fx_oil",
            "query": f"US 10-year Treasury yield closing price DXY dollar index level WTI crude oil price {date_str}",
            "recency": "week",
        },
        {
            "name": "breadth",
            "query": f"stock market breadth NYSE advancers decliners S&P 500 sector performance best worst sectors {date_str}",
            "recency": "week",
        },
        {
            "name": "geopolitical",
            "query": "geopolitical risk military conflict sanctions trade war oil supply disruption market impact",
            "recency": "day",
        },
    ]

    client = PerplexityClient()
    results = []
    for entry in queries:
        response = client.search(
            query=entry["query"],
            max_results=10,
            country="US",
            max_tokens_per_page=512,
            search_recency_filter=entry["recency"],
        )
        results.append(
            {
                "name": entry["name"],
                "query": entry["query"],
                "recency": entry["recency"],
                "response": response,
            }
        )

    return {
        "as_of": as_of,
        "date": date_str,
        "tz": tz,
        "watchlist": watchlist_list,
        "queries": results,
    }


def _missing_api_keys() -> List[str]:
    missing = []
    if not os.getenv("PERPLEXITY_API_KEY"):
        missing.append("PERPLEXITY_API_KEY")
    if not os.getenv("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")
    return missing


def dedupe_headlines(headlines: List[HeadlineItem]) -> List[HeadlineItem]:
    seen_titles = set()
    seen_sources = set()
    deduped: List[HeadlineItem] = []

    for headline in headlines:
        title_key = _normalize_title(headline.title)
        source_keys = [_normalize_source(src) for src in headline.sources if src]

        if title_key in seen_titles:
            continue
        if any(source_key in seen_sources for source_key in source_keys):
            continue

        deduped.append(headline)
        seen_titles.add(title_key)
        for source_key in source_keys:
            seen_sources.add(source_key)

    return deduped


def _normalize_title(title: str) -> str:
    return " ".join(title.lower().split())


def _normalize_source(source: str) -> str:
    cleaned = source.strip().lower()
    if cleaned.endswith("/"):
        cleaned = cleaned[:-1]
    return cleaned


def _write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def _write_text(path: Path, content: str) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(content.rstrip() + "\n")


def _build_mock_research(*, date_str: str, tz: str, watchlist: Iterable[str]) -> dict:
    tzinfo = ZoneInfo(tz)
    date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
    as_of = datetime.combine(date_obj, time(7, 0), tzinfo=tzinfo).isoformat()
    watchlist_list = [w for w in watchlist if w]

    return {
        "as_of": as_of,
        "earnings": [
            {
                "company": "Applied Materials",
                "ticker": "AMAT",
                "date": date_str,
                "time_hint": "After market close",
                "notes": "Equipment demand commentary and margin outlook.",
                "sources": ["https://example.com/earnings/amat"],
            },
            {
                "company": "Airbnb",
                "ticker": "ABNB",
                "date": _shift_date(date_obj, 2),
                "time_hint": "After market close",
                "notes": "Booking trends and forward guidance focus.",
                "sources": ["https://example.com/earnings/abnb"],
            },
            {
                "company": "Walmart",
                "ticker": "WMT",
                "date": _shift_date(date_obj, 3),
                "time_hint": "Before market open",
                "notes": "Consumer spending and inventory levels.",
                "sources": ["https://example.com/earnings/wmt"],
            },
        ],
        "events": [
            {
                "event": "US CPI release",
                "date_time_local": _shift_datetime(date_obj, 1, 7, 30, tzinfo),
                "region": "US",
                "why_it_matters": "Inflation trend influences rate expectations.",
                "sources": ["https://example.com/events/cpi"],
            },
            {
                "event": "Fed speakers (regional presidents)",
                "date_time_local": _shift_datetime(date_obj, 2, 12, 0, tzinfo),
                "region": "US",
                "why_it_matters": "Signals on policy path and inflation tolerance.",
                "sources": ["https://example.com/events/fed"],
            },
            {
                "event": "US Treasury 10Y auction",
                "date_time_local": _shift_datetime(date_obj, 3, 12, 0, tzinfo),
                "region": "US",
                "why_it_matters": "Auction demand can move yields and risk sentiment.",
                "sources": ["https://example.com/events/treasury"],
            },
        ],
        "headlines": _mock_headlines(watchlist_list),
        "market_state": {
            "spy": {
                "last": "502.35",
                "prev_ohlc": {"o": "500.10", "h": "503.88", "l": "498.72", "c": "502.35"},
                "premarket_change": "+0.3%",
                "sources": ["https://example.com/market/spy"],
            },
            "futures": {
                "es_change": "+0.25%",
                "overnight_high": "5045",
                "overnight_low": "5012",
                "notes": "Thin overnight session; held above prior close.",
                "sources": ["https://example.com/market/es"],
            },
            "volatility": {
                "vix": "14.8",
                "vix_change": "-0.6",
                "vvix": "82.5",
                "term_structure": "contango",
                "expected_move_1d": "0.7%",
                "expected_move_7d": "1.8%",
                "sources": ["https://example.com/market/vix"],
            },
            "rates_fx_oil": {
                "us10y": "4.28%",
                "us10y_change": "-3 bps",
                "dxy": "104.2",
                "dxy_change": "-0.15%",
                "wti": "76.40",
                "wti_change": "-0.8%",
                "sources": ["https://example.com/market/rates"],
            },
            "breadth": {
                "adv_dec": "1.4:1 favoring advancers",
                "notes": "Breadth improved late session on rotation into cyclicals.",
                "leaders": "Industrials, Materials",
                "laggards": "Utilities, Staples",
                "sources": ["https://example.com/market/breadth"],
            },
            "options_positioning": {
                "put_wall": "4950",
                "call_wall": "5100",
                "gamma_flip": "5020",
                "zero_dte_notes": "0DTE volume elevated; dealers short gamma below 5020.",
                "sources": ["https://example.com/market/options"],
            },
        },
        "weekly_context": {
            "themes": [
                {
                    "theme": "Rates volatility eased as inflation fears moderated",
                    "evidence": [
                        "Front-end yields stabilized while 10Y traded in a tighter range",
                        "Equity risk premia held steady with modest sector rotation",
                    ],
                    "sources": ["https://example.com/themes/rates"],
                },
                {
                    "theme": "AI capex optimism supported mega-cap tech",
                    "evidence": [
                        "Cloud spend commentary remained constructive",
                        "Semi equipment orders showed resilience",
                    ],
                    "sources": ["https://example.com/themes/ai"],
                },
            ],
            "market_moves": [
                {
                    "asset": "S&P 500",
                    "move": "+1.4%",
                    "window": "last 5 sessions",
                    "sources": ["https://example.com/moves/spx"],
                },
                {
                    "asset": "NASDAQ 100",
                    "move": "+2.1%",
                    "window": "last 5 sessions",
                    "sources": ["https://example.com/moves/ndx"],
                },
                {
                    "asset": "US 10Y Treasury yield",
                    "move": "-6 bps",
                    "window": "last 5 sessions",
                    "sources": ["https://example.com/moves/10y"],
                },
            ],
        },
    }


def _mock_headlines(watchlist: List[str]) -> List[dict]:
    base = [
        {
            "title": "US futures edge higher ahead of CPI print",
            "topic": "Macro",
            "tickers": ["SPY", "QQQ"],
            "impact": "high",
            "one_line_take": "Risk appetite steady as investors await inflation data.",
            "sources": ["https://example.com/headlines/futures"],
        },
        {
            "title": "Oil slips on demand uncertainty as inventories rise",
            "topic": "Energy",
            "tickers": [],
            "impact": "medium",
            "one_line_take": "Crude prices soften after inventory data shows a build.",
            "sources": ["https://example.com/headlines/oil"],
        },
        {
            "title": "Treasury yields dip as buyers return to duration",
            "topic": "Rates",
            "tickers": [],
            "impact": "medium",
            "one_line_take": "Long-end yields retreat with softer inflation expectations.",
            "sources": ["https://example.com/headlines/yields"],
        },
        {
            "title": "Semiconductor stocks extend rebound on upbeat demand checks",
            "topic": "Semiconductors",
            "tickers": ["NVDA"],
            "impact": "high",
            "one_line_take": "Channel checks point to resilient AI server demand.",
            "sources": ["https://example.com/headlines/semis"],
        },
        {
            "title": "Large-cap tech leads premarket as cloud spend stabilizes",
            "topic": "Technology",
            "tickers": ["MSFT"],
            "impact": "medium",
            "one_line_take": "Enterprise IT budgets show incremental improvement.",
            "sources": ["https://example.com/headlines/tech"],
        },
        {
            "title": "Banks mixed after weekend headlines on regional liquidity",
            "topic": "Financials",
            "tickers": [],
            "impact": "low",
            "one_line_take": "Regulatory commentary keeps attention on funding trends.",
            "sources": ["https://example.com/headlines/banks"],
        },
        {
            "title": "Consumer discretionary names steady ahead of key retail earnings",
            "topic": "Consumer",
            "tickers": ["AMZN"],
            "impact": "low",
            "one_line_take": "Investors watch guidance for demand normalization.",
            "sources": ["https://example.com/headlines/retail"],
        },
        {
            "title": "Gold flat as dollar drifts lower",
            "topic": "Commodities",
            "tickers": [],
            "impact": "low",
            "one_line_take": "Safe-haven demand muted with stable risk tone.",
            "sources": ["https://example.com/headlines/gold"],
        },
        {
            "title": "Asia equities mixed after export data surprise",
            "topic": "Global Markets",
            "tickers": [],
            "impact": "medium",
            "one_line_take": "Regional data keeps focus on growth momentum.",
            "sources": ["https://example.com/headlines/asia"],
        },
        {
            "title": "European stocks open higher as banks and autos gain",
            "topic": "Global Markets",
            "tickers": [],
            "impact": "medium",
            "one_line_take": "Cyclical sectors lead early trading in Europe.",
            "sources": ["https://example.com/headlines/europe"],
        },
        {
            "title": "Volatility index drifts lower into data-heavy week",
            "topic": "Derivatives",
            "tickers": [],
            "impact": "low",
            "one_line_take": "Options markets price modest near-term swings.",
            "sources": ["https://example.com/headlines/vix"],
        },
        {
            "title": "Dollar softens against majors as rate cuts re-priced",
            "topic": "FX",
            "tickers": [],
            "impact": "medium",
            "one_line_take": "FX markets react to shifting policy expectations.",
            "sources": ["https://example.com/headlines/dollar"],
        },
    ]

    watchlist_items = []
    for ticker in watchlist:
        watchlist_items.append(
            {
                "title": f"{ticker} in focus as investors position ahead of catalysts",
                "topic": "Watchlist",
                "tickers": [ticker],
                "impact": "medium",
                "one_line_take": "Pre-event positioning lifts volumes and attention.",
                "sources": [f"https://example.com/headlines/{ticker.lower()}"],
            }
        )

    combined = base + watchlist_items
    return combined[:25]


def _build_mock_brief(
    research: dict, date_str: str, tz: str, watchlist: Iterable[str]
) -> str:
    watchlist_text = ", ".join([w for w in watchlist if w]) or "None"
    lines = [
        f"# Morning Market Brief - {date_str} ({tz})",
        "",
        f"_As of {research.get('as_of', 'Not specified')}_",
        "",
        "## Quick Takeaways",
    ]

    headlines = research.get("headlines", [])
    for headline in headlines[:3]:
        lines.append(f"- {headline.get('one_line_take', 'Not specified')}")

    lines.extend(
        [
            "",
            "## Earnings (Today + Next 7 Days)",
        ]
    )
    for item in research.get("earnings", []):
        lines.append(
            f"- {item.get('date', 'Not specified')}: {item.get('company', 'Unknown')} "
            f"({item.get('ticker', 'N/A')}) - {item.get('time_hint', 'Time TBD')}"
        )

    lines.extend(["", "## Macro & Events (Next 7 Days)"])
    for item in research.get("events", []):
        lines.append(
            f"- {item.get('date_time_local', 'Not specified')}: {item.get('event', 'Event')} "
            f"({item.get('region', 'Region')})"
        )

    lines.extend(["", "## Headlines (Last ~18 Hours)"])
    for item in headlines:
        tickers = ", ".join(item.get("tickers", []))
        ticker_text = f" [{tickers}]" if tickers else ""
        lines.append(
            f"- {item.get('title', 'Headline')}{ticker_text} - {item.get('one_line_take', '')}"
        )

    weekly = research.get("weekly_context", {})
    lines.extend(["", "## Weekly Context (Last 5 Trading Days)", "### Themes"])
    for theme in weekly.get("themes", []):
        lines.append(f"- {theme.get('theme', 'Theme')}")

    lines.append("### Market Moves")
    for move in weekly.get("market_moves", []):
        lines.append(
            f"- {move.get('asset', 'Asset')}: {move.get('move', 'Move')} "
            f"({move.get('window', 'Window')})"
        )

    ms = research.get("market_state", {})
    vol = ms.get("volatility", {})
    fut = ms.get("futures", {})
    rfx = ms.get("rates_fx_oil", {})
    opt = ms.get("options_positioning", {})

    lines.extend(["", "## Market State"])
    lines.append(
        f"- **SPY last:** {ms.get('spy', {}).get('last', 'N/A')} | "
        f"**ES overnight:** {fut.get('overnight_low', 'N/A')}–{fut.get('overnight_high', 'N/A')} "
        f"({fut.get('es_change', 'N/A')})"
    )
    lines.append(
        f"- **VIX:** {vol.get('vix', 'N/A')} ({vol.get('vix_change', 'N/A')}) | "
        f"**Term structure:** {vol.get('term_structure', 'N/A')} | "
        f"**Expected move 1d:** {vol.get('expected_move_1d', 'N/A')}"
    )
    lines.append(
        f"- **US 10Y:** {rfx.get('us10y', 'N/A')} ({rfx.get('us10y_change', 'N/A')}) | "
        f"**DXY:** {rfx.get('dxy', 'N/A')} ({rfx.get('dxy_change', 'N/A')}) | "
        f"**WTI:** {rfx.get('wti', 'N/A')} ({rfx.get('wti_change', 'N/A')})"
    )
    if opt.get("put_wall") != "Not specified" or opt.get("call_wall") != "Not specified":
        lines.append(
            f"- **Options:** put wall {opt.get('put_wall', 'N/A')} | "
            f"call wall {opt.get('call_wall', 'N/A')} | "
            f"gamma flip {opt.get('gamma_flip', 'N/A')}"
        )

    lines.extend([
        "",
        "## SPY Tomorrow: Scenario Map (Not a recommendation)",
        "- If CPI prints in-line and 10Y yield holds steady, expect range-bound "
        "action near prior close with VIX drifting lower.",
        "- If yields spike on a hot CPI print, SPY likely tests overnight lows "
        "and may approach the put wall; watch for dealer hedging flows.",
        "- If inflation comes in soft and futures hold above the gamma flip level, "
        "upside toward the call wall is possible with breadth broadening.",
    ])

    lines.extend(["", f"_Watchlist: {watchlist_text}_"])

    return "\n".join(lines)


def _shift_date(date_obj: date_type, days: int) -> str:
    shifted = date_type.fromordinal(date_obj.toordinal() + days)
    return shifted.strftime("%Y-%m-%d")


def _shift_datetime(
    date_obj: date_type, days: int, hour: int, minute: int, tzinfo: ZoneInfo
) -> str:
    shifted = date_type.fromordinal(date_obj.toordinal() + days)
    dt = datetime.combine(shifted, time(hour, minute), tzinfo=tzinfo)
    return dt.isoformat()
