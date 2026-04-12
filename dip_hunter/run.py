"""CLI entry point and pipeline orchestrator for dip-hunter."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from .brief import build_dip_hunter_brief
from .config import DEFAULT_CAPITAL, PORTFOLIO_PATH, UNIVERSE_PATH
from .portfolio import load_portfolio, save_portfolio, seed_portfolio
from .scanner import (
    build_mock_live_prices,
    build_mock_market_context,
    build_mock_scans,
    fetch_live_prices,
    fetch_market_context,
    load_universe,
    scan_universe,
)
from .schemas import DipHunterBrief
from .signals import (
    evaluate_holdings,
    generate_buy_signals,
    generate_rotate_signals,
    generate_sell_signals,
)
from .tracker import compute_track_record, save_daily_picks, update_outcomes

logger = logging.getLogger(__name__)


def run_dip_hunter(
    *,
    date: str | None = None,
    tz: str = "America/Chicago",
    out_dir: str | Path = "./out",
    mock_mode: bool = False,
    capital: float = DEFAULT_CAPITAL,
    universe_path: str | Path | None = None,
    portfolio_path: str | Path | None = None,
) -> Path:
    """Full dip-hunter daily pipeline."""
    tzinfo = ZoneInfo(tz)
    if date:
        target_date = datetime.strptime(date, "%Y-%m-%d").date()
    else:
        target_date = datetime.now(tzinfo).date()
    date_str = target_date.strftime("%Y-%m-%d")
    as_of = datetime.now(tzinfo).isoformat(timespec="seconds")

    output_dir = Path(out_dir) / date_str
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load universe
    u_path = Path(universe_path) if universe_path else UNIVERSE_PATH
    universe = load_universe(u_path)

    # Load or seed portfolio
    p_path = Path(portfolio_path) if portfolio_path else PORTFOLIO_PATH

    if mock_mode:
        logger.info("Running dip-hunter in MOCK mode.")
        market = build_mock_market_context()
        scans = build_mock_scans()
        live_prices = build_mock_live_prices()

        # Sort mock scans by dip_score
        scans.sort(key=lambda s: s.dip_score, reverse=True)

        # Seed portfolio with mock prices if no file exists
        if not p_path.exists():
            portfolio = seed_portfolio(
                [
                    {"ticker": "AMZN", "shares": 20},
                    {"ticker": "UNH", "shares": 15},
                    {"ticker": "QCOM", "shares": 50},
                    {"ticker": "CVS", "shares": 50},
                ],
                live_prices,
                capital_total=capital,
                date=date_str,
            )
        else:
            portfolio = load_portfolio(p_path)
    else:
        logger.info("Fetching live market data...")
        market = fetch_market_context()

        # Fetch live prices for portfolio holdings first
        portfolio = load_portfolio(p_path)
        if not portfolio.holdings:
            # First run — seed from user's positions
            held_tickers = ["AMZN", "UNH", "QCOM", "CVS"]
            live_prices = fetch_live_prices(held_tickers)
            portfolio = seed_portfolio(
                [{"ticker": t, "shares": s} for t, s in
                 zip(held_tickers, [20, 15, 50, 50])],
                live_prices,
                capital_total=capital,
                date=date_str,
            )
            save_portfolio(portfolio, p_path)
            logger.info("Seeded portfolio with current prices.")
        else:
            held_tickers = [h.ticker for h in portfolio.holdings]
            live_prices = fetch_live_prices(held_tickers)

        logger.info("Scanning stock universe...")
        scans = scan_universe(universe)

        # Add live prices for holdings
        for scan in scans:
            if scan.ticker in live_prices:
                pass  # scanner already fetched live data
        # Ensure we have prices for all holdings
        for h in portfolio.holdings:
            if h.ticker not in live_prices:
                for scan in scans:
                    if scan.ticker == h.ticker:
                        live_prices[h.ticker] = scan.price
                        break

    # Evaluate current holdings
    holdings_status = evaluate_holdings(portfolio, live_prices, today=date_str)

    # Generate signals
    sell_signals = generate_sell_signals(holdings_status)
    buy_signals = generate_buy_signals(scans, portfolio)
    rotate_signals = generate_rotate_signals(sell_signals, buy_signals)

    # Portfolio value
    portfolio_value = sum(
        hs.current_price * hs.shares for hs in holdings_status
    ) + portfolio.capital_available
    total_pnl = sum(hs.unrealized_pnl for hs in holdings_status)
    total_cost = sum(hs.avg_cost * hs.shares for hs in holdings_status)
    total_pnl_pct = round(total_pnl / total_cost * 100, 2) if total_cost > 0 else 0.0

    # --- Track record: update past picks, save today's picks ---
    update_outcomes(live_prices, today=date_str)
    track_record = compute_track_record()

    # Save today's picks
    if buy_signals:
        today_picks = [
            {
                "ticker": b.ticker,
                "price": b.current_price,
                "confidence": b.confidence,
                "confidence_level": b.confidence_level,
                "dip_score": b.dip_score,
                "tier": b.tier,
                "expected_bounce": b.expected_bounce_range,
            }
            for b in buy_signals
        ]
        save_daily_picks(date_str, today_picks)

    # Build brief
    brief = DipHunterBrief(
        as_of=as_of,
        date=date_str,
        spy_price=market["spy_price"],
        spy_change_pct=market["spy_change_pct"],
        vix=market["vix"],
        portfolio_value=round(portfolio_value, 2),
        total_unrealized_pnl=round(total_pnl, 2),
        total_unrealized_pnl_pct=total_pnl_pct,
        holdings_status=holdings_status,
        buy_signals=buy_signals,
        sell_signals=sell_signals,
        rotate_signals=rotate_signals,
        top_dips=scans[:10],
        track_record=track_record,
    )

    # Write outputs
    brief_md = build_dip_hunter_brief(brief)
    md_path = output_dir / "dip_hunter.md"
    with open(md_path, "w") as f:
        f.write(brief_md.rstrip() + "\n")

    json_path = output_dir / "dip_hunter.json"
    with open(json_path, "w") as f:
        json.dump(brief.model_dump(), f, indent=2, default=str)
        f.write("\n")

    logger.info("Wrote dip_hunter.md → %s", md_path)
    logger.info("Wrote dip_hunter.json → %s", json_path)

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"  Dip Hunter Daily Picks: {md_path}")
    print(f"  Holdings:         {len(holdings_status)}")
    print(f"  Portfolio value:  ${portfolio_value:,.2f}")
    print(f"  Unrealized P&L:   ${total_pnl:+,.2f} ({total_pnl_pct:+.1f}%)")
    print(f"  Sell signals:     {len(sell_signals)}")
    if buy_signals:
        for b in buy_signals:
            print(f"  PICK: {b.ticker:<6} {b.confidence_level} ({b.confidence}%) "
                  f"tier={b.tier} score={b.dip_score:.1f}")
    else:
        print("  No picks today")
    print(f"  Rotate signals:   {len(rotate_signals)}")
    if track_record and track_record.get("hit_rate_10pct") is not None:
        print(f"  Track record:     {track_record['hit_rate_10pct']:.0f}% hit rate "
              f"({track_record['total_with_20d_outcome']} measured)")
    print(f"{'=' * 60}\n")

    return md_path


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Dip Hunter — daily stock rotation system")
    parser.add_argument("--date", help="Date YYYY-MM-DD (default: today)")
    parser.add_argument("--tz", default="America/Chicago", help="Timezone")
    parser.add_argument("--out", default="./out", help="Output directory")
    parser.add_argument("--mock", action="store_true", help="Use mock data (no yfinance)")
    parser.add_argument("--capital", type=float, default=DEFAULT_CAPITAL, help="Total capital ($)")
    parser.add_argument("--universe", help="Path to universe JSON")
    parser.add_argument("--portfolio", help="Path to portfolio JSON")
    args = parser.parse_args(argv)

    try:
        run_dip_hunter(
            date=args.date,
            tz=args.tz,
            out_dir=args.out,
            mock_mode=args.mock,
            capital=args.capital,
            universe_path=args.universe,
            portfolio_path=args.portfolio,
        )
        return 0
    except Exception:
        logger.exception("Dip hunter failed.")
        return 1
