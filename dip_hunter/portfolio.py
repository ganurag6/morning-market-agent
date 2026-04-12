"""Portfolio state management — load, save, seed, record trades."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict

from .config import PORTFOLIO_PATH
from .schemas import Holding, PortfolioState, TradeRecord

logger = logging.getLogger(__name__)


def load_portfolio(path: Path | None = None) -> PortfolioState:
    """Load portfolio from JSON file. Returns empty state if missing."""
    path = path or PORTFOLIO_PATH
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        logger.info("Loaded portfolio from %s (%d holdings)", path, len(data.get("holdings", [])))
        return PortfolioState(**data)
    logger.info("No portfolio file found at %s — starting fresh.", path)
    return PortfolioState()


def save_portfolio(portfolio: PortfolioState, path: Path | None = None) -> None:
    """Write portfolio state to JSON."""
    path = path or PORTFOLIO_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(portfolio.model_dump(), f, indent=2, default=str)
        f.write("\n")
    logger.info("Saved portfolio to %s", path)


def seed_portfolio(
    holdings: list[dict],
    live_prices: Dict[str, float],
    capital_total: float = 15000.0,
    date: str = "",
) -> PortfolioState:
    """Create initial portfolio from user's current positions using live prices as cost basis."""
    date = date or datetime.now().strftime("%Y-%m-%d")
    parsed = []
    deployed = 0.0
    for h in holdings:
        ticker = h["ticker"]
        shares = h["shares"]
        price = live_prices.get(ticker, h.get("avg_cost", 0.0))
        cost = price * shares
        deployed += cost
        parsed.append(Holding(
            ticker=ticker,
            shares=shares,
            avg_cost=round(price, 2),
            entry_date=date,
            strategy_notes=h.get("strategy_notes", ""),
        ))

    return PortfolioState(
        updated_at=datetime.now().isoformat(timespec="seconds"),
        capital_total=capital_total,
        capital_deployed=round(deployed, 2),
        capital_available=round(capital_total - deployed, 2),
        holdings=parsed,
    )


def record_buy(
    portfolio: PortfolioState,
    ticker: str,
    shares: int,
    price: float,
    date: str,
) -> PortfolioState:
    """Add a new holding and record the trade."""
    cost = round(shares * price, 2)

    # Check if already holding — average in
    existing = next((h for h in portfolio.holdings if h.ticker == ticker), None)
    if existing:
        total_shares = existing.shares + shares
        total_cost = existing.avg_cost * existing.shares + price * shares
        existing.avg_cost = round(total_cost / total_shares, 2)
        existing.shares = total_shares
    else:
        portfolio.holdings.append(Holding(
            ticker=ticker,
            shares=shares,
            avg_cost=round(price, 2),
            entry_date=date,
        ))

    portfolio.capital_deployed = round(portfolio.capital_deployed + cost, 2)
    portfolio.capital_available = round(portfolio.capital_available - cost, 2)

    portfolio.trade_history.append(TradeRecord(
        ticker=ticker, action="BUY", shares=shares,
        price=price, date=date, reason="dip_buy",
    ))
    return portfolio


def record_sell(
    portfolio: PortfolioState,
    ticker: str,
    shares: int,
    price: float,
    date: str,
    reason: str = "take_profit",
) -> PortfolioState:
    """Remove/reduce a holding and record the trade with P&L."""
    holding = next((h for h in portfolio.holdings if h.ticker == ticker), None)
    if not holding:
        logger.warning("Cannot sell %s — not in portfolio.", ticker)
        return portfolio

    sell_shares = min(shares, holding.shares)
    pnl = round((price - holding.avg_cost) * sell_shares, 2)
    proceeds = round(sell_shares * price, 2)
    cost_basis = round(sell_shares * holding.avg_cost, 2)

    if sell_shares >= holding.shares:
        portfolio.holdings = [h for h in portfolio.holdings if h.ticker != ticker]
    else:
        holding.shares -= sell_shares

    portfolio.capital_deployed = round(portfolio.capital_deployed - cost_basis, 2)
    portfolio.capital_available = round(portfolio.capital_available + proceeds, 2)

    portfolio.trade_history.append(TradeRecord(
        ticker=ticker, action="SELL", shares=sell_shares,
        price=price, date=date, reason=reason, pnl=pnl,
    ))
    return portfolio
