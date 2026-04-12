"""Buy/sell/hold/rotate signal generation."""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, List

from .confidence import (
    HISTORICAL_STATS,
    compute_confidence,
    get_tier,
    should_skip_ticker,
)
from .config import (
    DIP_SCORE_ACTIONABLE,
    MAX_POSITION_PCT,
    MAX_POSITIONS,
    MIN_POSITION_PCT,
    STOP_LOSS_PCT,
    TAKE_PROFIT_PCT,
)
from .schemas import (
    BuySignal,
    HoldingStatus,
    PortfolioState,
    RotateSignal,
    SellSignal,
    StockScan,
)

logger = logging.getLogger(__name__)


def evaluate_holdings(
    portfolio: PortfolioState,
    live_prices: Dict[str, float],
    today: str = "",
) -> List[HoldingStatus]:
    """Compute current P&L and signal for each holding."""
    today = today or datetime.now().strftime("%Y-%m-%d")
    statuses: list[HoldingStatus] = []

    for h in portfolio.holdings:
        price = live_prices.get(h.ticker, h.avg_cost)
        pnl = round((price - h.avg_cost) * h.shares, 2)
        pnl_pct = round((price / h.avg_cost - 1) * 100, 2) if h.avg_cost > 0 else 0.0

        try:
            entry = datetime.strptime(h.entry_date, "%Y-%m-%d")
            days_held = (datetime.strptime(today, "%Y-%m-%d") - entry).days
        except ValueError:
            days_held = 0

        # Determine signal
        if pnl_pct >= TAKE_PROFIT_PCT:
            signal = "TAKE_PROFIT"
            reason = f"Up {pnl_pct:+.1f}% — bounce target reached. Consider taking profit."
        elif pnl_pct <= STOP_LOSS_PCT:
            signal = "STOP_LOSS"
            reason = f"Down {pnl_pct:+.1f}% — below stop loss. Thesis may be broken."
        elif pnl_pct >= TAKE_PROFIT_PCT * 0.7:
            signal = "WATCH"
            reason = f"Up {pnl_pct:+.1f}% — approaching take-profit target."
        else:
            signal = "HOLD"
            reason = f"P&L {pnl_pct:+.1f}% — within range, hold position."

        statuses.append(HoldingStatus(
            ticker=h.ticker,
            shares=h.shares,
            avg_cost=h.avg_cost,
            entry_date=h.entry_date,
            current_price=price,
            unrealized_pnl=pnl,
            unrealized_pnl_pct=pnl_pct,
            days_held=days_held,
            signal=signal,
            signal_reason=reason,
        ))

    return statuses


def generate_buy_signals(
    scans: List[StockScan],
    portfolio: PortfolioState,
    max_signals: int = 3,
) -> List[BuySignal]:
    """Top dip candidates not already held, with confidence scoring.

    Filters out C-tier stocks (poor bouncers) and ranks by confidence × dip_score.
    """
    held_tickers = {h.ticker for h in portfolio.holdings}
    open_slots = portfolio.max_positions - len(portfolio.holdings)
    available = portfolio.capital_available

    if open_slots <= 0 or available <= 0:
        return []

    # Filter: actionable dip score, not held, skip C-tier
    candidates = [
        s for s in scans
        if s.ticker not in held_tickers
        and s.dip_score >= DIP_SCORE_ACTIONABLE
        and not should_skip_ticker(s.ticker)
    ]

    # Score each candidate with confidence and sort by confidence × dip_score
    scored: list[tuple[StockScan, int, str, str]] = []
    for scan in candidates:
        conf, level, explanation = compute_confidence(scan)
        scored.append((scan, conf, level, explanation))

    # Sort: confidence first, then dip_score as tiebreaker
    scored.sort(key=lambda x: (x[1], x[0].dip_score), reverse=True)

    signals: list[BuySignal] = []
    for i, (scan, conf, level, explanation) in enumerate(scored[:max_signals], 1):
        shares, dollars = compute_position_size(
            scan.price, available, portfolio.capital_total,
        )
        if shares <= 0:
            continue

        stop_price = round(scan.price * (1 + STOP_LOSS_PCT / 100), 2)
        tp_price = round(scan.price * (1 + TAKE_PROFIT_PCT / 100), 2)
        entry_target = round(scan.price * 0.995, 2)  # 0.5% below market

        # Historical stats
        stats = HISTORICAL_STATS.get(scan.ticker)
        tier = get_tier(scan.ticker)
        win_rate = stats.win_rate if stats else None
        avg_bounce = stats.avg_bounce if stats else None

        # Expected bounce range
        if stats and stats.tier == "A":
            expected = f"+10-{min(int(stats.avg_bounce), 40)}% in 20-40 days"
        elif stats and stats.tier == "B":
            expected = f"+8-{min(int(stats.avg_bounce), 25)}% in 20-40 days"
        else:
            expected = "uncertain — no strong historical pattern"

        top_signals = scan.signals[:4]
        reasoning = (
            f"{scan.ticker} ({scan.sector}): dip score {scan.dip_score:.1f}/10. "
            f"Down {scan.return_20d_pct or 0:+.1f}% over 20d, "
            f"RSI {scan.rsi_14 or 0:.0f}, "
            f"{scan.dist_from_20sma_pct:+.1f}% from 20-SMA. "
            f"Signals: {', '.join(top_signals)}."
        )

        signals.append(BuySignal(
            rank=i,
            ticker=scan.ticker,
            sector=scan.sector,
            current_price=scan.price,
            dip_score=scan.dip_score,
            entry_target=entry_target,
            stop_loss=stop_price,
            take_profit=tp_price,
            position_size_shares=shares,
            position_size_dollars=dollars,
            signals=scan.signals,
            reasoning=reasoning,
            confidence=conf,
            confidence_level=level,
            confidence_reasoning=explanation,
            tier=tier,
            historical_win_rate=win_rate,
            historical_avg_bounce=avg_bounce,
            expected_bounce_range=expected,
        ))

    return signals


def generate_sell_signals(holdings_status: List[HoldingStatus]) -> List[SellSignal]:
    """Generate sell signals for holdings at take-profit or stop-loss."""
    signals: list[SellSignal] = []
    for hs in holdings_status:
        if hs.signal in ("TAKE_PROFIT", "STOP_LOSS"):
            signals.append(SellSignal(
                ticker=hs.ticker,
                action=hs.signal,
                current_price=hs.current_price,
                avg_cost=hs.avg_cost,
                pnl_pct=hs.unrealized_pnl_pct,
                reasoning=hs.signal_reason,
            ))
    return signals


def generate_rotate_signals(
    sell_signals: List[SellSignal],
    buy_signals: List[BuySignal],
) -> List[RotateSignal]:
    """Pair sell signals with buy candidates for rotation recommendations."""
    rotations: list[RotateSignal] = []
    available_buys = list(buy_signals)

    for sell in sell_signals:
        if not available_buys:
            break
        buy = available_buys.pop(0)
        rotations.append(RotateSignal(
            sell_ticker=sell.ticker,
            sell_reason=sell.action,
            sell_pnl_pct=sell.pnl_pct,
            buy_ticker=buy.ticker,
            buy_dip_score=buy.dip_score,
            buy_entry_target=buy.entry_target,
            reasoning=(
                f"Sell {sell.ticker} ({sell.action}, {sell.pnl_pct:+.1f}%) → "
                f"Buy {buy.ticker} (dip score {buy.dip_score:.1f}, "
                f"entry ${buy.entry_target:.2f}). "
                f"Rotate capital into fresh dip opportunity."
            ),
        ))

    return rotations


def compute_position_size(
    price: float,
    available_capital: float,
    total_capital: float,
) -> tuple[int, float]:
    """Compute shares and dollar amount respecting position limits."""
    max_dollars = total_capital * MAX_POSITION_PCT
    min_dollars = total_capital * MIN_POSITION_PCT
    target = min(max_dollars, available_capital)

    if target < min_dollars:
        # Not enough capital for minimum position
        if available_capital >= price:
            target = available_capital
        else:
            return 0, 0.0

    shares = int(target / price)
    if shares <= 0:
        return 0, 0.0

    dollars = round(shares * price, 2)
    return shares, dollars
