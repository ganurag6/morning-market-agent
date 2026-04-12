"""Backtest the dip-hunter strategy over historical data."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf

from .config import SCAN_SLEEP_SEC
from .scanner import compute_dip_score, compute_rsi, _clamp

logger = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    ticker: str
    buy_date: str
    buy_price: float
    shares: int
    cost: float
    sell_date: str = ""
    sell_price: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    exit_reason: str = ""
    days_held: int = 0


@dataclass
class BacktestResult:
    start_date: str
    end_date: str
    starting_capital: float
    ending_capital: float
    total_return: float
    total_return_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win_pct: float
    avg_loss_pct: float
    avg_days_held: float
    max_drawdown_pct: float
    trades: List[BacktestTrade] = field(default_factory=list)
    open_positions: List[BacktestTrade] = field(default_factory=list)
    equity_curve: List[dict] = field(default_factory=list)


def fetch_backtest_data(
    tickers: List[str],
    period: str = "1y",
) -> Dict[str, pd.DataFrame]:
    """Fetch historical daily data for all tickers."""
    data: Dict[str, pd.DataFrame] = {}
    logger.info("Fetching historical data for %d tickers...", len(tickers))
    for ticker in tickers:
        try:
            hist = yf.Ticker(ticker).history(period=period, interval="1d")
            if not hist.empty:
                if isinstance(hist.columns, pd.MultiIndex):
                    hist = hist.droplevel("Ticker", axis=1)
                data[ticker] = hist
            time.sleep(SCAN_SLEEP_SEC)
        except Exception as e:
            logger.warning("Failed to fetch %s: %s", ticker, e)
    logger.info("Fetched data for %d tickers.", len(data))
    return data


def _score_on_date(
    ticker_data: pd.DataFrame,
    idx: int,
    sector_data: Optional[pd.DataFrame] = None,
) -> Optional[float]:
    """Compute dip score for a stock at a specific date index using only past data."""
    if idx < 60:
        return None

    closes = ticker_data["Close"].iloc[:idx + 1]
    close = float(closes.iloc[-1])

    # Returns
    ret_5d = None
    if len(closes) >= 6:
        ret_5d = (close / float(closes.iloc[-6]) - 1) * 100
    ret_20d = None
    if len(closes) >= 21:
        ret_20d = (close / float(closes.iloc[-21]) - 1) * 100

    # SMAs
    sma20 = float(closes.rolling(20).mean().iloc[-1])
    dist_20sma = (close - sma20) / sma20 * 100

    dist_50sma = None
    if len(closes) >= 50:
        sma50 = float(closes.rolling(50).mean().iloc[-1])
        dist_50sma = (close - sma50) / sma50 * 100

    # 52-week high approximation
    highs = ticker_data["High"].iloc[:idx + 1]
    high_period = float(highs.tail(252).max()) if len(highs) >= 252 else float(highs.max())
    dist_52w = (1 - close / high_period) * 100 if high_period > 0 else None

    # RSI
    rsi = compute_rsi(closes, period=14)

    # Volume ratio
    volumes = ticker_data["Volume"].iloc[:idx + 1]
    avg_vol = float(volumes.rolling(20).mean().iloc[-1])
    vol_ratio = float(volumes.iloc[-1]) / avg_vol if avg_vol > 0 else 1.0

    # Sector relative (simplified — use SPY as benchmark)
    sector_rel = None
    if sector_data is not None and len(sector_data) > idx and idx >= 21:
        spy_closes = sector_data["Close"].iloc[:idx + 1]
        spy_ret = (float(spy_closes.iloc[-1]) / float(spy_closes.iloc[-21]) - 1) * 100
        if ret_20d is not None:
            sector_rel = spy_ret - ret_20d

    _, _, dip_score, _ = compute_dip_score(
        return_5d=ret_5d,
        return_20d=ret_20d,
        dist_50sma=dist_50sma,
        dist_from_52w_high=dist_52w,
        rsi=rsi,
        volume_ratio=vol_ratio,
        dist_20sma=dist_20sma,
        sector_rel=sector_rel,
    )
    return dip_score


def run_backtest(
    tickers: List[str],
    starting_capital: float = 10000.0,
    max_per_position: float = 2500.0,
    take_profit_pct: float = 10.0,
    stop_loss_pct: float = -8.0,
    dip_threshold: float = 5.0,
    max_positions: int = 4,
    period: str = "1y",
) -> BacktestResult:
    """Run full backtest over historical data."""

    # Fetch all data upfront
    all_tickers = list(set(tickers + ["SPY"]))
    data = fetch_backtest_data(all_tickers, period=period)
    spy_data = data.get("SPY")

    # Find common date range
    all_dates = None
    for ticker, df in data.items():
        if ticker == "SPY":
            continue
        dates = set(df.index)
        if all_dates is None:
            all_dates = dates
        else:
            all_dates = all_dates.intersection(dates)

    if not all_dates:
        raise RuntimeError("No common dates found across tickers.")

    sorted_dates = sorted(all_dates)
    # Start after 60 days of warmup
    start_idx = 60

    capital = starting_capital
    positions: List[BacktestTrade] = []
    closed_trades: List[BacktestTrade] = []
    equity_curve: List[dict] = []
    peak_equity = starting_capital

    logger.info("Backtesting from %s to %s (%d trading days)",
                sorted_dates[start_idx].strftime("%Y-%m-%d"),
                sorted_dates[-1].strftime("%Y-%m-%d"),
                len(sorted_dates) - start_idx)

    for day_idx in range(start_idx, len(sorted_dates)):
        date = sorted_dates[day_idx]
        date_str = date.strftime("%Y-%m-%d")

        # --- Check exits on existing positions ---
        still_open = []
        for pos in positions:
            ticker_df = data.get(pos.ticker)
            if ticker_df is None:
                still_open.append(pos)
                continue

            # Find this date in ticker's data
            mask = ticker_df.index == date
            if not mask.any():
                still_open.append(pos)
                continue

            current_price = float(ticker_df.loc[mask, "Close"].iloc[0])
            pnl_pct = (current_price / pos.buy_price - 1) * 100

            if pnl_pct >= take_profit_pct:
                # Take profit
                pos.sell_date = date_str
                pos.sell_price = current_price
                pos.pnl = round((current_price - pos.buy_price) * pos.shares, 2)
                pos.pnl_pct = round(pnl_pct, 2)
                pos.exit_reason = "TAKE_PROFIT"
                pos.days_held = (date - datetime.strptime(pos.buy_date, "%Y-%m-%d").replace(tzinfo=date.tzinfo)).days
                capital += pos.shares * current_price
                closed_trades.append(pos)
            elif pnl_pct <= stop_loss_pct:
                # Stop loss
                pos.sell_date = date_str
                pos.sell_price = current_price
                pos.pnl = round((current_price - pos.buy_price) * pos.shares, 2)
                pos.pnl_pct = round(pnl_pct, 2)
                pos.exit_reason = "STOP_LOSS"
                pos.days_held = (date - datetime.strptime(pos.buy_date, "%Y-%m-%d").replace(tzinfo=date.tzinfo)).days
                capital += pos.shares * current_price
                closed_trades.append(pos)
            else:
                still_open.append(pos)

        positions = still_open

        # --- Scan for new entries ---
        if len(positions) < max_positions and capital >= 500:
            scores: list[tuple[str, float, float]] = []
            held_tickers = {p.ticker for p in positions}

            for ticker, df in data.items():
                if ticker == "SPY" or ticker in held_tickers:
                    continue

                # Find index of this date in ticker's data
                try:
                    loc = df.index.get_loc(date)
                except KeyError:
                    continue

                score = _score_on_date(df, loc, spy_data)
                if score is not None and score >= dip_threshold:
                    price = float(df["Close"].iloc[loc])
                    scores.append((ticker, score, price))

            # Sort by score, buy top candidates
            scores.sort(key=lambda x: x[1], reverse=True)

            for ticker, score, price in scores:
                if len(positions) >= max_positions:
                    break
                alloc = min(max_per_position, capital)
                if alloc < price:
                    continue
                shares = int(alloc / price)
                if shares <= 0:
                    continue
                cost = round(shares * price, 2)
                capital -= cost

                positions.append(BacktestTrade(
                    ticker=ticker,
                    buy_date=date_str,
                    buy_price=round(price, 2),
                    shares=shares,
                    cost=cost,
                ))

        # --- Track equity ---
        position_value = 0.0
        for pos in positions:
            ticker_df = data.get(pos.ticker)
            if ticker_df is not None:
                mask = ticker_df.index == date
                if mask.any():
                    position_value += float(ticker_df.loc[mask, "Close"].iloc[0]) * pos.shares
                else:
                    position_value += pos.buy_price * pos.shares

        total_equity = capital + position_value
        peak_equity = max(peak_equity, total_equity)
        drawdown = (total_equity / peak_equity - 1) * 100

        equity_curve.append({
            "date": date_str,
            "equity": round(total_equity, 2),
            "cash": round(capital, 2),
            "positions": len(positions),
            "drawdown_pct": round(drawdown, 2),
        })

    # --- Force-close any remaining positions at last price ---
    last_date = sorted_dates[-1]
    last_date_str = last_date.strftime("%Y-%m-%d")
    for pos in positions:
        ticker_df = data.get(pos.ticker)
        if ticker_df is not None and not ticker_df.empty:
            current_price = float(ticker_df["Close"].iloc[-1])
        else:
            current_price = pos.buy_price
        pos.sell_date = last_date_str
        pos.sell_price = round(current_price, 2)
        pos.pnl = round((current_price - pos.buy_price) * pos.shares, 2)
        pos.pnl_pct = round((current_price / pos.buy_price - 1) * 100, 2)
        pos.exit_reason = "OPEN"
        pos.days_held = (last_date - datetime.strptime(pos.buy_date, "%Y-%m-%d").replace(tzinfo=last_date.tzinfo)).days

    all_trades = closed_trades + positions

    # --- Compute stats ---
    ending_capital = equity_curve[-1]["equity"] if equity_curve else starting_capital
    total_return = ending_capital - starting_capital
    total_return_pct = (ending_capital / starting_capital - 1) * 100

    winners = [t for t in all_trades if t.pnl > 0]
    losers = [t for t in all_trades if t.pnl < 0]
    win_rate = len(winners) / len(all_trades) * 100 if all_trades else 0

    avg_win = sum(t.pnl_pct for t in winners) / len(winners) if winners else 0
    avg_loss = sum(t.pnl_pct for t in losers) / len(losers) if losers else 0
    avg_days = sum(t.days_held for t in all_trades) / len(all_trades) if all_trades else 0

    max_dd = min(e["drawdown_pct"] for e in equity_curve) if equity_curve else 0

    return BacktestResult(
        start_date=sorted_dates[start_idx].strftime("%Y-%m-%d"),
        end_date=last_date_str,
        starting_capital=starting_capital,
        ending_capital=round(ending_capital, 2),
        total_return=round(total_return, 2),
        total_return_pct=round(total_return_pct, 2),
        total_trades=len(all_trades),
        winning_trades=len(winners),
        losing_trades=len(losers),
        win_rate=round(win_rate, 1),
        avg_win_pct=round(avg_win, 2),
        avg_loss_pct=round(avg_loss, 2),
        avg_days_held=round(avg_days, 1),
        max_drawdown_pct=round(max_dd, 2),
        trades=all_trades,
        open_positions=positions,
        equity_curve=equity_curve,
    )


def print_backtest_report(result: BacktestResult) -> str:
    """Generate a markdown report of backtest results."""
    lines = [
        "# Dip Hunter Backtest Results",
        "",
        f"**Period:** {result.start_date} to {result.end_date}",
        "",
        "## Performance Summary",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Starting Capital | ${result.starting_capital:,.2f} |",
        f"| Ending Capital | ${result.ending_capital:,.2f} |",
        f"| **Total Return** | **${result.total_return:+,.2f} ({result.total_return_pct:+.1f}%)** |",
        f"| Total Trades | {result.total_trades} |",
        f"| Winning Trades | {result.winning_trades} |",
        f"| Losing Trades | {result.losing_trades} |",
        f"| **Win Rate** | **{result.win_rate:.1f}%** |",
        f"| Avg Win | +{result.avg_win_pct:.1f}% |",
        f"| Avg Loss | {result.avg_loss_pct:.1f}% |",
        f"| Avg Days Held | {result.avg_days_held:.0f} days |",
        f"| Max Drawdown | {result.max_drawdown_pct:.1f}% |",
        "",
        "## All Trades",
        "| # | Ticker | Buy Date | Buy $ | Sell Date | Sell $ | P&L | P&L% | Days | Exit |",
        "|---|--------|----------|-------|-----------|--------|-----|------|------|------|",
    ]

    for i, t in enumerate(result.trades, 1):
        lines.append(
            f"| {i} | {t.ticker} | {t.buy_date} | ${t.buy_price:.2f} | "
            f"{t.sell_date} | ${t.sell_price:.2f} | ${t.pnl:+,.2f} | "
            f"{t.pnl_pct:+.1f}% | {t.days_held} | {t.exit_reason} |"
        )

    # Monthly equity curve
    lines.extend([
        "",
        "## Monthly Equity Curve",
        "| Month | Equity | Positions | Drawdown |",
        "|-------|--------|-----------|----------|",
    ])
    seen_months = set()
    for e in result.equity_curve:
        month = e["date"][:7]
        if month not in seen_months:
            seen_months.add(month)
            lines.append(
                f"| {month} | ${e['equity']:,.2f} | {e['positions']} | {e['drawdown_pct']:.1f}% |"
            )
    # Always include last day
    if result.equity_curve:
        last = result.equity_curve[-1]
        lines.append(
            f"| {last['date']} (final) | ${last['equity']:,.2f} | {last['positions']} | {last['drawdown_pct']:.1f}% |"
        )

    return "\n".join(lines)
