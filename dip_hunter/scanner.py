"""Universe scanning and dip scoring engine."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yfinance as yf

from .config import (
    BOUNCE_WEIGHT,
    DEPTH_WEIGHT,
    HISTORY_PERIOD,
    SCAN_SLEEP_SEC,
    UNIVERSE_PATH,
)
from .schemas import StockScan

logger = logging.getLogger(__name__)

# Sector ETF mapping for relative strength
SECTOR_ETFS = {
    "Technology": "XLK",
    "Semiconductors": "SMH",
    "Healthcare": "XLV",
    "Financials": "XLF",
    "Consumer": "XLY",
    "Industrials": "XLI",
    "Energy": "XLE",
    "REITs": "XLRE",
    "Utilities": "XLU",
    "Communication": "XLC",
}


def load_universe(path: Path | None = None) -> Dict[str, List[str]]:
    """Load stock universe from JSON. Returns {sector: [tickers]}."""
    path = path or UNIVERSE_PATH
    with open(path) as f:
        data = json.load(f)
    return data.get("sectors", {})


def compute_rsi(closes: pd.Series, period: int = 14) -> Optional[float]:
    """Compute RSI using standard Wilder smoothing."""
    if len(closes) < period + 1:
        return None
    delta = closes.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean().iloc[-1]
    avg_loss = loss.rolling(window=period, min_periods=period).mean().iloc[-1]
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100.0 - (100.0 / (1.0 + rs)), 2)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(value, hi))


def compute_dip_score(
    *,
    return_5d: Optional[float],
    return_20d: Optional[float],
    dist_50sma: Optional[float],
    dist_from_52w_high: Optional[float],
    rsi: Optional[float],
    volume_ratio: float,
    dist_20sma: float,
    sector_rel: Optional[float],
) -> tuple[float, float, float, list[str]]:
    """Compute depth, bounce, and composite dip scores. Returns (depth, bounce, dip, signals)."""
    signals: list[str] = []

    # --- Depth score (0-10) ---
    d_5d = _clamp(abs(min(return_5d or 0, 0)) / 2, 0, 3)
    d_20d = _clamp(abs(min(return_20d or 0, 0)) / 3, 0, 3)
    d_50sma = _clamp(abs(min(dist_50sma or 0, 0)) / 2, 0, 2)
    d_52w = _clamp(abs(dist_from_52w_high or 0) / 10, 0, 2) if dist_from_52w_high else 0
    depth = d_5d + d_20d + d_50sma + d_52w

    if (return_5d or 0) < -5:
        signals.append("5d_crash")
    elif (return_5d or 0) < -3:
        signals.append("5d_selloff")
    if (return_20d or 0) < -10:
        signals.append("deeply_oversold")
    elif (return_20d or 0) < -5:
        signals.append("oversold")
    if (dist_50sma or 0) < -3:
        signals.append("well_below_50sma")
    if (dist_from_52w_high or 0) > 15:
        signals.append("near_52w_low")

    # --- Bounce score (0-10) ---
    b_rsi = _clamp((40 - (rsi or 50)) / 4, 0, 3)
    b_vol = _clamp((volume_ratio - 1) * 2, 0, 2)
    b_20sma = _clamp(abs(min(dist_20sma, 0)) / 1.5, 0, 2)
    b_sector = _clamp((sector_rel or 0) / 3, 0, 2)
    b_mrev = 1.0 if (return_20d or 0) < -8 and (rsi or 50) < 35 else 0
    bounce = b_rsi + b_vol + b_20sma + b_sector + b_mrev

    if (rsi or 50) < 25:
        signals.append("rsi_deeply_oversold")
    elif (rsi or 50) < 30:
        signals.append("rsi_oversold")
    if volume_ratio > 1.5:
        signals.append("high_volume")
    if dist_20sma < -3:
        signals.append("well_below_20sma")
    elif dist_20sma < -1:
        signals.append("below_20sma")

    dip = round(depth * DEPTH_WEIGHT + bounce * BOUNCE_WEIGHT, 2)
    return round(depth, 2), round(bounce, 2), dip, signals


def scan_stock(
    ticker: str,
    sector: str,
    sector_returns: Dict[str, float],
) -> Optional[StockScan]:
    """Fetch data for one stock and compute dip score."""
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period=HISTORY_PERIOD, interval="1d")
        if hist.empty or len(hist) < 20:
            return None
        if isinstance(hist.columns, pd.MultiIndex):
            hist = hist.droplevel("Ticker", axis=1)

        close = float(hist["Close"].iloc[-1])

        # Returns
        ret_5d = None
        if len(hist) >= 6:
            ret_5d = round((close / float(hist["Close"].iloc[-6]) - 1) * 100, 2)
        ret_20d = None
        if len(hist) >= 21:
            ret_20d = round((close / float(hist["Close"].iloc[-21]) - 1) * 100, 2)
        ret_60d = None
        if len(hist) >= 61:
            ret_60d = round((close / float(hist["Close"].iloc[-61]) - 1) * 100, 2)

        # SMAs
        sma20 = float(hist["Close"].rolling(20).mean().iloc[-1])
        dist_20sma = round((close - sma20) / sma20 * 100, 2)

        dist_50sma = None
        if len(hist) >= 50:
            sma50 = float(hist["Close"].rolling(50).mean().iloc[-1])
            dist_50sma = round((close - sma50) / sma50 * 100, 2)

        # 52-week high approximation (use available history)
        high_all = float(hist["High"].max())
        dist_52w = round((1 - close / high_all) * 100, 2) if high_all > 0 else None

        # RSI
        rsi = compute_rsi(hist["Close"])

        # Volume ratio
        avg_vol = float(hist["Volume"].rolling(20).mean().iloc[-1])
        vol_ratio = round(float(hist["Volume"].iloc[-1]) / avg_vol, 2) if avg_vol > 0 else 1.0

        # Sector relative strength
        sector_ret = sector_returns.get(sector)
        sector_rel = None
        if sector_ret is not None and ret_20d is not None:
            sector_rel = round(sector_ret - ret_20d, 2)

        depth, bounce, dip, signals = compute_dip_score(
            return_5d=ret_5d,
            return_20d=ret_20d,
            dist_50sma=dist_50sma,
            dist_from_52w_high=dist_52w,
            rsi=rsi,
            volume_ratio=vol_ratio,
            dist_20sma=dist_20sma,
            sector_rel=sector_rel,
        )

        return StockScan(
            ticker=ticker,
            sector=sector,
            price=round(close, 2),
            return_5d_pct=ret_5d,
            return_20d_pct=ret_20d,
            return_60d_pct=ret_60d,
            dist_from_20sma_pct=dist_20sma,
            dist_from_50sma_pct=dist_50sma,
            dist_from_52w_high_pct=dist_52w,
            rsi_14=rsi,
            volume_ratio=vol_ratio,
            sector_return_20d_pct=sector_ret,
            depth_score=depth,
            bounce_score=bounce,
            dip_score=dip,
            signals=signals,
        )
    except Exception as e:
        logger.warning("Failed to scan %s: %s", ticker, e)
        return None


def fetch_sector_returns() -> Dict[str, float]:
    """Fetch 20-day returns for sector ETFs."""
    returns: Dict[str, float] = {}
    for sector, etf in SECTOR_ETFS.items():
        try:
            hist = yf.Ticker(etf).history(period="2mo", interval="1d")
            if hist.empty or len(hist) < 21:
                continue
            if isinstance(hist.columns, pd.MultiIndex):
                hist = hist.droplevel("Ticker", axis=1)
            close = float(hist["Close"].iloc[-1])
            close_20d = float(hist["Close"].iloc[-21])
            returns[sector] = round((close / close_20d - 1) * 100, 2)
            time.sleep(SCAN_SLEEP_SEC)
        except Exception as e:
            logger.warning("Failed to fetch sector ETF %s: %s", etf, e)
    return returns


def fetch_market_context() -> Dict[str, float]:
    """Fetch SPY price/change and VIX for market context."""
    result = {"spy_price": 0.0, "spy_change_pct": 0.0, "vix": 0.0}
    try:
        spy = yf.Ticker("SPY").history(period="5d", interval="1d")
        if not spy.empty:
            if isinstance(spy.columns, pd.MultiIndex):
                spy = spy.droplevel("Ticker", axis=1)
            close = float(spy["Close"].iloc[-1])
            prev = float(spy["Close"].iloc[-2]) if len(spy) >= 2 else close
            result["spy_price"] = round(close, 2)
            result["spy_change_pct"] = round((close / prev - 1) * 100, 2)
    except Exception as e:
        logger.warning("Failed to fetch SPY: %s", e)

    time.sleep(SCAN_SLEEP_SEC)

    try:
        vix = yf.Ticker("^VIX").history(period="5d", interval="1d")
        if not vix.empty:
            if isinstance(vix.columns, pd.MultiIndex):
                vix = vix.droplevel("Ticker", axis=1)
            result["vix"] = round(float(vix["Close"].iloc[-1]), 2)
    except Exception as e:
        logger.warning("Failed to fetch VIX: %s", e)

    return result


def scan_universe(
    universe: Dict[str, List[str]],
    sector_returns: Dict[str, float] | None = None,
) -> List[StockScan]:
    """Scan all stocks in the universe. Returns sorted by dip_score descending."""
    if sector_returns is None:
        sector_returns = fetch_sector_returns()

    scans: list[StockScan] = []
    for sector, tickers in universe.items():
        for ticker in tickers:
            result = scan_stock(ticker, sector, sector_returns)
            if result:
                scans.append(result)
            time.sleep(SCAN_SLEEP_SEC)

    scans.sort(key=lambda s: s.dip_score, reverse=True)
    logger.info("Scanned %d stocks. Top dip: %s (%.2f)",
                len(scans),
                scans[0].ticker if scans else "N/A",
                scans[0].dip_score if scans else 0)
    return scans


def fetch_live_prices(tickers: List[str]) -> Dict[str, float]:
    """Fetch current prices for a list of tickers."""
    prices: Dict[str, float] = {}
    for ticker in tickers:
        try:
            hist = yf.Ticker(ticker).history(period="1d")
            if not hist.empty:
                if isinstance(hist.columns, pd.MultiIndex):
                    hist = hist.droplevel("Ticker", axis=1)
                prices[ticker] = round(float(hist["Close"].iloc[-1]), 2)
            time.sleep(SCAN_SLEEP_SEC)
        except Exception as e:
            logger.warning("Failed to fetch price for %s: %s", ticker, e)
    return prices


# ---------------------------------------------------------------------------
# Mock data for testing
# ---------------------------------------------------------------------------
def build_mock_scans() -> List[StockScan]:
    """Hardcoded scan results for mock mode."""
    return [
        StockScan(ticker="MU", sector="AI Infrastructure", price=88.50,
                  return_5d_pct=-6.2, return_20d_pct=-14.5, return_60d_pct=-18.0,
                  dist_from_20sma_pct=-5.8, dist_from_50sma_pct=-8.2,
                  dist_from_52w_high_pct=30.0, rsi_14=22.5, volume_ratio=2.2,
                  sector_return_20d_pct=-2.0,
                  depth_score=8.2, bounce_score=8.5, dip_score=8.32,
                  signals=["5d_crash", "deeply_oversold", "rsi_deeply_oversold", "high_volume", "well_below_20sma"]),
        StockScan(ticker="MRVL", sector="AI Infrastructure", price=58.20,
                  return_5d_pct=-4.5, return_20d_pct=-12.0, return_60d_pct=-20.0,
                  dist_from_20sma_pct=-4.2, dist_from_50sma_pct=-7.5,
                  dist_from_52w_high_pct=35.0, rsi_14=25.0, volume_ratio=2.1,
                  sector_return_20d_pct=2.0,
                  depth_score=7.8, bounce_score=8.0, dip_score=7.88,
                  signals=["5d_selloff", "deeply_oversold", "rsi_deeply_oversold", "high_volume", "near_52w_low"]),
        StockScan(ticker="AMD", sector="AI Infrastructure", price=142.00,
                  return_5d_pct=-5.8, return_20d_pct=-11.5, return_60d_pct=-15.0,
                  dist_from_20sma_pct=-4.5, dist_from_50sma_pct=-6.0,
                  dist_from_52w_high_pct=22.0, rsi_14=26.0, volume_ratio=1.8,
                  sector_return_20d_pct=1.5,
                  depth_score=7.5, bounce_score=7.8, dip_score=7.62,
                  signals=["5d_crash", "oversold", "rsi_oversold", "high_volume", "well_below_20sma"]),
        StockScan(ticker="DELL", sector="AI-Powered Hardware", price=95.00,
                  return_5d_pct=-3.8, return_20d_pct=-9.0, return_60d_pct=-12.0,
                  dist_from_20sma_pct=-3.5, dist_from_50sma_pct=-5.0,
                  dist_from_52w_high_pct=18.0, rsi_14=30.0, volume_ratio=1.5,
                  sector_return_20d_pct=0.5,
                  depth_score=6.2, bounce_score=5.5, dip_score=5.92,
                  signals=["oversold", "rsi_oversold", "below_20sma"]),
        StockScan(ticker="SNOW", sector="Cloud & AI Platforms", price=135.00,
                  return_5d_pct=-3.5, return_20d_pct=-8.0, return_60d_pct=-14.0,
                  dist_from_20sma_pct=-3.0, dist_from_50sma_pct=-4.5,
                  dist_from_52w_high_pct=20.0, rsi_14=33.0, volume_ratio=1.3,
                  sector_return_20d_pct=1.5,
                  depth_score=5.5, bounce_score=4.8, dip_score=5.22,
                  signals=["oversold", "below_20sma"]),
        StockScan(ticker="AMZN", sector="Consumer", price=185.00,
                  return_5d_pct=2.0, return_20d_pct=3.5, return_60d_pct=8.0,
                  dist_from_20sma_pct=1.5, dist_from_50sma_pct=2.0,
                  dist_from_52w_high_pct=5.0, rsi_14=55.0, volume_ratio=0.9,
                  depth_score=0.5, bounce_score=0.3, dip_score=0.42,
                  signals=[]),
        StockScan(ticker="UNH", sector="Healthcare", price=475.00,
                  return_5d_pct=-0.7, return_20d_pct=-3.0, return_60d_pct=-5.0,
                  dist_from_20sma_pct=-1.2, dist_from_50sma_pct=-2.0,
                  dist_from_52w_high_pct=10.0, rsi_14=42.0, volume_ratio=1.0,
                  depth_score=2.5, bounce_score=1.5, dip_score=2.10,
                  signals=["below_20sma"]),
        StockScan(ticker="QCOM", sector="Semiconductors", price=162.50,
                  return_5d_pct=0.3, return_20d_pct=-1.5, return_60d_pct=2.0,
                  dist_from_20sma_pct=0.5, rsi_14=48.0, volume_ratio=0.95,
                  depth_score=0.5, bounce_score=0.2, dip_score=0.38,
                  signals=[]),
        StockScan(ticker="CVS", sector="Healthcare", price=55.00,
                  return_5d_pct=0.5, return_20d_pct=1.0, return_60d_pct=-2.0,
                  dist_from_20sma_pct=0.8, rsi_14=50.0, volume_ratio=0.85,
                  depth_score=0.3, bounce_score=0.1, dip_score=0.22,
                  signals=[]),
        StockScan(ticker="AMAT", sector="AI Infrastructure", price=155.00,
                  return_5d_pct=-3.0, return_20d_pct=-8.5, return_60d_pct=-14.0,
                  dist_from_20sma_pct=-3.2, dist_from_50sma_pct=-5.0,
                  dist_from_52w_high_pct=20.0, rsi_14=29.0, volume_ratio=1.4,
                  sector_return_20d_pct=-2.0,
                  depth_score=6.0, bounce_score=5.5, dip_score=5.80,
                  signals=["oversold", "rsi_oversold", "well_below_20sma"]),
    ]


def build_mock_market_context() -> Dict[str, float]:
    return {"spy_price": 520.30, "spy_change_pct": -0.80, "vix": 22.50}


def build_mock_live_prices() -> Dict[str, float]:
    return {
        "AMZN": 185.00, "UNH": 475.00, "QCOM": 162.50, "CVS": 55.00,
        "MU": 88.50, "AMD": 142.00, "MRVL": 58.20, "DELL": 95.00,
        "SNOW": 135.00, "AMAT": 155.00,
    }
