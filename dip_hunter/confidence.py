"""Confidence scoring engine based on historical dip-bounce data.

Uses 1-year backtest results (April 2025 – April 2026) to assign
confidence levels to dip-buy candidates.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from .schemas import StockScan


# --- Historical bounce stats from backtest ---
# Format: (dips, avg_bounce_pct, best_bounce_pct, win_rate_10pct)
# win_rate_10pct = % of dips that bounced at least 10%

@dataclass
class TickerStats:
    dips: int
    avg_bounce: float
    best_bounce: float
    win_rate: float  # 0-100, hit rate for 10%+ bounce
    tier: str  # "A", "B", "C"


HISTORICAL_STATS: Dict[str, TickerStats] = {
    # A-tier: avg bounce >30%, win rate 89%+
    "MU":   TickerStats(dips=11, avg_bounce=73.2, best_bounce=118.9, win_rate=100, tier="A"),
    "AMD":  TickerStats(dips=13, avg_bounce=42.9, best_bounce=74.9,  win_rate=100, tier="A"),
    "AMAT": TickerStats(dips=9,  avg_bounce=41.1, best_bounce=68.1,  win_rate=100, tier="A"),
    "MRVL": TickerStats(dips=12, avg_bounce=40.2, best_bounce=74.4,  win_rate=100, tier="A"),
    "CAT":  TickerStats(dips=4,  avg_bounce=35.1, best_bounce=44.0,  win_rate=100, tier="A"),
    "DELL": TickerStats(dips=11, avg_bounce=31.7, best_bounce=67.0,  win_rate=100, tier="A"),
    "SMCI": TickerStats(dips=13, avg_bounce=29.9, best_bounce=81.1,  win_rate=100, tier="A"),
    "TSM":  TickerStats(dips=5,  avg_bounce=28.1, best_bounce=40.0,  win_rate=100, tier="A"),
    # B-tier: avg bounce 15-30%, win rate 60%+
    "LLY":   TickerStats(dips=11, avg_bounce=23.2, best_bounce=55.6,  win_rate=73,  tier="B"),
    "PLTR":  TickerStats(dips=10, avg_bounce=22.7, best_bounce=55.9,  win_rate=70,  tier="B"),
    "GOOGL": TickerStats(dips=7,  avg_bounce=22.6, best_bounce=52.5,  win_rate=71,  tier="B"),
    "UNH":   TickerStats(dips=8,  avg_bounce=22.6, best_bounce=54.5,  win_rate=100, tier="B"),
    "MRK":   TickerStats(dips=7,  avg_bounce=22.5, best_bounce=36.2,  win_rate=86,  tier="B"),
    "AVGO":  TickerStats(dips=8,  avg_bounce=22.2, best_bounce=33.5,  win_rate=88,  tier="B"),
    "HPE":   TickerStats(dips=8,  avg_bounce=20.3, best_bounce=33.2,  win_rate=88,  tier="B"),
    "LMT":   TickerStats(dips=9,  avg_bounce=20.2, best_bounce=51.8,  win_rate=56,  tier="B"),
    "TMO":   TickerStats(dips=9,  avg_bounce=19.9, best_bounce=34.4,  win_rate=67,  tier="B"),
    "CEG":   TickerStats(dips=11, avg_bounce=19.3, best_bounce=35.2,  win_rate=82,  tier="B"),
    "SNOW":  TickerStats(dips=10, avg_bounce=19.1, best_bounce=44.5,  win_rate=70,  tier="B"),
    "GS":    TickerStats(dips=5,  avg_bounce=18.9, best_bounce=31.7,  win_rate=80,  tier="B"),
    "AMZN":  TickerStats(dips=8,  avg_bounce=14.8, best_bounce=19.6,  win_rate=100, tier="B"),
    "CVS":   TickerStats(dips=9,  avg_bounce=16.5, best_bounce=40.5,  win_rate=100, tier="B"),
    "DE":    TickerStats(dips=8,  avg_bounce=17.4, best_bounce=43.6,  win_rate=62,  tier="B"),
    "RTX":   TickerStats(dips=4,  avg_bounce=18.3, best_bounce=24.0,  win_rate=75,  tier="B"),
    "MS":    TickerStats(dips=6,  avg_bounce=16.4, best_bounce=24.4,  win_rate=83,  tier="B"),
    "GE":    TickerStats(dips=5,  avg_bounce=15.6, best_bounce=21.9,  win_rate=60,  tier="B"),
    "AAPL":  TickerStats(dips=8,  avg_bounce=16.7, best_bounce=33.0,  win_rate=62,  tier="B"),
    "META":  TickerStats(dips=7,  avg_bounce=14.0, best_bounce=25.2,  win_rate=57,  tier="B"),
    "NEE":   TickerStats(dips=7,  avg_bounce=15.9, best_bounce=24.5,  win_rate=71,  tier="B"),
    "WMT":   TickerStats(dips=7,  avg_bounce=15.8, best_bounce=33.3,  win_rate=71,  tier="B"),
    "XOM":   TickerStats(dips=7,  avg_bounce=16.0, best_bounce=45.7,  win_rate=71,  tier="B"),
    "COP":   TickerStats(dips=10, avg_bounce=17.3, best_bounce=43.5,  win_rate=60,  tier="B"),
    "VST":   TickerStats(dips=12, avg_bounce=13.9, best_bounce=30.2,  win_rate=58,  tier="B"),
    # C-tier: avg bounce <15%, win rate <50% — avoid dip buying
    "MSFT":  TickerStats(dips=6,  avg_bounce=4.8,  best_bounce=9.5,   win_rate=0,   tier="C"),
    "V":     TickerStats(dips=7,  avg_bounce=5.6,  best_bounce=11.3,  win_rate=14,  tier="C"),
    "MA":    TickerStats(dips=7,  avg_bounce=7.5,  best_bounce=12.5,  win_rate=29,  tier="C"),
    "PG":    TickerStats(dips=8,  avg_bounce=8.4,  best_bounce=22.0,  win_rate=25,  tier="C"),
    "KO":    TickerStats(dips=3,  avg_bounce=7.0,  best_bounce=11.3,  win_rate=33,  tier="C"),
    "NVDA":  TickerStats(dips=7,  avg_bounce=13.8, best_bounce=24.0,  win_rate=71,  tier="B"),
    "ORCL":  TickerStats(dips=9,  avg_bounce=14.9, best_bounce=47.2,  win_rate=67,  tier="B"),
    "HON":   TickerStats(dips=7,  avg_bounce=13.2, best_bounce=31.1,  win_rate=29,  tier="C"),
    "JPM":   TickerStats(dips=6,  avg_bounce=10.6, best_bounce=14.3,  win_rate=67,  tier="B"),
    "ISRG":  TickerStats(dips=10, avg_bounce=12.9, best_bounce=38.0,  win_rate=30,  tier="C"),
    "ABBV":  TickerStats(dips=9,  avg_bounce=9.3,  best_bounce=19.8,  win_rate=44,  tier="C"),
    "SO":    TickerStats(dips=4,  avg_bounce=9.6,  best_bounce=17.0,  win_rate=50,  tier="C"),
    "COST":  TickerStats(dips=6,  avg_bounce=9.0,  best_bounce=20.0,  win_rate=33,  tier="C"),
    "JNJ":   TickerStats(dips=2,  avg_bounce=11.8, best_bounce=19.8,  win_rate=50,  tier="C"),
    "QCOM":  TickerStats(dips=9,  avg_bounce=11.2, best_bounce=28.7,  win_rate=44,  tier="C"),
    "CVX":   TickerStats(dips=4,  avg_bounce=13.3, best_bounce=37.4,  win_rate=50,  tier="C"),
}


# Score weights
W_BOUNCE_RATE = 0.40
W_DIP_DEPTH = 0.20
W_RSI = 0.15
W_VOLUME = 0.10
W_SECTOR = 0.15


def _score_bounce_history(stats: Optional[TickerStats]) -> float:
    """Score 0-100 based on historical bounce reliability."""
    if stats is None:
        return 25.0  # unknown ticker — low default
    # Win rate is already 0-100
    # Boost A-tier, slight penalty for C-tier
    base = stats.win_rate
    if stats.tier == "A":
        base = min(100, base * 1.1)
    elif stats.tier == "C":
        base = base * 0.6
    return base


def _score_dip_depth(scan: StockScan) -> float:
    """Score 0-100 based on how deep the dip is. Deeper = higher score."""
    # Use 20d return as primary depth indicator
    ret = scan.return_20d_pct
    if ret is None:
        ret = scan.return_5d_pct or 0
    drop = abs(min(ret, 0))
    # 0% drop = 0, 5% drop = 40, 10% drop = 70, 15%+ drop = 90-100
    if drop <= 0:
        return 0
    elif drop <= 5:
        return drop * 8
    elif drop <= 10:
        return 40 + (drop - 5) * 6
    elif drop <= 15:
        return 70 + (drop - 10) * 4
    else:
        return min(100, 90 + (drop - 15) * 2)


def _score_rsi(scan: StockScan) -> float:
    """Score 0-100 based on RSI. Lower RSI = higher confidence."""
    rsi = scan.rsi_14
    if rsi is None:
        return 30.0
    # RSI 20 = 100, RSI 30 = 75, RSI 40 = 40, RSI 50+ = 10
    if rsi <= 20:
        return 100
    elif rsi <= 30:
        return 100 - (rsi - 20) * 2.5
    elif rsi <= 40:
        return 75 - (rsi - 30) * 3.5
    elif rsi <= 50:
        return 40 - (rsi - 40) * 3
    else:
        return max(0, 10 - (rsi - 50))


def _score_volume(scan: StockScan) -> float:
    """Score 0-100 based on volume spike. Higher volume = capitulation."""
    vr = scan.volume_ratio
    # vol_ratio 1.0 = normal = 20, 1.5x = 50, 2.0x = 80, 3.0x+ = 100
    if vr <= 1.0:
        return 20
    elif vr <= 1.5:
        return 20 + (vr - 1.0) * 60
    elif vr <= 2.0:
        return 50 + (vr - 1.5) * 60
    elif vr <= 3.0:
        return 80 + (vr - 2.0) * 20
    else:
        return 100


def _score_sector_alignment(scan: StockScan) -> float:
    """Score 0-100 based on whether sector is also dipping.

    If sector is UP but stock is DOWN → stock-specific weakness → higher confidence
    (mean-reversion more likely when it's the stock, not the market).
    """
    stock_ret = scan.return_20d_pct
    sector_ret = scan.sector_return_20d_pct
    if stock_ret is None or sector_ret is None:
        return 40.0  # neutral

    gap = (sector_ret or 0) - (stock_ret or 0)
    # gap > 0 means stock underperformed sector → good for mean reversion
    if gap >= 10:
        return 90
    elif gap >= 5:
        return 70
    elif gap >= 2:
        return 50
    elif gap >= 0:
        return 35
    else:
        # Stock outperformed sector (not really a dip)
        return 15


def compute_confidence(scan: StockScan) -> tuple[int, str, str]:
    """Compute confidence score for a dip-buy candidate.

    Returns:
        (score, level, explanation)
        score: 0-100
        level: "HIGH", "MEDIUM", "LOW"
        explanation: human-readable reasoning
    """
    stats = HISTORICAL_STATS.get(scan.ticker)

    s_bounce = _score_bounce_history(stats)
    s_depth = _score_dip_depth(scan)
    s_rsi = _score_rsi(scan)
    s_vol = _score_volume(scan)
    s_sector = _score_sector_alignment(scan)

    raw = (
        s_bounce * W_BOUNCE_RATE
        + s_depth * W_DIP_DEPTH
        + s_rsi * W_RSI
        + s_vol * W_VOLUME
        + s_sector * W_SECTOR
    )
    score = int(round(min(100, max(0, raw))))

    if score >= 75:
        level = "HIGH"
    elif score >= 50:
        level = "MEDIUM"
    else:
        level = "LOW"

    # Build explanation
    parts = []
    if stats:
        parts.append(
            f"Historically bounces {stats.win_rate:.0f}% of the time "
            f"(avg +{stats.avg_bounce:.0f}%, {stats.dips} samples)"
        )
        parts.append(f"Tier {stats.tier}")
    else:
        parts.append("No historical data — unknown bounce pattern")

    ret_20d = scan.return_20d_pct or scan.return_5d_pct or 0
    if ret_20d < -10:
        parts.append(f"Deep dip ({ret_20d:+.1f}%)")
    elif ret_20d < -5:
        parts.append(f"Moderate dip ({ret_20d:+.1f}%)")

    if scan.rsi_14 and scan.rsi_14 < 30:
        parts.append(f"RSI oversold ({scan.rsi_14:.0f})")
    elif scan.rsi_14 and scan.rsi_14 < 40:
        parts.append(f"RSI approaching oversold ({scan.rsi_14:.0f})")

    if scan.volume_ratio > 1.5:
        parts.append(f"Volume spike ({scan.volume_ratio:.1f}x)")

    explanation = ". ".join(parts) + "."

    return score, level, explanation


def get_tier(ticker: str) -> str:
    """Return tier for a ticker: A, B, C, or ? if unknown."""
    stats = HISTORICAL_STATS.get(ticker)
    return stats.tier if stats else "?"


def should_skip_ticker(ticker: str) -> bool:
    """Return True if this ticker should be excluded from dip buys."""
    stats = HISTORICAL_STATS.get(ticker)
    if stats is None:
        return False  # unknown — allow but with low confidence
    return stats.tier == "C"
