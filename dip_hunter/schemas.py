"""Pydantic models for the dip-hunter system."""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class StockScan(BaseModel):
    """Result of scanning a single stock for dip quality."""

    ticker: str
    sector: str
    price: float
    return_5d_pct: Optional[float] = None
    return_20d_pct: Optional[float] = None
    return_60d_pct: Optional[float] = None
    dist_from_20sma_pct: float = 0.0
    dist_from_50sma_pct: Optional[float] = None
    dist_from_52w_high_pct: Optional[float] = None
    rsi_14: Optional[float] = None
    volume_ratio: float = 1.0
    sector_return_20d_pct: Optional[float] = None
    depth_score: float = 0.0
    bounce_score: float = 0.0
    dip_score: float = 0.0
    signals: List[str] = Field(default_factory=list)


class Holding(BaseModel):
    """A single portfolio position."""

    ticker: str
    shares: int
    avg_cost: float
    entry_date: str
    strategy_notes: str = ""


class TradeRecord(BaseModel):
    """Historical trade for P&L tracking."""

    ticker: str
    action: Literal["BUY", "SELL"]
    shares: int
    price: float
    date: str
    reason: str
    pnl: Optional[float] = None


class PortfolioState(BaseModel):
    """Persisted portfolio state."""

    updated_at: str = ""
    capital_total: float = 15000.0
    capital_deployed: float = 0.0
    capital_available: float = 15000.0
    max_positions: int = 5
    holdings: List[Holding] = Field(default_factory=list)
    trade_history: List[TradeRecord] = Field(default_factory=list)


class HoldingStatus(BaseModel):
    """Current status of a held position with live pricing."""

    ticker: str
    shares: int
    avg_cost: float
    entry_date: str
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    days_held: int
    signal: Literal["HOLD", "TAKE_PROFIT", "STOP_LOSS", "WATCH"]
    signal_reason: str


class BuySignal(BaseModel):
    """Recommendation to buy a dip candidate."""

    rank: int
    ticker: str
    sector: str
    current_price: float
    dip_score: float
    entry_target: float
    stop_loss: float
    take_profit: float
    position_size_shares: int
    position_size_dollars: float
    signals: List[str] = Field(default_factory=list)
    reasoning: str
    # Confidence fields
    confidence: int = 0  # 0-100
    confidence_level: str = ""  # HIGH / MEDIUM / LOW
    confidence_reasoning: str = ""
    tier: str = ""  # A / B / C
    historical_win_rate: Optional[float] = None
    historical_avg_bounce: Optional[float] = None
    expected_bounce_range: str = ""  # e.g. "+10-15% in 20 days"


class SellSignal(BaseModel):
    """Recommendation to sell a current holding."""

    ticker: str
    action: Literal["TAKE_PROFIT", "STOP_LOSS"]
    current_price: float
    avg_cost: float
    pnl_pct: float
    reasoning: str


class RotateSignal(BaseModel):
    """Recommendation to rotate: sell one position and buy another."""

    sell_ticker: str
    sell_reason: str
    sell_pnl_pct: float
    buy_ticker: str
    buy_dip_score: float
    buy_entry_target: float
    reasoning: str


class DipHunterBrief(BaseModel):
    """Top-level output of a dip-hunter daily run."""

    as_of: str
    date: str
    spy_price: float
    spy_change_pct: float
    vix: float
    portfolio_value: float
    total_unrealized_pnl: float
    total_unrealized_pnl_pct: float
    holdings_status: List[HoldingStatus] = Field(default_factory=list)
    buy_signals: List[BuySignal] = Field(default_factory=list)
    sell_signals: List[SellSignal] = Field(default_factory=list)
    rotate_signals: List[RotateSignal] = Field(default_factory=list)
    top_dips: List[StockScan] = Field(default_factory=list)
    track_record: Optional[dict] = None
    disclaimer: str = (
        "DISCLAIMER: Algorithmically generated signals. Not financial advice. "
        "Always do your own research and manage risk appropriately."
    )
