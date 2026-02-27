"""Pydantic models for the trading recommendation system."""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class MarketSnapshot(BaseModel):
    """Pre-market data snapshot at time of analysis."""

    as_of: str
    spy_price: float
    spy_prev_close: float
    spy_premarket_change_pct: float
    spy_5d_change_pct: Optional[float] = None
    spy_20d_change_pct: Optional[float] = None
    spy_distance_from_20sma_pct: Optional[float] = None
    spy_volume_ratio: Optional[float] = None
    vix: float
    vix_prev_close: Optional[float] = None
    vix_change: Optional[float] = None
    us10y_yield: Optional[float] = None
    us10y_change_bps: Optional[float] = None
    dxy: Optional[float] = None
    dxy_change_pct: Optional[float] = None
    gap_pct: float


class RuleSignal(BaseModel):
    """A single trading rule evaluation result."""

    rule_id: str
    rule_name: str
    triggered: bool
    direction: Literal["long", "short", "neutral"]
    confidence: Literal["high", "medium", "low", "failed"]
    win_rate: float
    sample_size: int
    current_value: Optional[float] = None
    threshold: Optional[float] = None
    target_ticker: Optional[str] = None
    reasoning: str


class TradeRecommendation(BaseModel):
    """A specific actionable trade recommendation."""

    ticker: str
    direction: Literal["call", "put"]
    strike: Optional[float] = None
    expiry: Optional[str] = None
    entry_timing: str
    allocation_dollars: float
    max_loss_dollars: float
    stop_loss_pct: float = 50.0
    take_profit_pct: float = 100.0
    triggered_rules: List[str] = Field(default_factory=list)
    reasoning: str
    confidence: Literal["high", "medium", "low"]


class SympathyPlay(BaseModel):
    """A correlated/beta play off a primary catalyst."""

    primary_ticker: str
    primary_catalyst: str
    sympathy_ticker: str
    beta: float
    direction: Literal["call", "put"]
    entry_timing: Optional[str] = None
    reasoning: str


class PortfolioSummary(BaseModel):
    """Aggregate view of all recommendations."""

    total_allocation: float
    max_portfolio_risk: float
    num_trades: int
    net_directional_bias: Literal["bullish", "bearish", "neutral"]
    rules_triggered_count: int
    rules_total_count: int = 17


class RecommendationPack(BaseModel):
    """Top-level output of the recommendation engine."""

    as_of: str
    date: str
    market_snapshot: MarketSnapshot
    active_signals: List[RuleSignal] = Field(default_factory=list)
    recommendations: List[TradeRecommendation] = Field(default_factory=list)
    sympathy_plays: List[SympathyPlay] = Field(default_factory=list)
    portfolio_summary: PortfolioSummary
    is_intraday_rerun: bool = False
    disclaimer: str = (
        "DISCLAIMER: These are algorithmically generated signals based on "
        "backtested rules with limited sample sizes. Past performance does not "
        "guarantee future results. This is not financial advice. Always do your "
        "own research and manage risk appropriately."
    )
