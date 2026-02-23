from __future__ import annotations

from typing import List, Optional, Literal

from pydantic import BaseModel, ConfigDict, Field


class EarningsItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    company: str
    ticker: str
    date: str
    time_hint: Optional[str] = None
    notes: str
    sources: List[str] = Field(default_factory=list)


class EventItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event: str
    date_time_local: Optional[str] = None
    region: str
    why_it_matters: str
    sources: List[str] = Field(default_factory=list)


class HeadlineItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    topic: str
    tickers: List[str] = Field(default_factory=list)
    impact: Literal["high", "medium", "low"]
    one_line_take: str
    sources: List[str] = Field(default_factory=list)


class ThemeItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    theme: str
    evidence: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)


class MarketMoveItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asset: str
    move: str
    window: str
    sources: List[str] = Field(default_factory=list)


class WeeklyContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    themes: List[ThemeItem] = Field(default_factory=list)
    market_moves: List[MarketMoveItem] = Field(default_factory=list)


class SpyOHLC(BaseModel):
    model_config = ConfigDict(extra="forbid")

    o: str = "Not specified"
    h: str = "Not specified"
    l: str = "Not specified"
    c: str = "Not specified"


class SpyData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    last: str = "Not specified"
    prev_ohlc: SpyOHLC = Field(default_factory=SpyOHLC)
    premarket_change: str = "Not specified"
    sources: List[str] = Field(default_factory=list)


class FuturesData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    es_change: str = "Not specified"
    overnight_high: str = "Not specified"
    overnight_low: str = "Not specified"
    notes: str = "Not specified"
    sources: List[str] = Field(default_factory=list)


class VolatilityData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vix: str = "Not specified"
    vix_change: str = "Not specified"
    vvix: str = "Not specified"
    term_structure: str = "Not specified"
    expected_move_1d: str = "Not specified"
    expected_move_7d: str = "Not specified"
    sources: List[str] = Field(default_factory=list)


class RatesFxOilData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    us10y: str = "Not specified"
    us10y_change: str = "Not specified"
    dxy: str = "Not specified"
    dxy_change: str = "Not specified"
    wti: str = "Not specified"
    wti_change: str = "Not specified"
    sources: List[str] = Field(default_factory=list)


class BreadthData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    adv_dec: str = "Not specified"
    notes: str = "Not specified"
    leaders: str = "Not specified"
    laggards: str = "Not specified"
    sources: List[str] = Field(default_factory=list)


class OptionsPositioningData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    put_wall: str = "Not specified"
    call_wall: str = "Not specified"
    gamma_flip: str = "Not specified"
    zero_dte_notes: str = "Not specified"
    sources: List[str] = Field(default_factory=list)


class MarketState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    spy: SpyData = Field(default_factory=SpyData)
    futures: FuturesData = Field(default_factory=FuturesData)
    volatility: VolatilityData = Field(default_factory=VolatilityData)
    rates_fx_oil: RatesFxOilData = Field(default_factory=RatesFxOilData)
    breadth: BreadthData = Field(default_factory=BreadthData)
    options_positioning: OptionsPositioningData = Field(default_factory=OptionsPositioningData)


class ResearchPack(BaseModel):
    model_config = ConfigDict(extra="forbid")

    as_of: str
    earnings: List[EarningsItem] = Field(default_factory=list)
    events: List[EventItem] = Field(default_factory=list)
    headlines: List[HeadlineItem] = Field(default_factory=list)
    weekly_context: WeeklyContext
    market_state: MarketState = Field(default_factory=MarketState)
