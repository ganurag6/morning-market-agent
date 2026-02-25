"""Tests for the trading recommendation system."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.rec_schema import (
    MarketSnapshot,
    PortfolioSummary,
    RecommendationPack,
    RuleSignal,
    TradeRecommendation,
    SympathyPlay,
)
from agent.recommendations import (
    RuleEngine,
    build_recommendation_brief,
    _build_mock_snapshot,
    _build_mock_recommendations,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def snapshot_high_vix():
    """Market snapshot with VIX above 20."""
    return MarketSnapshot(
        as_of="2026-02-25T08:15:00-06:00",
        spy_price=684.50,
        spy_prev_close=687.35,
        spy_premarket_change_pct=-0.41,
        spy_5d_change_pct=-1.80,
        spy_20d_change_pct=-3.50,
        spy_distance_from_20sma_pct=-0.90,
        spy_volume_ratio=1.15,
        vix=21.30,
        vix_prev_close=19.55,
        vix_change=1.75,
        us10y_yield=4.085,
        us10y_change_bps=-5.5,
        dxy=97.80,
        dxy_change_pct=0.30,
        gap_pct=-0.41,
    )


@pytest.fixture
def snapshot_calm():
    """Market snapshot with calm conditions."""
    return MarketSnapshot(
        as_of="2026-02-25T08:15:00-06:00",
        spy_price=690.00,
        spy_prev_close=689.50,
        spy_premarket_change_pct=0.07,
        spy_5d_change_pct=1.20,
        spy_20d_change_pct=2.50,
        spy_distance_from_20sma_pct=0.80,
        spy_volume_ratio=0.95,
        vix=15.30,
        vix_prev_close=15.80,
        vix_change=-0.50,
        us10y_yield=4.10,
        us10y_change_bps=1.0,
        dxy=97.50,
        dxy_change_pct=-0.10,
        gap_pct=0.07,
    )


@pytest.fixture
def research_with_events():
    return {
        "date": "2026-02-25",
        "events": [
            {
                "event": "Consumer Confidence Index",
                "date_time_local": "2026-02-25T09:00:00-06:00",
                "region": "US",
            }
        ],
        "earnings": [
            {
                "company": "Nvidia",
                "ticker": "NVDA",
                "date": "2026-02-25",
                "time_hint": "After Close",
            }
        ],
        "headlines": [],
    }


@pytest.fixture
def research_empty():
    return {"date": "2026-02-25", "events": [], "earnings": [], "headlines": []}


@pytest.fixture
def engine():
    return RuleEngine()


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------
class TestSchema:
    def test_market_snapshot_valid(self, snapshot_high_vix):
        assert snapshot_high_vix.vix == 21.30
        assert snapshot_high_vix.gap_pct == -0.41

    def test_recommendation_pack_with_disclaimer(self):
        pack = RecommendationPack(
            as_of="2026-02-25T08:15:00-06:00",
            date="2026-02-25",
            market_snapshot=_build_mock_snapshot("2026-02-25T08:15:00-06:00"),
            portfolio_summary=PortfolioSummary(
                total_allocation=2000,
                max_portfolio_risk=1000,
                num_trades=2,
                net_directional_bias="bullish",
                rules_triggered_count=3,
            ),
        )
        assert pack.disclaimer.startswith("DISCLAIMER")
        assert len(pack.recommendations) == 0

    def test_trade_recommendation_fields(self):
        rec = TradeRecommendation(
            ticker="SPY",
            direction="call",
            strike=685.0,
            expiry="2026-02-28",
            entry_timing="At open",
            allocation_dollars=1500,
            max_loss_dollars=750,
            triggered_rules=["R4", "R8"],
            reasoning="Test reasoning",
            confidence="high",
        )
        assert rec.stop_loss_pct == 50.0
        assert rec.take_profit_pct == 100.0


# ---------------------------------------------------------------------------
# Rule evaluation tests
# ---------------------------------------------------------------------------
class TestRuleEngine:
    def test_r4_vix_above_20_triggers(self, engine, snapshot_high_vix, research_empty):
        signals = engine.evaluate_all(snapshot_high_vix, research_empty)
        r4 = next(s for s in signals if s.rule_id == "R4")
        assert r4.triggered is True
        assert r4.direction == "long"
        assert r4.confidence == "high"

    def test_r4_vix_below_20_no_trigger(self, engine, snapshot_calm, research_empty):
        signals = engine.evaluate_all(snapshot_calm, research_empty)
        r4 = next(s for s in signals if s.rule_id == "R4")
        assert r4.triggered is False

    def test_r8_gap_down_triggers(self, engine, snapshot_high_vix, research_empty):
        signals = engine.evaluate_all(snapshot_high_vix, research_empty)
        r8 = next(s for s in signals if s.rule_id == "R8")
        assert r8.triggered is True
        assert r8.direction == "long"

    def test_r8_no_gap_no_trigger(self, engine, snapshot_calm, research_empty):
        signals = engine.evaluate_all(snapshot_calm, research_empty)
        r8 = next(s for s in signals if s.rule_id == "R8")
        assert r8.triggered is False

    def test_r9_below_sma_triggers(self, engine, snapshot_high_vix, research_empty):
        signals = engine.evaluate_all(snapshot_high_vix, research_empty)
        r9 = next(s for s in signals if s.rule_id == "R9")
        assert r9.triggered is True

    def test_r9_above_sma_no_trigger(self, engine, snapshot_calm, research_empty):
        signals = engine.evaluate_all(snapshot_calm, research_empty)
        r9 = next(s for s in signals if s.rule_id == "R9")
        assert r9.triggered is False

    def test_r7_yield_drop_triggers(self, engine, snapshot_high_vix, research_empty):
        signals = engine.evaluate_all(snapshot_high_vix, research_empty)
        r7 = next(s for s in signals if s.rule_id == "R7")
        assert r7.triggered is True  # -5.5 bps < -5.0 threshold

    def test_r11_consecutive_down_no_trigger(self, engine, snapshot_high_vix, research_empty):
        # -1.80% is not below -2.0% threshold
        signals = engine.evaluate_all(snapshot_high_vix, research_empty)
        r11 = next(s for s in signals if s.rule_id == "R11")
        assert r11.triggered is False

    def test_r12_failed_rule_flagged(self, engine, snapshot_calm, research_empty):
        signals = engine.evaluate_all(snapshot_calm, research_empty)
        r12 = next(s for s in signals if s.rule_id == "R12")
        assert r12.confidence == "failed"

    def test_r13_compound_triggers(self, engine, snapshot_high_vix, research_empty):
        # VIX 21.3 > 18, yield change -5.5 < -2.0
        signals = engine.evaluate_all(snapshot_high_vix, research_empty)
        r13 = next(s for s in signals if s.rule_id == "R13")
        assert r13.triggered is True

    def test_r14_dxy_up_vix_up_triggers(self, engine, snapshot_high_vix, research_empty):
        # DXY change +0.30%, VIX change +1.75
        signals = engine.evaluate_all(snapshot_high_vix, research_empty)
        r14 = next(s for s in signals if s.rule_id == "R14")
        assert r14.triggered is True

    def test_event_dependent_rules_no_events(self, engine, snapshot_high_vix, research_empty):
        signals = engine.evaluate_all(snapshot_high_vix, research_empty)
        r2 = next(s for s in signals if s.rule_id == "R2")
        r3 = next(s for s in signals if s.rule_id == "R3")
        r5 = next(s for s in signals if s.rule_id == "R5")
        r6 = next(s for s in signals if s.rule_id == "R6")
        assert r2.triggered is False
        assert r3.triggered is False
        assert r5.triggered is False
        assert r6.triggered is False

    def test_all_14_rules_evaluated(self, engine, snapshot_high_vix, research_empty):
        signals = engine.evaluate_all(snapshot_high_vix, research_empty)
        assert len(signals) == 14
        rule_ids = {s.rule_id for s in signals}
        expected = {f"R{i}" for i in range(1, 15)}
        assert rule_ids == expected


# ---------------------------------------------------------------------------
# Brief builder tests
# ---------------------------------------------------------------------------
class TestBriefBuilder:
    def test_builds_markdown(self, snapshot_high_vix):
        pack = RecommendationPack(
            as_of="2026-02-25T08:15:00-06:00",
            date="2026-02-25",
            market_snapshot=snapshot_high_vix,
            active_signals=[
                RuleSignal(
                    rule_id="R4",
                    rule_name="VIX >20",
                    triggered=True,
                    direction="long",
                    confidence="high",
                    win_rate=1.0,
                    sample_size=7,
                    current_value=21.30,
                    threshold=20.0,
                    reasoning="VIX elevated.",
                )
            ],
            recommendations=[
                TradeRecommendation(
                    ticker="SPY",
                    direction="call",
                    strike=685.0,
                    expiry="2026-02-28",
                    entry_timing="At open",
                    allocation_dollars=1500,
                    max_loss_dollars=750,
                    triggered_rules=["R4"],
                    reasoning="VIX >20 buy signal.",
                    confidence="high",
                )
            ],
            portfolio_summary=PortfolioSummary(
                total_allocation=1500,
                max_portfolio_risk=750,
                num_trades=1,
                net_directional_bias="bullish",
                rules_triggered_count=1,
            ),
        )
        md = build_recommendation_brief(pack)
        assert "## Trading Recommendations" in md
        assert "Market Snapshot" in md
        assert "Active Rule Signals" in md
        assert "Trade 1:" in md
        assert "SPY" in md
        assert "DISCLAIMER" in md

    def test_no_trades_message(self, snapshot_calm):
        pack = RecommendationPack(
            as_of="2026-02-25T08:15:00-06:00",
            date="2026-02-25",
            market_snapshot=snapshot_calm,
            portfolio_summary=PortfolioSummary(
                total_allocation=0,
                max_portfolio_risk=0,
                num_trades=0,
                net_directional_bias="neutral",
                rules_triggered_count=0,
            ),
        )
        md = build_recommendation_brief(pack)
        assert "No trade recommendations today" in md


# ---------------------------------------------------------------------------
# Mock mode tests
# ---------------------------------------------------------------------------
class TestMockMode:
    def test_mock_snapshot_valid(self):
        snap = _build_mock_snapshot("2026-02-25T08:15:00-06:00")
        assert snap.spy_price > 0
        assert snap.vix > 0
        assert snap.gap_pct != 0 or snap.gap_pct == 0  # any value is valid

    def test_mock_recommendations_valid(self):
        recs, sympathy = _build_mock_recommendations()
        assert len(recs) > 0
        assert len(sympathy) > 0
        for r in recs:
            TradeRecommendation(**r)  # validates schema
        for s in sympathy:
            SympathyPlay(**s)  # validates schema

    def test_mock_mode_e2e(self, tmp_path):
        from agent.recommendations import run_recommendations

        result = run_recommendations(
            date="2026-02-25",
            tz="America/Chicago",
            watchlist=["NVDA"],
            out_dir=tmp_path,
            mock_mode=True,
        )
        assert result.recommendations_path.exists()
        assert result.brief_path.exists()
        assert result.mock_mode is True

        # Validate JSON output
        with open(result.recommendations_path) as f:
            data = json.load(f)
        assert "market_snapshot" in data
        assert "active_signals" in data
        assert "recommendations" in data
        assert len(data["active_signals"]) == 14

        # Validate markdown output
        md = result.brief_path.read_text()
        assert "## Trading Recommendations" in md
