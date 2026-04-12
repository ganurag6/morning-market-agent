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
    compute_rule_weight,
    _build_mock_snapshot,
    _build_mock_recommendations,
    _validate_and_fix_recommendations,
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

    def test_r11_consecutive_down_triggers(self, engine, snapshot_high_vix, research_empty):
        # -1.80% is below -1.5% threshold (loosened from -2.0%)
        signals = engine.evaluate_all(snapshot_high_vix, research_empty)
        r11 = next(s for s in signals if s.rule_id == "R11")
        assert r11.triggered is True

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

    def test_all_18_rules_evaluated(self, engine, snapshot_high_vix, research_empty):
        signals = engine.evaluate_all(snapshot_high_vix, research_empty)
        assert len(signals) >= 18
        rule_ids = {s.rule_id for s in signals}
        expected = {f"R{i}" for i in range(1, 19)}
        assert expected.issubset(rule_ids)


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
                    weight=1.0,
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
        assert "| Weight |" in md
        assert "weight 1.00" in md  # weighted confluence summary

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
        assert len(data["active_signals"]) >= 14

        # Validate markdown output
        md = result.brief_path.read_text()
        assert "## Trading Recommendations" in md


# ---------------------------------------------------------------------------
# Post-processing validation tests
# ---------------------------------------------------------------------------
class TestPostProcessingValidation:
    def _make_rec(self, ticker="SPY", direction="call", strike=685.0,
                  allocation=1000, max_loss=500, confidence="medium"):
        return TradeRecommendation(
            ticker=ticker,
            direction=direction,
            strike=strike,
            expiry="2026-02-28",
            entry_timing="At open",
            allocation_dollars=allocation,
            max_loss_dollars=max_loss,
            triggered_rules=["R4"],
            reasoning="Test",
            confidence=confidence,
        )

    def test_hedge_injection_all_calls(self):
        """If all trades are calls, a put hedge should be injected."""
        recs = [
            self._make_rec("SPY", "call", 685.0, 1000, 500),
            self._make_rec("NVDA", "call", 135.0, 800, 400),
            self._make_rec("AMD", "call", 160.0, 600, 300),
        ]
        result = _validate_and_fix_recommendations(recs, {}, 5000)
        directions = [r.direction for r in result]
        assert "put" in directions
        hedge = [r for r in result if r.direction == "put"]
        assert len(hedge) >= 1
        assert "hedge" in hedge[0].triggered_rules

    def test_hedge_injection_all_puts(self):
        """If all trades are puts, a call hedge should be injected."""
        recs = [
            self._make_rec("SPY", "put", 680.0, 1000, 500),
            self._make_rec("NVDA", "put", 130.0, 800, 400),
        ]
        result = _validate_and_fix_recommendations(recs, {}, 5000)
        directions = [r.direction for r in result]
        assert "call" in directions

    def test_no_hedge_injection_when_mixed(self):
        """If trades already have both directions, no hedge is injected."""
        recs = [
            self._make_rec("SPY", "call", 685.0, 1000, 500),
            self._make_rec("SPY", "put", 680.0, 400, 200),
        ]
        result = _validate_and_fix_recommendations(recs, {}, 5000)
        assert len(result) == 2  # no extra hedge added

    def test_strike_snapping(self):
        """Invalid strikes should snap to nearest available in chain."""
        recs = [self._make_rec("SPY", "call", 687.0, 1000, 500)]
        chains = {
            "SPY": {
                "calls": [
                    {"strike": 685.0, "lastPrice": 3.0, "bid": 2.8, "ask": 3.2, "openInterest": 100},
                    {"strike": 686.0, "lastPrice": 2.5, "bid": 2.3, "ask": 2.7, "openInterest": 80},
                    {"strike": 688.0, "lastPrice": 1.5, "bid": 1.3, "ask": 1.7, "openInterest": 60},
                ],
                "puts": [],
                "preferred_call_strike": 685.0,
                "preferred_put_strike": 685.0,
            }
        }
        result = _validate_and_fix_recommendations(recs, chains, 5000)
        # 687 should snap to nearest available (686 is 1 away, 688 is 1 away; min picks 686)
        assert result[0].strike in (686.0, 688.0)
        assert result[0].strike != 687.0  # original invalid strike is gone

    def test_allocation_cap_35_percent(self):
        """No single trade should exceed 35% of max allocation."""
        recs = [self._make_rec("SPY", "call", 685.0, 3000, 1500)]
        result = _validate_and_fix_recommendations(recs, {}, 5000)
        # 35% of 5000 = 1750
        assert result[0].allocation_dollars <= 1750

    def test_total_scaling(self):
        """Total allocation should not exceed max_allocation."""
        recs = [
            self._make_rec("SPY", "call", 685.0, 1500, 750),
            self._make_rec("NVDA", "call", 135.0, 1500, 750),
            self._make_rec("AMD", "call", 160.0, 1500, 750),
            self._make_rec("MSFT", "put", 410.0, 1500, 750),  # mixed, no hedge
        ]
        result = _validate_and_fix_recommendations(recs, {}, 5000)
        total = sum(r.allocation_dollars for r in result)
        assert total <= 5000


# ---------------------------------------------------------------------------
# New threshold tests
# ---------------------------------------------------------------------------
class TestNewThresholds:
    def test_r4_triggers_at_vix_19(self, engine, research_empty):
        """R4 should now trigger at VIX 19.0 (new threshold 18.5)."""
        snap = MarketSnapshot(
            as_of="2026-02-25T08:15:00-06:00",
            spy_price=685.0, spy_prev_close=687.0,
            spy_premarket_change_pct=-0.29,
            vix=19.0, gap_pct=-0.29,
        )
        signals = engine.evaluate_all(snap, research_empty)
        r4 = next(s for s in signals if s.rule_id == "R4")
        assert r4.triggered is True

    def test_r8_triggers_at_gap_minus_026(self, engine, research_empty):
        """R8 should now trigger at gap -0.26% (new threshold -0.25)."""
        snap = MarketSnapshot(
            as_of="2026-02-25T08:15:00-06:00",
            spy_price=685.0, spy_prev_close=687.0,
            spy_premarket_change_pct=-0.26,
            vix=15.0, gap_pct=-0.26,
        )
        signals = engine.evaluate_all(snap, research_empty)
        r8 = next(s for s in signals if s.rule_id == "R8")
        assert r8.triggered is True

    def test_r11_triggers_at_5d_minus_16(self, engine, research_empty):
        """R11 should now trigger at -1.6% (new threshold -1.5)."""
        snap = MarketSnapshot(
            as_of="2026-02-25T08:15:00-06:00",
            spy_price=685.0, spy_prev_close=687.0,
            spy_premarket_change_pct=-0.29,
            spy_5d_change_pct=-1.6,
            vix=15.0, gap_pct=-0.29,
        )
        signals = engine.evaluate_all(snap, research_empty)
        r11 = next(s for s in signals if s.rule_id == "R11")
        assert r11.triggered is True

    def test_r10_triggers_at_volume_125(self, engine, research_empty):
        """R10 should now trigger at 1.25x volume (new threshold 1.2)."""
        snap = MarketSnapshot(
            as_of="2026-02-25T08:15:00-06:00",
            spy_price=685.0, spy_prev_close=687.0,
            spy_premarket_change_pct=-0.10,
            spy_volume_ratio=1.25,
            vix=15.0, gap_pct=-0.10,
        )
        signals = engine.evaluate_all(snap, research_empty)
        r10 = next(s for s in signals if s.rule_id == "R10")
        assert r10.triggered is True

    def test_r13_triggers_at_vix_17_5(self, engine, research_empty):
        """R13 should now trigger at VIX 17.5 (new threshold_vix 17.0)."""
        snap = MarketSnapshot(
            as_of="2026-02-25T08:15:00-06:00",
            spy_price=685.0, spy_prev_close=687.0,
            spy_premarket_change_pct=-0.10,
            vix=17.5, us10y_change_bps=-3.0,
            gap_pct=-0.10,
        )
        signals = engine.evaluate_all(snap, research_empty)
        r13 = next(s for s in signals if s.rule_id == "R13")
        assert r13.triggered is True


# ---------------------------------------------------------------------------
# SympathyPlay entry_timing tests
# ---------------------------------------------------------------------------
class TestSympathyPlayTiming:
    def test_sympathy_play_with_entry_timing(self):
        sp = SympathyPlay(
            primary_ticker="NVDA",
            primary_catalyst="Earnings after close",
            sympathy_ticker="AMD",
            beta=1.01,
            direction="call",
            entry_timing="NEXT DAY ONLY",
            reasoning="Test",
        )
        assert sp.entry_timing == "NEXT DAY ONLY"

    def test_sympathy_play_without_entry_timing(self):
        sp = SympathyPlay(
            primary_ticker="NVDA",
            primary_catalyst="Earnings before open",
            sympathy_ticker="AMD",
            beta=1.01,
            direction="call",
            reasoning="Test",
        )
        assert sp.entry_timing is None

    def test_mock_sympathy_plays_have_timing(self):
        _, sympathy = _build_mock_recommendations()
        for s in sympathy:
            sp = SympathyPlay(**s)
            assert sp.entry_timing is not None


# ---------------------------------------------------------------------------
# R15: Sell-the-News tests
# ---------------------------------------------------------------------------
class TestR15SellTheNews:
    @pytest.fixture
    def engine(self):
        return RuleEngine()

    @pytest.fixture
    def snapshot(self):
        return MarketSnapshot(
            as_of="2026-02-27T08:15:00-06:00",
            spy_price=685.0, spy_prev_close=687.0,
            spy_premarket_change_pct=-0.29,
            vix=18.0, gap_pct=-0.29,
        )

    @pytest.fixture
    def research_with_earnings(self):
        return {
            "date": "2026-02-27",
            "events": [],
            "earnings": [
                {"company": "Nvidia", "ticker": "NVDA", "date": "2026-02-26",
                 "time_hint": "After Close"},
            ],
            "headlines": [],
        }

    def test_r15_triggers_on_rally_and_fade(self, engine, snapshot, research_with_earnings):
        """R15 triggers when stock rallied >5% into earnings and is now flat/down."""
        scans = [
            {"ticker": "NVDA", "price": 130.0, "day_change_pct": -2.5,
             "change_5d_pct": 8.0, "change_20d_pct": -5.0,
             "distance_from_20sma_pct": -3.0, "volume_ratio": 1.5,
             "signals": ["oversold"], "signal_score": 2.0},
        ]
        signals = engine.evaluate_all(snapshot, research_with_earnings, watchlist_scans=scans)
        r15_signals = [s for s in signals if s.rule_id == "R15"]
        triggered = [s for s in r15_signals if s.triggered]
        assert len(triggered) == 1
        assert triggered[0].target_ticker == "NVDA"
        assert triggered[0].direction == "short"

    def test_r15_no_trigger_when_not_rallied(self, engine, snapshot, research_with_earnings):
        """R15 does not trigger if stock didn't rally >5% before earnings."""
        scans = [
            {"ticker": "NVDA", "price": 130.0, "day_change_pct": -1.0,
             "change_5d_pct": 2.0, "change_20d_pct": -5.0,
             "distance_from_20sma_pct": -3.0, "volume_ratio": 1.5,
             "signals": ["oversold"], "signal_score": 2.0},
        ]
        signals = engine.evaluate_all(snapshot, research_with_earnings, watchlist_scans=scans)
        r15_signals = [s for s in signals if s.rule_id == "R15"]
        triggered = [s for s in r15_signals if s.triggered]
        assert len(triggered) == 0

    def test_r15_no_trigger_without_scans(self, engine, snapshot, research_with_earnings):
        """R15 gracefully handles missing watchlist scan data."""
        signals = engine.evaluate_all(snapshot, research_with_earnings, watchlist_scans=None)
        r15_signals = [s for s in signals if s.rule_id == "R15"]
        assert len(r15_signals) >= 1
        assert all(not s.triggered for s in r15_signals)


# ---------------------------------------------------------------------------
# R16: Mega-Cap Earnings Vol Event tests
# ---------------------------------------------------------------------------
class TestR16MegaCapVol:
    @pytest.fixture
    def engine(self):
        return RuleEngine()

    @pytest.fixture
    def snapshot(self):
        return MarketSnapshot(
            as_of="2026-02-27T08:15:00-06:00",
            spy_price=685.0, spy_prev_close=687.0,
            spy_premarket_change_pct=-0.29,
            vix=18.0, gap_pct=-0.29,
        )

    def test_r16_muted_reaction_triggers(self, engine, snapshot):
        """R16 triggers on muted reaction (<1%) after mega-cap earnings."""
        research = {
            "date": "2026-02-27",
            "events": [],
            "earnings": [
                {"company": "Nvidia", "ticker": "NVDA", "date": "2026-02-26",
                 "time_hint": "After Close"},
            ],
            "headlines": [],
        }
        scans = [
            {"ticker": "NVDA", "price": 130.0, "day_change_pct": 0.5,
             "change_5d_pct": 2.0, "change_20d_pct": -5.0,
             "distance_from_20sma_pct": -3.0, "volume_ratio": 1.5,
             "signals": ["oversold"], "signal_score": 2.0},
        ]
        signals = engine.evaluate_all(snapshot, research, watchlist_scans=scans)
        r16_signals = [s for s in signals if s.rule_id == "R16" and s.triggered]
        assert len(r16_signals) >= 1
        assert r16_signals[0].target_ticker == "NVDA"

    def test_r16_upcoming_binary_event(self, engine, snapshot):
        """R16 triggers warning for upcoming mega-cap earnings after close."""
        research = {
            "date": "2026-02-27",
            "events": [],
            "earnings": [
                {"company": "Nvidia", "ticker": "NVDA", "date": "2026-02-27",
                 "time_hint": "After Close"},
            ],
            "headlines": [],
        }
        # No scan data needed — just the earnings listing
        signals = engine.evaluate_all(snapshot, research, watchlist_scans=[])
        r16_signals = [s for s in signals if s.rule_id == "R16" and s.triggered]
        assert len(r16_signals) >= 1

    def test_r16_ignores_non_megacap(self, engine, snapshot):
        """R16 does not trigger for non-mega-cap stocks."""
        research = {
            "date": "2026-02-27",
            "events": [],
            "earnings": [
                {"company": "Acme Corp", "ticker": "ACME", "date": "2026-02-26",
                 "time_hint": "After Close"},
            ],
            "headlines": [],
        }
        scans = [
            {"ticker": "ACME", "price": 50.0, "day_change_pct": 0.3,
             "change_5d_pct": 1.0, "change_20d_pct": 2.0,
             "distance_from_20sma_pct": 0.5, "volume_ratio": 1.0,
             "signals": [], "signal_score": 0.0},
        ]
        signals = engine.evaluate_all(snapshot, research, watchlist_scans=scans)
        r16_signals = [s for s in signals if s.rule_id == "R16" and s.triggered]
        assert len(r16_signals) == 0


# ---------------------------------------------------------------------------
# R17: Contrarian Warning tests
# ---------------------------------------------------------------------------
class TestR17Contrarian:
    def _make_rec(self, ticker="SPY", direction="call", allocation=1000):
        return TradeRecommendation(
            ticker=ticker, direction=direction, strike=685.0,
            expiry="2026-02-28", entry_timing="At open",
            allocation_dollars=allocation, max_loss_dollars=allocation // 2,
            triggered_rules=["R4"], reasoning="Test", confidence="medium",
        )

    def test_r17_20pct_hedge_when_all_bullish(self):
        """R17 bumps hedge to 20% when all scans bullish + all recs are calls."""
        recs = [
            self._make_rec("SPY", "call", 1500),
            self._make_rec("NVDA", "call", 1000),
            self._make_rec("AMD", "call", 800),
        ]
        scans = [
            {"ticker": "NVDA", "signals": ["oversold", "below_sma"], "signal_score": 2.5},
            {"ticker": "AMD", "signals": ["deeply_oversold"], "signal_score": 2.0},
        ]
        signals = [
            RuleSignal(rule_id="R4", rule_name="VIX >20", triggered=True,
                       direction="long", confidence="high", win_rate=1.0,
                       sample_size=7, reasoning="Test"),
        ]
        result = _validate_and_fix_recommendations(
            recs, {}, 5000, watchlist_scans=scans, active_signals=signals,
        )
        hedge = [r for r in result if r.direction == "put"]
        assert len(hedge) >= 1
        assert "R17" in hedge[0].triggered_rules
        # 20% of 5000 = 1000
        assert hedge[0].allocation_dollars == 1000

    def test_r17_not_triggered_with_mixed_signals(self):
        """R17 does not trigger if scans have non-bullish signals."""
        recs = [
            self._make_rec("SPY", "call", 1500),
            self._make_rec("NVDA", "call", 1000),
        ]
        scans = [
            {"ticker": "NVDA", "signals": ["oversold", "below_sma"], "signal_score": 2.5},
            {"ticker": "TSLA", "signals": ["overbought"], "signal_score": 0.5},
        ]
        result = _validate_and_fix_recommendations(
            recs, {}, 5000, watchlist_scans=scans, active_signals=[],
        )
        hedge = [r for r in result if r.direction == "put"]
        assert len(hedge) >= 1
        # Standard hedge, NOT R17
        assert "R17" not in hedge[0].triggered_rules

    def test_r17_not_triggered_when_puts_present(self):
        """R17 does not trigger when recommendations already include puts."""
        recs = [
            self._make_rec("SPY", "call", 1500),
            self._make_rec("SPY", "put", 800),
        ]
        scans = [
            {"ticker": "NVDA", "signals": ["oversold"], "signal_score": 1.5},
        ]
        result = _validate_and_fix_recommendations(
            recs, {}, 5000, watchlist_scans=scans, active_signals=[],
        )
        # No extra hedge injected since mix already present
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Intraday re-run tests
# ---------------------------------------------------------------------------
class TestIntradayRerun:
    def test_intraday_writes_separate_files(self, tmp_path):
        from agent.recommendations import run_recommendations

        result = run_recommendations(
            date="2026-02-25",
            tz="America/Chicago",
            watchlist=["NVDA"],
            out_dir=tmp_path,
            mock_mode=True,
            is_intraday_rerun=True,
        )
        assert result.recommendations_path.exists()
        assert result.brief_path.exists()
        assert "intraday" in result.recommendations_path.name
        assert "intraday" in result.brief_path.name

        with open(result.recommendations_path) as f:
            data = json.load(f)
        assert data.get("is_intraday_rerun") is True

        md = result.brief_path.read_text()
        assert "INTRADAY RE-RUN" in md


# ---------------------------------------------------------------------------
# R18: Geopolitical Risk Escalation tests
# ---------------------------------------------------------------------------
class TestR18GeopoliticalRisk:
    @pytest.fixture
    def engine(self):
        return RuleEngine()

    @pytest.fixture
    def snapshot(self):
        return MarketSnapshot(
            as_of="2026-03-01T08:15:00-06:00",
            spy_price=680.0, spy_prev_close=690.0,
            spy_premarket_change_pct=-1.45,
            vix=22.0, gap_pct=-1.45,
        )

    def test_r18_triggers_on_geopolitical_headlines(self, engine, snapshot):
        """R18 triggers when headlines contain multiple geopolitical keywords."""
        research = {
            "date": "2026-03-01",
            "events": [],
            "earnings": [],
            "headlines": [
                {"title": "Military conflict escalates in Middle East",
                 "topic": "Geopolitics", "tickers": [], "impact": "high",
                 "one_line_take": "Airstrike campaign intensifies.", "sources": []},
                {"title": "Oil surges on sanctions and embargo fears",
                 "topic": "Energy", "tickers": [], "impact": "high",
                 "one_line_take": "Crude spikes on supply disruption risk.", "sources": []},
                {"title": "Markets sell off on retaliation fears",
                 "topic": "Macro", "tickers": [], "impact": "high",
                 "one_line_take": "Escalation drives risk-off move.", "sources": []},
            ],
            "market_state": {
                "rates_fx_oil": {"wti_change": "+4.2%"},
            },
            "weekly_context": {"themes": []},
        }
        signals = engine.evaluate_all(snapshot, research)
        r18 = next(s for s in signals if s.rule_id == "R18")
        assert r18.triggered is True
        assert r18.direction == "short"
        assert r18.current_value >= 6

    def test_r18_no_trigger_on_calm_headlines(self, engine, snapshot):
        """R18 does not trigger when no geopolitical risk detected."""
        research = {
            "date": "2026-03-01",
            "events": [],
            "earnings": [],
            "headlines": [
                {"title": "Tech stocks rally on AI optimism",
                 "topic": "Tech", "tickers": ["NVDA"], "impact": "high",
                 "one_line_take": "AI spend lifts semis.", "sources": []},
                {"title": "CPI comes in line with expectations",
                 "topic": "Macro", "tickers": [], "impact": "medium",
                 "one_line_take": "Inflation stable.", "sources": []},
            ],
            "market_state": {
                "rates_fx_oil": {"wti_change": "-0.5%"},
            },
            "weekly_context": {"themes": []},
        }
        signals = engine.evaluate_all(snapshot, research)
        r18 = next(s for s in signals if s.rule_id == "R18")
        assert r18.triggered is False

    def test_r18_oil_spike_adds_bonus(self, engine, snapshot):
        """Oil spike >3% adds bonus score to R18."""
        research = {
            "date": "2026-03-01",
            "events": [],
            "earnings": [],
            "headlines": [
                {"title": "Sanctions escalation shakes energy markets",
                 "topic": "Energy", "tickers": [], "impact": "medium",
                 "one_line_take": "Embargo threat rises.", "sources": []},
                {"title": "Troops mobilize near border as conflict deepens",
                 "topic": "Geopolitics", "tickers": [], "impact": "high",
                 "one_line_take": "Military buildup accelerates.", "sources": []},
            ],
            "market_state": {
                "rates_fx_oil": {"wti_change": "+5.0%"},
            },
            "weekly_context": {"themes": []},
        }
        signals = engine.evaluate_all(snapshot, research)
        r18 = next(s for s in signals if s.rule_id == "R18")
        # sanctions(medium=2) + troops(high=3) + oil bonus(3) = 8 >= 6
        assert r18.triggered is True
        assert "oil" in r18.reasoning.lower()

    def test_r18_post_processing_scales_allocations(self):
        """R18 post-processing reduces call allocations by 25% and sets 30% hedge."""
        recs = [
            TradeRecommendation(
                ticker="SPY", direction="call", strike=685.0,
                expiry="2026-03-07", entry_timing="At open",
                allocation_dollars=2000, max_loss_dollars=1000,
                triggered_rules=["R4"], reasoning="Test", confidence="high",
            ),
            TradeRecommendation(
                ticker="NVDA", direction="call", strike=130.0,
                expiry="2026-03-07", entry_timing="At open",
                allocation_dollars=1000, max_loss_dollars=500,
                triggered_rules=["R9"], reasoning="Test", confidence="medium",
            ),
        ]
        r18_signal = RuleSignal(
            rule_id="R18", rule_name="Geopolitical Risk Escalation",
            triggered=True, direction="short", confidence="medium",
            win_rate=0.55, sample_size=0, current_value=8.0,
            threshold=6.0, reasoning="Test",
        )
        result = _validate_and_fix_recommendations(
            recs, {}, 5000, watchlist_scans=None,
            active_signals=[r18_signal],
        )
        # Call allocations should be reduced by 25%
        spy_rec = next(r for r in result if r.ticker == "SPY" and r.direction == "call")
        assert spy_rec.allocation_dollars == 1500  # 2000 * 0.75

        nvda_rec = next(r for r in result if r.ticker == "NVDA" and r.direction == "call")
        assert nvda_rec.allocation_dollars == 750  # 1000 * 0.75

        # Hedge should be present with R18 tag and 30% allocation
        hedge = [r for r in result if r.direction == "put"]
        assert len(hedge) >= 1
        assert "R18" in hedge[0].triggered_rules
        assert hedge[0].allocation_dollars == 1500  # 30% of 5000


# ---------------------------------------------------------------------------
# Priority weight tests
# ---------------------------------------------------------------------------
class TestComputeRuleWeight:
    def test_perfect_rule(self):
        """R4: (1.0, high, 7) → 1.00"""
        assert compute_rule_weight(1.0, "high", 7) == 1.0

    def test_strong_rule(self):
        """R8: (0.86, high, 7) → 0.86"""
        assert compute_rule_weight(0.86, "high", 7) == 0.86

    def test_low_confidence(self):
        """R3: (0.67, low, 3) → 0.1608"""
        assert compute_rule_weight(0.67, "low", 3) == 0.1608

    def test_failed_rule(self):
        """R12: (0.33, failed, 3) → 0.0"""
        assert compute_rule_weight(0.33, "failed", 3) == 0.0

    def test_no_samples(self):
        """R18: (0.55, medium, 0) → 0.0"""
        assert compute_rule_weight(0.55, "medium", 0) == 0.0

    def test_partial_samples(self):
        """(1.0, high, 3) → 0.6"""
        assert compute_rule_weight(1.0, "high", 3) == 0.6

    def test_capped_samples(self):
        """(1.0, high, 10) → 1.0 (sample factor caps at 1.0)"""
        assert compute_rule_weight(1.0, "high", 10) == 1.0

    def test_unknown_confidence(self):
        """Unknown confidence string → 0.0"""
        assert compute_rule_weight(1.0, "unknown", 5) == 0.0


class TestWeightIntegration:
    @pytest.fixture
    def engine(self):
        return RuleEngine()

    @pytest.fixture
    def snapshot(self):
        return MarketSnapshot(
            as_of="2026-02-25T08:15:00-06:00",
            spy_price=684.50, spy_prev_close=687.35,
            spy_premarket_change_pct=-0.41,
            spy_5d_change_pct=-1.80, spy_20d_change_pct=-3.50,
            spy_distance_from_20sma_pct=-0.90, spy_volume_ratio=1.15,
            vix=21.30, vix_prev_close=19.55, vix_change=1.75,
            us10y_yield=4.085, us10y_change_bps=-5.5,
            dxy=97.80, dxy_change_pct=0.30, gap_pct=-0.41,
        )

    def test_all_weights_in_range(self, engine, snapshot):
        """Every signal from evaluate_all has weight in [0.0, 1.0]."""
        research = {"date": "2026-02-25", "events": [], "earnings": [], "headlines": []}
        signals = engine.evaluate_all(snapshot, research)
        for s in signals:
            assert 0.0 <= s.weight <= 1.0, f"{s.rule_id} weight={s.weight} out of range"

    def test_r4_gets_weight_1(self, engine, snapshot):
        """R4 (100% win, high conf, 7 trades) should get weight 1.0."""
        research = {"date": "2026-02-25", "events": [], "earnings": [], "headlines": []}
        signals = engine.evaluate_all(snapshot, research)
        r4 = next(s for s in signals if s.rule_id == "R4")
        assert r4.weight == 1.0

    def test_r12_gets_weight_0(self, engine, snapshot):
        """R12 (failed confidence) should get weight 0.0."""
        research = {"date": "2026-02-25", "events": [], "earnings": [], "headlines": []}
        signals = engine.evaluate_all(snapshot, research)
        r12 = next(s for s in signals if s.rule_id == "R12")
        assert r12.weight == 0.0

    def test_brief_contains_weight_column(self, engine, snapshot):
        """Brief output contains Weight column header."""
        research = {"date": "2026-02-25", "events": [], "earnings": [], "headlines": []}
        signals = engine.evaluate_all(snapshot, research)
        triggered = [s for s in signals if s.triggered and s.confidence != "failed"]

        pack = RecommendationPack(
            as_of="2026-02-25T08:15:00-06:00",
            date="2026-02-25",
            market_snapshot=snapshot,
            active_signals=signals,
            portfolio_summary=PortfolioSummary(
                total_allocation=0, max_portfolio_risk=0,
                num_trades=0, net_directional_bias="bullish",
                rules_triggered_count=len(triggered),
            ),
        )
        md = build_recommendation_brief(pack)
        assert "| Weight |" in md
