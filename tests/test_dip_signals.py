"""Tests for dip-hunter signal generation."""
from __future__ import annotations

import pytest

from dip_hunter.schemas import Holding, PortfolioState, StockScan
from dip_hunter.signals import (
    compute_position_size,
    evaluate_holdings,
    generate_buy_signals,
    generate_rotate_signals,
    generate_sell_signals,
)


@pytest.fixture
def portfolio():
    return PortfolioState(
        updated_at="2026-04-11",
        capital_total=15000.0,
        capital_deployed=8000.0,
        capital_available=7000.0,
        max_positions=5,
        holdings=[
            Holding(ticker="AMZN", shares=20, avg_cost=185.0, entry_date="2026-03-25"),
            Holding(ticker="CVS", shares=50, avg_cost=60.0, entry_date="2026-03-20"),
        ],
    )


@pytest.fixture
def scans():
    return [
        StockScan(ticker="PFE", sector="Healthcare", price=24.30,
                  return_20d_pct=-12.5, dist_from_20sma_pct=-4.8,
                  rsi_14=24.5, volume_ratio=1.8,
                  dip_score=7.68, signals=["deeply_oversold", "rsi_deeply_oversold"]),
        StockScan(ticker="NKE", sector="Consumer", price=72.50,
                  return_20d_pct=-9.5, dist_from_20sma_pct=-3.5,
                  rsi_14=27.0, volume_ratio=1.6,
                  dip_score=7.12, signals=["oversold", "rsi_oversold"]),
        StockScan(ticker="INTC", sector="Semis", price=28.80,
                  return_20d_pct=-11.0, dist_from_20sma_pct=-4.0,
                  rsi_14=26.0, volume_ratio=2.1,
                  dip_score=6.90, signals=["deeply_oversold"]),
        StockScan(ticker="AMZN", sector="Consumer", price=185.0,
                  dist_from_20sma_pct=1.5, dip_score=0.42, signals=[]),
        StockScan(ticker="DIS", sector="Consumer", price=95.0,
                  dist_from_20sma_pct=-2.0, dip_score=4.22, signals=["below_20sma"]),
    ]


class TestEvaluateHoldings:
    def test_hold_signal(self, portfolio):
        prices = {"AMZN": 190.0, "CVS": 58.0}
        statuses = evaluate_holdings(portfolio, prices, today="2026-04-11")
        assert len(statuses) == 2
        amzn = next(s for s in statuses if s.ticker == "AMZN")
        assert amzn.signal == "HOLD"
        assert amzn.unrealized_pnl_pct == pytest.approx(2.7, abs=0.1)

    def test_take_profit_signal(self, portfolio):
        prices = {"AMZN": 210.0, "CVS": 58.0}  # AMZN +13.5%
        statuses = evaluate_holdings(portfolio, prices, today="2026-04-11")
        amzn = next(s for s in statuses if s.ticker == "AMZN")
        assert amzn.signal == "TAKE_PROFIT"

    def test_stop_loss_signal(self, portfolio):
        prices = {"AMZN": 185.0, "CVS": 54.0}  # CVS -10%
        statuses = evaluate_holdings(portfolio, prices, today="2026-04-11")
        cvs = next(s for s in statuses if s.ticker == "CVS")
        assert cvs.signal == "STOP_LOSS"

    def test_watch_signal_near_tp(self, portfolio):
        prices = {"AMZN": 198.0, "CVS": 58.0}  # AMZN +7%
        statuses = evaluate_holdings(portfolio, prices, today="2026-04-11")
        amzn = next(s for s in statuses if s.ticker == "AMZN")
        assert amzn.signal == "WATCH"

    def test_days_held(self, portfolio):
        prices = {"AMZN": 185.0, "CVS": 55.0}
        statuses = evaluate_holdings(portfolio, prices, today="2026-04-11")
        amzn = next(s for s in statuses if s.ticker == "AMZN")
        assert amzn.days_held == 17  # March 25 to April 11


class TestGenerateBuySignals:
    def test_excludes_held_tickers(self, scans, portfolio):
        buys = generate_buy_signals(scans, portfolio)
        buy_tickers = {b.ticker for b in buys}
        assert "AMZN" not in buy_tickers

    def test_excludes_low_score(self, scans, portfolio):
        buys = generate_buy_signals(scans, portfolio)
        buy_tickers = {b.ticker for b in buys}
        assert "DIS" not in buy_tickers  # score 4.22 < 5.0

    def test_top_3_by_score(self, scans, portfolio):
        buys = generate_buy_signals(scans, portfolio)
        assert len(buys) <= 3
        assert buys[0].ticker == "PFE"
        assert buys[0].rank == 1

    def test_no_signals_when_full(self, scans):
        full_portfolio = PortfolioState(
            capital_total=15000, capital_available=0, max_positions=2,
            holdings=[
                Holding(ticker="A", shares=1, avg_cost=100, entry_date="2026-04-01"),
                Holding(ticker="B", shares=1, avg_cost=100, entry_date="2026-04-01"),
            ],
        )
        buys = generate_buy_signals(scans, full_portfolio)
        assert len(buys) == 0

    def test_position_sizing(self, scans, portfolio):
        buys = generate_buy_signals(scans, portfolio)
        for b in buys:
            assert b.position_size_shares > 0
            assert b.position_size_dollars > 0
            # Should not exceed 30% of total capital
            assert b.position_size_dollars <= portfolio.capital_total * 0.30 + 1


class TestGenerateSellSignals:
    def test_generates_for_tp_and_sl(self, portfolio):
        prices = {"AMZN": 210.0, "CVS": 54.0}
        statuses = evaluate_holdings(portfolio, prices, today="2026-04-11")
        sells = generate_sell_signals(statuses)
        assert len(sells) == 2
        actions = {s.action for s in sells}
        assert "TAKE_PROFIT" in actions
        assert "STOP_LOSS" in actions

    def test_no_sells_when_holding(self, portfolio):
        prices = {"AMZN": 190.0, "CVS": 58.0}
        statuses = evaluate_holdings(portfolio, prices, today="2026-04-11")
        sells = generate_sell_signals(statuses)
        assert len(sells) == 0


class TestRotateSignals:
    def test_pairs_sell_with_buy(self, scans, portfolio):
        prices = {"AMZN": 210.0, "CVS": 54.0}
        statuses = evaluate_holdings(portfolio, prices, today="2026-04-11")
        sells = generate_sell_signals(statuses)
        buys = generate_buy_signals(scans, portfolio)
        rotations = generate_rotate_signals(sells, buys)
        assert len(rotations) >= 1
        assert rotations[0].sell_ticker in {"AMZN", "CVS"}
        assert rotations[0].buy_ticker in {"PFE", "NKE", "INTC"}


class TestPositionSizing:
    def test_respects_max_pct(self):
        shares, dollars = compute_position_size(100.0, 10000.0, 15000.0)
        assert dollars <= 15000 * 0.30 + 1  # 30% max

    def test_zero_when_price_too_high(self):
        shares, dollars = compute_position_size(50000.0, 100.0, 15000.0)
        assert shares == 0

    def test_uses_available_capital(self):
        shares, dollars = compute_position_size(100.0, 2000.0, 15000.0)
        assert dollars <= 2000.0
