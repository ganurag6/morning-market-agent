"""Tests for dip-hunter portfolio management."""
from __future__ import annotations

import json

import pytest

from dip_hunter.portfolio import (
    load_portfolio,
    save_portfolio,
    seed_portfolio,
    record_buy,
    record_sell,
)
from dip_hunter.schemas import PortfolioState


class TestLoadSave:
    def test_load_missing_file_returns_empty(self, tmp_path):
        p = load_portfolio(tmp_path / "nonexistent.json")
        assert len(p.holdings) == 0
        assert p.capital_total == 15000.0

    def test_save_and_load_roundtrip(self, tmp_path):
        state = PortfolioState(
            updated_at="2026-04-11T08:00:00",
            capital_total=10000.0,
            capital_deployed=5000.0,
            capital_available=5000.0,
        )
        path = tmp_path / "portfolio.json"
        save_portfolio(state, path)
        loaded = load_portfolio(path)
        assert loaded.capital_total == 10000.0
        assert loaded.capital_deployed == 5000.0


class TestSeedPortfolio:
    def test_seed_with_live_prices(self):
        holdings = [
            {"ticker": "AMZN", "shares": 20},
            {"ticker": "CVS", "shares": 50},
        ]
        prices = {"AMZN": 185.0, "CVS": 55.0}
        state = seed_portfolio(holdings, prices, capital_total=15000.0, date="2026-04-11")

        assert len(state.holdings) == 2
        assert state.holdings[0].avg_cost == 185.0
        assert state.holdings[1].avg_cost == 55.0
        # deployed = 20*185 + 50*55 = 3700 + 2750 = 6450
        assert state.capital_deployed == 6450.0
        assert state.capital_available == 15000.0 - 6450.0


class TestRecordTrades:
    def _make_state(self):
        return PortfolioState(
            updated_at="2026-04-11",
            capital_total=10000.0,
            capital_deployed=2000.0,
            capital_available=8000.0,
            holdings=[{
                "ticker": "AMZN", "shares": 10,
                "avg_cost": 200.0, "entry_date": "2026-04-01",
            }],
        )

    def test_record_buy_new_ticker(self):
        state = self._make_state()
        state = record_buy(state, "NVDA", 5, 120.0, "2026-04-11")
        assert len(state.holdings) == 2
        nvda = next(h for h in state.holdings if h.ticker == "NVDA")
        assert nvda.shares == 5
        assert nvda.avg_cost == 120.0
        assert state.capital_deployed == 2600.0  # 2000 + 600
        assert state.capital_available == 7400.0  # 8000 - 600
        assert len(state.trade_history) == 1

    def test_record_buy_existing_ticker_averages_in(self):
        state = self._make_state()
        state = record_buy(state, "AMZN", 10, 180.0, "2026-04-11")
        amzn = next(h for h in state.holdings if h.ticker == "AMZN")
        assert amzn.shares == 20
        # avg = (10*200 + 10*180) / 20 = 190
        assert amzn.avg_cost == 190.0

    def test_record_sell_full_position(self):
        state = self._make_state()
        state = record_sell(state, "AMZN", 10, 220.0, "2026-04-11", "take_profit")
        assert len(state.holdings) == 0
        assert len(state.trade_history) == 1
        trade = state.trade_history[0]
        assert trade.pnl == 200.0  # (220-200)*10
        assert trade.reason == "take_profit"

    def test_record_sell_partial(self):
        state = self._make_state()
        state = record_sell(state, "AMZN", 5, 210.0, "2026-04-11", "take_profit")
        assert len(state.holdings) == 1
        assert state.holdings[0].shares == 5
        trade = state.trade_history[0]
        assert trade.pnl == 50.0  # (210-200)*5

    def test_record_sell_nonexistent_ticker(self):
        state = self._make_state()
        state = record_sell(state, "FAKE", 10, 100.0, "2026-04-11")
        assert len(state.trade_history) == 0  # no trade recorded
