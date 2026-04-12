"""Tests for dip-hunter scanner engine."""
from __future__ import annotations

import pytest

from dip_hunter.scanner import compute_rsi, compute_dip_score, build_mock_scans

import pandas as pd


class TestComputeRSI:
    def test_rsi_all_gains(self):
        closes = pd.Series([float(i) for i in range(1, 20)])
        rsi = compute_rsi(closes, period=14)
        assert rsi == 100.0

    def test_rsi_insufficient_data(self):
        closes = pd.Series([1.0, 2.0, 3.0])
        assert compute_rsi(closes, period=14) is None

    def test_rsi_typical_range(self):
        # Alternating up/down should produce RSI near 50
        closes = pd.Series([100 + (i % 2) * 2 for i in range(30)])
        rsi = compute_rsi(closes, period=14)
        assert rsi is not None
        assert 30 < rsi < 70


class TestComputeDipScore:
    def test_deeply_beaten_stock(self):
        depth, bounce, dip, signals = compute_dip_score(
            return_5d=-8.0, return_20d=-15.0, dist_50sma=-6.0,
            dist_from_52w_high=30.0, rsi=22.0, volume_ratio=2.0,
            dist_20sma=-5.0, sector_rel=8.0,
        )
        assert depth > 6.0
        assert bounce > 6.0
        assert dip > 6.0
        assert "5d_crash" in signals
        assert "deeply_oversold" in signals
        assert "rsi_deeply_oversold" in signals

    def test_healthy_stock_low_score(self):
        depth, bounce, dip, signals = compute_dip_score(
            return_5d=2.0, return_20d=5.0, dist_50sma=3.0,
            dist_from_52w_high=2.0, rsi=60.0, volume_ratio=0.9,
            dist_20sma=1.5, sector_rel=-1.0,
        )
        assert dip < 1.0
        assert len(signals) == 0

    def test_moderate_dip(self):
        depth, bounce, dip, signals = compute_dip_score(
            return_5d=-3.5, return_20d=-6.0, dist_50sma=-2.0,
            dist_from_52w_high=12.0, rsi=33.0, volume_ratio=1.3,
            dist_20sma=-2.0, sector_rel=3.0,
        )
        assert 3.0 < dip < 6.0
        assert "5d_selloff" in signals
        assert "oversold" in signals

    def test_none_values_handled(self):
        """Handles None gracefully for optional fields."""
        depth, bounce, dip, signals = compute_dip_score(
            return_5d=None, return_20d=None, dist_50sma=None,
            dist_from_52w_high=None, rsi=None, volume_ratio=1.0,
            dist_20sma=0.0, sector_rel=None,
        )
        assert dip == 0.0

    def test_score_caps_at_10(self):
        """Even extreme values shouldn't exceed component caps."""
        depth, bounce, dip, signals = compute_dip_score(
            return_5d=-50.0, return_20d=-50.0, dist_50sma=-50.0,
            dist_from_52w_high=90.0, rsi=5.0, volume_ratio=10.0,
            dist_20sma=-50.0, sector_rel=50.0,
        )
        assert depth <= 10.0
        assert bounce <= 10.0
        assert dip <= 10.0


class TestMockScans:
    def test_mock_scans_sorted_by_score(self):
        scans = build_mock_scans()
        assert len(scans) >= 8
        # Top picks should have highest scores
        assert scans[0].dip_score >= scans[-1].dip_score

    def test_mock_scans_have_required_fields(self):
        for scan in build_mock_scans():
            assert scan.ticker
            assert scan.sector
            assert scan.price > 0
