"""Core recommendation engine: market data, rule evaluation, trade generation."""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import date as date_type, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

from .openai_client import MissingAPIKeyError, OpenAIClient
from .rec_schema import (
    MarketSnapshot,
    PortfolioSummary,
    RecommendationPack,
    RuleSignal,
    SympathyPlay,
    TradeRecommendation,
)

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
RULES_PATH = ROOT / "data" / "trading_rules.json"
SYMPATHY_PATH = ROOT / "data" / "sympathy_map.json"
HISTORY_PATH = ROOT / "data" / "macro_spy_reactions.csv"


# ---------------------------------------------------------------------------
# Priority weight computation
# ---------------------------------------------------------------------------
CONFIDENCE_MULTIPLIER = {
    "high": 1.0,
    "medium": 0.7,
    "low": 0.4,
    "failed": 0.0,
}


def compute_rule_weight(win_rate: float, confidence: str, sample_size: int) -> float:
    """Derive a priority weight (0.0–1.0) from a rule's backtest stats."""
    conf_mult = CONFIDENCE_MULTIPLIER.get(confidence, 0.0)
    sample_factor = min(sample_size / 5.0, 1.0)
    return round(win_rate * conf_mult * sample_factor, 4)


# ---------------------------------------------------------------------------
# Market Data Fetcher
# ---------------------------------------------------------------------------
class MarketDataFetcher:
    """Fetch live pre-market data via yfinance."""

    def fetch_spy_snapshot(self) -> Dict[str, Any]:
        """Fetch SPY current/pre-market price and historical context."""
        logger.info("Fetching SPY snapshot...")
        ticker = yf.Ticker("SPY")

        # Daily history for SMA, volume, lookback returns
        hist = ticker.history(period="2mo", interval="1d")
        if hist.empty:
            raise RuntimeError("No SPY daily data from yfinance.")

        if isinstance(hist.columns, pd.MultiIndex):
            hist = hist.droplevel("Ticker", axis=1)

        prev_close = float(hist["Close"].iloc[-1])
        prev_open = float(hist["Open"].iloc[-1])
        prev_high = float(hist["High"].iloc[-1])
        prev_low = float(hist["Low"].iloc[-1])
        prev_volume = float(hist["Volume"].iloc[-1])

        # 20-day SMA
        sma20 = float(hist["Close"].rolling(20).mean().iloc[-1])
        distance_from_sma = (prev_close - sma20) / sma20 * 100

        # Volume ratio
        avg_vol_20d = float(hist["Volume"].rolling(20).mean().iloc[-1])
        volume_ratio = prev_volume / avg_vol_20d if avg_vol_20d > 0 else 1.0

        # Lookback returns
        if len(hist) >= 5:
            close_5d_ago = float(hist["Close"].iloc[-6])
            change_5d = (prev_close - close_5d_ago) / close_5d_ago * 100
        else:
            change_5d = None

        if len(hist) >= 20:
            close_20d_ago = float(hist["Close"].iloc[-21])
            change_20d = (prev_close - close_20d_ago) / close_20d_ago * 100
        else:
            change_20d = None

        # Pre-market: try to get today's data with prepost
        try:
            premarket = ticker.history(period="1d", interval="1m", prepost=True)
            if not premarket.empty:
                if isinstance(premarket.columns, pd.MultiIndex):
                    premarket = premarket.droplevel("Ticker", axis=1)
                current_price = float(premarket["Close"].iloc[-1])
            else:
                current_price = prev_close
        except Exception:
            current_price = prev_close

        premarket_change = (current_price - prev_close) / prev_close * 100
        gap_pct = premarket_change  # gap = premarket vs prev close

        return {
            "spy_price": round(current_price, 2),
            "spy_prev_close": round(prev_close, 2),
            "spy_prev_open": round(prev_open, 2),
            "spy_prev_high": round(prev_high, 2),
            "spy_prev_low": round(prev_low, 2),
            "spy_premarket_change_pct": round(premarket_change, 2),
            "spy_5d_change_pct": round(change_5d, 2) if change_5d is not None else None,
            "spy_20d_change_pct": round(change_20d, 2) if change_20d is not None else None,
            "spy_distance_from_20sma_pct": round(distance_from_sma, 2),
            "spy_volume_ratio": round(volume_ratio, 2),
            "gap_pct": round(gap_pct, 2),
        }

    def fetch_vix(self) -> Dict[str, float]:
        """Fetch current VIX level."""
        logger.info("Fetching VIX...")
        ticker = yf.Ticker("^VIX")
        hist = ticker.history(period="5d", interval="1d")
        if hist.empty:
            logger.warning("No VIX data available.")
            return {"vix": 0.0, "vix_prev_close": 0.0, "vix_change": 0.0}

        if isinstance(hist.columns, pd.MultiIndex):
            hist = hist.droplevel("Ticker", axis=1)

        vix = float(hist["Close"].iloc[-1])
        vix_prev = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else vix
        return {
            "vix": round(vix, 2),
            "vix_prev_close": round(vix_prev, 2),
            "vix_change": round(vix - vix_prev, 2),
        }

    def fetch_rates_fx(self) -> Dict[str, Optional[float]]:
        """Fetch 10Y yield and DXY."""
        logger.info("Fetching rates/FX...")
        result: Dict[str, Optional[float]] = {
            "us10y_yield": None,
            "us10y_change_bps": None,
            "dxy": None,
            "dxy_change_pct": None,
        }

        # 10Y Treasury
        try:
            tnx = yf.Ticker("^TNX")
            tnx_hist = tnx.history(period="5d", interval="1d")
            if not tnx_hist.empty:
                if isinstance(tnx_hist.columns, pd.MultiIndex):
                    tnx_hist = tnx_hist.droplevel("Ticker", axis=1)
                result["us10y_yield"] = round(float(tnx_hist["Close"].iloc[-1]), 3)
                if len(tnx_hist) >= 2:
                    prev = float(tnx_hist["Close"].iloc[-2])
                    result["us10y_change_bps"] = round(
                        (float(tnx_hist["Close"].iloc[-1]) - prev) * 100, 1
                    )
        except Exception as e:
            logger.warning("Failed to fetch ^TNX: %s", e)

        time.sleep(1)  # Rate limit yfinance

        # Dollar Index
        try:
            dxy = yf.Ticker("DX-Y.NYB")
            dxy_hist = dxy.history(period="5d", interval="1d")
            if not dxy_hist.empty:
                if isinstance(dxy_hist.columns, pd.MultiIndex):
                    dxy_hist = dxy_hist.droplevel("Ticker", axis=1)
                dxy_close = float(dxy_hist["Close"].iloc[-1])
                result["dxy"] = round(dxy_close, 3)
                if len(dxy_hist) >= 2:
                    dxy_prev = float(dxy_hist["Close"].iloc[-2])
                    result["dxy_change_pct"] = round(
                        (dxy_close - dxy_prev) / dxy_prev * 100, 2
                    )
        except Exception as e:
            logger.warning("Failed to fetch DX-Y.NYB: %s", e)

        return result

    def fetch_options_chain(
        self, ticker: str, min_dte: int = 2, max_dte: int = 14
    ) -> Dict[str, Any]:
        """Fetch options chain for strike selection."""
        logger.info("Fetching options chain for %s...", ticker)
        try:
            t = yf.Ticker(ticker)
            expiries = t.options
            if not expiries:
                return {"ticker": ticker, "expiry": None, "calls": [], "puts": []}

            today = date_type.today()
            valid_expiries = []
            for exp_str in expiries:
                exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
                dte = (exp_date - today).days
                if min_dte <= dte <= max_dte:
                    valid_expiries.append(exp_str)

            if not valid_expiries:
                # Fall back to nearest expiry
                valid_expiries = [expiries[0]]

            expiry = valid_expiries[0]
            chain = t.option_chain(expiry)

            # Get current price for filtering strikes
            hist = t.history(period="1d")
            if isinstance(hist.columns, pd.MultiIndex):
                hist = hist.droplevel("Ticker", axis=1)
            current_price = float(hist["Close"].iloc[-1]) if not hist.empty else 0

            # Filter to strikes within 1.5% of current price
            strike_range = current_price * 0.015

            calls_df = chain.calls
            calls_near = calls_df[
                (calls_df["strike"] >= current_price - strike_range)
                & (calls_df["strike"] <= current_price + strike_range)
            ]
            calls_list = []
            for _, row in calls_near.iterrows():
                calls_list.append({
                    "strike": float(row["strike"]),
                    "lastPrice": float(row.get("lastPrice", 0)),
                    "bid": float(row.get("bid", 0)),
                    "ask": float(row.get("ask", 0)),
                    "openInterest": int(row.get("openInterest", 0)),
                })

            puts_df = chain.puts
            puts_near = puts_df[
                (puts_df["strike"] >= current_price - strike_range)
                & (puts_df["strike"] <= current_price + strike_range)
            ]
            puts_list = []
            for _, row in puts_near.iterrows():
                puts_list.append({
                    "strike": float(row["strike"]),
                    "lastPrice": float(row.get("lastPrice", 0)),
                    "bid": float(row.get("bid", 0)),
                    "ask": float(row.get("ask", 0)),
                    "openInterest": int(row.get("openInterest", 0)),
                })

            # Preferred strikes: nearest ATM
            preferred_call = min(
                (c["strike"] for c in calls_list if c["strike"] >= current_price),
                default=current_price,
            )
            preferred_put = max(
                (p["strike"] for p in puts_list if p["strike"] <= current_price),
                default=current_price,
            )

            return {
                "ticker": ticker,
                "current_price": round(current_price, 2),
                "expiry": expiry,
                "calls": calls_list[:10],
                "puts": puts_list[:10],
                "preferred_call_strike": round(preferred_call, 2),
                "preferred_put_strike": round(preferred_put, 2),
            }
        except Exception as e:
            logger.warning("Failed to fetch options for %s: %s", ticker, e)
            return {"ticker": ticker, "expiry": None, "calls": [], "puts": []}

    def scan_watchlist(
        self,
        tickers: List[str],
        earnings_tickers: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Scan watchlist tickers for technical setups (oversold, momentum, etc.)."""
        logger.info("Scanning watchlist: %s", tickers)
        earnings_tickers = earnings_tickers or []
        scans = []
        for ticker_str in tickers:
            try:
                t = yf.Ticker(ticker_str)
                hist = t.history(period="2mo", interval="1d")
                if hist.empty:
                    continue
                if isinstance(hist.columns, pd.MultiIndex):
                    hist = hist.droplevel("Ticker", axis=1)

                close = float(hist["Close"].iloc[-1])
                prev_close = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else close
                day_change_pct = round((close - prev_close) / prev_close * 100, 2)

                # 5-day return
                if len(hist) >= 6:
                    c5 = float(hist["Close"].iloc[-6])
                    change_5d = round((close - c5) / c5 * 100, 2)
                else:
                    change_5d = None

                # 20-day return
                if len(hist) >= 21:
                    c20 = float(hist["Close"].iloc[-21])
                    change_20d = round((close - c20) / c20 * 100, 2)
                else:
                    change_20d = None

                # 20-day SMA distance
                sma20 = float(hist["Close"].rolling(20).mean().iloc[-1])
                dist_sma = round((close - sma20) / sma20 * 100, 2)

                # Volume ratio
                avg_vol = float(hist["Volume"].rolling(20).mean().iloc[-1])
                vol_ratio = round(float(hist["Volume"].iloc[-1]) / avg_vol, 2) if avg_vol > 0 else 1.0

                # 20-day low for near_support
                low_20d = float(hist["Low"].rolling(20).min().iloc[-1])

                # Determine signals with weights
                signals = []
                signal_score = 0.0

                # Extreme oversold: 5d < -5% AND 20d < -10%
                if (change_5d is not None and change_5d < -5
                        and change_20d is not None and change_20d < -10):
                    signals.append("extreme_oversold")
                    signal_score += 3.0
                elif change_20d is not None and change_20d < -10:
                    signals.append("deeply_oversold")
                    signal_score += 2.0
                elif change_20d is not None and change_20d < -5:
                    signals.append("oversold")
                    signal_score += 1.5

                # Near support: within 1% of 20-day low
                if low_20d > 0 and (close - low_20d) / low_20d * 100 < 1.0:
                    signals.append("near_support")
                    signal_score += 1.5

                if dist_sma < -3:
                    signals.append("well_below_sma")
                    signal_score += 2.0
                elif dist_sma < -1:
                    signals.append("below_sma")
                    signal_score += 1.0

                if change_5d is not None and change_5d < -3:
                    signals.append("5d_selloff")
                    signal_score += 1.0
                if day_change_pct > 3:
                    signals.append("momentum_breakout")
                    signal_score += 1.0
                if vol_ratio > 1.5:
                    signals.append("high_volume")
                    signal_score += 1.0
                if change_20d is not None and change_20d > 10:
                    signals.append("overbought")
                    signal_score += 0.5

                # Earnings soon cross-reference
                if ticker_str in earnings_tickers:
                    signals.append("earnings_soon")
                    signal_score += 1.5

                scans.append({
                    "ticker": ticker_str,
                    "price": round(close, 2),
                    "day_change_pct": day_change_pct,
                    "change_5d_pct": change_5d,
                    "change_20d_pct": change_20d,
                    "distance_from_20sma_pct": dist_sma,
                    "volume_ratio": vol_ratio,
                    "signals": signals,
                    "signal_score": round(signal_score, 1),
                })
                time.sleep(0.5)
            except Exception as e:
                logger.warning("Failed to scan %s: %s", ticker_str, e)
                continue

        # Sort: highest signal quality score first
        scans.sort(key=lambda x: x.get("signal_score", 0), reverse=True)
        return scans

    def get_market_snapshot(self, as_of: str) -> MarketSnapshot:
        """Assemble complete MarketSnapshot from all data sources."""
        spy = self.fetch_spy_snapshot()
        time.sleep(1)
        vix = self.fetch_vix()
        time.sleep(1)
        rates_fx = self.fetch_rates_fx()

        return MarketSnapshot(
            as_of=as_of,
            spy_price=spy["spy_price"],
            spy_prev_close=spy["spy_prev_close"],
            spy_premarket_change_pct=spy["spy_premarket_change_pct"],
            spy_5d_change_pct=spy.get("spy_5d_change_pct"),
            spy_20d_change_pct=spy.get("spy_20d_change_pct"),
            spy_distance_from_20sma_pct=spy.get("spy_distance_from_20sma_pct"),
            spy_volume_ratio=spy.get("spy_volume_ratio"),
            vix=vix["vix"],
            vix_prev_close=vix.get("vix_prev_close"),
            vix_change=vix.get("vix_change"),
            us10y_yield=rates_fx.get("us10y_yield"),
            us10y_change_bps=rates_fx.get("us10y_change_bps"),
            dxy=rates_fx.get("dxy"),
            dxy_change_pct=rates_fx.get("dxy_change_pct"),
            gap_pct=spy["gap_pct"],
        )


# ---------------------------------------------------------------------------
# Rule Engine
# ---------------------------------------------------------------------------
class RuleEngine:
    """Evaluate the 18 backtested trading rules against current conditions."""

    def __init__(self, rules_path: Path = RULES_PATH):
        with open(rules_path) as f:
            self.rules = json.load(f)

    def evaluate_all(
        self,
        snapshot: MarketSnapshot,
        research: Dict[str, Any],
        history_df: Optional[pd.DataFrame] = None,
        watchlist_scans: Optional[List[Dict[str, Any]]] = None,
    ) -> List[RuleSignal]:
        """Evaluate all rules and return list of RuleSignal."""
        signals = []
        for rule in self.rules:
            ctype = rule.get("condition_type", "")
            evaluator = self._get_evaluator(ctype)
            if evaluator:
                signal = evaluator(rule, snapshot, research, history_df, watchlist_scans)
                if isinstance(signal, list):
                    signals.extend(signal)
                else:
                    signals.append(signal)
            else:
                signals.append(
                    RuleSignal(
                        rule_id=rule["id"],
                        rule_name=rule["name"],
                        triggered=False,
                        direction=rule.get("direction", "neutral"),
                        confidence=rule.get("confidence", "low"),
                        win_rate=rule.get("win_rate", 0),
                        sample_size=rule.get("sample_size", 0),
                        reasoning=f"No evaluator for condition_type={ctype}",
                    )
                )

        # Compute priority weights for all signals
        for sig in signals:
            sig.weight = compute_rule_weight(sig.win_rate, sig.confidence, sig.sample_size)

        return signals

    def _get_evaluator(self, condition_type: str):
        """Dispatch to the right evaluator method."""
        dispatch = {
            "spy_prev_day_drop": self._eval_spy_prev_day_drop,
            "inflation_surprise": self._eval_inflation_surprise,
            "nfp_beat": self._eval_nfp_beat,
            "vix_level": self._eval_vix_level,
            "fomc_hold": self._eval_fomc_hold,
            "ism_contraction": self._eval_ism_contraction,
            "yield_drop": self._eval_yield_drop,
            "gap_down": self._eval_gap_down,
            "below_sma": self._eval_below_sma,
            "volume_spike": self._eval_volume_spike,
            "consecutive_down": self._eval_consecutive_down,
            "rally_short": self._eval_rally_short,
            "low_yield_high_vix": self._eval_low_yield_high_vix,
            "dxy_up_vix_up": self._eval_dxy_up_vix_up,
            "sell_the_news": self._eval_sell_the_news,
            "megacap_earnings_vol": self._eval_megacap_earnings_vol,
            "all_bullish_contrarian": self._eval_all_bullish_contrarian,
            "geopolitical_risk": self._eval_geopolitical_risk,
        }
        return dispatch.get(condition_type)

    def _eval_spy_prev_day_drop(self, rule, snapshot, research, history, watchlist_scans):
        """R1: SPY dropped >1% on a macro event day yesterday."""
        prev_change = snapshot.spy_premarket_change_pct
        # Use yesterday's actual change (premarket_change is proxy for gap)
        # In practice this checks if the prior session was a big drop
        # We approximate with: if SPY prev close is significantly lower than before
        val = prev_change
        triggered = val < rule["threshold"]
        return RuleSignal(
            rule_id=rule["id"],
            rule_name=rule["name"],
            triggered=triggered,
            direction="long",
            confidence=rule.get("confidence", "medium"),
            win_rate=rule["win_rate"],
            sample_size=rule["sample_size"],
            current_value=round(val, 2),
            threshold=rule["threshold"],
            reasoning=(
                f"SPY pre-market change: {val:+.2f}% (threshold: {rule['threshold']}%). "
                + ("Prior session saw a significant drop — buy signal." if triggered
                   else "No significant prior-day drop detected.")
            ),
        )

    def _eval_inflation_surprise(self, rule, snapshot, research, history, watchlist_scans):
        """R2: Positive surprise on CPI/PPI."""
        events = research.get("events", [])
        today_str = research.get("date", "")
        event_types = rule.get("event_types", [])

        for ev in events:
            ev_name = ev.get("event", "")
            dt = ev.get("date_time_local", "") or ""
            if today_str in dt and any(et.lower() in ev_name.lower() for et in event_types):
                return RuleSignal(
                    rule_id=rule["id"],
                    rule_name=rule["name"],
                    triggered=True,
                    direction="long",
                    confidence=rule.get("confidence", "medium"),
                    win_rate=rule["win_rate"],
                    sample_size=rule["sample_size"],
                    reasoning=(
                        f"CPI/PPI release scheduled today ({ev_name}). "
                        "Historically, positive surprises lead to gains. "
                        "CONDITIONAL: Wait for release data before entry."
                    ),
                )

        return RuleSignal(
            rule_id=rule["id"],
            rule_name=rule["name"],
            triggered=False,
            direction="long",
            confidence=rule.get("confidence", "medium"),
            win_rate=rule["win_rate"],
            sample_size=rule["sample_size"],
            reasoning="No CPI/PPI release scheduled today.",
        )

    def _eval_nfp_beat(self, rule, snapshot, research, history, watchlist_scans):
        """R3: NFP beat → Short."""
        events = research.get("events", [])
        today_str = research.get("date", "")

        for ev in events:
            ev_name = ev.get("event", "")
            dt = ev.get("date_time_local", "") or ""
            if today_str in dt and "nonfarm" in ev_name.lower() or "nfp" in ev_name.lower():
                return RuleSignal(
                    rule_id=rule["id"],
                    rule_name=rule["name"],
                    triggered=True,
                    direction="short",
                    confidence=rule.get("confidence", "low"),
                    win_rate=rule["win_rate"],
                    sample_size=rule["sample_size"],
                    reasoning=(
                        "NFP release today. If jobs beat expectations, consider short. "
                        "CONDITIONAL: Wait for 7:30 AM CT release data."
                    ),
                )

        return RuleSignal(
            rule_id=rule["id"],
            rule_name=rule["name"],
            triggered=False,
            direction="short",
            confidence=rule.get("confidence", "low"),
            win_rate=rule["win_rate"],
            sample_size=rule["sample_size"],
            reasoning="No NFP release scheduled today.",
        )

    def _eval_vix_level(self, rule, snapshot, research, history, watchlist_scans):
        """R4: VIX >20 → Buy. Best performing rule (100% win, 7/7)."""
        val = snapshot.vix
        triggered = val > rule["threshold"]
        return RuleSignal(
            rule_id=rule["id"],
            rule_name=rule["name"],
            triggered=triggered,
            direction="long",
            confidence=rule.get("confidence", "high"),
            win_rate=rule["win_rate"],
            sample_size=rule["sample_size"],
            current_value=round(val, 2),
            threshold=rule["threshold"],
            reasoning=(
                f"VIX at {val:.2f} (threshold: {rule['threshold']}). "
                + ("Elevated fear — historically 100% win rate buying here."
                   if triggered else "VIX below threshold, no signal.")
            ),
        )

    def _eval_fomc_hold(self, rule, snapshot, research, history, watchlist_scans):
        """R5: FOMC holds rates → Buy."""
        events = research.get("events", [])
        today_str = research.get("date", "")

        for ev in events:
            ev_name = ev.get("event", "")
            dt = ev.get("date_time_local", "") or ""
            if today_str in dt and "fomc" in ev_name.lower():
                return RuleSignal(
                    rule_id=rule["id"],
                    rule_name=rule["name"],
                    triggered=True,
                    direction="long",
                    confidence=rule.get("confidence", "medium"),
                    win_rate=rule["win_rate"],
                    sample_size=rule["sample_size"],
                    reasoning=(
                        "FOMC decision today. If rates held steady, buy signal triggers. "
                        "CONDITIONAL: Wait for 1:00 PM CT announcement."
                    ),
                )

        return RuleSignal(
            rule_id=rule["id"],
            rule_name=rule["name"],
            triggered=False,
            direction="long",
            confidence=rule.get("confidence", "medium"),
            win_rate=rule["win_rate"],
            sample_size=rule["sample_size"],
            reasoning="No FOMC decision scheduled today.",
        )

    def _eval_ism_contraction(self, rule, snapshot, research, history, watchlist_scans):
        """R6: ISM Mfg <50 → Buy."""
        events = research.get("events", [])
        today_str = research.get("date", "")

        for ev in events:
            ev_name = ev.get("event", "")
            dt = ev.get("date_time_local", "") or ""
            if today_str in dt and "ism" in ev_name.lower() and "manufactur" in ev_name.lower():
                return RuleSignal(
                    rule_id=rule["id"],
                    rule_name=rule["name"],
                    triggered=True,
                    direction="long",
                    confidence=rule.get("confidence", "medium"),
                    win_rate=rule["win_rate"],
                    sample_size=rule["sample_size"],
                    reasoning=(
                        "ISM Manufacturing scheduled today. If below 50 (contraction), buy signal. "
                        "CONDITIONAL: Wait for 9:00 AM CT release."
                    ),
                )

        return RuleSignal(
            rule_id=rule["id"],
            rule_name=rule["name"],
            triggered=False,
            direction="long",
            confidence=rule.get("confidence", "medium"),
            win_rate=rule["win_rate"],
            sample_size=rule["sample_size"],
            reasoning="No ISM Manufacturing release scheduled today.",
        )

    def _eval_yield_drop(self, rule, snapshot, research, history, watchlist_scans):
        """R7: 10Y yield drops >5bps → Buy."""
        val = snapshot.us10y_change_bps
        if val is None:
            return RuleSignal(
                rule_id=rule["id"],
                rule_name=rule["name"],
                triggered=False,
                direction="long",
                confidence=rule.get("confidence", "medium"),
                win_rate=rule["win_rate"],
                sample_size=rule["sample_size"],
                reasoning="10Y yield change data not available.",
            )
        triggered = val < rule["threshold"]
        return RuleSignal(
            rule_id=rule["id"],
            rule_name=rule["name"],
            triggered=triggered,
            direction="long",
            confidence=rule.get("confidence", "medium"),
            win_rate=rule["win_rate"],
            sample_size=rule["sample_size"],
            current_value=round(val, 1),
            threshold=rule["threshold"],
            reasoning=(
                f"10Y yield change: {val:+.1f} bps (threshold: {rule['threshold']} bps). "
                + ("Yields dropping sharply — risk-on signal." if triggered
                   else "No significant yield drop.")
            ),
        )

    def _eval_gap_down(self, rule, snapshot, research, history, watchlist_scans):
        """R8: Gap down >0.3% → Buy the dip."""
        val = snapshot.gap_pct
        triggered = val < rule["threshold"]
        return RuleSignal(
            rule_id=rule["id"],
            rule_name=rule["name"],
            triggered=triggered,
            direction="long",
            confidence=rule.get("confidence", "high"),
            win_rate=rule["win_rate"],
            sample_size=rule["sample_size"],
            current_value=round(val, 2),
            threshold=rule["threshold"],
            reasoning=(
                f"SPY gap: {val:+.2f}% (threshold: {rule['threshold']}%). "
                + ("Gap-down buy signal — historically fills with 86% success."
                   if triggered else "No significant gap down.")
            ),
        )

    def _eval_below_sma(self, rule, snapshot, research, history, watchlist_scans):
        """R9: SPY below 20-day SMA → Buy."""
        val = snapshot.spy_distance_from_20sma_pct
        if val is None:
            return RuleSignal(
                rule_id=rule["id"],
                rule_name=rule["name"],
                triggered=False,
                direction="long",
                confidence=rule.get("confidence", "high"),
                win_rate=rule["win_rate"],
                sample_size=rule["sample_size"],
                reasoning="SMA data not available.",
            )
        triggered = val < rule["threshold"]
        return RuleSignal(
            rule_id=rule["id"],
            rule_name=rule["name"],
            triggered=triggered,
            direction="long",
            confidence=rule.get("confidence", "high"),
            win_rate=rule["win_rate"],
            sample_size=rule["sample_size"],
            current_value=round(val, 2),
            threshold=rule["threshold"],
            reasoning=(
                f"SPY distance from 20-SMA: {val:+.2f}% (threshold: below 0%). "
                + ("Below SMA — mean reversion buy signal." if triggered
                   else "SPY trading above 20-day SMA.")
            ),
        )

    def _eval_volume_spike(self, rule, snapshot, research, history, watchlist_scans):
        """R10: Volume >1.3x average → Buy."""
        val = snapshot.spy_volume_ratio
        if val is None:
            return RuleSignal(
                rule_id=rule["id"],
                rule_name=rule["name"],
                triggered=False,
                direction="long",
                confidence=rule.get("confidence", "medium"),
                win_rate=rule["win_rate"],
                sample_size=rule["sample_size"],
                reasoning="Volume ratio data not available.",
            )
        triggered = val > rule["threshold"]
        return RuleSignal(
            rule_id=rule["id"],
            rule_name=rule["name"],
            triggered=triggered,
            direction="long",
            confidence=rule.get("confidence", "medium"),
            win_rate=rule["win_rate"],
            sample_size=rule["sample_size"],
            current_value=round(val, 2),
            threshold=rule["threshold"],
            reasoning=(
                f"Volume ratio: {val:.2f}x (threshold: {rule['threshold']}x). "
                + ("High volume capitulation — buy signal." if triggered
                   else "Volume not elevated enough.")
            ),
        )

    def _eval_consecutive_down(self, rule, snapshot, research, history, watchlist_scans):
        """R11: 5-day change <-2% → Buy."""
        val = snapshot.spy_5d_change_pct
        if val is None:
            return RuleSignal(
                rule_id=rule["id"],
                rule_name=rule["name"],
                triggered=False,
                direction="long",
                confidence=rule.get("confidence", "medium"),
                win_rate=rule["win_rate"],
                sample_size=rule["sample_size"],
                reasoning="5-day change data not available.",
            )
        triggered = val < rule["threshold"]
        return RuleSignal(
            rule_id=rule["id"],
            rule_name=rule["name"],
            triggered=triggered,
            direction="long",
            confidence=rule.get("confidence", "medium"),
            win_rate=rule["win_rate"],
            sample_size=rule["sample_size"],
            current_value=round(val, 2),
            threshold=rule["threshold"],
            reasoning=(
                f"SPY 5-day change: {val:+.2f}% (threshold: {rule['threshold']}%). "
                + ("Extended weakness — bounce-back buy signal." if triggered
                   else "No extended weakness detected.")
            ),
        )

    def _eval_rally_short(self, rule, snapshot, research, history, watchlist_scans):
        """R12: Rally >1% → Short. WARNING: FAILED rule (33% win rate)."""
        val = snapshot.spy_premarket_change_pct
        triggered = val > rule["threshold"]
        return RuleSignal(
            rule_id=rule["id"],
            rule_name=rule["name"],
            triggered=triggered,
            direction="short",
            confidence="failed",
            win_rate=rule["win_rate"],
            sample_size=rule["sample_size"],
            current_value=round(val, 2),
            threshold=rule["threshold"],
            reasoning=(
                f"SPY change: {val:+.2f}% (threshold: {rule['threshold']}%). "
                "WARNING: This rule FAILED backtesting (33% win rate). "
                + ("Triggered but NOT recommended — use as counter-signal only."
                   if triggered else "Not triggered.")
            ),
        )

    def _eval_low_yield_high_vix(self, rule, snapshot, research, history, watchlist_scans):
        """R13: Low yield + high VIX → Buy."""
        vix_val = snapshot.vix
        yield_change = snapshot.us10y_change_bps
        vix_thresh = rule.get("threshold_vix", 18.0)
        yield_thresh = rule.get("threshold_yield_change", -2.0)

        if yield_change is None:
            return RuleSignal(
                rule_id=rule["id"],
                rule_name=rule["name"],
                triggered=False,
                direction="long",
                confidence=rule.get("confidence", "high"),
                win_rate=rule["win_rate"],
                sample_size=rule["sample_size"],
                reasoning="Yield change data not available for compound rule.",
            )

        triggered = vix_val > vix_thresh and yield_change < yield_thresh
        return RuleSignal(
            rule_id=rule["id"],
            rule_name=rule["name"],
            triggered=triggered,
            direction="long",
            confidence=rule.get("confidence", "high"),
            win_rate=rule["win_rate"],
            sample_size=rule["sample_size"],
            current_value=round(vix_val, 2),
            reasoning=(
                f"VIX: {vix_val:.2f} (need >{vix_thresh}), "
                f"yield change: {yield_change:+.1f} bps (need <{yield_thresh}). "
                + ("Both conditions met — strong buy signal." if triggered
                   else "Compound condition not fully met.")
            ),
        )

    def _eval_dxy_up_vix_up(self, rule, snapshot, research, history, watchlist_scans):
        """R14: DXY up + VIX up → Buy the fear."""
        dxy_change = snapshot.dxy_change_pct
        vix_change = snapshot.vix_change

        if dxy_change is None or vix_change is None:
            return RuleSignal(
                rule_id=rule["id"],
                rule_name=rule["name"],
                triggered=False,
                direction="long",
                confidence=rule.get("confidence", "medium"),
                win_rate=rule["win_rate"],
                sample_size=rule["sample_size"],
                reasoning="DXY or VIX change data not available.",
            )

        triggered = dxy_change > 0 and vix_change > 0
        return RuleSignal(
            rule_id=rule["id"],
            rule_name=rule["name"],
            triggered=triggered,
            direction="long",
            confidence=rule.get("confidence", "medium"),
            win_rate=rule["win_rate"],
            sample_size=rule["sample_size"],
            reasoning=(
                f"DXY change: {dxy_change:+.2f}%, VIX change: {vix_change:+.2f}. "
                + ("Both rising — buy-the-fear signal." if triggered
                   else "Conditions not met for compound fear signal.")
            ),
        )

    def _eval_sell_the_news(self, rule, snapshot, research, history, watchlist_scans):
        """R15: Sell-the-News Post-Earnings Fade.

        Trigger: Ticker reported earnings yesterday after close (or today before
        open), rallied >5% in the 5 days before earnings, and is now trading
        flat or down.  Returns a list of RuleSignal (one per matching ticker).
        """
        signals: list[RuleSignal] = []
        if not watchlist_scans:
            signals.append(RuleSignal(
                rule_id=rule["id"],
                rule_name=rule["name"],
                triggered=False,
                direction="short",
                confidence=rule.get("confidence", "medium"),
                win_rate=rule.get("win_rate", 0.65),
                sample_size=rule.get("sample_size", 0),
                reasoning="No watchlist scan data available for sell-the-news evaluation.",
            ))
            return signals

        rally_thresh = rule.get("threshold_pre_earnings_rally_pct", 5.0)
        post_thresh = rule.get("threshold_post_earnings_change_pct", 0.0)
        earnings_list = research.get("earnings", [])

        # Build lookup: ticker → earnings info
        earnings_by_ticker = {}
        for e in earnings_list:
            t = e.get("ticker", "")
            if t:
                earnings_by_ticker[t] = e

        for scan in watchlist_scans:
            ticker = scan.get("ticker", "")
            if ticker not in earnings_by_ticker:
                continue

            earning = earnings_by_ticker[ticker]
            time_hint = (earning.get("time_hint") or "").lower()

            # We look for stocks that just reported: "after close" yesterday
            # or "before open" today.  The scan data reflects current-day prices.
            is_post_earnings = "after" in time_hint or "before" in time_hint

            if not is_post_earnings:
                continue

            change_5d = scan.get("change_5d_pct")
            day_change = scan.get("day_change_pct", 0.0)

            # Check: rallied >5% in prior 5 days AND now flat/down
            pre_rally = change_5d is not None and change_5d > rally_thresh
            post_fade = day_change <= post_thresh

            triggered = pre_rally and post_fade

            signals.append(RuleSignal(
                rule_id=rule["id"],
                rule_name=rule["name"],
                triggered=triggered,
                direction="short",
                confidence=rule.get("confidence", "medium"),
                win_rate=rule.get("win_rate", 0.65),
                sample_size=rule.get("sample_size", 0),
                current_value=day_change,
                target_ticker=ticker,
                reasoning=(
                    f"{ticker}: 5d rally {change_5d:+.1f}% (need >{rally_thresh}%), "
                    f"post-earnings {day_change:+.1f}% (need <={post_thresh}%). "
                    + (f"SELL-THE-NEWS detected — classic fade pattern on {ticker}. Consider puts."
                       if triggered
                       else f"Conditions not met for sell-the-news on {ticker}.")
                ),
            ))

        if not signals:
            signals.append(RuleSignal(
                rule_id=rule["id"],
                rule_name=rule["name"],
                triggered=False,
                direction="short",
                confidence=rule.get("confidence", "medium"),
                win_rate=rule.get("win_rate", 0.65),
                sample_size=rule.get("sample_size", 0),
                reasoning="No post-earnings tickers found in watchlist scans.",
            ))

        return signals

    def _eval_megacap_earnings_vol(self, rule, snapshot, research, history, watchlist_scans):
        """R16: Mega-Cap Earnings Vol Event.

        Case A: Mega-cap just reported with muted reaction (<1% move on a beat).
        Case B: Mega-cap reports today after close — binary event warning.
        Returns a list of RuleSignal (one per matching ticker).
        """
        signals: list[RuleSignal] = []
        mega_caps = rule.get("mega_cap_list", [
            "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA",
        ])
        premarket_thresh = rule.get("threshold_premarket_move_pct", 1.0)
        earnings_list = research.get("earnings", [])

        earnings_by_ticker = {}
        for e in earnings_list:
            t = e.get("ticker", "")
            if t:
                earnings_by_ticker[t] = e

        scans_by_ticker = {}
        if watchlist_scans:
            for s in watchlist_scans:
                scans_by_ticker[s.get("ticker", "")] = s

        for ticker in mega_caps:
            if ticker not in earnings_by_ticker:
                continue

            earning = earnings_by_ticker[ticker]
            time_hint = (earning.get("time_hint") or "").lower()
            scan = scans_by_ticker.get(ticker, {})
            day_change = scan.get("day_change_pct", 0.0)

            # Case A: Just reported (after close yesterday or before open today)
            # and reaction is muted (<1% absolute move)
            if "after" in time_hint or "before" in time_hint:
                muted = abs(day_change) < premarket_thresh
                if muted:
                    signals.append(RuleSignal(
                        rule_id=rule["id"],
                        rule_name=rule["name"],
                        triggered=True,
                        direction="short",
                        confidence=rule.get("confidence", "medium"),
                        win_rate=rule.get("win_rate", 0.60),
                        sample_size=rule.get("sample_size", 0),
                        current_value=day_change,
                        target_ticker=ticker,
                        reasoning=(
                            f"{ticker} reported earnings — muted reaction "
                            f"({day_change:+.1f}%, threshold <{premarket_thresh}%). "
                            "Market not rewarding the beat. Reduce long exposure."
                        ),
                    ))
                    continue

            # Case B: Reports today after close — upcoming binary event
            if "after" in time_hint:
                signals.append(RuleSignal(
                    rule_id=rule["id"],
                    rule_name=rule["name"],
                    triggered=True,
                    direction="short",
                    confidence=rule.get("confidence", "medium"),
                    win_rate=rule.get("win_rate", 0.60),
                    sample_size=rule.get("sample_size", 0),
                    current_value=None,
                    target_ticker=ticker,
                    reasoning=(
                        f"{ticker} reports earnings after close today. "
                        "Binary vol event — reduce sizing on {ticker} positions "
                        "and consider hedges."
                    ),
                ))

        if not signals:
            signals.append(RuleSignal(
                rule_id=rule["id"],
                rule_name=rule["name"],
                triggered=False,
                direction="short",
                confidence=rule.get("confidence", "medium"),
                win_rate=rule.get("win_rate", 0.60),
                sample_size=rule.get("sample_size", 0),
                reasoning="No mega-cap earnings events detected.",
            ))

        return signals

    def _eval_all_bullish_contrarian(self, rule, snapshot, research, history, watchlist_scans):
        """R17: All-Bullish Contrarian Warning (stub).

        Real logic runs post-recommendation in _validate_and_fix_recommendations().
        This stub always returns not-triggered.
        """
        return RuleSignal(
            rule_id=rule["id"],
            rule_name=rule["name"],
            triggered=False,
            direction="neutral",
            confidence=rule.get("confidence", "medium"),
            win_rate=rule.get("win_rate", 0.50),
            sample_size=rule.get("sample_size", 0),
            reasoning=(
                "Contrarian check deferred to post-recommendation validation. "
                "If all scan signals are bullish AND all recommendations are calls, "
                "a mandatory 20% hedge will be enforced."
            ),
        )

    def _eval_geopolitical_risk(self, rule, snapshot, research, history, watchlist_scans):
        """R18: Geopolitical Risk Escalation.

        Scans headlines and weekly_context themes for geopolitical keywords.
        Scores each match weighted by impact (high=3, medium=2, low=1).
        Oil spike >threshold adds a bonus. Triggered if score >= threshold.
        """
        keywords = [kw.lower() for kw in rule.get("geopolitical_keywords", [])]
        threshold_score = rule.get("threshold_headline_score", 6)
        oil_spike_thresh = rule.get("threshold_oil_spike_pct", 3.0)

        score = 0
        matches = []

        # Scan headlines
        headlines = research.get("headlines", [])
        for hl in headlines:
            title = (hl.get("title") or "").lower()
            take = (hl.get("one_line_take") or "").lower()
            impact = (hl.get("impact") or "low").lower()
            weight = {"high": 3, "medium": 2, "low": 1}.get(impact, 1)
            text = f"{title} {take}"
            for kw in keywords:
                if kw in text:
                    score += weight
                    matches.append(f"{kw} ({impact})")
                    break  # one match per headline

        # Scan weekly_context themes
        weekly = research.get("weekly_context", {})
        for theme_obj in weekly.get("themes", []):
            theme_text = (theme_obj.get("theme") or "").lower()
            evidence_texts = " ".join(
                (e or "").lower() for e in theme_obj.get("evidence", [])
            )
            combined = f"{theme_text} {evidence_texts}"
            for kw in keywords:
                if kw in combined:
                    score += 2  # themes are medium weight
                    matches.append(f"{kw} (theme)")
                    break

        # Oil spike reinforcement
        oil_bonus = False
        market_state = research.get("market_state", {})
        rfx = market_state.get("rates_fx_oil", {})
        wti_change_str = rfx.get("wti_change", "")
        try:
            wti_val = float(str(wti_change_str).replace("%", "").replace("+", "").strip())
            if wti_val > oil_spike_thresh:
                score += 3
                oil_bonus = True
                matches.append(f"oil +{wti_val:.1f}%")
        except (ValueError, TypeError):
            pass

        triggered = score >= threshold_score
        return RuleSignal(
            rule_id=rule["id"],
            rule_name=rule["name"],
            triggered=triggered,
            direction="short",
            confidence=rule.get("confidence", "medium"),
            win_rate=rule.get("win_rate", 0.55),
            sample_size=rule.get("sample_size", 0),
            current_value=float(score),
            threshold=float(threshold_score),
            reasoning=(
                f"Geopolitical risk score: {score} (threshold: {threshold_score}). "
                f"Matches: {', '.join(matches[:8]) if matches else 'none'}. "
                + (f"Oil spike reinforcement active. " if oil_bonus else "")
                + ("GEOPOLITICAL RISK ELEVATED — reduce long exposure, increase hedges."
                   if triggered
                   else "Geopolitical risk within normal range.")
            ),
        )


# ---------------------------------------------------------------------------
# Markdown brief builder
# ---------------------------------------------------------------------------
def build_recommendation_brief(rec_pack: RecommendationPack) -> str:
    """Generate the markdown text for the recommendations email section."""
    snap = rec_pack.market_snapshot
    run_label = "INTRADAY RE-RUN" if rec_pack.is_intraday_rerun else "Pre-Market"
    lines = [
        f"## Trading Recommendations - {rec_pack.date}",
    ]
    if rec_pack.is_intraday_rerun:
        lines.append("")
        lines.append("> **INTRADAY RE-RUN** — Updated signals based on mid-day market conditions.")
    lines.extend([
        "",
        f"### Market Snapshot ({run_label} {snap.as_of})",
        "| Metric | Value | Signal |",
        "|--------|-------|--------|",
    ])

    # SPY
    spy_signal = "Gap down" if snap.gap_pct < -0.2 else ("Gap up" if snap.gap_pct > 0.2 else "Flat")
    lines.append(f"| SPY | ${snap.spy_price:.2f} (pre: {snap.spy_premarket_change_pct:+.2f}%) | {spy_signal} |")

    # VIX
    vix_signal = "Elevated" if snap.vix > 20 else ("Low" if snap.vix < 15 else "Moderate")
    vix_change_str = f"{snap.vix_change:+.1f}" if snap.vix_change is not None else "N/A"
    lines.append(f"| VIX | {snap.vix:.1f} ({vix_change_str}) | {vix_signal} |")

    # 10Y
    if snap.us10y_yield is not None:
        yield_change_str = f"{snap.us10y_change_bps:+.0f} bps" if snap.us10y_change_bps else "N/A"
        lines.append(f"| 10Y Yield | {snap.us10y_yield:.3f}% ({yield_change_str}) | |")

    # DXY
    if snap.dxy is not None:
        dxy_str = f"{snap.dxy_change_pct:+.2f}%" if snap.dxy_change_pct else "N/A"
        lines.append(f"| DXY | {snap.dxy:.2f} ({dxy_str}) | |")

    # Trend context
    if snap.spy_5d_change_pct is not None:
        lines.append(f"| 5-Day Change | {snap.spy_5d_change_pct:+.2f}% | |")
    if snap.spy_distance_from_20sma_pct is not None:
        sma_signal = "Below SMA" if snap.spy_distance_from_20sma_pct < 0 else "Above SMA"
        lines.append(f"| Dist from 20 SMA | {snap.spy_distance_from_20sma_pct:+.2f}% | {sma_signal} |")

    # Active signals
    triggered = [s for s in rec_pack.active_signals if s.triggered]
    failed_triggers = [s for s in triggered if s.confidence == "failed"]
    active_triggers = [s for s in triggered if s.confidence != "failed"]

    # Split into SPY-level and per-ticker signals
    spy_triggers = [s for s in active_triggers if s.target_ticker is None]
    ticker_triggers = [s for s in active_triggers if s.target_ticker is not None]

    total_rules = rec_pack.portfolio_summary.rules_total_count
    lines.extend([
        "",
        f"### Active Rule Signals ({len(active_triggers)} of {total_rules} triggered)",
    ])

    if spy_triggers:
        lines.extend([
            "| Rule | Signal | Confidence | Weight | Detail |",
            "|------|--------|------------|--------|--------|",
        ])
        for s in spy_triggers:
            direction = "BUY" if s.direction == "long" else "SHORT"
            val_str = f"{s.current_value}" if s.current_value is not None else ""
            thresh_str = f" (threshold: {s.threshold})" if s.threshold is not None else ""
            lines.append(
                f"| {s.rule_id} {s.rule_name} | {direction} | "
                f"{s.confidence.upper()} | {s.weight:.2f} | {val_str}{thresh_str} — "
                f"{s.win_rate*100:.0f}% win rate, {s.sample_size} trades |"
            )
    elif not ticker_triggers:
        lines.append("_No rules triggered today._")

    # Per-ticker signals (R15/R16)
    if ticker_triggers:
        lines.extend(["", "**Per-Ticker Signals:**"])
        for s in ticker_triggers:
            direction = "BUY" if s.direction == "long" else "SHORT"
            ticker_label = f"[{s.target_ticker}]" if s.target_ticker else ""
            lines.append(
                f"- **{s.rule_id} {s.rule_name}** {ticker_label}: "
                f"{direction} ({s.confidence.upper()}) — {s.reasoning}"
            )

    if failed_triggers:
        lines.extend(["", "**Counter-signals (failed rules, for reference only):**"])
        for s in failed_triggers:
            lines.append(f"- {s.rule_id} {s.rule_name}: {s.reasoning}")

    # Confluence summary with weighted sums
    long_count = sum(1 for s in active_triggers if s.direction == "long")
    short_count = sum(1 for s in active_triggers if s.direction == "short")
    neutral_count = sum(1 for s in active_triggers if s.direction == "neutral")
    long_wt = sum(s.weight for s in active_triggers if s.direction == "long")
    short_wt = sum(s.weight for s in active_triggers if s.direction == "short")
    if active_triggers:
        confluence = f"**Confluence: {long_count} bullish (weight {long_wt:.2f}), {short_count} bearish (weight {short_wt:.2f})"
        if neutral_count:
            confluence += f", {neutral_count} neutral"
        confluence += " signals.**"
        lines.extend(["", confluence])

    # Trade recommendations
    lines.extend(["", "### Trade Recommendations"])
    if rec_pack.recommendations:
        for i, rec in enumerate(rec_pack.recommendations, 1):
            direction_str = "BUY CALL" if rec.direction == "call" else "BUY PUT"
            strike_str = f" {rec.strike}" if rec.strike else ""
            expiry_str = f" {rec.expiry}" if rec.expiry else ""
            lines.extend([
                f"",
                f"#### Trade {i}: {rec.ticker}{strike_str}{'C' if rec.direction == 'call' else 'P'}{expiry_str} ({direction_str})",
                f"- **Entry:** {rec.entry_timing}",
                f"- **Allocation:** ${rec.allocation_dollars:,.0f} | **Max Loss:** ${rec.max_loss_dollars:,.0f} ({rec.stop_loss_pct:.0f}% stop)",
                f"- **Take Profit:** {rec.take_profit_pct:.0f}% of premium",
                f"- **Rules:** {', '.join(rec.triggered_rules)}",
                f"- **Confidence:** {rec.confidence.upper()}",
                f"- **Reasoning:** {rec.reasoning}",
            ])
    else:
        lines.append("_No trade recommendations today — insufficient signal confluence._")

    # Sympathy plays
    if rec_pack.sympathy_plays:
        lines.extend(["", "### Sympathy Plays"])
        for sp in rec_pack.sympathy_plays:
            direction_str = "calls" if sp.direction == "call" else "puts"
            timing_str = f" [{sp.entry_timing}]" if sp.entry_timing else ""
            lines.append(
                f"- **{sp.sympathy_ticker}** (beta {sp.beta:.2f} to {sp.primary_ticker}): "
                f"{sp.reasoning} — consider {direction_str}{timing_str}"
            )

    # Portfolio summary
    ps = rec_pack.portfolio_summary
    lines.extend([
        "",
        "### Portfolio Summary",
        f"- **Total Allocation:** ${ps.total_allocation:,.0f}",
        f"- **Maximum Risk:** ${ps.max_portfolio_risk:,.0f}",
        f"- **Number of Trades:** {ps.num_trades}",
        f"- **Net Bias:** {ps.net_directional_bias.upper()} ({ps.rules_triggered_count}/{ps.rules_total_count} rules triggered)",
        "",
        "### Disclaimer",
        rec_pack.disclaimer,
    ])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Post-processing validation
# ---------------------------------------------------------------------------
def _validate_and_fix_recommendations(
    recs: List[TradeRecommendation],
    options_chains: Dict[str, Any],
    max_allocation: float,
    watchlist_scans: Optional[List[Dict[str, Any]]] = None,
    active_signals: Optional[List[RuleSignal]] = None,
) -> List[TradeRecommendation]:
    """Post-process recommendations to enforce hedge, strike, and allocation rules."""
    if not recs:
        return recs

    # --- R17: All-Bullish Contrarian Warning ---
    r17_triggered = False
    if watchlist_scans and active_signals:
        # Check if all scan signals are bullish (oversold/below_sma/near_support)
        bullish_scan_signals = {
            "extreme_oversold", "deeply_oversold", "oversold",
            "below_sma", "well_below_sma", "near_support",
            "5d_selloff", "earnings_soon",
        }
        all_scans_bullish = all(
            any(sig in bullish_scan_signals for sig in scan.get("signals", []))
            for scan in watchlist_scans
            if scan.get("signals")
        )
        # Check if all recommendations are calls (no puts)
        all_calls = all(r.direction == "call" for r in recs)

        r17_triggered = all_scans_bullish and all_calls and len(recs) >= 2
        if r17_triggered:
            logger.warning(
                "R17 CONTRARIAN WARNING: All scan signals bullish + all recs are calls. "
                "Enforcing 20%% hedge."
            )

    # --- R18: Geopolitical Risk Escalation ---
    r18_triggered = False
    if active_signals:
        for sig in active_signals:
            if sig.rule_id == "R18" and sig.triggered:
                r18_triggered = True
                break
    if r18_triggered:
        logger.warning(
            "R18 GEOPOLITICAL RISK: Elevated geopolitical risk detected. "
            "Scaling down long allocations by 25%% and increasing hedge to 30%%."
        )
        for rec in recs:
            if rec.direction == "call":
                rec.allocation_dollars = round(rec.allocation_dollars * 0.75)
                rec.max_loss_dollars = round(rec.max_loss_dollars * 0.75)

    # --- Weight-based allocation scaling ---
    if active_signals:
        weight_by_rule = {s.rule_id: s.weight for s in active_signals}
        for rec in recs:
            rule_weights = [
                weight_by_rule[r] for r in rec.triggered_rules
                if r in weight_by_rule
            ]
            if rule_weights:
                avg_weight = sum(rule_weights) / len(rule_weights)
                scale_factor = 0.5 + 0.5 * avg_weight
                rec.allocation_dollars = round(rec.allocation_dollars * scale_factor)
                rec.max_loss_dollars = round(rec.max_loss_dollars * scale_factor)

    # --- Hedge enforcement ---
    directions = {r.direction for r in recs}
    if len(directions) == 1:
        # All same direction — inject hedge
        hedge_direction = "put" if "call" in directions else "call"
        # R18: 30% hedge overrides R17's 20% and weight-based default
        if r18_triggered:
            hedge_pct = 0.30
        elif r17_triggered:
            hedge_pct = 0.20
        else:
            # Dynamic hedge sizing based on aggregate signal weight
            agg_weight = 0.0
            if active_signals:
                agg_weight = sum(
                    s.weight for s in active_signals
                    if s.triggered and s.confidence != "failed"
                )
            if agg_weight >= 3.0:
                hedge_pct = 0.125  # high conviction → smaller hedge
            elif agg_weight >= 1.5:
                hedge_pct = 0.175  # moderate conviction
            else:
                hedge_pct = 0.225  # low conviction → larger hedge
        hedge_alloc = round(max_allocation * hedge_pct)
        hedge_max_loss = round(hedge_alloc * 0.5)

        # Use SPY options chain for strike if available
        spy_chain = options_chains.get("SPY", {})
        if hedge_direction == "put":
            hedge_strike = spy_chain.get("preferred_put_strike")
        else:
            hedge_strike = spy_chain.get("preferred_call_strike")
        hedge_expiry = spy_chain.get("expiry")

        hedge_rules = ["hedge"]
        hedge_reasoning = (
            f"MANDATORY hedge trade. All other positions are "
            f"{'bullish' if hedge_direction == 'put' else 'bearish'}. "
            f"This {hedge_direction} limits downside if thesis is wrong."
        )
        if r18_triggered:
            hedge_rules.append("R18")
            hedge_reasoning = (
                "GEOPOLITICAL RISK (R18): Elevated geopolitical risk detected in headlines. "
                "Long allocations reduced by 25%. "
                f"Mandatory {hedge_pct*100:.0f}% hedge enforced. "
                f"This {hedge_direction} protects against ongoing geopolitical uncertainty."
            )
            if r17_triggered:
                hedge_rules.append("R17")
        elif r17_triggered:
            hedge_rules.append("R17")
            hedge_reasoning = (
                "CONTRARIAN WARNING (R17): All watchlist scans show bullish signals "
                "AND all recommendations are calls. One-directional positioning detected. "
                f"Mandatory {hedge_pct*100:.0f}% hedge enforced. "
                f"This {hedge_direction} protects against consensus-is-wrong risk."
            )

        hedge = TradeRecommendation(
            ticker="SPY",
            direction=hedge_direction,
            strike=hedge_strike,
            expiry=hedge_expiry,
            entry_timing="At open — portfolio hedge",
            allocation_dollars=hedge_alloc,
            max_loss_dollars=hedge_max_loss,
            stop_loss_pct=50.0,
            take_profit_pct=100.0,
            triggered_rules=hedge_rules,
            reasoning=hedge_reasoning,
            confidence="low",
        )
        recs.append(hedge)

    # --- Strike validation: snap to nearest in options chain ---
    for rec in recs:
        chain = options_chains.get(rec.ticker, {})
        if not chain or rec.strike is None:
            continue
        strike_list = (
            [c["strike"] for c in chain.get("calls", [])]
            if rec.direction == "call"
            else [p["strike"] for p in chain.get("puts", [])]
        )
        if strike_list:
            nearest = min(strike_list, key=lambda s: abs(s - rec.strike))
            if nearest != rec.strike:
                logger.info(
                    "Snapping %s %s strike %.2f → %.2f",
                    rec.ticker, rec.direction, rec.strike, nearest,
                )
                rec.strike = nearest

    # --- Allocation cap: no single trade > 35% ---
    cap = max_allocation * 0.35
    for rec in recs:
        if rec.allocation_dollars > cap:
            logger.info(
                "Capping %s allocation $%.0f → $%.0f (35%% max)",
                rec.ticker, rec.allocation_dollars, cap,
            )
            ratio = cap / rec.allocation_dollars
            rec.allocation_dollars = round(cap)
            rec.max_loss_dollars = round(rec.max_loss_dollars * ratio)

    # --- Total cap: scale down if exceeds max_allocation ---
    total = sum(r.allocation_dollars for r in recs)
    if total > max_allocation:
        scale = max_allocation / total
        logger.info(
            "Scaling all allocations by %.2f (total $%.0f > max $%.0f)",
            scale, total, max_allocation,
        )
        for rec in recs:
            rec.allocation_dollars = round(rec.allocation_dollars * scale)
            rec.max_loss_dollars = round(rec.max_loss_dollars * scale)

    return recs


# ---------------------------------------------------------------------------
# Mock data for testing without API keys
# ---------------------------------------------------------------------------
def _build_mock_snapshot(as_of: str) -> MarketSnapshot:
    return MarketSnapshot(
        as_of=as_of,
        spy_price=685.50,
        spy_prev_close=687.35,
        spy_premarket_change_pct=-0.27,
        spy_5d_change_pct=-1.45,
        spy_20d_change_pct=-3.20,
        spy_distance_from_20sma_pct=-0.85,
        spy_volume_ratio=1.15,
        vix=19.50,
        vix_prev_close=19.55,
        vix_change=0.95,
        us10y_yield=4.085,
        us10y_change_bps=-3.5,
        dxy=97.80,
        dxy_change_pct=0.15,
        gap_pct=-0.27,
    )


def _build_mock_recommendations() -> tuple[list[dict], list[dict]]:
    recs = [
        {
            "ticker": "SPY",
            "direction": "call",
            "strike": 686.0,
            "expiry": "2026-02-28",
            "entry_timing": "At open, limit near $3.50",
            "allocation_dollars": 1200,
            "max_loss_dollars": 600,
            "stop_loss_pct": 50,
            "take_profit_pct": 100,
            "triggered_rules": ["R4", "R8", "R9"],
            "reasoning": "VIX above 20 with gap down and below 20-SMA. Triple confluence buy signal. Historically 100% win rate when VIX >20.",
            "confidence": "high",
        },
        {
            "ticker": "AMD",
            "direction": "call",
            "strike": 160.0,
            "expiry": "2026-02-28",
            "entry_timing": "At open or on a dip to $158",
            "allocation_dollars": 800,
            "max_loss_dollars": 400,
            "stop_loss_pct": 50,
            "take_profit_pct": 150,
            "triggered_rules": ["R4", "sympathy_NVDA"],
            "reasoning": "AMD is deeply oversold (-14% over 20d) and has 1.01 beta to NVDA. With NVDA earnings today, AMD is the highest-conviction sympathy play.",
            "confidence": "high",
        },
        {
            "ticker": "MSFT",
            "direction": "call",
            "strike": 410.0,
            "expiry": "2026-03-07",
            "entry_timing": "Wait for 9:45 AM CT pullback entry",
            "allocation_dollars": 600,
            "max_loss_dollars": 300,
            "stop_loss_pct": 50,
            "take_profit_pct": 100,
            "triggered_rules": ["R9", "oversold"],
            "reasoning": "MSFT down -17% over 20 days, well below 20-SMA. Oversold bounce candidate with cloud/AI tailwinds from NVDA earnings sentiment.",
            "confidence": "medium",
        },
        {
            "ticker": "NVDA",
            "direction": "call",
            "strike": 135.0,
            "expiry": "2026-02-28",
            "entry_timing": "After earnings release, AH or next morning open",
            "allocation_dollars": 500,
            "max_loss_dollars": 250,
            "stop_loss_pct": 50,
            "take_profit_pct": 200,
            "triggered_rules": ["earnings_catalyst"],
            "reasoning": "Analysts expect EPS $1.53 (+72% YoY) on $65.7B revenue. 36 upward EPS revisions. Post-earnings IV crush makes this a calculated bet on a beat.",
            "confidence": "medium",
        },
        {
            "ticker": "SPY",
            "direction": "put",
            "strike": 680.0,
            "expiry": "2026-02-27",
            "entry_timing": "If SPY breaks below $682 intraday or NVDA misses",
            "allocation_dollars": 400,
            "max_loss_dollars": 200,
            "stop_loss_pct": 50,
            "take_profit_pct": 100,
            "triggered_rules": ["hedge"],
            "reasoning": "Portfolio hedge against gap-down acceleration. If NVDA disappoints, entire market sells off. Limits downside on the bullish positions above.",
            "confidence": "low",
        },
    ]
    sympathy = [
        {
            "primary_ticker": "NVDA",
            "primary_catalyst": "Earnings after close — EPS $1.53 expected",
            "sympathy_ticker": "AMD",
            "beta": 1.01,
            "direction": "call",
            "entry_timing": "NEXT DAY ONLY",
            "reasoning": "AMD moves 1:1 with NVDA on earnings. If NVDA beats, AMD opens up 5-8% next day. Already deeply oversold.",
        },
        {
            "primary_ticker": "NVDA",
            "primary_catalyst": "Earnings after close — AI capex commentary",
            "sympathy_ticker": "AVGO",
            "beta": 0.85,
            "direction": "call",
            "entry_timing": "NEXT DAY ONLY",
            "reasoning": "AVGO is a slower follower (0.85 beta) but benefits from positive AI infrastructure commentary. Look for Thursday AM entry.",
        },
    ]
    return recs, sympathy


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------
@dataclass
class RecommendationResult:
    date: str
    out_dir: Path
    recommendations_path: Path
    brief_path: Path
    mock_mode: bool


def run_recommendations(
    *,
    date: str | None = None,
    tz: str = "America/Chicago",
    watchlist: list[str],
    out_dir: str | Path,
    mock_mode: bool = False,
    max_allocation: float = 5000.0,
    is_intraday_rerun: bool = False,
) -> RecommendationResult:
    """Full recommendation pipeline."""
    tzinfo = ZoneInfo(tz)
    if date:
        target_date = datetime.strptime(date, "%Y-%m-%d").date()
    else:
        target_date = datetime.now(tzinfo).date()
    date_str = target_date.strftime("%Y-%m-%d")
    as_of = datetime.now(tzinfo).isoformat(timespec="seconds")

    output_dir = Path(out_dir) / date_str
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load research.json if available
    research_path = output_dir / "research.json"
    if research_path.exists():
        with open(research_path) as f:
            research = json.load(f)
        research["date"] = date_str
        logger.info("Loaded research.json from %s", research_path)
    else:
        research = {"date": date_str, "events": [], "earnings": [], "headlines": []}
        logger.warning("No research.json found — running with empty research context.")

    # Load history CSV
    history_df = None
    if HISTORY_PATH.exists():
        history_df = pd.read_csv(HISTORY_PATH)
        logger.info("Loaded history: %d rows from %s", len(history_df), HISTORY_PATH)

    # Load sympathy map
    sympathy_map = {}
    if SYMPATHY_PATH.exists():
        with open(SYMPATHY_PATH) as f:
            sympathy_map = json.load(f)

    options_chains: Dict[str, Any] = {}

    watchlist_scans: List[Dict[str, Any]] = []

    if mock_mode or not os.getenv("OPENAI_API_KEY"):
        logger.info("Running recommendations in MOCK mode.")
        snapshot = _build_mock_snapshot(as_of)
        engine = RuleEngine()
        signals = engine.evaluate_all(snapshot, research, history_df, watchlist_scans=None)

        mock_recs, mock_sympathy = _build_mock_recommendations()
        recs = [TradeRecommendation(**r) for r in mock_recs]
        sympathy_plays = [SympathyPlay(**s) for s in mock_sympathy]
    else:
        # Live mode
        logger.info("Fetching live market data...")
        fetcher = MarketDataFetcher()
        snapshot = fetcher.get_market_snapshot(as_of)

        # Build earnings tickers list for cross-referencing
        earnings_tickers = [
            e.get("ticker", "") for e in research.get("earnings", []) if e.get("ticker")
        ]

        # Scan watchlist BEFORE rule evaluation (R15/R16 need scan data)
        watchlist_scans = fetcher.scan_watchlist(watchlist, earnings_tickers=earnings_tickers)
        logger.info("  Scanned %d watchlist tickers", len(watchlist_scans))

        # Evaluate trading rules (including R15/R16 which use watchlist_scans)
        logger.info("Evaluating trading rules...")
        engine = RuleEngine()
        signals = engine.evaluate_all(
            snapshot, research, history_df, watchlist_scans=watchlist_scans,
        )

        triggered = [s for s in signals if s.triggered and s.confidence != "failed"]
        logger.info("  %d rules triggered (excluding failed rules)", len(triggered))

        # Build list of tickers to trade: SPY always + watchlist + sympathy plays
        trade_tickers = ["SPY"]
        for scan in watchlist_scans:
            if scan["ticker"] not in trade_tickers:
                trade_tickers.append(scan["ticker"])
        # Add sympathy plays for upcoming earnings
        for earning in research.get("earnings", [])[:5]:
            eticker = earning.get("ticker", "")
            if eticker in sympathy_map:
                for sp in sympathy_map[eticker].get("sympathy_plays", [])[:2]:
                    if sp["ticker"] not in trade_tickers:
                        trade_tickers.append(sp["ticker"])
            if eticker and eticker not in trade_tickers:
                trade_tickers.append(eticker)

        # Fetch options chains (max 8 tickers)
        options_chains = {}
        for ticker in trade_tickers[:8]:
            chain = fetcher.fetch_options_chain(ticker)
            if chain.get("expiry"):
                options_chains[ticker] = chain
            time.sleep(1)

        # Tag earnings with timing info for sympathy plays
        earnings_timing = {}
        for earning in research.get("earnings", []):
            eticker = earning.get("ticker", "")
            time_hint = (earning.get("time_hint") or "").lower()
            if "after" in time_hint:
                earnings_timing[eticker] = "NEXT DAY ONLY"
            elif "before" in time_hint:
                earnings_timing[eticker] = "Same day"
            else:
                earnings_timing[eticker] = "Unknown — verify timing"

        # Generate recommendations via OpenAI
        try:
            oai = OpenAIClient()
            signals_dicts = [s.model_dump() for s in signals]
            snapshot_dict = snapshot.model_dump()

            result = oai.generate_trade_recs(
                snapshot=snapshot_dict,
                signals=signals_dicts,
                research=research,
                options_chains=options_chains,
                sympathy_map=sympathy_map,
                max_allocation=max_allocation,
                watchlist=watchlist,
                watchlist_scans=watchlist_scans,
                earnings_timing=earnings_timing,
            )

            recs = [
                TradeRecommendation(**r)
                for r in result.get("recommendations", [])
            ]
            sympathy_plays = [
                SympathyPlay(**s)
                for s in result.get("sympathy_plays", [])
            ]
        except Exception as e:
            logger.error("OpenAI trade rec generation failed: %s", e)
            recs = []
            sympathy_plays = []

    # Post-process: validate and fix recommendations (R17 needs scans + signals)
    recs = _validate_and_fix_recommendations(
        recs, options_chains, max_allocation,
        watchlist_scans=watchlist_scans or None,
        active_signals=signals,
    )

    # Build portfolio summary
    total_alloc = sum(r.allocation_dollars for r in recs)
    max_risk = sum(r.max_loss_dollars for r in recs)
    triggered_signals = [s for s in signals if s.triggered and s.confidence != "failed"]
    long_weight = sum(s.weight for s in triggered_signals if s.direction == "long")
    short_weight = sum(s.weight for s in triggered_signals if s.direction == "short")

    if long_weight > short_weight:
        bias = "bullish"
    elif short_weight > long_weight:
        bias = "bearish"
    else:
        bias = "neutral"

    portfolio_summary = PortfolioSummary(
        total_allocation=total_alloc,
        max_portfolio_risk=max_risk,
        num_trades=len(recs),
        net_directional_bias=bias,
        rules_triggered_count=len(triggered_signals),
    )

    # Assemble RecommendationPack
    rec_pack = RecommendationPack(
        as_of=as_of,
        date=date_str,
        market_snapshot=snapshot,
        active_signals=signals,
        recommendations=recs,
        sympathy_plays=sympathy_plays,
        portfolio_summary=portfolio_summary,
        is_intraday_rerun=is_intraday_rerun,
    )

    # Write outputs — intraday re-runs write to separate files
    suffix = "_intraday" if is_intraday_rerun else ""
    rec_json_path = output_dir / f"recommendations{suffix}.json"
    rec_brief_path = output_dir / f"recommendations{suffix}.md"

    with open(rec_json_path, "w") as f:
        json.dump(rec_pack.model_dump(), f, indent=2, default=str)
        f.write("\n")

    brief_md = build_recommendation_brief(rec_pack)
    with open(rec_brief_path, "w") as f:
        f.write(brief_md.rstrip() + "\n")

    logger.info("Wrote recommendations.json → %s", rec_json_path)
    logger.info("Wrote recommendations.md → %s", rec_brief_path)

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"  Recommendations: {rec_json_path}")
    print(f"  Rules triggered: {len(triggered_signals)}/{portfolio_summary.rules_total_count}")
    print(f"  Trades:          {len(recs)}")
    print(f"  Total allocation:${total_alloc:,.0f}")
    print(f"  Max risk:        ${max_risk:,.0f}")
    print(f"  Bias:            {bias.upper()}")
    print(f"{'=' * 60}\n")

    return RecommendationResult(
        date=date_str,
        out_dir=output_dir,
        recommendations_path=rec_json_path,
        brief_path=rec_brief_path,
        mock_mode=mock_mode or not os.getenv("OPENAI_API_KEY"),
    )
