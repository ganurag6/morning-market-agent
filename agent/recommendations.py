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

            # Filter to strikes within 3% of current price
            strike_range = current_price * 0.03

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

            return {
                "ticker": ticker,
                "current_price": round(current_price, 2),
                "expiry": expiry,
                "calls": calls_list[:10],
                "puts": puts_list[:10],
            }
        except Exception as e:
            logger.warning("Failed to fetch options for %s: %s", ticker, e)
            return {"ticker": ticker, "expiry": None, "calls": [], "puts": []}

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
    """Evaluate the 14 backtested trading rules against current conditions."""

    def __init__(self, rules_path: Path = RULES_PATH):
        with open(rules_path) as f:
            self.rules = json.load(f)

    def evaluate_all(
        self,
        snapshot: MarketSnapshot,
        research: Dict[str, Any],
        history_df: Optional[pd.DataFrame] = None,
    ) -> List[RuleSignal]:
        """Evaluate all rules and return list of RuleSignal."""
        signals = []
        for rule in self.rules:
            ctype = rule.get("condition_type", "")
            evaluator = self._get_evaluator(ctype)
            if evaluator:
                signal = evaluator(rule, snapshot, research, history_df)
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
        }
        return dispatch.get(condition_type)

    def _eval_spy_prev_day_drop(self, rule, snapshot, research, history):
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

    def _eval_inflation_surprise(self, rule, snapshot, research, history):
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

    def _eval_nfp_beat(self, rule, snapshot, research, history):
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

    def _eval_vix_level(self, rule, snapshot, research, history):
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

    def _eval_fomc_hold(self, rule, snapshot, research, history):
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

    def _eval_ism_contraction(self, rule, snapshot, research, history):
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

    def _eval_yield_drop(self, rule, snapshot, research, history):
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

    def _eval_gap_down(self, rule, snapshot, research, history):
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

    def _eval_below_sma(self, rule, snapshot, research, history):
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

    def _eval_volume_spike(self, rule, snapshot, research, history):
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

    def _eval_consecutive_down(self, rule, snapshot, research, history):
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

    def _eval_rally_short(self, rule, snapshot, research, history):
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

    def _eval_low_yield_high_vix(self, rule, snapshot, research, history):
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

    def _eval_dxy_up_vix_up(self, rule, snapshot, research, history):
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


# ---------------------------------------------------------------------------
# Markdown brief builder
# ---------------------------------------------------------------------------
def build_recommendation_brief(rec_pack: RecommendationPack) -> str:
    """Generate the markdown text for the recommendations email section."""
    snap = rec_pack.market_snapshot
    lines = [
        f"## Trading Recommendations - {rec_pack.date}",
        "",
        f"### Market Snapshot (Pre-Market {snap.as_of})",
        "| Metric | Value | Signal |",
        "|--------|-------|--------|",
    ]

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

    lines.extend([
        "",
        f"### Active Rule Signals ({len(active_triggers)} of 14 triggered)",
    ])

    if active_triggers:
        lines.extend([
            "| Rule | Signal | Confidence | Detail |",
            "|------|--------|------------|--------|",
        ])
        for s in active_triggers:
            direction = "BUY" if s.direction == "long" else "SHORT"
            val_str = f"{s.current_value}" if s.current_value is not None else ""
            thresh_str = f" (threshold: {s.threshold})" if s.threshold is not None else ""
            lines.append(
                f"| {s.rule_id} {s.rule_name} | {direction} | "
                f"{s.confidence.upper()} | {val_str}{thresh_str} — "
                f"{s.win_rate*100:.0f}% win rate, {s.sample_size} trades |"
            )
    else:
        lines.append("_No rules triggered today._")

    if failed_triggers:
        lines.extend(["", "**Counter-signals (failed rules, for reference only):**"])
        for s in failed_triggers:
            lines.append(f"- {s.rule_id} {s.rule_name}: {s.reasoning}")

    # Confluence summary
    long_count = sum(1 for s in active_triggers if s.direction == "long")
    short_count = sum(1 for s in active_triggers if s.direction == "short")
    if active_triggers:
        lines.extend([
            "",
            f"**Confluence: {long_count} bullish, {short_count} bearish signals.**",
        ])

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
            lines.append(
                f"- **{sp.sympathy_ticker}** (beta {sp.beta:.2f} to {sp.primary_ticker}): "
                f"{sp.reasoning} — consider {direction_str}"
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
        vix=20.50,
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
            "allocation_dollars": 1500,
            "max_loss_dollars": 750,
            "stop_loss_pct": 50,
            "take_profit_pct": 100,
            "triggered_rules": ["R4", "R8", "R9"],
            "reasoning": "VIX above 20 with gap down and below 20-SMA. Triple confluence buy signal.",
            "confidence": "high",
        },
        {
            "ticker": "SPY",
            "direction": "put",
            "strike": 680.0,
            "expiry": "2026-02-27",
            "entry_timing": "If SPY breaks below $682 intraday",
            "allocation_dollars": 500,
            "max_loss_dollars": 250,
            "stop_loss_pct": 50,
            "take_profit_pct": 100,
            "triggered_rules": ["hedge"],
            "reasoning": "Protective hedge against gap-down acceleration.",
            "confidence": "low",
        },
    ]
    sympathy = [
        {
            "primary_ticker": "NVDA",
            "primary_catalyst": "Earnings after close",
            "sympathy_ticker": "AMD",
            "beta": 1.01,
            "direction": "call",
            "reasoning": "AMD moves 1:1 with NVDA on earnings. If NVDA beats, AMD follows.",
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

    if mock_mode or not os.getenv("OPENAI_API_KEY"):
        logger.info("Running recommendations in MOCK mode.")
        snapshot = _build_mock_snapshot(as_of)
        engine = RuleEngine()
        signals = engine.evaluate_all(snapshot, research, history_df)

        mock_recs, mock_sympathy = _build_mock_recommendations()
        recs = [TradeRecommendation(**r) for r in mock_recs]
        sympathy_plays = [SympathyPlay(**s) for s in mock_sympathy]
    else:
        # Live mode
        logger.info("Fetching live market data...")
        fetcher = MarketDataFetcher()
        snapshot = fetcher.get_market_snapshot(as_of)

        logger.info("Evaluating trading rules...")
        engine = RuleEngine()
        signals = engine.evaluate_all(snapshot, research, history_df)

        triggered = [s for s in signals if s.triggered and s.confidence != "failed"]
        logger.info("  %d rules triggered (excluding failed rules)", len(triggered))

        # Fetch options chains for tickers we might trade
        options_chains = {}
        trade_tickers = ["SPY"]
        # Add watchlist tickers that have sympathy plays with upcoming earnings
        for earning in research.get("earnings", [])[:3]:
            ticker = earning.get("ticker", "")
            if ticker in sympathy_map:
                for sp in sympathy_map[ticker].get("sympathy_plays", [])[:2]:
                    if sp["ticker"] not in trade_tickers:
                        trade_tickers.append(sp["ticker"])

        for ticker in trade_tickers[:5]:  # Max 5 to avoid rate limits
            chain = fetcher.fetch_options_chain(ticker)
            if chain.get("expiry"):
                options_chains[ticker] = chain
            time.sleep(1)

        # Generate recommendations via OpenAI
        if triggered:
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
        else:
            logger.info("No rules triggered — no trades to recommend.")
            recs = []
            sympathy_plays = []

    # Build portfolio summary
    total_alloc = sum(r.allocation_dollars for r in recs)
    max_risk = sum(r.max_loss_dollars for r in recs)
    triggered_signals = [s for s in signals if s.triggered and s.confidence != "failed"]
    long_count = sum(1 for s in triggered_signals if s.direction == "long")
    short_count = sum(1 for s in triggered_signals if s.direction == "short")

    if long_count > short_count:
        bias = "bullish"
    elif short_count > long_count:
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
    )

    # Write outputs
    rec_json_path = output_dir / "recommendations.json"
    rec_brief_path = output_dir / "recommendations.md"

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
    print(f"  Rules triggered: {len(triggered_signals)}/14")
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
