"""Pick tracking — save daily picks and measure accuracy over time."""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from .config import DATA_DIR, SCAN_SLEEP_SEC

logger = logging.getLogger(__name__)

PICKS_PATH = DATA_DIR / "picks_history.json"


def _load_history(path: Path | None = None) -> List[dict]:
    p = path or PICKS_PATH
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return []


def _save_history(history: List[dict], path: Path | None = None) -> None:
    p = path or PICKS_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(history, f, indent=2, default=str)
        f.write("\n")


def save_daily_picks(
    date: str,
    picks: List[dict],
    path: Path | None = None,
) -> None:
    """Save today's picks to history.

    Each pick dict should have: ticker, price, confidence, confidence_level,
    dip_score, tier, expected_bounce.
    """
    history = _load_history(path)

    # Don't duplicate — remove any existing entry for this date
    history = [h for h in history if h.get("date") != date]

    history.append({
        "date": date,
        "picks": picks,
        "outcomes": {},
    })
    _save_history(history, path)
    logger.info("Saved %d picks for %s", len(picks), date)


def update_outcomes(
    live_prices: Dict[str, float],
    today: str = "",
    path: Path | None = None,
) -> None:
    """Update outcomes for past picks using current prices."""
    if not today:
        today = datetime.now().strftime("%Y-%m-%d")
    today_dt = datetime.strptime(today, "%Y-%m-%d")

    history = _load_history(path)
    updated = False

    for entry in history:
        pick_date = entry["date"]
        pick_dt = datetime.strptime(pick_date, "%Y-%m-%d")
        days_since = (today_dt - pick_dt).days

        if days_since <= 0:
            continue

        outcomes = entry.get("outcomes", {})

        for horizon_label, horizon_days in [("5d", 5), ("10d", 10), ("20d", 20)]:
            if days_since >= horizon_days and horizon_label not in outcomes:
                # Record outcomes at this horizon
                horizon_outcomes = {}
                for pick in entry["picks"]:
                    ticker = pick["ticker"]
                    pick_price = pick["price"]
                    current = live_prices.get(ticker)
                    if current is not None:
                        ret_pct = round((current / pick_price - 1) * 100, 2)
                        horizon_outcomes[ticker] = {
                            "price": round(current, 2),
                            "return_pct": ret_pct,
                            "hit_10pct": ret_pct >= 10.0,
                        }
                if horizon_outcomes:
                    outcomes[horizon_label] = horizon_outcomes
                    updated = True

        entry["outcomes"] = outcomes

    if updated:
        _save_history(history, path)
        logger.info("Updated outcomes for past picks.")


def compute_track_record(path: Path | None = None) -> dict:
    """Compute accuracy stats from pick history.

    Returns dict with:
        total_picks, avg_5d_return, avg_10d_return, avg_20d_return,
        hit_rate_10pct, recent_picks (last 10 with outcomes),
        total_days_tracked
    """
    history = _load_history(path)

    all_returns_5d = []
    all_returns_10d = []
    all_returns_20d = []
    hits_10pct = 0
    total_with_outcome = 0
    recent_picks: List[dict] = []

    for entry in history:
        outcomes = entry.get("outcomes", {})
        for pick in entry["picks"]:
            ticker = pick["ticker"]
            rec = {
                "date": entry["date"],
                "ticker": ticker,
                "price": pick["price"],
                "confidence": pick.get("confidence", 0),
                "confidence_level": pick.get("confidence_level", "?"),
                "tier": pick.get("tier", "?"),
            }

            # 5-day outcome
            if "5d" in outcomes and ticker in outcomes["5d"]:
                o = outcomes["5d"][ticker]
                rec["return_5d"] = o["return_pct"]
                all_returns_5d.append(o["return_pct"])

            # 10-day outcome
            if "10d" in outcomes and ticker in outcomes["10d"]:
                o = outcomes["10d"][ticker]
                rec["return_10d"] = o["return_pct"]
                all_returns_10d.append(o["return_pct"])

            # 20-day outcome
            if "20d" in outcomes and ticker in outcomes["20d"]:
                o = outcomes["20d"][ticker]
                rec["return_20d"] = o["return_pct"]
                all_returns_20d.append(o["return_pct"])
                total_with_outcome += 1
                if o["return_pct"] >= 10.0:
                    hits_10pct += 1

            recent_picks.append(rec)

    # Sort recent picks by date descending
    recent_picks.sort(key=lambda x: x["date"], reverse=True)

    return {
        "total_picks": sum(len(e["picks"]) for e in history),
        "total_days_tracked": len(history),
        "avg_5d_return": round(sum(all_returns_5d) / len(all_returns_5d), 2) if all_returns_5d else None,
        "avg_10d_return": round(sum(all_returns_10d) / len(all_returns_10d), 2) if all_returns_10d else None,
        "avg_20d_return": round(sum(all_returns_20d) / len(all_returns_20d), 2) if all_returns_20d else None,
        "hit_rate_10pct": round(hits_10pct / total_with_outcome * 100, 1) if total_with_outcome > 0 else None,
        "total_with_20d_outcome": total_with_outcome,
        "recent_picks": recent_picks[:15],
    }
