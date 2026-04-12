"""Markdown brief generation for dip-hunter daily output."""
from __future__ import annotations

from .schemas import DipHunterBrief


def build_dip_hunter_brief(brief: DipHunterBrief) -> str:
    """Generate the markdown daily brief with confidence picks."""
    lines = [
        f"# Dip Hunter Daily Picks — {brief.date}",
        "",
        f"_As of {brief.as_of}_",
        "",
    ]

    # --- Track Record ---
    tr = brief.track_record
    if tr and tr.get("total_picks", 0) > 0:
        lines.extend([
            "## Track Record",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total picks made | {tr['total_picks']} |",
        ])
        if tr.get("avg_5d_return") is not None:
            lines.append(f"| Avg 5-day return | {tr['avg_5d_return']:+.1f}% |")
        if tr.get("avg_10d_return") is not None:
            lines.append(f"| Avg 10-day return | {tr['avg_10d_return']:+.1f}% |")
        if tr.get("avg_20d_return") is not None:
            lines.append(f"| Avg 20-day return | {tr['avg_20d_return']:+.1f}% |")
        if tr.get("hit_rate_10pct") is not None:
            lines.append(f"| **10%+ hit rate** | **{tr['hit_rate_10pct']:.0f}%** ({tr['total_with_20d_outcome']} picks measured) |")
        lines.append("")

        # Recent pick outcomes
        recent = tr.get("recent_picks", [])
        has_outcomes = [p for p in recent if "return_5d" in p or "return_10d" in p or "return_20d" in p]
        if has_outcomes:
            lines.extend([
                "### Recent Picks Performance",
                "| Date | Ticker | Confidence | Pick $ | 5d | 10d | 20d |",
                "|------|--------|------------|--------|----|----|-----|",
            ])
            for p in has_outcomes[:10]:
                r5 = f"{p['return_5d']:+.1f}%" if "return_5d" in p else "—"
                r10 = f"{p['return_10d']:+.1f}%" if "return_10d" in p else "—"
                r20 = f"{p['return_20d']:+.1f}%" if "return_20d" in p else "—"
                lines.append(
                    f"| {p['date']} | {p['ticker']} | {p['confidence_level']} ({p['confidence']}%) "
                    f"| ${p['price']:.2f} | {r5} | {r10} | {r20} |"
                )
            lines.append("")

    # --- Market Context ---
    lines.extend([
        "## Market Context",
        "| Metric | Value |",
        "|--------|-------|",
        f"| SPY | ${brief.spy_price:.2f} ({brief.spy_change_pct:+.2f}%) |",
        f"| VIX | {brief.vix:.1f} |",
        "",
    ])

    # --- Today's Picks (confidence-rated buy signals) ---
    if brief.buy_signals:
        lines.extend([
            "## Today's Picks",
            "",
        ])
        for b in brief.buy_signals:
            # Confidence badge
            if b.confidence_level == "HIGH":
                badge = "HIGH CONFIDENCE"
            elif b.confidence_level == "MEDIUM":
                badge = "MEDIUM CONFIDENCE"
            else:
                badge = "LOW CONFIDENCE"

            lines.extend([
                f"### {b.rank}. {b.ticker} — {badge} ({b.confidence}%)",
                "",
                "| Detail | Value |",
                "|--------|-------|",
                f"| Tier | **{b.tier}** |",
                f"| Current Price | ${b.current_price:.2f} |",
                f"| Dip Score | {b.dip_score:.1f}/10 |",
                f"| Entry Target | ${b.entry_target:.2f} |",
                f"| Stop Loss | ${b.stop_loss:.2f} (-8%) |",
                f"| Take Profit | ${b.take_profit:.2f} (+10%) |",
                f"| Position Size | {b.position_size_shares} shares (${b.position_size_dollars:,.0f}) |",
            ])
            if b.historical_win_rate is not None:
                lines.append(f"| Historical Win Rate | {b.historical_win_rate:.0f}% |")
            if b.historical_avg_bounce is not None:
                lines.append(f"| Historical Avg Bounce | +{b.historical_avg_bounce:.0f}% |")
            if b.expected_bounce_range:
                lines.append(f"| Expected Bounce | {b.expected_bounce_range} |")

            lines.extend([
                "",
                f"**Why:** {b.confidence_reasoning}",
                "",
                f"**Technical:** {b.reasoning}",
                "",
            ])
    else:
        lines.extend([
            "## Today's Picks",
            "",
            "_No high-confidence dip candidates today. Sitting on hands._",
            "",
        ])

    # --- Portfolio Status ---
    lines.extend(["## Portfolio Status"])

    if brief.holdings_status:
        lines.extend([
            "| Ticker | Shares | Avg Cost | Current | P&L | P&L% | Days | Signal |",
            "|--------|--------|----------|---------|-----|------|------|--------|",
        ])
        for hs in brief.holdings_status:
            lines.append(
                f"| {hs.ticker} | {hs.shares} | ${hs.avg_cost:.2f} | "
                f"${hs.current_price:.2f} | ${hs.unrealized_pnl:+,.2f} | "
                f"{hs.unrealized_pnl_pct:+.1f}% | {hs.days_held} | "
                f"**{hs.signal}** |"
            )

        total_pnl_sign = "+" if brief.total_unrealized_pnl >= 0 else ""
        lines.extend([
            "",
            f"**Portfolio value:** ${brief.portfolio_value:,.2f} | "
            f"**Total P&L:** {total_pnl_sign}${brief.total_unrealized_pnl:,.2f} "
            f"({brief.total_unrealized_pnl_pct:+.1f}%)",
        ])
    else:
        lines.append("_No positions held._")

    # Sell signals
    if brief.sell_signals:
        lines.extend(["", "## ACTION: Sell Signals"])
        for s in brief.sell_signals:
            emoji = "STOP LOSS" if s.action == "STOP_LOSS" else "TAKE PROFIT"
            lines.append(
                f"- **{s.ticker}**: {emoji} at ${s.current_price:.2f} "
                f"({s.pnl_pct:+.1f}% from ${s.avg_cost:.2f} entry). {s.reasoning}"
            )

    # Rotate signals
    if brief.rotate_signals:
        lines.extend(["", "## ACTION: Rotate Signals"])
        for r in brief.rotate_signals:
            lines.append(
                f"- **Sell {r.sell_ticker}** ({r.sell_reason}, {r.sell_pnl_pct:+.1f}%) "
                f"→ **Buy {r.buy_ticker}** (dip score {r.buy_dip_score:.1f}, "
                f"entry ${r.buy_entry_target:.2f}). {r.reasoning}"
            )

    # Top dips universe scan (compact)
    if brief.top_dips:
        lines.extend([
            "",
            "## Watchlist — Top Dips (Full Scan)",
            "| Ticker | Sector | Price | 5d | 20d | RSI | Score | Tier |",
            "|--------|--------|-------|----|-----|-----|-------|------|",
        ])
        for s in brief.top_dips[:10]:
            ret5 = f"{s.return_5d_pct:+.1f}%" if s.return_5d_pct is not None else "N/A"
            ret20 = f"{s.return_20d_pct:+.1f}%" if s.return_20d_pct is not None else "N/A"
            rsi = f"{s.rsi_14:.0f}" if s.rsi_14 is not None else "N/A"
            from .confidence import get_tier
            tier = get_tier(s.ticker)
            lines.append(
                f"| {s.ticker} | {s.sector} | ${s.price:.2f} | {ret5} | {ret20} | "
                f"{rsi} | **{s.dip_score:.1f}** | {tier} |"
            )

    # Disclaimer
    lines.extend([
        "",
        "---",
        f"_{brief.disclaimer}_",
    ])

    return "\n".join(lines)
