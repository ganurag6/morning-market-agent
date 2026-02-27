"""CLI entry point for the trading recommendations engine."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from .recommendations import run_recommendations


def main(argv: list[str] | None = None) -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Generate trading recommendations based on backtested rules."
    )
    parser.add_argument("--date", help="Date in YYYY-MM-DD", default=None)
    parser.add_argument("--tz", help="Timezone", default="America/Chicago")
    parser.add_argument(
        "--watchlist",
        help='Comma-separated tickers (e.g. "AAPL,MSFT,NVDA,AMD,AVGO")',
        default="",
    )
    parser.add_argument("--out", help="Output directory", default="./out")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Force mock mode with no external API calls",
    )
    parser.add_argument(
        "--max-allocation",
        type=float,
        default=5000.0,
        help="Maximum total dollar allocation across all trades",
    )
    parser.add_argument(
        "--time-check",
        action="store_true",
        help="Intraday re-run: writes to recommendations_intraday.* files",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    watchlist = [t.strip().upper() for t in args.watchlist.split(",") if t.strip()]

    result = run_recommendations(
        date=args.date,
        tz=args.tz,
        watchlist=watchlist,
        out_dir=Path(args.out),
        mock_mode=args.mock,
        max_allocation=args.max_allocation,
        is_intraday_rerun=args.time_check,
    )

    logging.getLogger(__name__).info(
        "Wrote recommendations to %s and brief to %s (mock_mode=%s)",
        result.recommendations_path,
        result.brief_path,
        result.mock_mode,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
