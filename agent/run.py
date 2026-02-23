from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from .pipeline import run_pipeline


def main(argv: list[str] | None = None) -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Generate the Morning Market Brief files."
    )
    parser.add_argument("--date", help="Date in YYYY-MM-DD", default=None)
    parser.add_argument("--tz", help="Timezone", default="America/Chicago")
    parser.add_argument(
        "--watchlist",
        help='Comma-separated tickers (e.g. "AAPL,MSFT,NVDA")',
        default="",
    )
    parser.add_argument("--out", help="Output directory", default="./out")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Force mock mode with no external API calls",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    watchlist = [t.strip().upper() for t in args.watchlist.split(",") if t.strip()]

    result = run_pipeline(
        date=args.date,
        tz=args.tz,
        watchlist=watchlist,
        out_dir=Path(args.out),
        mock_mode=args.mock if args.mock else None,
    )

    logging.getLogger(__name__).info(
        "Wrote research to %s and brief to %s (mock_mode=%s)",
        result.research_path,
        result.brief_path,
        result.mock_mode,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
