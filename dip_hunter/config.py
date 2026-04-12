"""Configuration constants for dip-hunter."""
from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(__file__).resolve().parent / "data"
UNIVERSE_PATH = DATA_DIR / "universe.json"
PORTFOLIO_PATH = DATA_DIR / "portfolio.json"

# Scoring weights
DEPTH_WEIGHT = 0.60
BOUNCE_WEIGHT = 0.40

# Signal thresholds
DIP_SCORE_ACTIONABLE = 5.0
RSI_OVERSOLD = 30

# Position sizing
DEFAULT_CAPITAL = 15000.0
MAX_POSITIONS = 5
MIN_POSITION_PCT = 0.15
MAX_POSITION_PCT = 0.30

# Take-profit / Stop-loss
TAKE_PROFIT_PCT = 10.0
STOP_LOSS_PCT = -8.0

# Yfinance
SCAN_SLEEP_SEC = 0.5
HISTORY_PERIOD = "6mo"
