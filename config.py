"""
Configuration for P2-ETF-ONLINE-LEARNING engine.
"""

import os
from datetime import datetime

# --- Hugging Face Repositories ---
HF_DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
HF_DATA_FILE = "master_data.parquet"
HF_OUTPUT_REPO = "P2SAMAPA/p2-etf-online-learning-results"

# --- Universe Definitions ---
FI_COMMODITIES_TICKERS = ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"]
EQUITY_SECTORS_TICKERS = [
    "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV",
    "XLI", "XLY", "XLP", "XLU", "GDX", "XME",
    "IWF", "XSD", "XBI", "IWM"
]
ALL_TICKERS = list(set(FI_COMMODITIES_TICKERS + EQUITY_SECTORS_TICKERS))

UNIVERSES = {
    "FI_COMMODITIES": FI_COMMODITIES_TICKERS,
    "EQUITY_SECTORS": EQUITY_SECTORS_TICKERS,
    "COMBINED": ALL_TICKERS
}

# --- Macro Features (available from 2008) ---
MACRO_COLS = ["VIX", "DXY", "T10Y2Y", "TBILL_3M"]

# --- Online Learning Parameters ---
MODEL_TYPE = "PARegressor"            # Passive‑Aggressive Regressor (River)
WARMUP_WINDOW = 252                  # Warm‑up: 1 year (was 504 → caused daily zeros)
ADWIN_CLOCK = 32                     # Check for drift every 32 samples
ADWIN_DELTA = 0.002                  # Significance level for ADWIN
MIN_OBSERVATIONS = 100               # Minimum samples before forecasting

# --- Feature Windows ---
RETURN_WINDOWS = [1, 5, 21, 63]      # Lagged returns as features
VOLATILITY_WINDOW = 21               # Window for rolling volatility
BENCHMARK_LOOKBACK = 63              # Lookback for relative strength

# --- Training Modes ---
DAILY_LOOKBACK = 504                 # Days for daily trading (252 warm‑up + 252 training)
GLOBAL_TRAIN_START = "2008-01-01"    # Start date for global training
SHRINKING_WINDOW_START_YEARS = list(range(2010, 2025))

# --- Date Handling ---
TODAY = datetime.now().strftime("%Y-%m-%d")

# --- Optional: Hugging Face Token ---
HF_TOKEN = os.environ.get("HF_TOKEN", None)
