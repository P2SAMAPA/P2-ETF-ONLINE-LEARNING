"""
Data loading and feature engineering for Online Learning engine.
"""

import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
import config

def load_master_data() -> pd.DataFrame:
    print(f"Downloading {config.HF_DATA_FILE} from {config.HF_DATA_REPO}...")
    file_path = hf_hub_download(
        repo_id=config.HF_DATA_REPO,
        filename=config.HF_DATA_FILE,
        repo_type="dataset",
        token=config.HF_TOKEN,
        cache_dir="./hf_cache"
    )
    df = pd.read_parquet(file_path)
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index().rename(columns={'index': 'Date'})
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def prepare_returns_matrix(df_wide: pd.DataFrame, tickers: list) -> pd.DataFrame:
    """Prepare a wide-format DataFrame of log returns with Date index."""
    available_tickers = [t for t in tickers if t in df_wide.columns]
    df_long = pd.melt(
        df_wide, id_vars=['Date'], value_vars=available_tickers,
        var_name='ticker', value_name='price'
    )
    df_long = df_long.sort_values(['ticker', 'Date'])
    df_long['log_return'] = df_long.groupby('ticker')['price'].transform(
        lambda x: np.log(x / x.shift(1))
    )
    df_long = df_long.dropna(subset=['log_return'])
    return df_long.pivot(index='Date', columns='ticker', values='log_return')[available_tickers].dropna()

def prepare_macro_features(df_wide: pd.DataFrame) -> pd.DataFrame:
    """Extract macro columns and return as DataFrame with Date index."""
    macro_cols = [c for c in config.MACRO_COLS if c in df_wide.columns]
    macro_df = df_wide[['Date'] + macro_cols].copy()
    macro_df = macro_df.set_index('Date').ffill().dropna()
    return macro_df

def build_feature_dict(returns: pd.DataFrame, macro: pd.DataFrame, ticker: str,
                       benchmark_returns: pd.Series = None):
    """
    Build a dictionary of features for a single time step.
    Includes: lagged returns, volatility, relative strength, macro.
    """
    ret_series = returns[ticker]
    features = {}
    for w in config.RETURN_WINDOWS:
        if len(ret_series) >= w:
            features[f'ret_{w}d'] = ret_series.iloc[-w:].mean()
        else:
            features[f'ret_{w}d'] = 0.0

    # 21‑day volatility
    if len(ret_series) >= config.VOLATILITY_WINDOW:
        features['vol_21d'] = ret_series.iloc[-config.VOLATILITY_WINDOW:].std()
    else:
        features['vol_21d'] = 0.0

    # Relative strength vs benchmark
    if benchmark_returns is not None and len(benchmark_returns) >= config.BENCHMARK_LOOKBACK:
        ticker_cum = (1 + ret_series.iloc[-config.BENCHMARK_LOOKBACK:]).prod()
        bench_cum = (1 + benchmark_returns.iloc[-config.BENCHMARK_LOOKBACK:]).prod()
        features['rel_strength'] = ticker_cum / bench_cum - 1.0
    else:
        features['rel_strength'] = 0.0

    # Current macro values
    for col in macro.columns:
        features[col] = macro[col].iloc[-1] if len(macro) > 0 else 0.0

    return features
