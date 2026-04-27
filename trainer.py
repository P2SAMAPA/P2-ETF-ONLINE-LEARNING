"""
Main training script – Online learning with Daily, Global, and Shrinking modes.
"""

import json
import pandas as pd
import numpy as np
from collections import defaultdict

import config
import data_manager
from online_model import OnlineLearner
import push_results


def stream_and_predict(returns, macro, tickers, benchmark_returns=None):
    """
    Stream all data through an OnlineLearner and produce forecasts for each ticker.
    Returns: dict of {ticker: {'forecast': float, 'state': dict}}
    """
    results = {}
    for ticker in tickers:
        model = OnlineLearner(model_type=config.MODEL_TYPE)
        # Build a list of feature dicts and targets
        for i in range(len(returns)):
            features = data_manager.build_feature_dict(returns.iloc[:i+1], macro.iloc[:i+1],
                                                       ticker, benchmark_returns.iloc[:i+1] if benchmark_returns is not None else None)
            target = returns[ticker].iloc[i]
            model.learn_one(features, target)

        # Forecast using the most recent feature
        latest_features = data_manager.build_feature_dict(returns, macro, ticker,
                                                          benchmark_returns)
        forecast = model.predict_one(latest_features)
        results[ticker] = {
            "forecast": forecast,
            "drift_detected": model.drift_detector.drift_detected,
            "drift_history": model.drift_history,
            "samples_seen": model.samples_seen
        }
    return results


def compute_top_picks(results):
    """Sort by forecast and return top 3."""
    sorted_items = sorted(results.items(), key=lambda x: x[1]['forecast'], reverse=True)
    return [{"ticker": t, "forecast": d['forecast'], "drift_detected": d['drift_detected']}
            for t, d in sorted_items[:3]]


def run_mode(returns, macro, tickers, mode_name, benchmark_returns=None):
    """Run online learning on a data slice and return results."""
    if len(returns) < config.MIN_OBSERVATIONS:
        return None

    forecasts = stream_and_predict(returns, macro, tickers, benchmark_returns)
    top3 = compute_top_picks(forecasts)
    all_forecasts = {t: d['forecast'] for t, d in forecasts.items()}
    drift_states = {t: {'drift_detected': d['drift_detected'],
                         'drift_history': d['drift_history']}
                    for t, d in forecasts.items()}

    return {
        "top_picks": top3,
        "all_forecasts": all_forecasts,
        "drift_states": drift_states,
        "training_start": str(returns.index[0].date()),
        "training_end": str(returns.index[-1].date()),
        "n_observations": len(returns)
    }


def run_shrinking_windows(returns, macro, tickers, benchmark_returns=None):
    """Run online learning on shrinking windows and compute consensus."""
    windows = []
    for start_year in config.SHRINKING_WINDOW_START_YEARS:
        sd = pd.Timestamp(f"{start_year}-01-01")
        ed = pd.Timestamp("2024-12-31")
        r = returns[(returns.index >= sd) & (returns.index <= ed)]
        if len(r) < config.MIN_OBSERVATIONS:
            continue
        m = macro.loc[r.index]
        b = benchmark_returns.loc[r.index] if benchmark_returns is not None else None
        forecasts = stream_and_predict(r, m, tickers, b)
        top3 = compute_top_picks(forecasts)
        best = top3[0]['ticker']
        windows.append({
            'window_start': start_year,
            'window_end': 2024,
            'ticker': best,
            'forecast': forecasts[best]['forecast']
        })

    if not windows:
        return None
    vote = {}
    for w in windows:
        vote[w['ticker']] = vote.get(w['ticker'], 0) + 1
    pick = max(vote, key=vote.get)
    conviction = vote[pick] / len(windows) * 100
    return {
        "ticker": pick,
        "conviction": conviction,
        "num_windows": len(windows),
        "windows": windows
    }


def main():
    import os
    token = os.getenv("HF_TOKEN")
    if not token:
        print("HF_TOKEN not set")
        return

    df = data_manager.load_master_data()
    macro = data_manager.prepare_macro_features(df)

    all_results = {}

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n=== {universe_name} ===")
        returns = data_manager.prepare_returns_matrix(df, tickers)
        if len(returns) < config.MIN_OBSERVATIONS:
            continue
        m = macro.loc[returns.index].dropna()
        returns = returns.loc[m.index]
        m = m.loc[returns.index]

        # Benchmark for relative strength: SPY for equity/combined, TLT for FI
        benchmark_ticker = "SPY" if "SPY" in df.columns else "TLT"
        benchmark_returns = None
        if benchmark_ticker in df.columns:
            benchmark_returns = data_manager.prepare_returns_matrix(df, [benchmark_ticker])
            benchmark_returns = benchmark_returns.loc[returns.index]

        universe_out = {}

        # Daily mode
        daily_out = run_mode(returns.iloc[-config.DAILY_LOOKBACK:],
                             m.iloc[-config.DAILY_LOOKBACK:],
                             tickers, "Daily", benchmark_returns.iloc[-config.DAILY_LOOKBACK:] if benchmark_returns is not None else None)
        if daily_out:
            universe_out['daily'] = daily_out
            print(f"  Daily top: {daily_out['top_picks'][0]['ticker']}")

        # Global mode
        global_out = run_mode(returns, m, tickers, "Global", benchmark_returns)
        if global_out:
            universe_out['global'] = global_out
            print(f"  Global top: {global_out['top_picks'][0]['ticker']}")

        # Shrinking windows consensus
        shrinking = run_shrinking_windows(returns, m, tickers, benchmark_returns)
        if shrinking:
            universe_out['shrinking'] = shrinking
            print(f"  Shrinking consensus: {shrinking['ticker']} ({shrinking['conviction']:.0f}%)")

        all_results[universe_name] = universe_out

    push_results.push_daily_result({"run_date": config.TODAY, "universes": all_results})
    print("\n=== Run Complete ===")


if __name__ == "__main__":
    main()
