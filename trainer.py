"""
Main training script for online learning engine.
Incrementally updates models on the full history, then forecasts next day.
"""

import json
import pandas as pd
import numpy as np
from collections import defaultdict

import config
import data_manager
from online_model import OnlineLearner
import push_results

def run_online_learning():
    print(f"=== P2-ETF-ONLINE-LEARNING Run: {config.TODAY} ===")
    df_master = data_manager.load_master_data()

    all_results = {}
    top_picks = {}
    state_info = {}

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n--- Processing Universe: {universe_name} ---")
        universe_results = {}
        universe_states = {}

        for ticker in tickers:
            print(f"  Streaming {ticker}...")
            returns = data_manager.prepare_return_series(df_master, ticker)
            if len(returns) < config.MIN_OBSERVATIONS:
                continue

            features = data_manager.build_features(returns, config.FEATURE_WINDOWS)
            model = OnlineLearner(model_type=config.MODEL_TYPE)

            # Stream all historical data through the model
            for idx, row in features.iterrows():
                x = {col: row[col] for col in features.columns if col != 'target'}
                y = row['target']
                model.learn_one(x, y)

            # Forecast next day using the most recent features
            latest = features.iloc[-1]
            x_latest = {col: latest[col] for col in features.columns if col != 'target'}
            forecast = model.predict_one(x_latest)

            universe_results[ticker] = {
                "ticker": ticker,
                "forecast": forecast,
                "last_return": returns.iloc[-1]
            }
            universe_states[ticker] = model.get_state()

        all_results[universe_name] = universe_results
        state_info[universe_name] = universe_states

        # Top 3 by forecast
        sorted_tickers = sorted(universe_results.items(), key=lambda x: x[1]["forecast"], reverse=True)
        top_picks[universe_name] = [{"ticker": t, "forecast": d["forecast"]} for t, d in sorted_tickers[:3]]

    output_payload = {
        "run_date": config.TODAY,
        "config": {
            "model_type": config.MODEL_TYPE,
            "window_size": config.WINDOW_SIZE,
            "feature_windows": config.FEATURE_WINDOWS
        },
        "daily_trading": {
            "universes": all_results,
            "top_picks": top_picks,
            "model_states": state_info
        }
    }

    push_results.push_daily_result(output_payload)
    print("\n=== Run Complete ===")

if __name__ == "__main__":
    run_online_learning()
