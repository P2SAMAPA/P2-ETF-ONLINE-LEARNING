# P2-ETF-ONLINE-LEARNING

**Incremental Learning with ADWIN Drift Detection for ETF Forecasting**

[![Daily Run](https://github.com/P2SAMAPA/P2-ETF-ONLINE-LEARNING/actions/workflows/daily_run.yml/badge.svg)](https://github.com/P2SAMAPA/P2-ETF-ONLINE-LEARNING/actions/workflows/daily_run.yml)
[![Hugging Face Dataset](https://img.shields.io/badge/🤗%20Dataset-p2--etf--online--learning--results-blue)](https://huggingface.co/datasets/P2SAMAPA/p2-etf-online-learning-results)

## Overview

`P2-ETF-ONLINE-LEARNING` uses **online learning** from the River library. A Passive‑Aggressive Regressor (or Hoeffding Tree) is updated incrementally with each new daily return. ADWIN drift detection monitors prediction errors and resets the model when the data distribution changes. Forecasts are generated from the current model state and ETFs are ranked by predicted next‑day return.

## Methodology

1. **Feature Engineering**: Lagged returns (1‑, 5‑, 21‑day means).
2. **Incremental Learning**: The model is streamed the full history (2008‑present) once to warm up, then forecasts the next day.
3. **Drift Detection**: ADWIN tracks absolute prediction error. When drift is signaled, the model is reset.
4. **Ranking**: Top 3 ETFs per universe by forecasted return.

## Usage

```bash
pip install -r requirements.txt
python trainer.py
streamlit run streamlit_app.py
