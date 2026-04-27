# P2-ETF-ONLINE-LEARNING

**Incremental Learning with ADWIN Drift Detection & Macro‑Informed Features**

[![Daily Run](https://github.com/P2SAMAPA/P2-ETF-ONLINE-LEARNING/actions/workflows/daily_run.yml/badge.svg)](https://github.com/P2SAMAPA/P2-ETF-ONLINE-LEARNING/actions/workflows/daily_run.yml)
[![Hugging Face Dataset](https://img.shields.io/badge/🤗%20Dataset-p2--etf--online--learning--results-blue)](https://huggingface.co/datasets/P2SAMAPA/p2-etf-online-learning-results)

## Overview

`P2-ETF-ONLINE-LEARNING` combines **River's Passive‑Aggressive Regressor** with **ADWIN drift detection** to incrementally learn ETF return patterns. Each new trading day updates the model, which automatically resets when the data distribution shifts. The engine now includes **macro conditioning** (VIX, DXY, T10Y2Y, TBILL_3M), richer technical features, and three distinct training modes.

## Methodology

1. **Feature Engineering**  
   - Lagged returns (1‑, 5‑, 21‑, 63‑day averages)  
   - 21‑day rolling volatility  
   - 63‑day relative strength vs. benchmark (SPY or TLT)  
   - Current macro values (VIX, DXY, T10Y2Y, TBILL_3M)

2. **Incremental Learning**  
   - The model is streamed the full history (2008‑present) once to warm up, then forecasts the next day.  
   - Daily mode uses only the most recent 504 trading days.

3. **Drift Detection (ADWIN)**  
   - Tracks absolute prediction error. When a statistically significant drift is detected, the model is reset.  
   - Drift status is prominently displayed in the dashboard (⚠️ Drift / ✓ Stable).

4. **Ranking**  
   - Top 3 ETFs per universe by forecasted next‑day return.

## Training Modes

- **Daily (504d)** – Most recent 2 years for current regime awareness.  
- **Global (2008‑YTD)** – Full history for long‑term learning.  
- **Shrinking Windows Consensus** – Runs the model on 15 rolling windows (2010‑2024) and selects the most frequently chosen ETF across all windows.

## Universe Coverage

| Universe | Tickers |
|----------|---------|
| **FI / Commodities** | TLT, VCIT, LQD, HYG, VNQ, GLD, SLV |
| **Equity Sectors** | SPY, QQQ, XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, GDX, XME, IWF, XSD, XBI, IWM |
| **Combined** | All tickers above |

## Usage

```bash
pip install -r requirements.txt
python trainer.py
streamlit run streamlit_app.py
Dashboard Features
Three sub‑tabs per universe: Daily, Global, Shrinking Consensus.

Hero Card with forecasted return and drift status badge.

Top 3 & All ETFs tables showing drift flags.

Next Trading Day via US market calendar.
