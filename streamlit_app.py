"""
Streamlit Dashboard for Online Learning Engine.
Displays incremental forecasts and drift status.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from huggingface_hub import HfApi, hf_hub_download
import json
import config
from us_calendar import USMarketCalendar

st.set_page_config(page_title="P2Quant Online Learning", page_icon="🔄", layout="wide")

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 600; color: #1f77b4; }
    .hero-card { background: linear-gradient(135deg, #1f77b4 0%, #2C5282 100%); border-radius: 16px; padding: 2rem; color: white; text-align: center; }
    .hero-ticker { font-size: 4rem; font-weight: 800; }
    .forecast-positive { color: #28a745; font-weight: 600; }
    .forecast-negative { color: #dc3545; font-weight: 600; }
    .drift-active { background: #ffc107; padding: 0.2rem 0.6rem; border-radius: 12px; color: black; }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def load_latest_results():
    try:
        api = HfApi(token=config.HF_TOKEN)
        files = api.list_repo_files(repo_id=config.HF_OUTPUT_REPO, repo_type="dataset")
        json_files = sorted([f for f in files if f.endswith('.json')], reverse=True)
        if not json_files:
            return None
        local_path = hf_hub_download(
            repo_id=config.HF_OUTPUT_REPO, filename=json_files[0],
            repo_type="dataset", token=config.HF_TOKEN, cache_dir="./hf_cache"
        )
        with open(local_path) as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return None

def forecast_badge(val):
    if val >= 0:
        return f'<span class="forecast-positive">+{val*100:.3f}%</span>'
    return f'<span class="forecast-negative">{val*100:.3f}%</span>'

def display_hero_card(ticker: str, forecast: float, drift_detected: bool):
    drift_badge = '<span class="drift-active">⚠️ Drift Detected</span>' if drift_detected else ''
    st.markdown(f"""
    <div class="hero-card">
        <div style="font-size: 1.2rem; opacity: 0.8;">🔄 TOP PICK (Online Learning)</div>
        <div class="hero-ticker">{ticker}</div>
        <div style="font-size: 2rem;">Forecast: {forecast_badge(forecast)}</div>
        <div style="margin-top: 1rem;">{drift_badge}</div>
    </div>
    """, unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.markdown("## ⚙️ Configuration")
calendar = USMarketCalendar()
st.sidebar.markdown(f"**📅 Next Trading Day:** {calendar.next_trading_day().strftime('%Y-%m-%d')}")
data = load_latest_results()
if data:
    st.sidebar.markdown(f"**Run Date:** {data.get('run_date', 'Unknown')}")
st.sidebar.divider()
st.sidebar.markdown("### 🔄 Online Learning Parameters")
st.sidebar.markdown(f"- Model: **{config.MODEL_TYPE}**")
st.sidebar.markdown(f"- Warm‑up: **{config.WINDOW_SIZE}** days")
st.sidebar.markdown(f"- Features: {config.FEATURE_WINDOWS}")

st.markdown('<div class="main-header">🔄 P2Quant Online Learning</div>', unsafe_allow_html=True)
st.markdown('<div>Passive‑Aggressive Regressor + ADWIN Drift Detection – Incremental ETF Forecasting</div>', unsafe_allow_html=True)

if data is None:
    st.warning("No data available.")
    st.stop()

daily = data['daily_trading']
states = daily.get('model_states', {})

tabs = st.tabs(["📊 Combined", "📈 Equity Sectors", "💰 FI/Commodities"])
universe_keys = ["COMBINED", "EQUITY_SECTORS", "FI_COMMODITIES"]

for tab, key in zip(tabs, universe_keys):
    with tab:
        top = daily['top_picks'].get(key, [])
        universe_data = daily['universes'].get(key, {})
        if top:
            pick = top[0]
            ticker = pick['ticker']
            drift = states.get(key, {}).get(ticker, {}).get('drift_detected', False)
            display_hero_card(ticker, pick['forecast'], drift)

        st.markdown("### All ETFs (Ranked by Forecast)")
        rows = []
        for t, d in universe_data.items():
            rows.append({
                'Ticker': t,
                'Forecast': f"{d['forecast']*100:.3f}%",
                'Last Return': f"{d['last_return']*100:.3f}%",
                'Drift': '⚠️' if states.get(key, {}).get(t, {}).get('drift_detected', False) else ''
            })
        df = pd.DataFrame(rows).sort_values('Forecast', ascending=False)
        st.dataframe(df, use_container_width=True, hide_index=True)
