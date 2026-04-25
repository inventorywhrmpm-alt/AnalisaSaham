import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="AI Saham Pro", layout="wide")

# --- SIDEBAR ---
st.sidebar.header("Konfigurasi Saham")
ticker_input = st.sidebar.text_input("Kode Saham (Contoh: SCMA, BBCA)", value="SCMA").upper()
ticker_yf = f"{ticker_input}.JK"

# --- 1. TRADINGVIEW WIDGET (PENGGANTI CANDLESTICK MANUAL) ---
st.subheader(f"Live Chart TradingView: {ticker_input}")

# Script Widget TradingView
tradingview_script = f"""
<div class="tradingview-widget-container" style="height:500px;">
  <div id="tradingview_chart"></div>
  <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
  <script type="text/javascript">
  new TradingView.widget({{
    "autosize": true,
    "symbol": "IDX:{ticker_input}",
    "interval": "D",
    "timezone": "Asia/Jakarta",
    "theme": "dark",
    "style": "1",
    "locale": "id",
    "toolbar_bg": "#f1f3f6",
    "enable_publishing": false,
    "hide_top_toolbar": false,
    "save_image": false,
    "container_id": "tradingview_chart"
  }});
  </script>
</div>
"""
# Tampilkan Widget
components.html(tradingview_script, height=500)

# --- 2. ENGINE PREDIKSI AI ---
try:
    df = yf.download(ticker_yf, start="2023-01-01", auto_adjust=True)
    
    if not df.empty:
        # Feature Engineering
        df['S_5'] = df['Close'].rolling(window=5).mean()
        df['V_5'] = df['Volume'].rolling(window=5).mean()
        df_ml = df.dropna()

        # Training kilat
        X = df_ml[['S_5', 'V_5']]
        y = df_ml['Close']
        model = RandomForestRegressor(n_estimators=100).fit(X, y.values.ravel())
        
        # Hasil Prediksi
        next_price = model.predict(X.tail(1))[0]
        
        st.write("---")
        st.subheader("🤖 Hasil Analisis AI")
        col1, col2 = st.columns(2)
        col1.metric("Ticker", ticker_input)
        col2.metric("Estimasi Harga Besok", f"Rp{next_price:.2f}")
        
    else:
        st.error("Data Yahoo Finance tidak tersedia untuk ticker ini.")
except Exception as e:
    st.error(f"Error AI: {e}")
