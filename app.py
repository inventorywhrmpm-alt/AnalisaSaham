import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

st.set_page_config(page_title="AI Saham Pro - Wyckoff Edition", layout="wide")
st.title("📈 AI Stock Predictor & Technical Analysis")

# --- SIDEBAR ---
st.sidebar.header("Konfigurasi Saham")
ticker_input = st.sidebar.text_input("Kode Saham (Contoh: SCMA, BBCA)", value="SCMA").upper()
ticker_yf = f"{ticker_input}.JK"

# --- 1. TRADINGVIEW WIDGET ---
st.subheader(f"Live Chart TradingView: {ticker_input}")
tradingview_script = f"""
<div class="tradingview-widget-container" style="height:600px; width:100%;">
  <div id="tradingview_chart" style="height:100%; width:100%;"></div>
  <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
  <script type="text/javascript">
  new TradingView.widget({{
    "autosize": true, "symbol": "IDX:{ticker_input}", "interval": "D",
    "timezone": "Asia/Jakarta", "theme": "dark", "style": "1", "locale": "id",
    "container_id": "tradingview_chart"
  }});
  </script>
</div>
"""
components.html(tradingview_script, height=620)

# --- 2. ENGINE ANALISA & AI ---
try:
    df = yf.download(ticker_yf, start="2023-01-01", auto_adjust=True)
    
    if not df.empty:
        # Perhitungan Indikator Teknis
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / loss)))

        # MACD
        df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

        # Fitur untuk AI
        df['S_5'] = df['Close'].rolling(window=5).mean()
        df['V_5'] = df['Volume'].rolling(window=5).mean()
        df_ml = df.dropna()

        # Training Model & Akurasi
        X = df_ml[['S_5', 'V_5', 'RSI', 'MACD']]
        y = df_ml['Close']
        split = int(len(df_ml) * 0.8)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X[:split], y[:split])
        
        y_pred = model.predict(X[split:])
        akurasi = r2_score(y[split:], y_pred)
        next_price = model.predict(X.tail(1))[0]

        # --- LOGIKA ANALISA ---
        latest = df_ml.iloc[-1]
        prev = df_ml.iloc[-2]

        # 1. Status Wyckoff (Sederhana berdasarkan Volume & Trend)
        if latest['Close'] > latest['S_5'] and latest['Volume'] > latest['V_5']:
            wyckoff = "Accumulation / Markup"
        elif latest['Close'] < latest['S_5'] and latest['Volume'] > latest['V_5']:
            wyckoff = "Distribution"
        else:
            wyckoff = "Neutral / Testing"

        # 2. MACD Status
        macd_status = "Bullish Crossover" if latest['MACD'] > latest['Signal'] else "Bearish Crossover"

        # 3. Divergence (Sederhana)
        div_status = "No Divergence"
        if latest['Close'] > prev['Close'] and latest['RSI'] < prev['RSI']:
            div_status = "Bearish Divergence"
        elif latest['Close'] < prev['Close'] and latest['RSI'] > prev['RSI']:
            div_status = "Bullish Divergence"

        # 4. Aksi (Signal)
        if latest['RSI'] < 30 or (latest['MACD'] > latest['Signal'] and div_status == "Bullish Divergence"):
            aksi = "BUY / ACCUMULATE"
            warna = "green"
        elif latest['RSI'] > 70 or (latest['MACD'] < latest['Signal'] and div_status == "Bearish Divergence"):
            aksi = "SELL / TAKE PROFIT"
            warna = "red"
        else:
            aksi = "WAIT / HOLD"
            warna = "yellow"

        # --- TAMPILAN DASHBOARD ---
        st.write("---")
        st.subheader("🤖 AI & Technical Analysis Summary")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Akurasi Model AI (R2)", f"{akurasi:.2%}")
        c2.metric("Estimasi Harga Besok", f"Rp{next_price:.2f}")
        c3.subheader(f"Status: :{warna}[{aksi}]")

        st.write("---")
        a1, a2, a3, a4 = st.columns(4)
        a1.write(f"**Wyckoff Phase:**\n{wyckoff}")
        a2.write(f"**MACD Status:**\n{macd_status}")
        a3.write(f"**RSI (14):**\n{latest['RSI']:.2f}")
        a4.write(f"**Divergence:**\n{div_status}")

except Exception as e:
    st.error(f"Error dalam analisa: {e}")
