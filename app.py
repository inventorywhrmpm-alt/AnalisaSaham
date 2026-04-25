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
ticker_input = st.sidebar.text_input("Kode Saham (Contoh: SCMA, BBCA, INET)", value="SCMA").upper()
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
    # Menarik data lebih panjang agar indikator stabil
    df = yf.download(ticker_yf, start="2023-01-01", auto_adjust=True)
    
    if not df.empty and len(df) > 30:
        # Perhitungan Indikator (Gunakan .iloc agar label konsisten)
        close_prices = df['Close']
        
        # RSI
        delta = close_prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / loss)))

        # MACD
        df['EMA12'] = close_prices.ewm(span=12, adjust=False).mean()
        df['EMA26'] = close_prices.ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

        # Fitur AI
        df['S_5'] = close_prices.rolling(window=5).mean()
        df['V_5'] = df['Volume'].rolling(window=5).mean()
        
        # Buat copy data bersih untuk ML
        df_ml = df.dropna().copy()

        # Training Model
        X = df_ml[['S_5', 'V_5', 'RSI', 'MACD']]
        y = df_ml['Close']
        split = int(len(df_ml) * 0.8)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X[:split], y[:split])
        
        y_pred = model.predict(X[split:])
        akurasi = r2_score(y[split:], y_pred)
        next_price = model.predict(X.tail(1))[0]

        # --- LOGIKA ANALISA (MENGGUNAKAN NILAI SKALAR AGAR TIDAK ERROR LABEL) ---
        latest_close = float(df_ml['Close'].iloc[-1])
        prev_close = float(df_ml['Close'].iloc[-2])
        latest_rsi = float(df_ml['RSI'].iloc[-1])
        prev_rsi = float(df_ml['RSI'].iloc[-2])
        latest_vol = float(df_ml['Volume'].iloc[-1])
        avg_vol = float(df_ml['V_5'].iloc[-1])
        ma5 = float(df_ml['S_5'].iloc[-1])

        # 1. Status Wyckoff
        if latest_close > ma5 and latest_vol > avg_vol:
            wyckoff = "Accumulation / Markup"
        elif latest_close < ma5 and latest_vol > avg_vol:
            wyckoff = "Distribution"
        else:
            wyckoff = "Neutral / Testing"

        # 2. MACD & Divergence
        macd_val = float(df_ml['MACD'].iloc[-1])
        sig_val = float(df_ml['Signal'].iloc[-1])
        macd_status = "Bullish Crossover" if macd_val > sig_val else "Bearish Crossover"

        div_status = "No Divergence"
        if latest_close > prev_close and latest_rsi < prev_rsi:
            div_status = "Bearish Divergence"
        elif latest_close < prev_close and latest_rsi > prev_rsi:
            div_status = "Bullish Divergence"

        # 3. Rekomendasi Aksi
        if latest_rsi < 35 or (macd_val > sig_val and div_status == "Bullish Divergence"):
            aksi, warna = "BUY / ACCUMULATE", "green"
        elif latest_rsi > 65 or (macd_val < sig_val and div_status == "Bearish Divergence"):
            aksi, warna = "SELL / TAKE PROFIT", "red"
        else:
            aksi, warna = "WAIT / HOLD", "yellow"

        # --- TAMPILAN ---
        st.write("---")
        st.subheader("🤖 AI & Technical Analysis Summary")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Akurasi Model (R2)", f"{akurasi:.2%}")
        c2.metric("Estimasi Harga Besok", f"Rp{next_price:.2f}")
        c3.markdown(f"**Aksi Saat Ini:**\n### :{warna}[{aksi}]")

        st.write("---")
        a1, a2, a3, a4 = st.columns(4)
        a1.info(f"**Wyckoff Phase**\n\n{wyckoff}")
        a2.info(f"**MACD Status**\n\n{macd_status}")
        a3.info(f"**RSI (14)**\n\n{latest_rsi:.2f}")
        a4.info(f"**Divergence**\n\n{div_status}")

    else:
        st.warning("Data tidak cukup untuk melakukan analisa. Coba saham dengan histori lebih panjang.")

except Exception as e:
    st.error(f"Error dalam analisa: {e}")
