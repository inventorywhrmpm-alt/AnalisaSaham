import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor

# Judul
st.set_page_config(page_title="AI Saham", layout="wide")
st.title("📈 AI Stock Predictor")

# Input
ticker_in = st.sidebar.text_input("Kode Saham (Contoh: SCMA, BBCA)", value="SCMA").upper()
ticker = f"{ticker_in}.JK"

# Fungsi Load Data
@st.cache_data
def load_data(symbol):
    data = yf.download(symbol, start="2022-01-01")
    return data

if ticker_in:
    try:
        df = load_data(ticker)
        
        if df.empty:
            st.warning("Data tidak ditemukan. Pastikan kode saham benar.")
        else:
            # Feature Engineering kilat
            df['S_5'] = df['Close'].rolling(window=5).mean()
            df['V_5'] = df['Volume'].rolling(window=5).mean()
            df = df.dropna()

            # Machine Learning Sederhana
            X = df[['S_5', 'V_5']]
            y = df['Close']
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(X, y.values.ravel())

            # Metrics
            next_price = model.predict(X.tail(1))[0]
            st.metric(f"Prediksi Harga Besok ({ticker_in})", f"Rp{next_price:.2f}")

            # GRAFIK CANDLESTICK
            st.subheader("Interactive Chart")
            fig = go.Figure(data=[go.Candlestick(
                x=df.index,
                open=df['Open'], high=df['High'],
                low=df['Low'], close=df['Close'],
                name='Market'
            )])
            
            fig.update_layout(template="plotly_dark", height=500, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
