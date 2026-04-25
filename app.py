import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor

# --- KONFIGURASI ---
st.set_page_config(page_title="AI Saham Kita", layout="wide")
st.title("📈 AI Stock Predictor")

# --- SIDEBAR ---
ticker_in = st.sidebar.text_input("Kode Saham (Contoh: SCMA, BBCA)", value="SCMA").upper()
ticker = f"{ticker_in}.JK"

@st.cache_data
def load_data(symbol):
    # Mengambil data yang cukup banyak agar indikator tidak kosong
    data = yf.download(symbol, start="2022-01-01")
    return data

if ticker_in:
    try:
        df = load_data(ticker)
        
        if df.empty:
            st.warning("Data tidak ditemukan. Pastikan kode saham benar.")
        else:
            # 1. Feature Engineering
            # Menggunakan copy() agar tidak merusak data asli untuk grafik candlestick
            df_ml = df.copy()
            df_ml['S_5'] = df_ml['Close'].rolling(window=5).mean()
            df_ml['V_5'] = df_ml['Volume'].rolling(window=5).mean()
            
            # Hapus data kosong (NaN) agar model bisa belajar
            df_ml = df_ml.dropna()

            if len(df_ml) > 10:
                # 2. Machine Learning
                X = df_ml[['S_5', 'V_5']]
                y = df_ml['Close']
                
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X, y.values.ravel())

                # 3. Prediksi
                # Mengambil baris terakhir untuk prediksi besok
                last_features = X.tail(1)
                next_price = model.predict(last_features)[0]

                # Tampilkan Metrik Prediksi
                st.subheader(f"Hasil Analisis AI: {ticker_in}")
                st.metric(label="Estimasi Harga Besok", value=f"Rp{next_price:.2f}")

                # 4. Grafik Candlestick
                st.subheader("Grafik Pergerakan Harga")
                fig = go.Figure(data=[go.Candlestick(
                    x=df.index,
                    open=df['Open'], high=df['High'],
                    low=df['Low'], close=df['Close'],
                    name='Harga Pasar'
                )])
                
                fig.update_layout(template="plotly_dark", height=500, xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Data terlalu sedikit untuk melakukan prediksi.")

    except Exception as e:
        st.error(f"Terjadi kesalahan teknis: {e}")
