import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

st.set_page_config(page_title="AI Saham Kita v1", layout="wide")
st.title("📈 AI Stock Predictor & Dashboard")

# --- SIDEBAR ---
ticker_in = st.sidebar.text_input("Kode Saham (Tanpa .JK)", value="SCMA").upper()
ticker = f"{ticker_in}.JK"

@st.cache_data
def load_data(symbol):
    # Menggunakan auto_adjust=True untuk memastikan kolom konsisten
    data = yf.download(symbol, start="2022-01-01", auto_adjust=True)
    return data

if ticker_in:
    try:
        df = load_data(ticker)
        
        if df.empty or len(df) < 20:
            st.warning(f"Data untuk {ticker_in} tidak cukup atau tidak ditemukan.")
        else:
            # --- PREPROCESSING ---
            # Menghitung Fitur
            df['S_5'] = df['Close'].rolling(window=5).mean()
            df['V_5'] = df['Volume'].rolling(window=5).mean()
            
            # Tambahkan RSI Sederhana untuk akurasi lebih baik
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            df['RSI'] = 100 - (100 / (1 + (gain / loss)))
            
            # Dataset untuk ML
            df_ml = df.dropna()
            
            # --- MACHINE LEARNING ---
            X = df_ml[['S_5', 'V_5', 'RSI']]
            y = df_ml['Close']
            
            # Split data (80% train, 20% test)
            split = int(len(df_ml) * 0.8)
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]
            
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Prediksi untuk Akurasi & Besok
            y_pred = model.predict(X_test)
            next_price = model.predict(X.tail(1))[0]
            
            # --- TAMPILAN METRIK ---
            st.subheader(f"Statistik Prediksi: {ticker_in}")
            m1, m2, m3 = st.columns(3)
            
            # Hitung Akurasi (R2 Score)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            m1.metric("Akurasi Model (R2)", f"{r2:.2%}")
            m2.metric("Rata-rata Error (MAE)", f"Rp{mae:.2f}")
            m3.metric("Estimasi Harga Besok", f"Rp{next_price:.2f}")

            # --- GRAFIK CANDLESTICK ---
            st.subheader("Grafik Candlestick & Prediksi AI")
            fig = go.Figure()

            # Candlestick
            fig.add_trace(go.Candlestick(
                x=df.index,
                open=df['Open'], high=df['High'],
                low=df['Low'], close=df['Close'],
                name='Harga Pasar'
            ))

            # Garis Prediksi pada data Test
            fig.add_trace(go.Scatter(
                x=y_test.index, y=y_pred, 
                line=dict(color='orange', width=2), 
                name='AI Prediction (Test Area)'
            ))

            fig.update_layout(
                template="plotly_dark", 
                height=600, 
                xaxis_rangeslider_visible=False,
                yaxis_title="Harga (IDR)"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("💡 Garis **Oren** menunjukkan kemampuan AI dalam menebak data histori terbaru.")

    except Exception as e:
        st.error(f"Terjadi error: {e}")
