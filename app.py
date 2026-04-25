import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

st.set_page_config(page_title="AI Saham Kita Pro", layout="wide")
st.title("📈 AI Stock Predictor & Dashboard")

# --- SIDEBAR (Start & End Date Ditambahkan) ---
st.sidebar.header("Konfigurasi Data")
ticker_input = st.sidebar.text_input("Kode Saham (Contoh: SCMA, BBCA)", value="SCMA").upper()
ticker = f"{ticker_input}.JK"

# Input Tanggal
start_date = st.sidebar.date_input("Tanggal Mulai", value=pd.to_datetime("2021-01-01"))
end_date = st.sidebar.date_input("Tanggal Akhir", value=pd.to_datetime("today"))

@st.cache_data
def load_data(symbol, start, end):
    data = yf.download(symbol, start=start, end=end, auto_adjust=True)
    return data

if ticker_input:
    try:
        df = load_data(ticker, start_date, end_date)
        
        if df.empty or len(df) < 30:
            st.warning("Data tidak cukup atau tidak ditemukan. Coba rentang tanggal yang lebih lama.")
        else:
            # --- FEATURE ENGINEERING ---
            # Menghindari error kolom, kita pastikan ambil kolom Close & Volume yang benar
            df['S_5'] = df['Close'].rolling(window=5).mean()
            df['S_20'] = df['Close'].rolling(window=20).mean() # Fitur tambahan untuk akurasi
            df['V_5'] = df['Volume'].rolling(window=5).mean()
            
            # RSI Sederhana
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            df['RSI'] = 100 - (100 / (1 + (gain / loss)))
            
            # Bersihkan NaN
            df_ml = df.dropna().copy()

            # --- MACHINE LEARNING ---
            X = df_ml[['S_5', 'S_20', 'V_5', 'RSI']]
            y = df_ml['Close']
            
            # Split data secara kronologis (Time Series)
            split = int(len(df_ml) * 0.8)
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]
            
            # Training Model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Prediksi
            y_pred = model.predict(X_test)
            next_price = model.predict(X.tail(1))[0]

            # --- METRICS ---
            st.subheader(f"Statistik Prediksi: {ticker_input}")
            m1, m2, m3 = st.columns(3)
            
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            # Jika R2 minus, kita beri peringatan warna merah
            m1.metric("Akurasi Model (R2)", f"{r2:.2%}", delta="Rendah" if r2 < 0 else None, delta_color="inverse")
            m2.metric("Rata-rata Error (MAE)", f"Rp{mae:.2f}")
            m3.metric("Estimasi Harga Besok", f"Rp{next_price:.2f}")

            # --- GRAFIK CANDLESTICK ---
            st.subheader("Grafik Candlestick & Prediksi AI")
            
            fig = go.Figure()

            # Plot Candlestick
            fig.add_trace(go.Candlestick(
                x=df.index,
                open=df['Open'], high=df['High'],
                low=df['Low'], close=df['Close'],
                name='Market'
            ))

            # Plot Garis Prediksi pada Data Uji
            fig.add_trace(go.Scatter(
                x=y_test.index, y=y_pred, 
                line=dict(color='#FFA500', width=2), 
                name='AI Prediction'
            ))

            fig.update_layout(
                template="plotly_dark", 
                height=600, 
                xaxis_rangeslider_visible=False,
                yaxis_title="Harga (IDR)",
                margin=dict(l=10, r=10, t=10, b=10)
            )
            
            # Paksa tampilkan grafik dengan use_container_width
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Terjadi error: {e}")
