# fix_model_scaler.py
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib, os

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# Create dummy training data with 12 columns (same as your Streamlit features)
# Columns: Open, High, Low, Close, Volume, HL_PCT, PCT_change, MA7, MA21, EMA21, STD21, Momentum_7
X = np.random.rand(500, 12) * 100
y = X[:, 3] * 1.02 + np.random.randn(500) * 2  # close + noise

# Fit scaler and model
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

lr = LinearRegression()
lr.fit(X_scaled, y)

# Save both
joblib.dump(scaler, "models/standard_scaler.joblib")
joblib.dump(lr, "models/linear_regression.joblib")

print("✅ Recreated 'standard_scaler.joblib' and 'linear_regression.joblib' for 12 features.")
# app.py
import streamlit as st
import yfinance as yf
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

st.title("Stock Price Next-Day Prediction")

ticker = st.text_input("Ticker (e.g., AAPL)", value="AAPL")
start_date = st.date_input("Start date", value=pd.to_datetime("2015-01-01"))
end_date = st.date_input("End date", value=pd.to_datetime("today"))

if st.button("Fetch & Predict"):
    df = yf.download(ticker, start=start_date.isoformat(), end=end_date.isoformat())
    if df.empty:
        st.error("No data found for this ticker/date range.")
    else:
        st.line_chart(df['Close'])
        st.write("Using saved Linear Regression model for demonstration...")

        # Load scalers & model
        scaler = joblib.load('models/standard_scaler.joblib')
        lr = joblib.load('models/linear_regression.joblib')

        # latest features
        latest = df.copy().tail(60)  # ensure we have enough days for features like MA21
        latest_feat = latest.copy()
        latest_feat['HL_PCT'] = (latest_feat['High'] - latest_feat['Low']) / latest_feat['Close']
        latest_feat['PCT_change'] = (latest_feat['Close'] - latest_feat['Open']) / latest_feat['Open']
        latest_feat['MA7'] = latest_feat['Close'].rolling(window=7).mean()
        latest_feat['MA21'] = latest_feat['Close'].rolling(window=21).mean()
        latest_feat['EMA21'] = latest_feat['Close'].ewm(span=21, adjust=False).mean()
        latest_feat['STD21'] = latest_feat['Close'].rolling(window=21).std()
        latest_feat['Momentum_7'] = latest_feat['Close'] - latest_feat['Close'].shift(7)
        latest_feat = latest_feat.dropna()

        if latest_feat.empty:
            st.error("Not enough recent data to compute features. Try a longer history.")
        else:
            last_row = latest_feat.iloc[-1][[
                'Open','High','Low','Close','Volume','HL_PCT','PCT_change','MA7','MA21','EMA21','STD21','Momentum_7'
            ]].values.reshape(1,-1)
            last_row_scaled = scaler.transform(last_row)
            pred = lr.predict(last_row_scaled)[0]
            st.success(f"Predicted next-day Close price for {ticker}: {pred:.2f}")
