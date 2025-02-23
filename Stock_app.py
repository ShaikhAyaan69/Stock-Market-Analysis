import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

# Streamlit UI Design
st.set_page_config(page_title="📈 Stock Price Predictor", layout="wide")
st.title("📈 Stock Price Prediction using Linear Regression")
st.markdown("🚀 **Upload stock market data and predict future prices!**")

# File Upload
uploaded_file = st.file_uploader("📂 Upload your stock data CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Ensure Date column exists
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
        st.success("✅ File uploaded successfully!")
        st.write("### 📊 Data Preview")
        st.dataframe(df.head())

        # Select Feature & Target side by side
        col1, col2 = st.columns(2)
        with col1:
            feature = st.selectbox("🎯 Select Feature Column", df.columns)
        with col2:
            target = st.selectbox("🎯 Select Target Column", df.columns, index=len(df.columns) - 1)

        # Prepare Data
        df['Lag_1'] = df[target].shift(1)
        df['Lag_7'] = df[target].shift(7)
        df.dropna(inplace=True)

        # Train-Test Split
        train = df.loc['2012-01-01':'2018-12-31']
        test = df.loc['2019-01-01':]

        X_train = train[['Lag_1', 'Lag_7']]
        y_train = train[target]
        X_test = test[['Lag_1', 'Lag_7']]
        y_test = test[target]
        
        # Scale the features
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train Linear Regression Model
        lr_model = LinearRegression()
        lr_model.fit(X_train_scaled, y_train)

        # Predict on test set
        y_pred_lr = lr_model.predict(X_test_scaled)

        # Future Predictions
        future_days = st.slider("📆 Select Future Days to Predict", min_value=1, max_value=60, value=30)
        future_dates = pd.date_range(start=X_test.index[-1] + pd.Timedelta(days=1), periods=future_days, freq="D")
        future_features = X_test_scaled[-future_days:]  # Modify based on feature engineering
        future_predictions = lr_model.predict(future_features)
        future_predictions_series = pd.Series(future_predictions, index=future_dates)

        # Display Plots Side by Side
        st.subheader("📉 Stock Price Predictions")
        col1, col2 = st.columns(2)

        with col1:
            st.write("### 📉 Actual vs Predicted Prices")
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(y_test.index, y_test, label="🟦 Actual Prices", color="blue")
            ax.plot(y_test.index, y_pred_lr, label="🔴 Predicted Prices", color="red", linestyle="dashed")
            ax.set_xlabel("Date")
            ax.set_ylabel("Stock Price")
            ax.set_title("📈 Stock Price Prediction")
            ax.legend()
            plt.xticks(fontsize=8, rotation=45)  # Adjust text size and rotation
            st.pyplot(fig)

        with col2:
            st.write("### 🔮 Future Stock Price Prediction")
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.plot(future_dates, future_predictions_series, label="🟢 Future Predictions", color="green", linestyle="dashed")
            ax2.set_xlabel("Date")
            ax2.set_ylabel("Stock Price")
            ax2.set_title("📈 Future Stock Price Forecast")
            ax2.legend()
            plt.xticks(fontsize=8, rotation=45)  # Adjust text size and rotation
            st.pyplot(fig2)

        # Show Forecast Table
        future_df = pd.DataFrame({"Date": future_dates, "Predicted Price": future_predictions})
        future_df.set_index("Date", inplace=True)
        st.write(future_df)

    else:
        st.error("❌ CSV must contain a 'Date' column for proper visualization.")
