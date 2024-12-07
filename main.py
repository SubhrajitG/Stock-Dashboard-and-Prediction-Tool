import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px

# Add custom CSS for styling
st.markdown(
    """
    <style>
    /* Background and Font Styling */
    body {
        background-color: #f5f5f5;
        font-family: 'Arial', sans-serif;
    }
    h1 {
        color: #4CAF50;
        text-align: center;
        margin-top: 20px;
        font-weight: bold;
    }
    .sidebar .sidebar-content {
        background-color: #f0f0f0;
        padding: 20px;
    }
    .stDataFrame, .stPlotlyChart, .stMetric {
        border-radius: 10px;
        box-shadow: 0px 4px 8px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .metric-container {
        display: flex;
        justify-content: space-between;
    }
    .metric {
        text-align: center;
        padding: 10px;
        border: 1px solid #ddd;
        border-radius: 8px;
        background: #ffffff;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    h2 {
        color: #0033cc;
        font-weight: 600;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Set the Streamlit app title
st.markdown('<h1>📊 Stock Dashboard</h1>', unsafe_allow_html=True)

# Sidebar inputs for user interaction
st.sidebar.header("Input Parameters")
ticker = st.sidebar.text_input("Enter Stock Ticker:", "AAPL")  # Default ticker: AAPL
start_date = st.sidebar.date_input("Select Start Date:", value=pd.to_datetime("2022-01-01"))
end_date = st.sidebar.date_input("Select End Date:", value=pd.to_datetime("2024-12-31"))

# Ensure valid input dates
if start_date > end_date:
    st.error("Error: Start date must be before end date.")
else:
    # Fetch stock data
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        
        # Check if data is returned
        if data.empty:
            st.warning(f"No data found for ticker '{ticker}' in the given date range.")
        else:
            # Display the data
            st.subheader(f"Stock Data for {ticker}")
            st.dataframe(data.style.highlight_max(axis=0, color="lightgreen"))

            # Extract adjusted close prices and ensure they are 1D
            adj_close = data['Adj Close']
            
            # Plot using Plotly
            fig = px.line(
                x=adj_close.index, 
                y=adj_close,  # Directly use Series; it has matching index and values
                title=f"{ticker} Stock Price Over Time", 
                labels={'x': 'Date', 'y': 'Adjusted Close Price'},
                color_discrete_sequence=["#1f77b4"]
            )
            st.plotly_chart(fig)
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Tabs for various sections
pricing_data, fundamental_data, news, stock_price_prediction_data = st.tabs(
    ["📈 Pricing Data", "📚 Fundamental Data", "📰 News", "🤖 Stock Price Prediction Data"]
)

# Pricing Data Tab
with pricing_data:
    st.header("Price Movement")
    data2 = data
    data2['% Change'] = data['Adj Close'] / data['Adj Close'].shift(1) - 1
    data2.dropna(inplace=True)
    st.dataframe(data2.style.background_gradient(cmap="coolwarm"))
    annual_return = data2['% Change'].mean() * 252 * 100
    st.metric(label="Annual Return (%)", value=f"{annual_return:.2f}")
    stdev = np.std(data2['% Change']) * np.sqrt(252)
    st.metric(label="Standard Deviation (%)", value=f"{stdev * 100:.2f}")
    risk_adj_return = annual_return / (stdev * 100)
    st.metric(label="Risk Adjustment Return", value=f"{risk_adj_return:.2f}")


    # News Data Tab
from stocknews import StockNews
with news:
    st.header(f'📰 News for {ticker}')
    sn = StockNews(ticker, save_news=False)
    df_news = sn.read_rss()
    for i in range(10):
        st.subheader(f'🗞️ News {i+1}')
        st.write(f"**Published:** {df_news['published'][i]}")
        st.write(f"**Title:** {df_news['title'][i]}")
        st.write(f"**Summary:** {df_news['summary'][i]}")
        title_sentiment = df_news['sentiment_title'][i]
        st.info(f"**Title Sentiment:** {title_sentiment}")
        news_sentiment = df_news['sentiment_summary'][i]
        st.info(f"**News Sentiment:** {news_sentiment}")

# Stock Price Prediction Tab
with stock_price_prediction_data:
    st.header("Stock Market Predictor")
    
    # Load the pre-trained model
    model_path = "C:/Users/subhr/Documents/Stock Price Prediction/Stock Predictions Model.keras"
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"Failed to load model: {e}")

    # Prepare the data for prediction
    st.subheader("Stock Data")
    st.dataframe(data)

    # Splitting data into training and testing sets
    data_train = pd.DataFrame(data.Close[0: int(len(data) * 0.80)])
    data_test = pd.DataFrame(data.Close[int(len(data) * 0.80): len(data)])
    
    scaler = MinMaxScaler(feature_range=(0, 1))

    # Preparing test data
    pas_100_days = data_train.tail(100)
    data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
    data_test_scaled = scaler.fit_transform(data_test)

    # Generate Moving Averages and plots
    st.subheader("Price vs 50-Day Moving Average")
    ma_50_days = data.Close.rolling(50).mean()
    fig1 = plt.figure(figsize=(8, 6))
    plt.plot(ma_50_days, "r", label="50-Day MA")
    plt.plot(data.Close, "g", label="Stock Price")
    plt.legend()
    st.pyplot(fig1)

    st.subheader("Price vs 50-Day MA vs 100-Day MA")
    ma_100_days = data.Close.rolling(100).mean()
    fig2 = plt.figure(figsize=(8, 6))
    plt.plot(ma_50_days, "r", label="50-Day MA")
    plt.plot(ma_100_days, "b", label="100-Day MA")
    plt.plot(data.Close, "g", label="Stock Price")
    plt.legend()
    st.pyplot(fig2)

    st.subheader("Price vs 100-Day MA vs 200-Day MA")
    ma_200_days = data.Close.rolling(200).mean()
    fig3 = plt.figure(figsize=(8, 6))
    plt.plot(ma_100_days, "r", label="100-Day MA")
    plt.plot(ma_200_days, "b", label="200-Day MA")
    plt.plot(data.Close, "g", label="Stock Price")
    plt.legend()
    st.pyplot(fig3)

    # Preparing input for the model
    x = []
    y = []

    for i in range(100, data_test_scaled.shape[0]):
        x.append(data_test_scaled[i - 100:i])
        y.append(data_test_scaled[i, 0])

    x, y = np.array(x), np.array(y)

    try:
        # Predicting stock prices
        predictions = model.predict(x)
        scale = 1 / scaler.scale_[0]
        predictions = predictions * scale
        y = y * scale

        # Plotting original vs predicted prices
        st.subheader("Original Stock Prices vs Predicted Stock Prices")
        fig4 = plt.figure(figsize=(8, 6))
        plt.plot(predictions, "r", label="Predicted Price")
        plt.plot(y, "g", label="Original Price")
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.legend()
        st.pyplot(fig4)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
