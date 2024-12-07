import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px

# Apply global style to the app
st.markdown(
    """
    <style>
    .main {
        background-color: #f5f5f5;  /* Light gray background for the main section */
    }
    .sidebar .sidebar-content {
        background-color: #1f4e78;  /* Dark blue sidebar background */
        color: white;               /* White text in sidebar */
    }
    h1, h2, h3, h4 {
        color: #1f4e78;             /* Consistent dark blue headings */
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #e8f4fc;  /* Light blue for tabs */
        color: #000000;             /* Black text for tabs */
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #c2e0f4;  /* Highlight effect on hover */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Set the Streamlit app title
st.title("📈 Stock Dashboard")

# Sidebar inputs for user interaction
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
            st.subheader(f"📊 Stock Data for {ticker}")
            st.dataframe(data.style.format("{:.2f}").set_table_styles(
                [{'selector': 'thead th', 'props': [('background-color', '#1f4e78'), 
                                                    ('color', 'white')]}]
            ))

            # Extract adjusted close prices and ensure they are 1D
            adj_close = data['Adj Close']
            
            # Plot using Plotly
            fig = px.line(
                x=adj_close.index, 
                y=adj_close,  # Directly use Series; it has matching index and values
                title=f"{ticker} Stock Price Over Time", 
                labels={'x': 'Date', 'y': 'Adjusted Close Price'},
                template="plotly_white",
                color_discrete_sequence=["#1f4e78"]
            )
            fig.update_layout(title_font_color="#1f4e78")
            st.plotly_chart(fig)
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Tabs for various sections
pricing_data, fundamental_data, news, stock_price_prediction_data = st.tabs(
    ["📈 Pricing Data", "📊 Fundamental Data", "📰 News", "🔮 Stock Price Prediction"]
)

# Pricing Data Tab
with pricing_data:
    st.header("📈 Price Movement")
    data2 = data
    data2["% Change"] = data['Adj Close'] / data['Adj Close'].shift(1) - 1
    data2.dropna(inplace=True)
    st.write(data2.style.highlight_max(axis=0, color="#c2e0f4"))
    annual_return = data2['% Change'].mean() * 252 * 100
    st.metric("Annual Return", f"{annual_return:.2f}%")
    stdev = np.std(data2['% Change']) * np.sqrt(252)
    st.metric("Standard Deviation", f"{stdev * 100:.2f}%")
    st.metric("Risk Adjustment Return", f"{stdev * 100:.2f}%")

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

# Stock Fundamental Data
from alpha_vantage.fundamentaldata import FundamentalData
with fundamental_data:
    key = 'HXQPXAEDJBJPWREY'
    fd = FundamentalData(key, output_format='pandas')
    st.subheader('Company Overview')
    company_overview = fd.get_company_overview(ticker)[0]
    cv = company_overview.T[2:]
    cv.columns = list(company_overview.T.iloc[0])
    st.write(cv)
    st.subheader('Balance Sheet')
    balance_sheet = fd.get_balance_sheet_annual(ticker)[0]
    bs = balance_sheet.T[2:]
    bs.columns = list(balance_sheet.T.iloc[0])
    st.write(bs)
    st.subheader('Income Statement')
    income_statement = fd.get_income_statement_annual(ticker)[0]
    is1 = income_statement.T[2:]
    is1.columns = list(income_statement.T.iloc[0])
    st.write(is1)
    st.subheader('Dividends Data')
    dividends=fd.get_dividends(ticker)[0]
    d=dividends.T[2:]
    d.columns=list(dividends.T.iloc[0])
    st.write(d)

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
