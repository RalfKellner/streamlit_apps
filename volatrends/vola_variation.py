import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model

# Set up the Streamlit app layout
st.title("Stock Returns Analysis")
st.sidebar.header("User Inputs")

# Sidebar inputs for the ticker, start date, end date, and frequency
ticker = st.sidebar.text_input("Ticker Symbol", value="AAPL")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2022-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2023-01-01"))
frequency = st.sidebar.selectbox("Data Frequency", ("Daily", "Weekly", "Monthly"))

data = yf.download(ticker, start = start_date, end = end_date)

daily_returns = data.Close.pct_change().dropna()
weekly_returns = data.resample("W").last().Close.pct_change().dropna().iloc[:-1]
monthly_returns = data.resample("ME").last().Close.pct_change().dropna().iloc[:-1]

if frequency == "Daily":
    returns = daily_returns
elif frequency == "Weekly":
    returns = weekly_returns
else:
    returns = monthly_returns

am = arch_model(returns*100, vol="Garch", p=1, o=0, q=1, dist="Normal")
garch_model = am.fit()

# Plot absolute returns with a horizontal line at the standard deviation
st.subheader(f"Absolute {frequency} Returns for {ticker}")
if not data['Close'].isnull().all():
    fig, ax = plt.subplots()
    returns.abs().plot(ax = ax, label = "absolute returns", alpha = 0.50)
    returns.rolling(window = 20).std(ddof = 1).plot(ax = ax, label = "rolling standard deviation 20 days")

    # Calculate and plot the standard deviation
    std_dev = returns.std(ddof=1)
    returns.ewm(alpha = 0.1).std().plot(ax = ax, label = "ewm standard deviation")
    std_dev = returns.std(ddof=1)
    garch_model.conditional_volatility.divide(100).plot(ax = ax, label = "garch volatility")
    ax.axhline(y=std_dev, color='grey', linestyle='--', label='unconditional standard deviation')

    ax.set_title(f"Absolute {frequency} Returns with Standard Deviation Line")
    ax.set_xlabel("Date")
    ax.set_ylabel(f"Absolute {frequency} Return")
    ax.legend()

    # Display the plot in the Streamlit app
    st.pyplot(fig)
else:
    st.write("No return data available for the selected period and frequency.")



