import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

def buy_hold_sell(ret):
    if ret > 15:
        return "BUY"
    elif ret > 5:
        return "HOLD"
    else:
        return "SELL"


def risk_score(vol):
    if vol < 1:
        return 2
    elif vol < 2:
        return 4
    elif vol < 3:
        return 6
    elif vol < 4:
        return 8
    else:
        return 10


def max_drawdown(returns):
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    return drawdown.min() * 100

@st.cache_data(ttl=1800)
def fetch_stock_data(symbols, period):
    return yf.download(
        symbols,
        period=period,
        group_by="ticker",
        threads=False
    )


def build_stock_table(symbols, period):
    rows = []
    price_data = {}

    try:
        data = fetch_stock_data(symbols, period)
    except Exception:
        st.error("Yahoo Finance rate-limited. Please wait a few minutes and retry.")
        st.stop()

    for symbol in symbols:
        try:
            hist = data[symbol].dropna()
            if hist.empty:
                continue

            start = hist["Close"].iloc[0]
            end = hist["Close"].iloc[-1]
            returns = hist["Close"].pct_change().dropna()
            years = len(hist) / 252

            cagr = ((end / start) ** (1 / years) - 1) * 100
            vol = returns.std() * 100
            mdd = max_drawdown(returns)
            ret_pct = ((end - start) / start) * 100

            rows.append({
                "Stock": symbol,
                "Start Price ($)": round(start, 2),
                "End Price ($)": round(end, 2),
                "Return (%)": round(ret_pct, 2),
                "CAGR (%)": round(cagr, 2),
                "Volatility (%)": round(vol, 2),
                "Risk Score": risk_score(vol),
                "Max Drawdown (%)": round(mdd, 2),
                "Recommendation": buy_hold_sell(ret_pct)
            })

            price_data[symbol] = hist["Close"]

        except Exception:
            continue

    return pd.DataFrame(rows), pd.DataFrame(price_data)


st.set_page_config(
    page_title="Investment Analytics Dashboard",
    page_icon="📈",
    layout="wide"
)

st.markdown(
    "<h1 style='text-align:center;color:#4CAF50;'>📈 Investment Analytics Dashboard</h1>",
    unsafe_allow_html=True
)

st.sidebar.header("Configuration")

symbols_input = st.sidebar.text_input(
    "Stock Symbols (comma separated)",
    "AAPL, MSFT, GOOGL"
)

period = st.sidebar.selectbox(
    "Time Period",
    ["3mo", "6mo", "1y", "2y"]
)

capital = st.sidebar.number_input(
    "Total Capital ($)",
    min_value=1000,
    value=100000,
    step=1000
)

symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]


if "summary_df" not in st.session_state:
    st.session_state.summary_df = None
    st.session_state.price_df = None


if st.sidebar.button("Generate Report"):

    summary_df, price_df = build_stock_table(symbols, period)

    if summary_df.empty:
        st.warning("No valid stock data found.")
        st.stop()

    st.session_state.summary_df = summary_df
    st.session_state.price_df = price_df



if st.session_state.summary_df is not None:

    df = st.session_state.summary_df
    prices = st.session_state.price_df

    st.subheader("📊 Stock Performance Summary")
    st.dataframe(df, use_container_width=True)

    st.subheader("💼 Portfolio Allocation (Risk Adjusted)")
    weights = (10 - df["Risk Score"]) / (10 - df["Risk Score"]).sum()
    alloc_df = pd.DataFrame({
        "Stock": df["Stock"],
        "Allocation (%)": (weights * 100).round(2),
        "Capital ($)": (weights * capital).round(2)
    })
    st.dataframe(alloc_df, use_container_width=True)

    st.subheader("📈 Price Trend")
    fig = go.Figure()
    for col in prices.columns:
        fig.add_trace(go.Scatter(x=prices.index, y=prices[col], mode="lines", name=col))
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🔗 Correlation Heatmap")
    corr = prices.pct_change().corr()
    heatmap = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu")
    st.plotly_chart(heatmap, use_container_width=True)

    st.subheader("📥 Export Report")
    st.download_button(
        label="Download CSV",
        data=df.to_csv(index=False),
        file_name="investment_report.csv",
        mime="text/csv"
    )

    st.success("✅ Report generated successfully.")
