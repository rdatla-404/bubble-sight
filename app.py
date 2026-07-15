# =============================================================
# app.py
# Stock Bubble Detector -- Streamlit web app
# =============================================================
# A GUI wrapper around stock_bubble_detector.py / train_bubble_model.py.
# All the actual data fetching, feature engineering, and ML scoring
# logic lives in those two files -- this file is presentation only.
#
# Run with:
#   streamlit run app.py
#
# Libraries required:
#   pip install -r requirements.txt
# =============================================================

from datetime import datetime, timedelta

import streamlit as st
import pandas as pd

import stock_bubble_detector as sbd
import train_bubble_model as trainer

st.set_page_config(
    page_title="Stock Bubble Detector",
    page_icon="chart",
    layout="wide",
)

RISK_COLORS = {
    "CRITICAL BUBBLE RISK": "#D32F2F",
    "HIGH BUBBLE RISK": "#F57C00",
    "MODERATE BUBBLE RISK": "#FBC02D",
    "LOW BUBBLE RISK": "#689F38",
    "HEALTHY -- No Bubble Detected": "#388E3C",
    "MODEL NOT TRAINED": "#757575",
    "INSUFFICIENT DATA": "#757575",
}

FEATURE_LABELS = {
    "rsi": "RSI",
    "ma_ratio": "Price / 200-day MA",
    "volatility": "Volatility",
    "accel": "Acceleration",
    "max_drawdown": "Max Drawdown",
}


# -------------------------------------------------------------
# Sidebar: model status + retrain control
# -------------------------------------------------------------
with st.sidebar:
    st.header("Model status")
    info = sbd.get_model_info()

    if info is None:
        st.warning("No trained model found yet.")
    else:
        st.success("Model loaded")
        st.caption(f"Trained on: {info['trained_on']}")
        st.caption(f"Training snapshots: {info['n_training_rows']}")
        st.caption(f"Ticker universe: {info['ticker_universe_size']} tickers")

    st.divider()
    st.caption(
        "Training downloads several years of history for ~60 tickers "
        "and can take a few minutes."
    )
    if st.button("Train / retrain model", use_container_width=True):
        progress_box = st.empty()
        with st.spinner("Downloading history and fitting the model..."):
            progress_box.info("Pulling ticker history -- this is the slow part.")
            rows = trainer.build_training_rows()
            progress_box.info(f"Collected {len(rows)} snapshots. Fitting model...")
            trainer.train_and_save(rows)
        sbd._MODEL_BUNDLE = None  # force reload of the new model
        progress_box.success("Model trained and saved.")
        st.rerun()

    st.divider()
    st.caption(
        "Bubble score comes from an Isolation Forest trained to recognize "
        "unusual RSI / MA-ratio / volatility / acceleration combinations, "
        "gated so it only fires in the 'overheating' direction -- not on "
        "crashes or other unrelated anomalies."
    )


# -------------------------------------------------------------
# Main input
# -------------------------------------------------------------
st.title("Stock Bubble Detector")
st.caption("ML-based bubble risk scoring, powered by Yahoo Finance data.")

col_a, col_b, col_c = st.columns([2, 1, 1])
with col_a:
    ticker_input = st.text_input("Ticker symbol", value="NVDA").upper().strip()
with col_b:
    years = st.slider("Years of history", min_value=1, max_value=5, value=2)
with col_c:
    st.write("")
    st.write("")
    analyze_clicked = st.button("Analyze", type="primary", use_container_width=True)

if "last_ticker" not in st.session_state:
    st.session_state.last_ticker = None


# -------------------------------------------------------------
# Analysis
# -------------------------------------------------------------
if analyze_clicked and ticker_input:
    end_date = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=365 * years)).strftime("%Y-%m-%d")

    with st.spinner(f"Downloading {ticker_input} data..."):
        data = sbd.load_stock_data(ticker_input, start_date, end_date)

    if data is None or data.empty:
        st.error(
            f"Could not load data for '{ticker_input}'. "
            "Double-check the ticker symbol and try again."
        )
    else:
        st.session_state.last_ticker = ticker_input
        st.session_state.data = data

if st.session_state.get("last_ticker") and "data" in st.session_state:
    ticker = st.session_state.last_ticker
    data = st.session_state.data

    # ---- Summary metrics ----
    st.subheader(f"{ticker} summary")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Price", f"${sbd.get_current_price(data):,.2f}")
    m2.metric("Total Return", f"{sbd.get_total_return(data):+.1f}%")
    m3.metric("Volatility (ann.)", f"{sbd.get_volatility(data):.1f}%")
    m4.metric("Max Drawdown", f"{sbd.get_max_drawdown(data):.1f}%")
    m5.metric("RSI (14d)", f"{float(sbd.get_rsi(data).iloc[-1]):.1f}")
    ratio = sbd.get_price_to_200ma_ratio(data)
    m6.metric("Price / 200MA", f"{float(ratio.iloc[-1]):.2f}x" if ratio is not None else "N/A")

    st.divider()

    # ---- Bubble score ----
    result = sbd.run_bubble_analysis(ticker, data)
    risk_color = RISK_COLORS.get(result["risk_level"], "#757575")

    left, right = st.columns([1, 2])
    with left:
        if result["bubble_score"] is not None:
            st.markdown(
                f"""
                <div style="text-align:center; padding: 1.5rem;
                            border-radius: 12px; background-color:{risk_color}22;
                            border: 2px solid {risk_color};">
                    <div style="font-size: 2.5rem; font-weight: 700; color:{risk_color};">
                        {result['bubble_score']} / 100
                    </div>
                    <div style="font-size: 1.1rem; font-weight: 600; color:{risk_color};">
                        {result['risk_level']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info(f"{result['risk_level']}")
            for w in result["warnings"]:
                st.caption(w)

    with right:
        if result["sub_scores"]:
            z_df = pd.DataFrame(
                {
                    "Feature": [FEATURE_LABELS.get(k, k) for k in result["sub_scores"]],
                    "Z-score": list(result["sub_scores"].values()),
                }
            ).set_index("Feature")
            st.caption("Feature z-scores vs. training population (|z| > 2 is unusual)")
            st.bar_chart(z_df)

    if result["warnings"] and result["bubble_score"] is not None:
        st.subheader("Warning flags")
        for w in result["warnings"]:
            st.warning(w)

    # ---- Strategies ----
    if result["bubble_score"] is not None:
        st.subheader("Suggested strategies")
        current_price = sbd.get_current_price(data)
        ma_200_value = (
            float(sbd.get_moving_average(data, 200).iloc[-1]) if len(data) >= 200 else None
        )
        strategies = sbd.generate_strategies(
            ticker, current_price, result["bubble_score"], ma_200_value
        )
        for s in strategies:
            with st.container(border=True):
                st.markdown(f"**[{s['priority']}] {s['action']}**")
                st.caption(s["detail"])

    # ---- Chart ----
    st.subheader("Price & RSI chart")
    fig = sbd.plot_stock_detail(ticker, data, show=False, save=False)
    st.pyplot(fig)

elif not analyze_clicked:
    st.info("Enter a ticker and click Analyze to get started.")
