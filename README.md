# Stock Bubble Detector

### ML-powered overheating detection for any publicly traded stock

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/ML-Isolation%20Forest-orange.svg)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/App-Streamlit-FF4B4B.svg)](https://streamlit.io)
[![Data](https://img.shields.io/badge/Data-Yahoo%20Finance-purple.svg)](https://pypi.org/project/yfinance/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Enter any stock ticker. An Isolation Forest anomaly model — trained across dozens of tickers' historical behavior — flags when a stock's momentum, valuation extension, volatility, and acceleration look statistically unlike normal market conditions. Gated to fire only on overheating, not crashes. Get a **Bubble Score (0–100)** and personalized strategies.

---

## Why This Exists

Every major market bubble showed the same handful of warning signs before it collapsed:

| Bubble | Year | Crash | Key Signal |
|--------|------|-------|------------|
| Dot-Com | 2000 | NASDAQ −78% | RSI > 85, parabolic price acceleration |
| Housing Crisis | 2008 | S&P 500 −57% | Extreme volatility, momentum collapse |
| Crypto Mania | 2021 | Bitcoin −77% | Price 3× above 200-day MA |
| Meme Stocks | 2021 | GameStop −90% | RSI > 90, extreme acceleration |

Rather than hand-picking thresholds for those signals, this project trains an **Isolation Forest** on real historical data across a diverse universe of tickers, and lets the model learn what "normal" looks like on its own. A stock is flagged when its current feature combination sits statistically outside that learned normal — and only when the anomaly points toward overheating, not toward a crash.

---

## Features

- **ML-based scoring** — Isolation Forest anomaly detection, not fixed point thresholds
- **Directional gating** — a stock cratering 40% won't score as a bubble; only "overheating" anomalies count
- **Three ways to use it** — interactive CLI, Jupyter notebook walkthrough, or a Streamlit web app
- **Company search** — look up a ticker by company name from a 7,000+ symbol NASDAQ/NYSE/AMEX directory
- **Model retraining** — retrain from the app's sidebar or via a configurable CLI script
- **Tested** — 27 unit tests on the core metric calculations
- **Built-in sanity checks** — validates the trained model against known historical bubble/calm periods

---

## Project Structure

```
stock-bubble-detector/
│
├── stock_bubble_detector.py    # Core: data loading, feature engineering, ML scoring, CLI
├── train_bubble_model.py       # Trains the Isolation Forest model (run this first)
├── app.py                      # Streamlit web app
├── all_tickers.csv             # NASDAQ/NYSE/AMEX ticker directory for company-name search
├── stock_bubble_detector.ipynb # Step-by-step notebook walkthrough
├── requirements.txt            # Pinned dependencies
├── tests/
│   └── test_bubble_detector.py # Unit tests (pytest)
└── README.md                   # This file
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model (one-time, ~5–10 minutes)
```bash
python train_bubble_model.py
```
This downloads several years of history across a curated, sector-diverse universe of ~60 tickers, builds a feature snapshot every 252 trading days per ticker (spaced out deliberately — see [How Training Works](#how-training-works)), fits the model, and runs a quick sanity check against known historical bubble/calm periods. Override the defaults without touching source:
```bash
python train_bubble_model.py --years 4 --contamination 0.03 --step-days 63
```

### 3. Run it

**CLI:**
```bash
python stock_bubble_detector.py
```
```
  STOCK BUBBLE ANALYZER -- INPUT
  Enter stock ticker symbol (e.g. AAPL, TSLA, NVDA): NVDA
  How many years of data to analyze? (1-5, default 2): 2

  BUBBLE REPORT: NVDA
  Bubble Score : 82.4 / 100  (ML anomaly score)
  Risk Level   : CRITICAL BUBBLE RISK

  Feature Z-Scores (vs. training population, |z| > 2 is unusual):
    RSI                  z= +2.41  [########------------]
    Price / 200-day MA   z= +3.02  [##########----------]
    Volatility           z= +1.15  [###-----------------]
    Acceleration         z= +2.88  [#########-----------]
    Max Drawdown         z= -0.62  [--------------------]
```

**Web app:**
```bash
streamlit run app.py
```

**Notebook:**
```bash
jupyter notebook stock_bubble_detector.ipynb
```

**Tests:**
```bash
pytest tests/
```

---

## How Scoring Works

1. Five continuous features are computed for the ticker: **RSI (14-day)**, **price ÷ 200-day moving average**, **annualized volatility (trailing 90 days)**, **acceleration** (30-day return vs. the prior 30 days), and **max drawdown (trailing 252 days)**.
2. The trained Isolation Forest scores how statistically unusual that combination is versus its training population, producing a raw anomaly score.
3. The score is normalized to **0–100** against the training set's own score distribution.
4. A **directional gate** checks whether the anomaly points toward overheating (elevated RSI, MA ratio, and acceleration) rather than a crash or an unrelated kind of outlier. If it doesn't, the score is capped low regardless of how anomalous the raw signal was.

| Bubble Score | Risk Level | Suggested Action |
|-------------|-----------|-------------------|
| 0–9 | Healthy | Hold or accumulate |
| 10–29 | Low | Monitor |
| 30–49 | Moderate | Watch closely, no new buys |
| 50–69 | High | Trim position, consider a trailing stop |
| 70–100 | Critical | Reduce exposure, consider hedging |

---

## How Training Works

`train_bubble_model.py` pulls history for a curated, sector-diverse set of ~60 large, liquid tickers (not "every ticker on the market" — see [Limitations](#honest-limitations) for why) and takes a feature snapshot from each one every **252 trading days** by default. That spacing is deliberate: it's the longest lookback window any single feature uses (max drawdown), so consecutive snapshots for the same ticker don't share overlapping price data — they're closer to independent observations rather than near-duplicates of each other. You can lower `--step-days` for a denser, faster training set, at the cost of reintroducing that correlation.

After training, `validate_against_known_cases()` runs the model against a handful of dates with well-known outcomes (e.g. NVDA's 2024 run-up, a calm period for JNJ) and prints a pass/fail table. This is a sanity check, not statistical proof the model is accurate — four data points can't validate a model. Treat a mismatch as a prompt to look closer, not a verdict.

---

## Honest Limitations

- **Curated ticker universe, not "all tickers."** Training on the full ~7,000-ticker market would drag the model's sense of "normal" toward thinly-traded penny stocks, making it worse at flagging bubbles in stocks people actually watch. `all_tickers.csv` is a lookup directory for the app's search feature only — it's intentionally not the training set.
- **Isolation Forest flags statistical rarity, not "bubble" specifically.** The directional gate reduces false positives on crashes, but this is still anomaly detection, not a certified bubble classifier.
- **Small validation set.** The known-case sanity check uses a handful of hand-picked historical dates. It catches obviously broken models, not subtle miscalibration.
- **No guarantee of predictive power.** A high score describes how unusual a stock's current behavior is relative to recent market history — it is not a forecast, and past patterns are not a guarantee of what happens next.

This tool is built for learning and exploration, not as financial advice.

---

## Requirements

```
yfinance, pandas, numpy, matplotlib, scikit-learn, joblib, streamlit, jupyterlab, notebook, pytest
```
See `requirements.txt` for pinned version ranges.

---

## License

MIT License — free to use, modify, and distribute for any purpose.

---

*Built with Python · scikit-learn · yfinance · pandas · Streamlit*
