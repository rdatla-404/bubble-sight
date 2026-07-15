# =============================================================
# train_bubble_model.py
# Trains the Isolation Forest anomaly model used by
# stock_bubble_detector.py.
#
# Run this once (and re-run periodically to keep it current):
#   python train_bubble_model.py
#
# It downloads several years of history for a broad universe of
# tickers, slides a window through each ticker's history to build
# thousands of feature snapshots, and fits an Isolation Forest to
# learn what "normal" market behavior looks like. Snapshots that
# don't fit that pattern are the anomalies stock_bubble_detector.py
# flags at inference time.
#
# Libraries required:
#   pip install yfinance pandas numpy scikit-learn joblib
# =============================================================

import numpy as np
import joblib
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

from stock_bubble_detector import (
    load_stock_data,
    compute_feature_vector,
    FEATURE_NAMES,
    MODEL_PATH,
)

# A broad, sector-diverse universe. More tickers = a better-calibrated
# sense of "normal." Feel free to extend this list.
TICKER_UNIVERSE = [
    # Tech / large growth
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD', 'CRM', 'ADBE',
    'ORCL', 'CSCO', 'INTC', 'AVGO', 'NFLX', 'SHOP', 'UBER', 'PLTR', 'SNOW', 'NOW',
    # Financials
    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'V', 'MA', 'AXP', 'BLK', 'SCHW',
    # Healthcare
    'JNJ', 'UNH', 'PFE', 'MRK', 'ABBV', 'LLY', 'TMO', 'ABT', 'CVS', 'BMY',
    # Consumer / industrial / energy / staples (defensive + cyclical mix)
    'WMT', 'PG', 'KO', 'PEP', 'COST', 'HD', 'MCD', 'NKE', 'DIS', 'SBUX',
    'XOM', 'CVX', 'CAT', 'BA', 'GE', 'HON', 'UPS', 'LMT', 'DE', 'MMM',
    'XLU', 'XLV', 'XLP',  # sector ETFs help anchor "defensive/normal" behavior
]

YEARS_OF_HISTORY = 6
SNAPSHOT_STEP_DAYS = 5     # take a feature snapshot every 5 trading days
MIN_HISTORY_DAYS = 250     # need >=200 for MA ratio, plus buffer


def build_training_rows():
    """
    Download history for the ticker universe and slide a window through
    each series to build a table of feature snapshots.

    Returns:
        list of dict -- each dict is one feature snapshot (one row).
    """
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=365 * YEARS_OF_HISTORY)).strftime('%Y-%m-%d')

    rows = []
    for ticker in TICKER_UNIVERSE:
        data = load_stock_data(ticker, start_date, end_date)
        if data is None or len(data) < MIN_HISTORY_DAYS:
            print(f"  Skipping {ticker} -- insufficient history.")
            continue

        count_for_ticker = 0
        for i in range(MIN_HISTORY_DAYS, len(data), SNAPSHOT_STEP_DAYS):
            snapshot = data.iloc[:i + 1]
            features = compute_feature_vector(snapshot)
            if features is not None:
                rows.append(features)
                count_for_ticker += 1

        print(f"  {ticker}: {count_for_ticker} snapshots")

    return rows


def train_and_save(rows):
    """
    Fit the scaler + Isolation Forest on the collected feature rows
    and save everything the inference code needs to bubble_model.pkl.
    """
    if len(rows) < 200:
        raise RuntimeError(
            f"Only collected {len(rows)} training rows -- too few to train "
            "a reliable model. Check your network access and ticker list."
        )

    x = np.array([[row[name] for name in FEATURE_NAMES] for row in rows])

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    model = IsolationForest(
        n_estimators=300,
        contamination=0.05,   # assume ~5% of snapshots look genuinely unusual
        random_state=42,
    )
    model.fit(x_scaled)

    # Normalize inference scores against this training run's distribution.
    scores = model.score_samples(x_scaled)
    score_low, score_high = np.percentile(scores, [1, 99])

    bundle = {
        'model': model,
        'scaler': scaler,
        'feature_names': FEATURE_NAMES,
        'score_low': float(score_low),
        'score_high': float(score_high),
        'trained_on': datetime.today().strftime('%Y-%m-%d'),
        'n_training_rows': len(rows),
        'ticker_universe_size': len(TICKER_UNIVERSE),
    }

    joblib.dump(bundle, MODEL_PATH)
    print(f"\n  Saved model to {MODEL_PATH}")
    print(f"  Trained on {len(rows)} snapshots across {len(TICKER_UNIVERSE)} tickers.")


def main():
    print("=" * 55)
    print("  TRAINING BUBBLE DETECTION MODEL")
    print("=" * 55)
    print(f"  Universe: {len(TICKER_UNIVERSE)} tickers, {YEARS_OF_HISTORY} years each\n")

    rows = build_training_rows()
    train_and_save(rows)


if __name__ == "__main__":
    main()
