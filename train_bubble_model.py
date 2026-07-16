
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

import argparse
import logging
from datetime import datetime, timedelta

import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

from stock_bubble_detector import (
    load_stock_data,
    compute_feature_vector,
    run_bubble_analysis,
    reload_model,
    FEATURE_NAMES,
    MODEL_PATH,
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

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

# A handful of dates with well-known outcomes, used to sanity-check the
# trained model (see validate_against_known_cases). These are rough
# expectations, not ground truth labels -- markets are messy and the
# model can reasonably disagree on borderline cases. Their purpose is
# to catch a badly broken model (e.g. gate polarity flipped, scaler
# mismatched), not to certify accuracy.
KNOWN_TEST_CASES = [
    # (ticker, as-of date, expected label, note)
    ('NVDA', '2024-06-18', 'overheated', 'Pre-split run-up, RSI/MA-ratio extended'),
    ('TSLA', '2021-11-01', 'overheated', 'Near 2021 all-time-high blow-off'),
    ('JNJ',  '2023-06-01', 'calm',       'Low-beta defensive, unremarkable period'),
    ('KO',   '2023-09-01', 'calm',       'Low-beta defensive, unremarkable period'),
]

DEFAULT_YEARS_OF_HISTORY = 6

# The longest lookback any single feature uses is 252 trading days
# (max_drawdown). Spacing snapshots at least this far apart means
# consecutive snapshots for the same ticker don't share the same
# underlying price window -- i.e. they're close to statistically
# independent observations, not near-duplicates of each other.
#
# Trade-off, stated plainly: this cuts snapshot COUNT drastically
# (~6 per ticker over 6 years instead of ~250) in exchange for each
# snapshot actually meaning something distinct. Fewer, real
# observations beat many correlated copies for a model that's
# learning "what does normal look like." If you deliberately want a
# denser, more autocorrelated training set (e.g. for a quick local
# experiment), pass a smaller --step-days -- just know you're
# reintroducing the correlation this default avoids.
NONOVERLAP_LOOKBACK_DAYS = 252
DEFAULT_SNAPSHOT_STEP_DAYS = NONOVERLAP_LOOKBACK_DAYS

MIN_HISTORY_DAYS = 250           # need >=200 for MA ratio, plus buffer
DEFAULT_CONTAMINATION = 0.05
MAX_ACCEPTABLE_SKIP_FRACTION = 0.20  # warn loudly if we lose >20% of tickers


def build_training_rows(years: int = DEFAULT_YEARS_OF_HISTORY,
                         step_days: int = DEFAULT_SNAPSHOT_STEP_DAYS,
                         tickers: list = None) -> list:
    """
    Download history for the ticker universe and slide a window through
    each series to build a table of feature snapshots.

    Parameters:
        years     : int  -- Years of history to pull per ticker
        step_days : int  -- Trading days between snapshots
        tickers   : list -- Ticker universe to use (defaults to TICKER_UNIVERSE)

    Returns:
        list of dict -- each dict is one feature snapshot (one row).
    """
    tickers = tickers if tickers is not None else TICKER_UNIVERSE
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=365 * years)).strftime('%Y-%m-%d')

    if step_days < NONOVERLAP_LOOKBACK_DAYS:
        logger.warning(
            f"step_days={step_days} is below the {NONOVERLAP_LOOKBACK_DAYS}-day "
            "max feature lookback -- snapshots for the same ticker will share "
            "overlapping price windows and won't be statistically independent. "
            "Fine for a quick experiment, not recommended for the model you "
            "actually rely on."
        )

    rows = []
    skipped = 0
    for ticker in tickers:
        data = load_stock_data(ticker, start_date, end_date)
        if data is None or len(data) < MIN_HISTORY_DAYS:
            logger.warning(f"Skipping {ticker} -- insufficient history.")
            skipped += 1
            continue

        count_for_ticker = 0
        for i in range(MIN_HISTORY_DAYS, len(data), step_days):
            snapshot = data.iloc[:i + 1]
            features = compute_feature_vector(snapshot)
            if features is not None:
                rows.append(features)
                count_for_ticker += 1

        logger.info(f"{ticker}: {count_for_ticker} snapshots")

    skip_fraction = skipped / len(tickers) if tickers else 0
    if skip_fraction > MAX_ACCEPTABLE_SKIP_FRACTION:
        logger.warning(
            f"{skipped}/{len(tickers)} tickers were skipped ({skip_fraction:.0%}) -- "
            "likely rate-limiting or network issues. The trained model may be "
            "based on a much smaller, less diverse universe than intended. "
            "Consider re-running later or reducing request volume."
        )

    return rows


def train_and_save(rows: list,
                    contamination: float = DEFAULT_CONTAMINATION,
                    ticker_universe_size: int = None) -> dict:
    """
    Fit the scaler + Isolation Forest on the collected feature rows
    and save everything the inference code needs to bubble_model.pkl.

    Returns:
        The saved bundle dict (also written to MODEL_PATH).
    """
    if len(rows) < 200:
        raise RuntimeError(
            f"Only collected {len(rows)} training rows -- too few to train "
            "a reliable model. Check your network access and ticker list."
        )

    ticker_universe_size = (
        ticker_universe_size if ticker_universe_size is not None else len(TICKER_UNIVERSE)
    )

    x = np.array([[row[name] for name in FEATURE_NAMES] for row in rows])

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    model = IsolationForest(
        n_estimators=300,
        contamination=contamination,
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
        'ticker_universe_size': ticker_universe_size,
        'contamination': contamination,
    }

    joblib.dump(bundle, MODEL_PATH)
    logger.info(f"Saved model to {MODEL_PATH}")
    logger.info(f"Trained on {len(rows)} snapshots across {ticker_universe_size} tickers.")
    return bundle


def validate_against_known_cases(years_buffer: int = 2) -> None:
    """
    Sanity-check the freshly trained model against a handful of dates
    with well-known outcomes (see KNOWN_TEST_CASES). This is NOT a
    rigorous backtest -- four data points can't validate a model -- it's
    a smoke test to catch a badly broken model (inverted gate, mismatched
    scaler, garbage scores) before you trust it.

    Prints a pass/fail table. Does not raise on mismatch -- these labels
    are fuzzy by nature, so a mismatch is a prompt to look closer, not
    proof the model is wrong.
    """
    reload_model()  # make sure we're validating the model we just trained

    logger.info("Validating model against known historical cases...")
    print(f"\n{'Ticker':<8}{'As-of Date':<14}{'Expected':<14}{'Score':<8}{'Risk Level':<24}{'Match?'}")
    print("-" * 85)

    for ticker, as_of, expected, note in KNOWN_TEST_CASES:
        as_of_dt = datetime.strptime(as_of, '%Y-%m-%d')
        start_date = (as_of_dt - timedelta(days=365 * years_buffer)).strftime('%Y-%m-%d')
        end_date = as_of

        data = load_stock_data(ticker, start_date, end_date)
        if data is None or len(data) < MIN_HISTORY_DAYS:
            print(f"{ticker:<8}{as_of:<14}{expected:<14}{'N/A':<8}{'insufficient data':<24}skip")
            continue

        result = run_bubble_analysis(ticker, data)
        if result['bubble_score'] is None:
            print(f"{ticker:<8}{as_of:<14}{expected:<14}{'N/A':<8}{result['risk_level']:<24}skip")
            continue

        predicted = 'overheated' if result['bubble_score'] >= 40 else 'calm'
        match = 'YES' if predicted == expected else 'NO -- review'
        print(f"{ticker:<8}{as_of:<14}{expected:<14}{result['bubble_score']:<8}"
              f"{result['risk_level']:<24}{match}")

    print(
        "\nNote: mismatches aren't necessarily bugs -- these are rough "
        "expectations for known periods, not ground truth. Use them as a "
        "prompt to inspect z-scores and the directional gate, not as a "
        "pass/fail gate on deploying the model.\n"
    )


def main():
    parser = argparse.ArgumentParser(description="Train the bubble-detection Isolation Forest model.")
    parser.add_argument('--years', type=int, default=DEFAULT_YEARS_OF_HISTORY,
                         help=f"Years of history per ticker (default {DEFAULT_YEARS_OF_HISTORY})")
    parser.add_argument('--step-days', type=int, default=DEFAULT_SNAPSHOT_STEP_DAYS,
                         help=(f"Trading days between snapshots (default "
                               f"{DEFAULT_SNAPSHOT_STEP_DAYS}, matching the longest "
                               f"feature lookback so snapshots don't overlap -- values "
                               f"below {NONOVERLAP_LOOKBACK_DAYS} trade independence for "
                               f"snapshot count)"))
    parser.add_argument('--contamination', type=float, default=DEFAULT_CONTAMINATION,
                         help=f"Expected anomaly fraction for Isolation Forest (default {DEFAULT_CONTAMINATION})")
    parser.add_argument('--skip-validation', action='store_true',
                         help="Skip the known-case sanity check after training")
    args = parser.parse_args()

    logger.info("=" * 55)
    logger.info("TRAINING BUBBLE DETECTION MODEL")
    logger.info("=" * 55)
    logger.info(
        f"Universe: {len(TICKER_UNIVERSE)} tickers, {args.years} years each, "
        f"snapshot every {args.step_days} days, contamination={args.contamination}"
    )

    rows = build_training_rows(years=args.years, step_days=args.step_days)
    train_and_save(rows, contamination=args.contamination)

    if not args.skip_validation:
        validate_against_known_cases()


if __name__ == "__main__":
    main()
