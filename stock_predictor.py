"""
Stock Direction Prediction Bot
==============================
Trains a machine-learning model to predict whether a stock's close will be
UP or DOWN at a chosen horizon (tomorrow / next week / 2 weeks / a month
out), using technical-indicator features built from historical price/volume
data, PLUS how each stock is moving relative to the broad market (SPY).

Rather than training a separate model per ticker, one shared model per
horizon is trained on data POOLED across TICKERS plus a broader
TRAINING_POOL_TICKERS universe — far more training examples than any single
stock (or just your watchlist) has on its own, so it learns general up/down
patterns instead of overfitting to one stock's quirks. Every horizon reuses
the same fetched price history, so adding horizons doesn't add network calls.

IMPORTANT / DISCLAIMER
-----------------------
- This is an educational / research tool, NOT financial advice.
- Public stock prices are close to a "random walk" — short-term direction is
  genuinely hard to predict, and a backtested accuracy of ~52-56% on a
  binary up/down call is actually a fairly typical (unimpressive-sounding
  but realistic) result for this kind of model. Treat outputs skeptically.
- Past performance of the model on historical data does NOT guarantee
  future results. Never trade real money based solely on this script.

Requirements
------------
pip install yfinance scikit-learn pandas numpy matplotlib

Usage
-----
python stock_predictor.py
(edit the TICKERS list below to change which stocks are analyzed)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, brier_score_loss

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN"]   # edit this list freely — your "watchlist"
LOOKBACK_PERIOD = "5y"                        # how much history to pull
TEST_FRACTION = 0.2                           # last 20% of history = test set
RANDOM_STATE = 42
MARKET_TICKER = "SPY"                         # broad-market benchmark for relative features
DEAD_ZONE_PCT = 0.0015                        # next-day moves smaller than this (0.15%) are
                                               # treated as noise and dropped from training —
                                               # scaled up for longer horizons, see label_target()

# Extra tickers pulled in ONLY to give the shared model more training examples —
# a broad, sector-diverse set of large, liquid stocks. They're fetched and used for
# training/context even if they're not on your personal watchlist.
TRAINING_POOL_TICKERS = [
    "META", "NVDA", "TSLA", "JPM", "BAC", "WFC", "V", "MA",
    "JNJ", "PFE", "UNH", "XOM", "CVX", "PG", "KO", "PEP",
    "WMT", "HD", "DIS", "NFLX", "ADBE", "CRM", "INTC", "CSCO", "VZ",
]

# Prediction horizons offered by the app, in trading days.
HORIZONS = {
    "Tomorrow": 1,
    "Next Week": 5,
    "2 Weeks": 10,
    "1 Month": 21,
}


# --------------------------------------------------------------------------
# 1. DATA FETCHING
# --------------------------------------------------------------------------
def fetch_price_history(ticker: str, period: str = LOOKBACK_PERIOD) -> pd.DataFrame:
    """Download historical OHLCV data for a ticker using yfinance."""
    import yfinance as yf
    df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}. Check the ticker symbol.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.title)
    return df


def fetch_market_context(period: str = LOOKBACK_PERIOD) -> pd.DataFrame:
    """Download the broad-market benchmark used to compute how much of a
    stock's daily move is market-wide noise vs. stock-specific signal."""
    market_raw = fetch_price_history(MARKET_TICKER, period=period)
    market_close = market_raw["Close"]
    market = pd.DataFrame(index=market_raw.index)
    market["Market_Return_1d"] = market_close.pct_change()
    market["Market_Return_5d"] = market_close.pct_change(5)
    market_sma20 = market_close.rolling(20).mean()
    market["Market_Price_vs_SMA20"] = (market_close - market_sma20) / market_sma20
    return market


# --------------------------------------------------------------------------
# 2. FEATURE ENGINEERING
# --------------------------------------------------------------------------
def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def add_features(df: pd.DataFrame, market: pd.DataFrame = None) -> pd.DataFrame:
    """Build technical-indicator + market-relative features from raw OHLCV data.

    `market` is the output of fetch_market_context() — pass the SAME one in
    for every ticker in a batch so it's only downloaded once.
    """
    out = df.copy()
    close = out["Close"]

    out["Return_1d"] = close.pct_change()
    out["Return_5d"] = close.pct_change(5)
    out["Return_10d"] = close.pct_change(10)

    out["SMA_5"] = close.rolling(5).mean()
    out["SMA_20"] = close.rolling(20).mean()
    out["SMA_50"] = close.rolling(50).mean()
    out["SMA5_vs_SMA20"] = (out["SMA_5"] - out["SMA_20"]) / out["SMA_20"]
    out["Price_vs_SMA20"] = (close - out["SMA_20"]) / out["SMA_20"]

    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    # Expressed as % of price (not raw dollars) so these are comparable across
    # tickers at very different share prices — needed since a shared model is
    # trained on multiple tickers pooled together.
    out["MACD"] = macd / close
    out["MACD_signal"] = macd_signal / close
    out["MACD_hist"] = (macd - macd_signal) / close

    out["RSI_14"] = compute_rsi(close, 14)
    out["Volatility_10d"] = out["Return_1d"].rolling(10).std()
    out["Volume_change"] = out["Volume"].pct_change()
    out["Volume_avg_ratio"] = out["Volume"] / out["Volume"].rolling(20).mean()

    high_low = out["High"] - out["Low"]
    out["Daily_range_pct"] = high_low / close

    # Market-relative features: how much of this move is the stock itself,
    # vs. the whole market moving the same way that day.
    if market is not None:
        aligned_market = market.reindex(out.index).ffill()
        out["Relative_Return_1d"] = out["Return_1d"] - aligned_market["Market_Return_1d"]
        out["Relative_Return_5d"] = out["Return_5d"] - aligned_market["Market_Return_5d"]
        out["Relative_Strength"] = out["Price_vs_SMA20"] - aligned_market["Market_Price_vs_SMA20"]
    else:
        out["Relative_Return_1d"] = 0.0
        out["Relative_Return_5d"] = 0.0
        out["Relative_Strength"] = 0.0

    return out


def label_target(featured_df: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    """Add/overwrite the Target column for a given prediction horizon (in
    trading days from today): 1 if the close `horizon_days` out rises by more
    than the dead zone, 0 if it falls by more than the dead zone. Smaller
    moves are ambiguous noise and left as NaN so they get dropped from
    training/testing — this sharpens the label instead of forcing a
    coin-flip call on a nearly-flat move.

    The dead zone scales with sqrt(horizon_days) — the standard random-walk
    volatility scaling — since a "flat" move over a month is naturally much
    bigger in absolute terms than a "flat" move over a single day.
    """
    out = featured_df.copy()
    close = out["Close"]
    dead_zone = DEAD_ZONE_PCT * (horizon_days ** 0.5)

    future_close = close.shift(-horizon_days)
    future_return = (future_close - close) / close
    out["Target"] = np.nan
    out.loc[future_return > dead_zone, "Target"] = 1.0
    out.loc[future_return < -dead_zone, "Target"] = 0.0

    return out


FEATURE_COLUMNS = [
    "Return_1d", "Return_5d", "Return_10d",
    "SMA5_vs_SMA20", "Price_vs_SMA20",
    "MACD", "MACD_signal", "MACD_hist",
    "RSI_14", "Volatility_10d",
    "Volume_change", "Volume_avg_ratio",
    "Daily_range_pct",
    "Relative_Return_1d", "Relative_Return_5d", "Relative_Strength",
]

# Human-readable labels for FEATURE_COLUMNS, used anywhere these are shown to a user.
FEATURE_DISPLAY_NAMES = {
    "Return_1d": "1-Day Price Change",
    "Return_5d": "5-Day Price Change",
    "Return_10d": "10-Day Price Change",
    "SMA5_vs_SMA20": "Short vs Medium-Term Trend",
    "Price_vs_SMA20": "Price vs 20-Day Average",
    "MACD": "Momentum Trend (MACD)",
    "MACD_signal": "Momentum Trend Signal",
    "MACD_hist": "Momentum Trend Strength",
    "RSI_14": "Overbought / Oversold Level (RSI)",
    "Volatility_10d": "Recent Volatility",
    "Volume_change": "Trading Volume Change",
    "Volume_avg_ratio": "Trading Volume vs Average",
    "Daily_range_pct": "Daily Price Swing %",
    "Relative_Return_1d": "1-Day Move vs. Market",
    "Relative_Return_5d": "5-Day Move vs. Market",
    "Relative_Strength": "Trend Strength vs. Market",
}


# --------------------------------------------------------------------------
# 3. DATASET PREPARATION
# --------------------------------------------------------------------------
def prepare_dataset(featured_df: pd.DataFrame):
    """Split into feature rows, their up/down labels (for whatever horizon
    label_target() was called with), and the latest row (today's features,
    used for the live prediction — always kept regardless of whether it has
    a settled label yet, since it's the most recent trading day)."""
    usable_rows = featured_df.dropna(subset=FEATURE_COLUMNS).copy()
    if usable_rows.empty:
        return usable_rows[FEATURE_COLUMNS], usable_rows.get("Target", pd.Series(dtype=int)), None

    today_row = usable_rows.iloc[[-1]]
    historical_rows = usable_rows.iloc[:-1].dropna(subset=["Target"])

    features = historical_rows[FEATURE_COLUMNS]
    labels = historical_rows["Target"].astype(int)
    features_today = today_row[FEATURE_COLUMNS]
    return features, labels, features_today


# --------------------------------------------------------------------------
# 4. MODEL TRAINING (pooled across tickers) / EVALUATION
# --------------------------------------------------------------------------
def make_classifier() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_depth=4,
        max_iter=300,
        learning_rate=0.05,
        min_samples_leaf=20,
        l2_regularization=0.1,
        random_state=RANDOM_STATE,
    )


def _calibration_check(classifier, features_test: pd.DataFrame, labels_test: pd.Series, predicted_labels) -> dict:
    """Check whether the model's confidence is trustworthy: when it says
    it's 70% confident, is it actually right about 70% of the time?

    Confidence here means P(the class the model actually called) — i.e.
    whichever is bigger, P(up) or P(down), matching what the app displays
    as "confidence in this call". Bucketed into equal-sized quantile bins
    (since confidence tends to cluster in a narrow band, e.g. 50-65%, for
    this kind of weak-signal problem — fixed-width bins would leave most
    of them empty) and compared against the bucket's actual accuracy.

    Brier score is a single-number summary of the same idea: mean squared
    error between predicted P(up) and the actual 0/1 outcome — 0 is
    perfect, 0.25 is what a constant 50% guess scores, higher is worse.
    """
    proba_up = classifier.predict_proba(features_test)[:, 1]
    confidence = np.maximum(proba_up, 1 - proba_up)
    correct = (np.asarray(predicted_labels) == labels_test.to_numpy()).astype(int)
    brier = brier_score_loss(labels_test, proba_up)

    n_bins = 8
    try:
        bin_true, bin_pred = calibration_curve(correct, confidence, n_bins=n_bins, strategy="quantile")
    except ValueError:
        bin_true, bin_pred = np.array([]), np.array([])

    return {
        "bin_pred": bin_pred.tolist(),
        "bin_true": bin_true.tolist(),
        "brier_score": float(brier),
        "n_test": int(len(labels_test)),
    }


def _train_one_horizon(featured_by_ticker: dict, horizon_days: int, base_skipped: list) -> dict:
    """Label + split + pool + train + evaluate for a single horizon, reusing
    already-fetched/feature-engineered data for every ticker."""
    per_ticker = {}
    train_features_parts, train_labels_parts = [], []
    test_features_parts, test_labels_parts = [], []
    skipped = list(base_skipped)

    for ticker, featured in featured_by_ticker.items():
        labeled = label_target(featured, horizon_days)
        features, labels, features_today = prepare_dataset(labeled)

        if features_today is None or len(features) < 100:
            skipped.append((ticker, "not enough history after feature engineering"))
            continue

        split_index = int(len(features) * (1 - TEST_FRACTION))
        features_train, features_test = features.iloc[:split_index], features.iloc[split_index:]
        labels_train, labels_test = labels.iloc[:split_index], labels.iloc[split_index:]

        if len(features_train) == 0 or len(features_test) == 0:
            skipped.append((ticker, "not enough history to form a train/test split"))
            continue

        train_features_parts.append(features_train)
        train_labels_parts.append(labels_train)
        test_features_parts.append(features_test)
        test_labels_parts.append(labels_test)

        per_ticker[ticker] = {
            "featured": labeled,
            "features_test": features_test,
            "labels_test": labels_test,
            "features_today": features_today,
        }

    if not train_features_parts:
        return {"horizon_days": horizon_days, "error": "No tickers had enough usable history to train on.",
                "skipped": skipped}

    pooled_features_train = pd.concat(train_features_parts, ignore_index=True)
    pooled_labels_train = pd.concat(train_labels_parts, ignore_index=True)
    pooled_features_test = pd.concat(test_features_parts, ignore_index=True)
    pooled_labels_test = pd.concat(test_labels_parts, ignore_index=True)

    classifier = make_classifier()
    classifier.fit(pooled_features_train, pooled_labels_train)

    overall_predicted = classifier.predict(pooled_features_test)
    overall_accuracy = accuracy_score(pooled_labels_test, overall_predicted)
    overall_baseline_accuracy = max(pooled_labels_test.mean(), 1 - pooled_labels_test.mean())

    importance = permutation_importance(
        classifier, pooled_features_test, pooled_labels_test,
        n_repeats=10, random_state=RANDOM_STATE,
    )
    feature_importances = dict(zip(FEATURE_COLUMNS, importance.importances_mean))

    calibration = _calibration_check(classifier, pooled_features_test, pooled_labels_test, overall_predicted)

    results = {}
    for ticker, data in per_ticker.items():
        labels_test = data["labels_test"]
        predicted_labels = classifier.predict(data["features_test"])
        accuracy = accuracy_score(labels_test, predicted_labels)
        baseline_accuracy = max(labels_test.mean(), 1 - labels_test.mean())

        proba = classifier.predict_proba(data["features_today"])[0]  # [P(down), P(up)]
        prob_up = proba[1]

        results[ticker] = {
            "ticker": ticker,
            "featured": data["featured"],
            "accuracy": accuracy,
            "baseline_accuracy": baseline_accuracy,
            "prediction": "UP" if prob_up >= 0.5 else "DOWN",
            "prob_up": prob_up,
            "latest_date": data["features_today"].index[-1],
        }

    return {
        "horizon_days": horizon_days,
        "classifier": classifier,
        "overall_accuracy": overall_accuracy,
        "overall_baseline_accuracy": overall_baseline_accuracy,
        "feature_importances": feature_importances,
        "calibration": calibration,
        "per_ticker": results,
        "skipped": skipped,
    }


def train_multi_horizon_model(tickers: list, horizons: dict = HORIZONS) -> dict:
    """Fetch + feature-engineer each ticker ONCE (the expensive network part),
    then train one shared classifier PER horizon on top of that same data —
    so switching horizons in the UI is instant, no re-fetching required.

    Returns {horizon_label: {classifier, overall_accuracy, overall_baseline_accuracy,
    feature_importances, per_ticker, skipped}} — see _train_one_horizon().
    """
    market = fetch_market_context()

    featured_by_ticker = {}
    base_skipped = []
    for ticker in tickers:
        try:
            raw = fetch_price_history(ticker)
            featured_by_ticker[ticker] = add_features(raw, market)
        except Exception as e:
            base_skipped.append((ticker, str(e)))

    if not featured_by_ticker:
        raise ValueError("None of the requested tickers had usable price history.")

    return {
        horizon_label: _train_one_horizon(featured_by_ticker, horizon_days, base_skipped)
        for horizon_label, horizon_days in horizons.items()
    }


# --------------------------------------------------------------------------
# 5. DASHBOARD VISUALIZATION
# --------------------------------------------------------------------------
def build_dashboard(results: list, feature_importances: dict, out_path: str = "prediction_dashboard.png"):
    results = [r for r in results if r is not None]
    n = len(results)
    if n == 0:
        print("No results to plot.")
        return

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(n, 2, figsize=(13, 4 * n))
    if n == 1:
        axes = axes.reshape(1, 2)

    importances = pd.Series(feature_importances)
    importances.index = [FEATURE_DISPLAY_NAMES[c] for c in importances.index]
    importances = importances.sort_values(ascending=True)

    for i, r in enumerate(results):
        ticker = r["ticker"]

        # --- Left panel: price history with SMA20 ---
        ax = axes[i, 0]
        hist = r["featured"].tail(180)
        ax.plot(hist.index, hist["Close"], label="Close", linewidth=1.6, color="#4C72B0")
        ax.plot(hist.index, hist["SMA_20"], label="SMA 20", linewidth=1.2,
                color="#DD8452", linestyle="--")
        ax.set_title(f"{ticker} — last 180 trading days", fontweight="bold")
        ax.set_ylabel("Price ($)")
        ax.legend(loc="upper left", fontsize=8, frameon=True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.autofmt_xdate()

        # --- Right panel: feature importance (shared model, same for every ticker) ---
        ax2 = axes[i, 1]
        ax2.barh(importances.index, importances.values, color="#4C72B0")
        pred_txt = (f"Next-day prediction: {r['prediction']}  "
                    f"(P(up)={r.get('prob_up', float('nan')):.0%})") if r["prediction"] else "No live prediction"
        acc_txt = f"Test accuracy: {r['accuracy']:.1%} (baseline {r['baseline_accuracy']:.1%})"
        ax2.set_title(f"{ticker} — shared-model feature importance\n{pred_txt}\n{acc_txt}", fontsize=10)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

    fig.suptitle(
        "Stock Direction Predictions — Educational Model, NOT Financial Advice",
        fontsize=14, fontweight="bold", y=1.0 + 0.01 * n,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nDashboard saved to: {out_path}")
    plt.show()


def print_summary_table(results: list):
    results = [r for r in results if r is not None and r["prediction"] is not None]
    if not results:
        return
    print(f"\n{'=' * 60}\nSUMMARY\n{'=' * 60}")
    print(f"{'Ticker':<8}{'Prediction':<12}{'P(up)':<10}{'Test Acc':<10}{'Baseline':<10}")
    for r in results:
        print(f"{r['ticker']:<8}{r['prediction']:<12}{r['prob_up']:<10.1%}"
              f"{r['accuracy']:<10.1%}{r['baseline_accuracy']:<10.1%}")
    print(
        "\nReminder: these are short-term directional guesses from a model trained\n"
        "on historical technical indicators. They are frequently wrong. Do not use\n"
        "this as your sole basis for a real trading decision."
    )


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
def main():
    all_tickers = sorted(set(TICKERS) | set(TRAINING_POOL_TICKERS))
    by_horizon = train_multi_horizon_model(all_tickers, horizons={"Tomorrow": 1})
    pooled = by_horizon["Tomorrow"]

    for ticker, error in pooled["skipped"]:
        print(f"  Skipping {ticker}: {error}")

    results = []
    for ticker in TICKERS:
        r = pooled["per_ticker"].get(ticker)
        if r is None:
            results.append(None)
            continue
        print(f"\n{'=' * 60}\n{ticker}\n{'=' * 60}")
        print(f"  Test accuracy:      {r['accuracy']:.1%}")
        print(f"  Naive baseline:     {r['baseline_accuracy']:.1%}  (always guessing the majority class)")
        print(f"  {'Model beats baseline' if r['accuracy'] > r['baseline_accuracy'] else 'Model does NOT beat baseline'}")
        print(f"  Latest prediction ({r['latest_date'].date()}): "
              f"{r['prediction']}  (P(up) = {r['prob_up']:.1%})")
        results.append(r)

    print(
        f"\nShared model trained on {len(all_tickers)} tickers ({len(TICKERS)} watchlist + "
        f"{len(TRAINING_POOL_TICKERS)} training-pool) — pooled test accuracy: "
        f"{pooled['overall_accuracy']:.1%} (baseline {pooled['overall_baseline_accuracy']:.1%})"
    )

    print_summary_table(results)
    build_dashboard(results, pooled["feature_importances"])


if __name__ == "__main__":
    main()
