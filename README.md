# Stock Direction Predictor

A desktop app that trains a machine-learning model to guess whether a stock's
closing price will be UP or DOWN at a chosen horizon — tomorrow, next week, 2
weeks, or a month out.

**Educational / research project only — not financial advice.** Short-term
stock direction is close to a random walk; a backtested accuracy of
~52-58% on a binary up/down call is a realistic, unimpressive-sounding
result for this kind of model. Treat all outputs skeptically, and never
trade real money based solely on this tool.

## What it does

- Downloads historical price data (via [yfinance](https://github.com/ranaroussi/yfinance)) for your watchlist plus a broad training pool of large-cap tickers, and the S&P 500 (SPY) as a market benchmark.
- Engineers technical-indicator features per stock: moving averages, RSI, MACD, volatility, volume trends, and how the stock is moving *relative to the market*.
- Trains one shared [`HistGradientBoostingClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html) per horizon on data **pooled across every ticker** (not one model per stock) — far more training examples than any single stock has on its own.
- Reports test accuracy against a naive "always guess the majority class" baseline, so you can tell whether the model is actually adding value or just riding a stock's upward drift.
- Shows a live next-move prediction with a confidence score, a price chart, and a permutation-importance chart of what the model is actually paying attention to.

## Files

- `stock_predictor.py` — the data-fetching, feature-engineering, and model-training engine. Also runnable directly as a CLI (`python stock_predictor.py`).
- `stock_predictor_gui.py` — a Tkinter desktop app: manage a watchlist, pick a prediction horizon, train, and browse results.

## Setup

```bash
pip install -r requirements.txt
python stock_predictor_gui.py
```

Edit `TICKERS` near the top of `stock_predictor.py` to change the default watchlist, or add/remove tickers from within the app (saved to a local `watchlist.json`, not committed to this repo).

## How it's trained

- **Pooled training**: one shared model per horizon is trained on your watchlist plus `TRAINING_POOL_TICKERS`, a fixed set of large, liquid, sector-diverse stocks — giving the model far more examples of the same patterns than a single ticker's history could provide.
- **Market-relative features**: each stock's move is compared against SPY, separating "this stock did something" from "the whole market did something."
- **Sharpened labels**: moves smaller than a small dead-zone threshold (scaled by horizon) are treated as ambiguous noise and dropped from training, instead of forcing a coin-flip label on a nearly-flat day.
- **Time-ordered train/test split**: no shuffling, so there's no look-ahead bias — the model is always evaluated on data chronologically after what it trained on.
