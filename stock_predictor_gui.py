"""
Stock Direction Predictor — Desktop App
========================================
Tkinter GUI wrapper around stock_predictor.py's data-fetch + ML pipeline.

- Maintain a watchlist of tickers, add/remove freely. It's saved to
  watchlist.json next to this script, so it survives closing the app.
- Click "Train & Predict All" to fetch history for your watchlist PLUS a
  broad training-pool of large-cap tickers (more examples for the shared
  model to learn from), then train one shared classifier PER horizon
  (Tomorrow / Next Week / 2 Weeks / 1 Month) — all reusing the same fetched
  data, so no extra network calls per horizon.
- Pick a horizon at the top of the prediction card to instantly switch
  which one is displayed (already cached, no retraining needed). Click any
  ticker to see its big, color-coded UP / DOWN call for that horizon, plus
  supporting charts.
- Fetching + training runs on a background thread so the window stays
  responsive; results are cached until you retrain.

IMPORTANT: Educational tool only. NOT financial advice. See
stock_predictor.py's module docstring for the full disclaimer.

Requirements
------------
pip install yfinance scikit-learn pandas numpy matplotlib

Usage
-----
python stock_predictor_gui.py
"""

import json
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk, messagebox

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.dates as mdates
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from stock_predictor import (
    TICKERS,
    TRAINING_POOL_TICKERS,
    MARKET_TICKER,
    HORIZONS,
    FEATURE_COLUMNS,
    FEATURE_DISPLAY_NAMES,
    train_multi_horizon_model,
)

# --------------------------------------------------------------------------
# COLOR PALETTE (validated categorical / status colors — see dataviz skill)
# --------------------------------------------------------------------------
COLORS = {
    "page_bg":        "#f9f9f7",
    "card_bg":        "#fcfcfb",
    "border":         "#e1e0d9",
    "text_primary":   "#0b0b0b",
    "text_secondary": "#52514e",
    "text_muted":     "#898781",
    "grid":           "#e1e0d9",
    "axis":           "#c3c2b7",

    "header_bg":      "#184f95",
    "header_text":    "#ffffff",
    "header_subtext": "#cde2fb",

    "series_price":   "#2a78d6",   # categorical slot 1 — blue
    "series_avg":     "#eb6834",   # categorical slot 2 — orange

    "accent":         "#2a78d6",
    "accent_dark":    "#184f95",

    "up_bg":          "#e5f7e6",
    "up_ink":         "#0f5c0f",
    "up_accent":      "#0ca30c",
    "up_trough":      "#cdeccd",

    "down_bg":        "#fbeaea",
    "down_ink":       "#8a2323",
    "down_accent":    "#d03b3b",
    "down_trough":    "#f5d0d0",

    "neutral_ink":    "#898781",
}

# Sequential blue ramp (light -> dark) for magnitude-ranked bars
IMPORTANCE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]

# Watchlist is saved next to this script, so it survives closing and reopening the app.
WATCHLIST_FILE = Path(__file__).resolve().parent / "watchlist.json"


def load_watchlist() -> list:
    try:
        saved = json.loads(WATCHLIST_FILE.read_text(encoding="utf-8"))
        if isinstance(saved, list) and all(isinstance(t, str) for t in saved):
            return saved
    except (OSError, ValueError):
        pass
    return list(TICKERS)


def save_watchlist(tickers: list):
    try:
        WATCHLIST_FILE.write_text(json.dumps(tickers, indent=2), encoding="utf-8")
    except OSError:
        pass  # non-critical — worst case the watchlist just doesn't persist


class StockPredictorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Stock Direction Predictor")
        self.geometry("1180x760")
        self.minsize(980, 640)
        self.configure(bg=COLORS["page_bg"])

        self.results_by_horizon = {}  # horizon label -> {per_ticker, skipped, overall_*, feature_importances}
        self.work_queue = queue.Queue()
        self.busy = False

        self.tickers = load_watchlist()   # source of truth; the Listbox is just its display
        self.horizon_var = tk.StringVar(value=next(iter(HORIZONS)))  # "Tomorrow"
        self.chart_view_var = tk.StringVar(value="Feature Importance")

        self._build_styles()
        self._build_layout()
        self._poll_queue()

        self._refresh_watchlist_display()
        if self.tickers:
            self.watchlist.selection_set(0)

    # ------------------------------------------------------------------
    # Per-horizon data helpers
    # ------------------------------------------------------------------
    def _horizon_data(self):
        return self.results_by_horizon.get(self.horizon_var.get())

    def _per_ticker_results(self):
        data = self._horizon_data()
        return data["per_ticker"] if data and "error" not in data else {}

    def _skipped_reasons(self):
        data = self._horizon_data()
        return dict(data["skipped"]) if data else {}

    # ------------------------------------------------------------------
    # Styling
    # ------------------------------------------------------------------
    def _build_styles(self):
        style = ttk.Style(self)
        style.theme_use("clam")

        style.configure("TFrame", background=COLORS["page_bg"])
        style.configure("Card.TFrame", background=COLORS["card_bg"])

        style.configure("TLabel", background=COLORS["page_bg"],
                         foreground=COLORS["text_primary"], font=("Segoe UI", 10))
        style.configure("Card.TLabel", background=COLORS["card_bg"],
                         foreground=COLORS["text_primary"], font=("Segoe UI", 10))
        style.configure("SectionTitle.TLabel", background=COLORS["card_bg"],
                         foreground=COLORS["text_primary"], font=("Segoe UI", 12, "bold"))
        style.configure("Muted.Card.TLabel", background=COLORS["card_bg"],
                         foreground=COLORS["text_muted"], font=("Segoe UI", 9))
        style.configure("Header.TLabel", background=COLORS["header_bg"],
                         foreground=COLORS["header_text"], font=("Segoe UI", 18, "bold"))
        style.configure("SubHeader.TLabel", background=COLORS["header_bg"],
                         foreground=COLORS["header_subtext"], font=("Segoe UI", 9, "italic"))
        style.configure("StatValue.Card.TLabel", background=COLORS["card_bg"],
                         foreground=COLORS["text_primary"], font=("Segoe UI", 20, "bold"))
        style.configure("StatLabel.Card.TLabel", background=COLORS["card_bg"],
                         foreground=COLORS["text_secondary"], font=("Segoe UI", 9))

        style.configure("Accent.TButton", background=COLORS["accent"], foreground="white",
                         font=("Segoe UI", 10, "bold"), padding=8, borderwidth=0)
        style.map("Accent.TButton",
                  background=[("active", COLORS["accent_dark"]), ("disabled", COLORS["text_muted"])])

        style.configure("Ghost.TButton", background=COLORS["card_bg"], foreground=COLORS["accent"],
                         font=("Segoe UI", 9), padding=6, borderwidth=1, relief="solid")
        style.map("Ghost.TButton", background=[("active", COLORS["border"])])

        style.configure("Danger.TButton", background=COLORS["card_bg"], foreground=COLORS["down_accent"],
                         font=("Segoe UI", 9), padding=6, borderwidth=1, relief="solid")
        style.map("Danger.TButton", background=[("active", COLORS["down_bg"])])

        # Horizon picker — radiobuttons styled as a flat segmented control.
        style.configure("Horizon.Toolbutton", background=COLORS["card_bg"], foreground=COLORS["text_secondary"],
                         font=("Segoe UI", 9, "bold"), padding=(12, 6), borderwidth=1, relief="solid")
        style.map("Horizon.Toolbutton",
                  background=[("selected", COLORS["accent"]), ("active", COLORS["border"])],
                  foreground=[("selected", "white")])

        style.configure("Up.Horizontal.TProgressbar", troughcolor=COLORS["up_trough"],
                         background=COLORS["up_accent"], bordercolor=COLORS["up_trough"],
                         lightcolor=COLORS["up_accent"], darkcolor=COLORS["up_accent"])
        style.configure("Down.Horizontal.TProgressbar", troughcolor=COLORS["down_trough"],
                         background=COLORS["down_accent"], bordercolor=COLORS["down_trough"],
                         lightcolor=COLORS["down_accent"], darkcolor=COLORS["down_accent"])
        style.configure("Neutral.Horizontal.TProgressbar", troughcolor=COLORS["border"],
                         background=COLORS["text_muted"], bordercolor=COLORS["border"],
                         lightcolor=COLORS["text_muted"], darkcolor=COLORS["text_muted"])

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_layout(self):
        # ---------------- Header banner ----------------
        header = tk.Frame(self, bg=COLORS["header_bg"])
        header.pack(side=tk.TOP, fill=tk.X)
        inner = tk.Frame(header, bg=COLORS["header_bg"])
        inner.pack(fill=tk.X, padx=16, pady=10)
        ttk.Label(inner, text="📊 Stock Direction Predictor", style="Header.TLabel").pack(anchor="w")
        ttk.Label(inner, text="Educational tool only — NOT financial advice. Predictions are frequently wrong.",
                  style="SubHeader.TLabel").pack(anchor="w")

        main = ttk.Frame(self, style="TFrame")
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=14, pady=14)

        # ---------------- Left: watchlist card ----------------
        left = tk.Frame(main, bg=COLORS["card_bg"], width=240,
                         highlightbackground=COLORS["border"], highlightthickness=1)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 14))
        left.pack_propagate(False)
        left_pad = tk.Frame(left, bg=COLORS["card_bg"])
        left_pad.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        ttk.Label(left_pad, text="📋 Watchlist", style="SectionTitle.TLabel").pack(anchor="w")

        add_frame = tk.Frame(left_pad, bg=COLORS["card_bg"])
        add_frame.pack(fill=tk.X, pady=(8, 6))
        self.new_ticker_var = tk.StringVar()
        entry = ttk.Entry(add_frame, textvariable=self.new_ticker_var)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        entry.bind("<Return>", lambda e: self._add_ticker())
        ttk.Button(add_frame, text="Add", width=6, style="Accent.TButton", cursor="hand2",
                   command=self._add_ticker).pack(side=tk.LEFT, padx=(6, 0))

        list_frame = tk.Frame(left_pad, bg=COLORS["card_bg"])
        list_frame.pack(fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)
        self.watchlist = tk.Listbox(
            list_frame, yscrollcommand=scrollbar.set, exportselection=False,
            font=("Consolas", 11), activestyle="dotbox",
            bg=COLORS["card_bg"], fg=COLORS["text_primary"],
            selectbackground=COLORS["accent"], selectforeground="white",
            highlightbackground=COLORS["border"], relief=tk.FLAT, borderwidth=0,
        )
        scrollbar.config(command=self.watchlist.yview)
        self.watchlist.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.watchlist.bind("<<ListboxSelect>>", self._on_watchlist_select)
        self.watchlist.configure(cursor="hand2")

        ttk.Button(left_pad, text="Remove from Watchlist", style="Danger.TButton", cursor="hand2",
                   command=self._remove_ticker).pack(fill=tk.X, pady=(8, 0))

        self.analyze_btn = ttk.Button(left_pad, text="🔮 Train & Predict All", style="Accent.TButton",
                                       cursor="hand2", command=self._train_all)
        self.analyze_btn.pack(fill=tk.X, pady=(16, 6))
        ttk.Label(left_pad,
                  text=f"Trains on your watchlist plus a {len(TRAINING_POOL_TICKERS)}-ticker training "
                       f"pool for more examples, across all {len(HORIZONS)} horizons at once.",
                  style="Muted.Card.TLabel", wraplength=210, justify=tk.LEFT).pack(anchor="w")
        self.save_btn = ttk.Button(left_pad, text="💾 Save Report as Image", style="Ghost.TButton",
                                    cursor="hand2", command=self._save_chart)
        self.save_btn.pack(fill=tk.X, pady=(12, 0))

        # ---------------- Right: horizon picker + prediction + stats + charts ----------------
        right = ttk.Frame(main, style="TFrame")
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # --- Horizon picker ---
        horizon_row = ttk.Frame(right, style="TFrame")
        horizon_row.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(horizon_row, text="Predict for:", font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT, padx=(0, 8))
        for label in HORIZONS:
            ttk.Radiobutton(horizon_row, text=label, value=label, variable=self.horizon_var,
                             style="Horizon.Toolbutton", cursor="hand2",
                             command=self._on_horizon_change).pack(side=tk.LEFT, padx=(0, 6))

        # --- Prediction hero card ---
        self.prediction_card = tk.Frame(right, bg=COLORS["card_bg"],
                                         highlightbackground=COLORS["border"], highlightthickness=1)
        self.prediction_card.pack(fill=tk.X, pady=(0, 12))
        pred_pad = tk.Frame(self.prediction_card, bg=COLORS["card_bg"])
        pred_pad.pack(fill=tk.X, padx=18, pady=14)

        self.pred_heading_var = tk.StringVar(value="Prediction")
        self.pred_heading_lbl = tk.Label(pred_pad, textvariable=self.pred_heading_var,
                                          bg=COLORS["card_bg"], fg=COLORS["text_secondary"],
                                          font=("Segoe UI", 11, "bold"), anchor="w")
        self.pred_heading_lbl.pack(anchor="w")

        self.pred_direction_var = tk.StringVar(value="Select a ticker and click “Train & Predict All”")
        self.pred_direction_lbl = tk.Label(pred_pad, textvariable=self.pred_direction_var,
                                            bg=COLORS["card_bg"], fg=COLORS["neutral_ink"],
                                            font=("Segoe UI", 26, "bold"), anchor="w")
        self.pred_direction_lbl.pack(anchor="w", pady=(2, 6))

        conf_row = tk.Frame(pred_pad, bg=COLORS["card_bg"])
        conf_row.pack(fill=tk.X)
        self.confidence_bar = ttk.Progressbar(conf_row, style="Neutral.Horizontal.TProgressbar",
                                               length=260, maximum=100, value=0)
        self.confidence_bar.pack(side=tk.LEFT)
        self.pred_confidence_var = tk.StringVar(value="")
        tk.Label(conf_row, textvariable=self.pred_confidence_var, bg=COLORS["card_bg"],
                 fg=COLORS["text_secondary"], font=("Segoe UI", 10)).pack(side=tk.LEFT, padx=(10, 0))

        self.pred_note_var = tk.StringVar(
            value=f"The model's best guess, based on recent technical indicators and how the stock is "
                  f"moving relative to the market ({MARKET_TICKER})."
        )
        tk.Label(pred_pad, textvariable=self.pred_note_var, bg=COLORS["card_bg"],
                 fg=COLORS["text_muted"], font=("Segoe UI", 9), wraplength=700, justify=tk.LEFT
                 ).pack(anchor="w", pady=(8, 0))

        self.pooled_summary_var = tk.StringVar(value="")
        tk.Label(pred_pad, textvariable=self.pooled_summary_var, bg=COLORS["card_bg"],
                 fg=COLORS["text_muted"], font=("Segoe UI", 8, "italic"), wraplength=700, justify=tk.LEFT
                 ).pack(anchor="w", pady=(4, 0))

        # --- Stat tiles row ---
        stats_row = tk.Frame(right, bg=COLORS["page_bg"])
        stats_row.pack(fill=tk.X, pady=(0, 12))

        self.accuracy_tile = self._make_stat_tile(stats_row, "🎯 Model Accuracy")
        self.accuracy_tile["frame"].pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))
        self.baseline_tile = self._make_stat_tile(stats_row, "📐 Baseline Accuracy")
        self.baseline_tile["frame"].pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6)
        self.verdict_tile = self._make_stat_tile(stats_row, "⚖️ vs. Guessing the Majority")
        self.verdict_tile["frame"].pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6, 0))

        # --- Right-hand chart toggle ---
        chart_toggle_row = ttk.Frame(right, style="TFrame")
        chart_toggle_row.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(chart_toggle_row, text="Right chart:", font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=(0, 8))
        for label in ("Feature Importance", "Confidence Calibration"):
            ttk.Radiobutton(chart_toggle_row, text=label, value=label, variable=self.chart_view_var,
                             style="Horizon.Toolbutton", cursor="hand2",
                             command=self._on_chart_view_change).pack(side=tk.LEFT, padx=(0, 6))

        # --- Charts card ---
        charts_card = tk.Frame(right, bg=COLORS["card_bg"],
                                highlightbackground=COLORS["border"], highlightthickness=1)
        charts_card.pack(fill=tk.BOTH, expand=True)
        charts_pad = tk.Frame(charts_card, bg=COLORS["card_bg"])
        charts_pad.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        self.figure = Figure(figsize=(8, 5), dpi=100, facecolor=COLORS["card_bg"], constrained_layout=True)
        self.ax_price = self.figure.add_subplot(121)
        self.ax_importance = self.figure.add_subplot(122)
        self.canvas = FigureCanvasTkAgg(self.figure, master=charts_pad)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._style_empty_axes()

        # ---------------- Status bar ----------------
        self.status_var = tk.StringVar(value="Ready.")
        status_bar = tk.Label(self, textvariable=self.status_var, bg=COLORS["border"],
                               fg=COLORS["text_secondary"], anchor="w", font=("Segoe UI", 9), padx=10, pady=4)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def _make_stat_tile(self, parent, label_text):
        frame = tk.Frame(parent, bg=COLORS["card_bg"],
                          highlightbackground=COLORS["border"], highlightthickness=1)
        pad = tk.Frame(frame, bg=COLORS["card_bg"])
        pad.pack(fill=tk.BOTH, expand=True, padx=14, pady=10)
        value_var = tk.StringVar(value="—")
        value_lbl = tk.Label(pad, textvariable=value_var, bg=COLORS["card_bg"],
                              fg=COLORS["text_primary"], font=("Segoe UI", 20, "bold"))
        value_lbl.pack(anchor="w")
        tk.Label(pad, text=label_text, bg=COLORS["card_bg"], fg=COLORS["text_secondary"],
                 font=("Segoe UI", 9)).pack(anchor="w")
        return {"frame": frame, "value_var": value_var, "value_lbl": value_lbl}

    def _style_empty_axes(self):
        importance_title = "What Influenced This Prediction" if self.chart_view_var.get() == "Feature Importance" \
            else "Confidence Calibration"
        for ax, title in ((self.ax_price, "Price Trend"), (self.ax_importance, importance_title)):
            ax.clear()
            ax.set_facecolor(COLORS["card_bg"])
            ax.set_title(title, fontsize=10, fontweight="bold", color=COLORS["text_primary"])
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            for spine in ("left", "bottom"):
                ax.spines[spine].set_color(COLORS["axis"])
            ax.tick_params(colors=COLORS["text_muted"])
        self.canvas.draw()

    # ------------------------------------------------------------------
    # Watchlist management
    # ------------------------------------------------------------------
    def _add_ticker(self):
        raw = self.new_ticker_var.get().strip().upper()
        if not raw:
            return
        if raw in self.tickers:
            messagebox.showinfo("Already added", f"{raw} is already in the watchlist.")
            return
        self.tickers.append(raw)
        save_watchlist(self.tickers)
        self.new_ticker_var.set("")
        self._refresh_watchlist_display()
        self.watchlist.selection_clear(0, tk.END)
        self.watchlist.selection_set(tk.END)
        self.watchlist.see(tk.END)
        self._on_watchlist_select()

    def _remove_ticker(self):
        sel = self.watchlist.curselection()
        if not sel:
            return
        index = sel[0]
        self.tickers.pop(index)
        save_watchlist(self.tickers)
        self._refresh_watchlist_display()
        if self.tickers:
            self.watchlist.selection_set(min(index, len(self.tickers) - 1))
            self._on_watchlist_select()

    def _selected_ticker(self):
        sel = self.watchlist.curselection()
        if not sel or sel[0] >= len(self.tickers):
            return None
        return self.tickers[sel[0]]

    def _refresh_watchlist_display(self):
        """Redraw the watchlist with a small colored icon per ticker showing its
        last known call (for the currently selected horizon) — an at-a-glance
        summary of the whole list, not just whichever one happens to be selected."""
        per_ticker = self._per_ticker_results()
        skipped_reasons = self._skipped_reasons()

        selected = self.watchlist.curselection()
        selected_index = selected[0] if selected else None

        self.watchlist.delete(0, tk.END)
        for i, ticker in enumerate(self.tickers):
            if ticker in per_ticker:
                prediction = per_ticker[ticker]["prediction"]
                if prediction == "UP":
                    icon, color = "▲", COLORS["up_accent"]
                elif prediction == "DOWN":
                    icon, color = "▼", COLORS["down_accent"]
                else:
                    icon, color = "·", COLORS["text_muted"]
            elif ticker in skipped_reasons:
                icon, color = "!", COLORS["text_muted"]
            else:
                icon, color = "·", COLORS["text_muted"]
            self.watchlist.insert(tk.END, f"{icon}  {ticker}")
            self.watchlist.itemconfig(i, fg=color)

        if selected_index is not None and selected_index < len(self.tickers):
            self.watchlist.selection_set(selected_index)

    def _on_watchlist_select(self, event=None):
        ticker = self._selected_ticker()
        if ticker is None:
            return
        per_ticker = self._per_ticker_results()
        if ticker in per_ticker:
            self._render_result(per_ticker[ticker])
            self.status_var.set(f"Showing {ticker}'s {self.horizon_var.get()} result from the last training run.")
        else:
            self._render_placeholder(ticker, self._skipped_reasons().get(ticker))

    def _on_horizon_change(self):
        self._refresh_watchlist_display()
        self._on_watchlist_select()

    def _on_chart_view_change(self):
        self._on_watchlist_select()

    # ------------------------------------------------------------------
    # Training (threaded) — shared models pooled across watchlist + training pool,
    # one per horizon, all reusing the same fetched data
    # ------------------------------------------------------------------
    def _train_all(self):
        if self.busy:
            return
        if not self.tickers:
            messagebox.showinfo("Watchlist is empty", "Add at least one ticker to the watchlist first.")
            return

        all_tickers = sorted(set(self.tickers) | set(TRAINING_POOL_TICKERS))

        self.busy = True
        self.analyze_btn.state(["disabled"])
        self.status_var.set(
            f"Fetching history for {len(all_tickers)} ticker(s) ({len(self.tickers)} watchlist + "
            f"{len(all_tickers) - len(self.tickers)} training pool) plus {MARKET_TICKER}, then training "
            f"{len(HORIZONS)} horizon models... this can take a couple of minutes."
        )
        self.pred_heading_var.set("Training shared models...")
        self.pred_direction_var.set("⏳ Working on it...")
        self.pred_direction_lbl.configure(fg=COLORS["neutral_ink"])
        self.pred_confidence_var.set("")
        self.confidence_bar.configure(style="Neutral.Horizontal.TProgressbar", mode="indeterminate")
        self.confidence_bar.start(12)

        thread = threading.Thread(target=self._run_training, args=(all_tickers,), daemon=True)
        thread.start()

    def _run_training(self, all_tickers):
        """Runs on a background thread — must not touch Tk widgets directly."""
        try:
            by_horizon = train_multi_horizon_model(all_tickers, horizons=HORIZONS)
            self.work_queue.put(("ok", by_horizon))
        except Exception as e:
            self.work_queue.put(("error", str(e)))

    def _poll_queue(self):
        try:
            while True:
                status, payload = self.work_queue.get_nowait()
                self.busy = False
                self.analyze_btn.state(["!disabled"])
                self.confidence_bar.stop()
                self.confidence_bar.configure(mode="determinate")
                if status == "ok":
                    self.results_by_horizon = payload
                    current = self._horizon_data()
                    if current and "error" not in current:
                        self.pooled_summary_var.set(
                            f"Shared model trained on {len(current['per_ticker'])} ticker(s) for "
                            f"“{self.horizon_var.get()}” — pooled test accuracy "
                            f"{current['overall_accuracy']:.1%} (baseline {current['overall_baseline_accuracy']:.1%})"
                        )
                    skipped_reasons = self._skipped_reasons()
                    if skipped_reasons:
                        self.status_var.set(f"Done. Skipped (insufficient data): {', '.join(skipped_reasons)}")
                    else:
                        self.status_var.set("Done.")
                    self._refresh_watchlist_display()
                    self._on_watchlist_select()
                else:
                    self.status_var.set(f"Training failed: {payload}")
                    self.pred_heading_var.set("Prediction")
                    self.pred_direction_var.set("⚠ Training failed")
                    self.pred_direction_lbl.configure(fg=COLORS["down_accent"])
                    messagebox.showerror("Training failed", str(payload))
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def _set_prediction_card_color(self, bg):
        self.prediction_card.configure(bg=bg)
        for child in self.prediction_card.winfo_children():
            child.configure(bg=bg)
            for grandchild in child.winfo_children():
                if isinstance(grandchild, (tk.Label, tk.Frame)):
                    grandchild.configure(bg=bg)
                    if isinstance(grandchild, tk.Frame):
                        for g2 in grandchild.winfo_children():
                            if isinstance(g2, tk.Label):
                                g2.configure(bg=bg)

    def _render_placeholder(self, ticker, reason=None):
        self._set_prediction_card_color(COLORS["card_bg"])
        self.pred_heading_lbl.configure(fg=COLORS["text_secondary"])
        self.pred_heading_var.set(f"{self.horizon_var.get()} Prediction for {ticker}")
        if reason:
            self.pred_direction_var.set("— Skipped —")
            self.pred_confidence_var.set(reason)
        else:
            self.pred_direction_var.set("Not analyzed yet")
            self.pred_confidence_var.set("Click “Train & Predict All” to include this ticker.")
        self.pred_direction_lbl.configure(fg=COLORS["neutral_ink"])
        self.confidence_bar.configure(style="Neutral.Horizontal.TProgressbar", value=0)

        self.accuracy_tile["value_var"].set("—")
        self.baseline_tile["value_var"].set("—")
        self.verdict_tile["value_var"].set("—")
        self.verdict_tile["value_lbl"].configure(fg=COLORS["text_primary"])

        self._style_empty_axes()

    def _render_result(self, result):
        ticker = result["ticker"]
        horizon_label = self.horizon_var.get()
        accuracy = result["accuracy"]
        baseline_accuracy = result["baseline_accuracy"]
        beats_baseline = accuracy > baseline_accuracy

        # --- Prediction hero card ---
        is_up = result["prediction"] == "UP"
        is_down = result["prediction"] == "DOWN"

        if is_up:
            bg, ink, accent, trough_style = COLORS["up_bg"], COLORS["up_ink"], COLORS["up_accent"], "Up.Horizontal.TProgressbar"
            arrow, word = "▲", "UP"
            confidence = result["prob_up"]
        elif is_down:
            bg, ink, accent, trough_style = COLORS["down_bg"], COLORS["down_ink"], COLORS["down_accent"], "Down.Horizontal.TProgressbar"
            arrow, word = "▼", "DOWN"
            confidence = 1 - result["prob_up"]
        else:
            bg, ink, accent, trough_style = COLORS["card_bg"], COLORS["neutral_ink"], COLORS["text_muted"], "Neutral.Horizontal.TProgressbar"
            arrow, word = "?", "NO PREDICTION"
            confidence = 0

        self._set_prediction_card_color(bg)

        date_txt = f" as of {result['latest_date'].date()}" if result["latest_date"] is not None else ""
        self.pred_heading_var.set(f"{horizon_label} Prediction for {ticker}{date_txt}")
        self.pred_heading_lbl.configure(fg=ink)
        self.pred_direction_var.set(f"{arrow}  {word}")
        self.pred_direction_lbl.configure(fg=accent)

        self.confidence_bar.configure(style=trough_style, value=confidence * 100)
        if result["prediction"]:
            self.pred_confidence_var.set(f"{confidence:.0%} confidence in this call")
        else:
            self.pred_confidence_var.set("No live prediction available for this ticker.")

        # --- Stat tiles ---
        self.accuracy_tile["value_var"].set(f"{accuracy:.1%}")
        self.baseline_tile["value_var"].set(f"{baseline_accuracy:.1%}")
        if beats_baseline:
            self.verdict_tile["value_var"].set("✓ Beats it")
            self.verdict_tile["value_lbl"].configure(fg=COLORS["up_accent"])
        else:
            self.verdict_tile["value_var"].set("✗ Does not beat it")
            self.verdict_tile["value_lbl"].configure(fg=COLORS["down_accent"])

        # --- Charts ---
        self.ax_price.clear()
        self.ax_importance.clear()
        for ax in (self.ax_price, self.ax_importance):
            ax.set_facecolor(COLORS["card_bg"])

        hist = result["featured"].tail(180)
        self.ax_price.fill_between(hist.index, hist["Close"], hist["Close"].min(),
                                    color=COLORS["series_price"], alpha=0.10, linewidth=0)
        self.ax_price.plot(hist.index, hist["Close"], label="Closing Price",
                            linewidth=2, color=COLORS["series_price"])
        self.ax_price.plot(hist.index, hist["SMA_20"], label="20-Day Average",
                            linewidth=1.4, color=COLORS["series_avg"], linestyle="--")

        last_date, last_price = hist.index[-1], hist["Close"].iloc[-1]
        self.ax_price.scatter([last_date], [last_price], color=COLORS["series_price"], s=34,
                               zorder=5, edgecolor=COLORS["card_bg"], linewidth=1.5)
        self.ax_price.annotate(f"${last_price:,.2f}", xy=(last_date, last_price),
                                xytext=(6, 8), textcoords="offset points", fontsize=8,
                                fontweight="bold", color=COLORS["series_price"])

        self.ax_price.set_title(f"{ticker} — Last 180 Trading Days", fontsize=10,
                                 fontweight="bold", color=COLORS["text_primary"])
        self.ax_price.set_ylabel("Price", color=COLORS["text_secondary"])
        self.ax_price.yaxis.set_major_formatter(FuncFormatter(lambda x, _pos: f"${x:,.0f}"))
        legend = self.ax_price.legend(loc="upper left", fontsize=8, frameon=False)
        for text in legend.get_texts():
            text.set_color(COLORS["text_secondary"])
        locator = mdates.AutoDateLocator(minticks=4, maxticks=7)
        self.ax_price.xaxis.set_major_locator(locator)
        self.ax_price.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        self.ax_price.tick_params(axis="x", colors=COLORS["text_muted"])
        self.ax_price.tick_params(axis="y", colors=COLORS["text_muted"])
        self.ax_price.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
        self.ax_price.margins(x=0.02)
        for spine in ("top", "right"):
            self.ax_price.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            self.ax_price.spines[spine].set_color(COLORS["axis"])

        horizon_data = self._horizon_data()
        if self.chart_view_var.get() == "Feature Importance":
            self._draw_importance_chart(horizon_data)
        else:
            self._draw_calibration_chart(horizon_data)

        self.canvas.draw()

    def _draw_importance_chart(self, horizon_data):
        importances = horizon_data["feature_importances"] if horizon_data and "error" not in horizon_data else {}
        order = sorted(range(len(FEATURE_COLUMNS)), key=lambda i: importances.get(FEATURE_COLUMNS[i], 0))
        names = [FEATURE_DISPLAY_NAMES[FEATURE_COLUMNS[i]] for i in order]
        raw_values = [importances.get(FEATURE_COLUMNS[i], 0) for i in order]
        # Raw permutation importance is a fraction of accuracy (e.g. 0.006) — shown
        # instead as "accuracy points lost if this signal were scrambled" (0.6),
        # which reads naturally on an axis instead of a wall of tiny decimals.
        values = [v * 100 for v in raw_values]
        max_val = max(raw_values) if raw_values else 1
        bar_colors = []
        for v in raw_values:
            # Permutation importance can go slightly negative for a useless/noisy
            # feature (shuffling it randomly helped a bit) — treat anything at or
            # below zero as "least important" rather than let it produce a
            # negative, out-of-range ramp index.
            ratio = max(v, 0) / max_val if max_val > 0 else 0
            step = max(0, min(int(ratio * (len(IMPORTANCE_RAMP) - 1)), len(IMPORTANCE_RAMP) - 1))
            bar_colors.append(IMPORTANCE_RAMP[step])
        bars = self.ax_importance.barh(names, values, color=bar_colors, height=0.65)
        span = max(values + [0.1]) - min(values + [0])
        for bar, v in zip(bars, values):
            offset = span * 0.03
            if v >= 0:
                self.ax_importance.text(v + offset, bar.get_y() + bar.get_height() / 2, f"{v:.1f}",
                                         va="center", ha="left", fontsize=7.5, color=COLORS["text_secondary"])
            else:
                self.ax_importance.text(v - offset, bar.get_y() + bar.get_height() / 2, f"{v:.1f}",
                                         va="center", ha="right", fontsize=7.5, color=COLORS["text_secondary"])
        self.ax_importance.set_title("What Influenced This Prediction", fontsize=9.5,
                                      fontweight="bold", color=COLORS["text_primary"], wrap=True)
        self.ax_importance.set_xlabel("Accuracy Points Lost if Signal Were Scrambled",
                                       fontsize=7.5, color=COLORS["text_secondary"], wrap=True)
        self.ax_importance.xaxis.set_major_formatter(FuncFormatter(lambda x, _pos: f"{x:.1f}"))
        self.ax_importance.tick_params(axis="y", labelsize=8, colors=COLORS["text_secondary"])
        self.ax_importance.tick_params(axis="x", colors=COLORS["text_muted"])
        self.ax_importance.grid(axis="x", color=COLORS["grid"], linewidth=0.8)
        self.ax_importance.margins(x=0.18)
        self.ax_importance.axvline(0, color=COLORS["axis"], linewidth=0.8)
        for spine in ("top", "right"):
            self.ax_importance.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            self.ax_importance.spines[spine].set_color(COLORS["axis"])

    def _draw_calibration_chart(self, horizon_data):
        """Reliability diagram: when the model says it's X% confident, is it
        actually right about X% of the time? Points above the dashed diagonal
        mean the model is under-confident there; below means over-confident."""
        calibration = horizon_data.get("calibration") if horizon_data and "error" not in horizon_data else None
        bin_pred = calibration["bin_pred"] if calibration else []
        bin_true = calibration["bin_true"] if calibration else []

        if bin_pred:
            self.ax_importance.plot([0.4, 1], [0.4, 1], linestyle="--", linewidth=1.2,
                                     color=COLORS["axis"], label="Perfect calibration")
            self.ax_importance.plot(bin_pred, bin_true, marker="o", markersize=6, linewidth=2,
                                     color=COLORS["accent"], label="Model")
            self.ax_importance.set_xlim(0.4, 1.0)
            self.ax_importance.set_ylim(0.0, 1.0)
            self.ax_importance.xaxis.set_major_formatter(FuncFormatter(lambda x, _pos: f"{x:.0%}"))
            self.ax_importance.yaxis.set_major_formatter(FuncFormatter(lambda x, _pos: f"{x:.0%}"))
            self.ax_importance.set_xlabel("Predicted Confidence", fontsize=7.5, color=COLORS["text_secondary"])
            self.ax_importance.set_ylabel("Actually Correct", fontsize=7.5, color=COLORS["text_secondary"])
            legend = self.ax_importance.legend(loc="upper left", fontsize=7.5, frameon=False)
            for text in legend.get_texts():
                text.set_color(COLORS["text_secondary"])
            self.ax_importance.text(
                0.98, 0.04, "Above line = under-confident\nBelow line = over-confident",
                fontsize=7, color=COLORS["text_muted"], ha="right", va="bottom",
                transform=self.ax_importance.transAxes,
            )
            self.ax_importance.set_title(
                f"Confidence Calibration — Shared Model\nBrier score {calibration['brier_score']:.3f} "
                f"(lower is better) · n={calibration['n_test']:,}",
                fontsize=9, fontweight="bold", color=COLORS["text_primary"], wrap=True)
        else:
            self.ax_importance.set_title("Confidence Calibration", fontsize=9.5,
                                          fontweight="bold", color=COLORS["text_primary"])
            self.ax_importance.text(0.5, 0.5, "Not enough test data yet", ha="center", va="center",
                                     fontsize=9, color=COLORS["text_muted"],
                                     transform=self.ax_importance.transAxes)

        self.ax_importance.tick_params(axis="both", colors=COLORS["text_muted"])
        self.ax_importance.grid(True, color=COLORS["grid"], linewidth=0.8)
        for spine in ("top", "right"):
            self.ax_importance.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            self.ax_importance.spines[spine].set_color(COLORS["axis"])

    def _save_chart(self):
        ticker = self._selected_ticker()
        if ticker is None or ticker not in self._per_ticker_results():
            messagebox.showinfo("Nothing to save", "Get a prediction for this ticker first.")
            return
        out_path = f"{ticker}_{self.horizon_var.get().replace(' ', '_')}_dashboard.png"
        self.figure.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=COLORS["card_bg"])
        self.status_var.set(f"Saved chart to {out_path}")
        messagebox.showinfo("Saved", f"Chart saved to:\n{out_path}")


if __name__ == "__main__":
    app = StockPredictorApp()
    app.mainloop()
