# config.py - Central configuration for Price Prediction Application
import os

# Application Settings
APP_TITLE = "Price Predictor - Terminal View"
APP_GEOMETRY = "1500x850"
DEFAULT_SYMBOL = "AAPL"
DEFAULT_ASSET_TYPE = "Stock"

# Model Configuration
SEQUENCE_LENGTH = 60
TRAIN_SPLIT = 0.85
VALIDATION_SPLIT = 0.15
MAX_EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 0.001

# Data Settings
HISTORICAL_PERIOD = "1y"
RECENT_DISPLAY_DAYS = 180
VOLATILITY_WINDOW = 20
RSI_PERIOD = 14

# Prediction Horizons (days)
PREDICTION_DAYS = [1, 7, 30, 90]
PREDICTION_LABELS = {
    1: "1 Day",
    7: "1 Week",
    30: "1 Month",
    90: "3 Months"
}

# Model Hyperparameters
LSTM_CONFIG = {
    "units": [128, 64, 32],
    "dropout": [0.3, 0.3, 0.2],
    "dense_units": 32,
    "epochs": 50,
    "batch_size": 16,
    "learning_rate": 0.001,
    "early_stopping_patience": 10,
    "reduce_lr_patience": 5,
}

GRU_CONFIG = {
    "units": [128, 64, 32],
    "dropout": [0.3, 0.3, 0.25],
    "dense_units": [64, 32],
    "epochs": 80,
    "batch_size": 24,
    "learning_rate": 0.0008,
    "clipnorm": 1.0,
    "early_stopping_patience": 12,
    "reduce_lr_patience": 6,
}

PROPHET_CONFIG = {
    "daily_seasonality": True,
    "weekly_seasonality": True,
    "yearly_seasonality": True,
    "changepoint_prior_scale": 0.08,
    "seasonality_prior_scale": 15.0,
    "holidays_prior_scale": 20.0,
    "seasonality_mode": "multiplicative",
    "interval_width": 0.95,
    "regressors": ["rsi", "atr_pct", "volume_ma", "weekday", "month"],
}

ARIMA_CONFIG = {
    "start_p": 1,
    "start_q": 1,
    "max_p": 5,
    "max_q": 5,
    "max_P": 2,
    "max_Q": 2,
    "m": 5,
    "seasonal": True,
    "stepwise": True,
    "n_fits": 20,
}

ENSEMBLE_CONFIG = {
    "performance_weight": 0.4,
    "confidence_weight": 0.3,
    "variance_weight": 0.3,
    "high_vol_threshold": 0.02,
    "disagreement_threshold_pct": 1.5,
}

# Terminal Color Theme (Bloomberg-style)
THEME = {
    "bg": "#000000",
    "panel_bg": "#111111",
    "primary": "#F4F725",
    "accent_blue": "#33A1FF",
    "accent_green": "#34E265",
    "accent_orange": "#FB8B1E",
    "danger": "#FF433D",
    "text": "#E0E0E0",
    "muted": "#7A7A7A",
    "input_bg": "#050505",
    "grid_color": "#333333",
    "border_color": "#444444",
}

# Plot Settings
PLOT_STYLE = "dark_background"
PLOT_CONFIG = {
    "axes.facecolor": "#000000",
    "figure.facecolor": "#000000",
    "axes.edgecolor": "#444444",
    "axes.labelcolor": "#E0E0E0",
    "xtick.color": "#B0B0B0",
    "ytick.color": "#B0B0B0",
    "grid.color": "#333333",
    "grid.linestyle": "--",
    "grid.alpha": 0.4,
    "text.color": "#E0E0E0",
    "legend.edgecolor": "#333333",
}

# Stat Card Configuration
STAT_CARDS = [
    ("Current Price", "accent_blue"),
    ("24h Change", "accent_orange"),
    ("Predicted (1D)", "accent_green"),
    ("Accuracy", "#9C27B0"),
]

# Model Availability Flags (set at runtime)
TF_AVAILABLE = False
PROPHET_AVAILABLE = False
ARIMA_BASE_AVAILABLE = False
PMDARIMA_AVAILABLE = False
TA_AVAILABLE = False
YF_AVAILABLE = False

# File Paths
MODEL_CHECKPOINT_DIR = "checkpoints"
LOGS_DIR = "logs"
