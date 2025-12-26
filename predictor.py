import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from keras.src.callbacks import ReduceLROnPlateau
from keras.src.layers import Bidirectional, BatchNormalization
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import threading
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings('ignore')

# --- OPTIONAL: technical indicators for Prophet regressors ---
try:
    import talib as ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    print("WARNING: TA-Lib not installed. Prophet regressors will be simplified. Run: pip install TA-Lib")

# --- yfinance ---
try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False
    print("ERROR: yfinance not installed. Run: pip install yfinance")

# --- TensorFlow / Keras ---
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    from tensorflow.keras.optimizers import Adam
    import tensorflow as tf

    tf.get_logger().setLevel('ERROR')
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("WARNING: TensorFlow/Keras not installed. LSTM/GRU models disabled.")

# --- Prophet ---
try:
    from prophet import Prophet
    import logging

    logging.getLogger('prophet').setLevel(logging.ERROR)
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("WARNING: prophet not installed. Prophet model disabled.")

# --- ARIMA / SARIMAX / pmdarima ---
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.arima.model import ARIMA
    ARIMA_BASE_AVAILABLE = True
except ImportError:
    ARIMA_BASE_AVAILABLE = False
    print("WARNING: statsmodels not installed. ARIMA/SARIMAX disabled.")

try:
    from pmdarima import auto_arima
    PMDARIMA_AVAILABLE = True
except ImportError:
    PMDARIMA_AVAILABLE = False
    print("WARNING: pmdarima not installed. Auto-ARIMA disabled.")

ARIMA_AVAILABLE = ARIMA_BASE_AVAILABLE and PMDARIMA_AVAILABLE

# ============ GLOBAL PLOT THEME (BLOOMBERG-LIKE TERMINAL) ============
plt.style.use('dark_background')
plt.rcParams.update({
    'axes.facecolor': '#000000',
    'figure.facecolor': '#000000',
    'axes.edgecolor': '#444444',
    'axes.labelcolor': '#E0E0E0',
    'xtick.color': '#B0B0B0',
    'ytick.color': '#B0B0B0',
    'grid.color': '#333333',
    'grid.linestyle': '--',
    'grid.alpha': 0.4,
    'text.color': '#E0E0E0',
    'legend.edgecolor': '#333333'
})


class PredictionModels:
    def __init__(self, seq_length=60):
        self.seq_length = seq_length
        self.scaler = MinMaxScaler()

    def prepare_data(self, prices: np.ndarray):
        try:
            prices = np.asarray(prices, dtype=float)
            scaled_data = self.scaler.fit_transform(prices.reshape(-1, 1))
            X, y = [], []
            for i in range(self.seq_length, len(scaled_data)):
                X.append(scaled_data[i - self.seq_length:i, 0])
                y.append(scaled_data[i, 0])
            X, y = np.array(X), np.array(y)
            if len(X) == 0:
                return None, None, None
            X = X.reshape((X.shape[0], X.shape[1], 1))
            return X, y, scaled_data
        except Exception as e:
            print(f"Data preparation error: {e}")
            return None, None, None

    # ================= LSTM =================
    def predict_lstm(self, prices, days_ahead=1):
        if (not TF_AVAILABLE) or len(prices) < self.seq_length + 50:
            return None, 0

        try:
            X, y, scaled_data = self.prepare_data(prices)
            if X is None:
                return None, 0

            split = int(0.85 * len(X))
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]

            model = Sequential([
                Bidirectional(LSTM(128, return_sequences=True, input_shape=(X.shape[1], 1))),
                Dropout(0.3),
                Bidirectional(LSTM(64, return_sequences=True)),
                Dropout(0.3),
                Bidirectional(LSTM(32)),
                Dropout(0.2),
                Dense(32, activation='relu'),
                Dense(1)
            ])

            model.compile(optimizer=Adam(learning_rate=0.001), loss='huber')

            early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)

            model.fit(
                X_train,
                y_train,
                epochs=50,
                batch_size=16,
                validation_split=0.15,
                callbacks=[early_stop, reduce_lr],
                verbose=0
            )

            predictions = []
            current_seq = scaled_data[-self.seq_length:].reshape(1, self.seq_length, 1)

            for _ in range(days_ahead):
                pred = model.predict(current_seq, verbose=0)
                pred_val = float(pred[0, 0])
                smoothed_pred = 0.7 * pred_val + 0.3 * float(current_seq[0, -1, 0])
                predictions.append(smoothed_pred)
                current_seq = np.append(
                    current_seq[:, 1:, :],
                    np.array(smoothed_pred).reshape(1, 1, 1),
                    axis=1
                )

            test_pred = model.predict(X_test, verbose=0).flatten()
            y_test_inv = self.scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
            test_pred_inv = self.scaler.inverse_transform(test_pred.reshape(-1, 1)).flatten()

            mape = np.mean(np.abs((y_test_inv - test_pred_inv) / (y_test_inv + 1e-8))) * 100
            accuracy = max(0.0, min(100.0, 100.0 - mape))

            predictions = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
            return float(predictions[-1][0]), float(accuracy)

        except Exception as e:
            print(f"LSTM error: {e}")
            return None, 0

    # ================= GRU =================
    def predict_gru(self, prices, days_ahead=1):
        if (not TF_AVAILABLE) or len(prices) < self.seq_length + 50:
            return None, 0

        try:
            X, y, scaled_data = self.prepare_data(prices)
            if X is None:
                return None, 0

            split = int(0.85 * len(X))
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]

            model = Sequential([
                Bidirectional(GRU(128, return_sequences=True, input_shape=(X.shape[1], 1))),
                Dropout(0.3),
                Bidirectional(GRU(64, return_sequences=True)),
                Dropout(0.3),
                Bidirectional(GRU(32)),
                Dropout(0.25),
                Dense(64, activation='relu'),
                BatchNormalization(),
                Dropout(0.2),
                Dense(32, activation='relu'),
                Dense(1)
            ])

            model.compile(
                optimizer=Adam(learning_rate=0.0008, clipnorm=1.0),
                loss='huber',
                metrics=['mae']
            )

            early_stop = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.4, patience=6, min_lr=1e-7)
            model_checkpoint = ModelCheckpoint('best_gru_model.h5', save_best_only=True, monitor='val_loss')

            model.fit(
                X_train,
                y_train,
                epochs=80,
                batch_size=24,
                validation_split=0.12,
                callbacks=[early_stop, reduce_lr, model_checkpoint],
                verbose=0
            )

            predictions = []
            current_seq = scaled_data[-self.seq_length:].reshape(1, self.seq_length, 1)

            for _ in range(days_ahead):
                pred = model.predict(current_seq, verbose=0)
                pred_val = float(pred[0, 0])
                confidence = 1.0 / (1.0 + abs(pred_val - float(current_seq[0, -1, 0])))
                alpha = 0.75 * confidence + 0.25
                smoothed_pred = alpha * pred_val + (1 - alpha) * float(current_seq[0, -1, 0])

                predictions.append(smoothed_pred)
                current_seq = np.append(
                    current_seq[:, 1:, :],
                    np.array(smoothed_pred).reshape(1, 1, 1),
                    axis=1
                )

            test_pred = model.predict(X_test, verbose=0).flatten()
            y_test_orig = self.scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
            test_pred_orig = self.scaler.inverse_transform(test_pred.reshape(-1, 1)).flatten()

            mape = np.mean(np.abs((y_test_orig - test_pred_orig) / (y_test_orig + 1e-8))) * 100
            directional_acc = np.mean(
                np.sign(np.diff(y_test_orig)) == np.sign(np.diff(test_pred_orig))
            ) * 100

            accuracy = max(0.0, min(100.0, 95.0 - mape + directional_acc * 0.3))

            predictions = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
            return float(predictions[-1][0]), float(accuracy)

        except Exception as e:
            print(f"GRU error: {e}")
            return None, 0

    # ================= Prophet =================
    def predict_prophet(self, df: pd.DataFrame, days_ahead=1):
        if (not PROPHET_AVAILABLE) or len(df) < 60:
            return None, 0

        try:
            df = df.copy()
            df = df.dropna(subset=['Close'])
            prophet_df = pd.DataFrame({
                'ds': df.index,
                'y': df['Close'].values
            }).reset_index(drop=True)

            # fallback if TA-Lib missing
            if TA_AVAILABLE:
                prophet_df['rsi'] = ta.RSI(df['Close'].values, timeperiod=14)
                prophet_df['atr_pct'] = ta.ATR(df['High'].values, df['Low'].values, df['Close'].values, timeperiod=14) / df['Close'].values
            else:
                prophet_df['rsi'] = 50.0
                prophet_df['atr_pct'] = df['Close'].pct_change().rolling(14).std().fillna(0)

            if 'Volume' in df.columns:
                prophet_df['volume_ma'] = df['Volume'].rolling(20).mean() / df['Volume']
            else:
                prophet_df['volume_ma'] = 1.0

            prophet_df['rsi'] = prophet_df['rsi'].fillna(50)
            prophet_df['atr_pct'] = prophet_df['atr_pct'].fillna(0)
            prophet_df['volume_ma'] = prophet_df['volume_ma'].replace([np.inf, -np.inf], 1.0).fillna(1.0)

            prophet_df['weekday'] = prophet_df['ds'].dt.weekday
            prophet_df['month'] = prophet_df['ds'].dt.month

            train_size = int(0.85 * len(prophet_df))
            train_df, test_df = prophet_df[:train_size], prophet_df[train_size:]

            model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=True,
                changepoint_prior_scale=0.08,
                seasonality_prior_scale=15.0,
                holidays_prior_scale=20.0,
                seasonality_mode='multiplicative',
                interval_width=0.95
            )

            model.add_regressor('rsi', prior_scale=0.5, mode='multiplicative')
            model.add_regressor('atr_pct', prior_scale=0.3, mode='multiplicative')
            model.add_regressor('volume_ma', prior_scale=0.4, mode='multiplicative')
            model.add_regressor('weekday')
            model.add_regressor('month')

            model.fit(train_df)

            future = model.make_future_dataframe(periods=days_ahead)
            future = future.merge(
                prophet_df[['ds', 'rsi', 'atr_pct', 'volume_ma']],
                on='ds',
                how='left'
            )

            future['rsi'] = future['rsi'].fillna(prophet_df['rsi'].iloc[-1])
            future['atr_pct'] = future['atr_pct'].fillna(prophet_df['atr_pct'].iloc[-1])
            future['volume_ma'] = future['volume_ma'].fillna(prophet_df['volume_ma'].iloc[-1])
            future['weekday'] = future['ds'].dt.weekday
            future['month'] = future['ds'].dt.month

            forecast = model.predict(future)
            prediction = float(forecast['yhat'].iloc[-1])

            mape_scores = []
            if len(test_df) > 10:
                for i in range(10, len(test_df), 5):
                    val_train = prophet_df[:train_size + i]
                    val_test = prophet_df[train_size + i:train_size + i + 5]

                    if len(val_test) == 0:
                        break

                    val_model = Prophet(
                        changepoint_prior_scale=0.08,
                        seasonality_mode='multiplicative'
                    )
                    val_model.add_regressor('rsi', prior_scale=0.5, mode='multiplicative')
                    val_model.add_regressor('atr_pct', prior_scale=0.3, mode='multiplicative')
                    val_model.add_regressor('volume_ma', prior_scale=0.4, mode='multiplicative')

                    val_model.fit(val_train)

                    val_future = val_model.make_future_dataframe(periods=len(val_test))
                    val_future = val_future.merge(
                        val_train[['ds', 'rsi', 'atr_pct', 'volume_ma']],
                        on='ds',
                        how='left'
                    )
                    val_future['rsi'] = val_future['rsi'].fillna(val_train['rsi'].iloc[-1])
                    val_future['atr_pct'] = val_future['atr_pct'].fillna(val_train['atr_pct'].iloc[-1])
                    val_future['volume_ma'] = val_future['volume_ma'].fillna(val_train['volume_ma'].iloc[-1])

                    val_forecast = val_model.predict(val_future)
                    val_actual = val_test['y'].values
                    val_pred = val_forecast['yhat'].tail(len(val_test)).values

                    mape_scores.append(
                        np.mean(np.abs((val_actual - val_pred) / (val_actual + 1e-8))) * 100
                    )

            mape = np.mean(mape_scores) if mape_scores else 15.0
            accuracy = max(0.0, min(100.0, 95.0 - mape))
            return prediction, float(accuracy)

        except Exception as e:
            print(f"Prophet error: {e}")
            return None, 0

    # ================= ARIMA / SARIMAX =================
    def _compute_rsi(self, prices, period=14):
        prices = np.asarray(prices, dtype=float)
        delta = np.diff(prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        gain_series = pd.Series(gain)
        loss_series = pd.Series(loss)
        avg_gain = gain_series.rolling(period).mean()
        avg_loss = loss_series.rolling(period).mean()
        rs = avg_gain / (avg_loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.fillna(50)
        return np.concatenate([np.full(period, 50.0), rsi.values])

    def _extrapolate_exog(self, df, train_size, days_ahead):
        last_returns = float(df['returns'].iloc[train_size - 1])
        last_vol = float(df['volatility'].iloc[train_size - 1])
        last_rsi = float(df['rsi'].iloc[train_size - 1])

        future_exog = pd.DataFrame({
            'returns': np.full(days_ahead, last_returns),
            'volatility': np.full(days_ahead, last_vol),
            'rsi': np.full(days_ahead, last_rsi)
        })
        return future_exog

    def predict_arima(self, prices, days_ahead=1):
        if (not ARIMA_AVAILABLE) or len(prices) < 100:
            return None, 0

        try:
            prices = np.asarray(prices, dtype=float)
            df = pd.DataFrame(
                {'price': prices},
                index=pd.date_range(start='2020-01-01', periods=len(prices), freq='D')
            )

            df['returns'] = df['price'].pct_change()
            df['volatility'] = df['returns'].rolling(20).std()
            df['rsi'] = self._compute_rsi(df['price'].values, 14)
            df['returns'] = df['returns'].fillna(0)
            df['volatility'] = df['volatility'].fillna(df['volatility'].mean())

            train_size = int(0.85 * len(df))
            train_data = df['price'][:train_size]

            model = auto_arima(
                train_data,
                start_p=1, start_q=1, max_p=5, max_q=5,
                start_P=0, start_Q=0, max_P=2, max_Q=2, m=5,
                seasonal=True, stepwise=True, suppress_warnings=True,
                error_action='ignore', trace=False,
                exogenous=df[['returns', 'volatility', 'rsi']][:train_size],
                scoring='mse', n_fits=20
            )

            sarimax_model = SARIMAX(
                train_data,
                exog=df[['returns', 'volatility', 'rsi']][:train_size],
                order=model.order,
                seasonal_order=model.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            fitted = sarimax_model.fit(disp=False, maxiter=200)

            future_exog = self._extrapolate_exog(df, train_size, days_ahead)
            forecast = fitted.forecast(steps=days_ahead, exog=future_exog)
            prediction = float(forecast.iloc[-1] if hasattr(forecast, 'iloc') else forecast[-1])

            mape_scores = []
            fold_size = max(5, len(df) // 15)

            for i in range(10, len(df) - fold_size, fold_size):
                fold_train_size = train_size - i
                if fold_train_size <= 20:
                    break

                fold_train = df['price'][:fold_train_size]
                fold_exog = df[['returns', 'volatility', 'rsi']][:fold_train_size]
                fold_test = df['price'][fold_train_size:fold_train_size + fold_size]

                if len(fold_test) == 0:
                    break

                try:
                    fold_model = auto_arima(
                        fold_train,
                        exogenous=fold_exog,
                        start_p=1,
                        max_p=3,
                        max_q=3,
                        seasonal=False,
                        stepwise=True,
                        suppress_warnings=True
                    )
                    fold_forecast = fold_model.predict(
                        n_periods=len(fold_test),
                        X=fold_exog.tail(len(fold_test))
                    )
                    fold_mape = np.mean(
                        np.abs((fold_test.values - fold_forecast) / (fold_test.values + 1e-8))
                    ) * 100
                    mape_scores.append(fold_mape)
                except Exception:
                    continue

            mape = np.mean(mape_scores) if mape_scores else 15.0
            accuracy = max(0.0, min(100.0, 92.0 - mape))

            return prediction, float(accuracy)

        except Exception as e:
            print(f"ARIMA error: {e}")
            return None, 0

    # ================= Ensemble =================
    def _estimate_volatility(self, prices, window=20):
        prices = np.asarray(prices, dtype=float)
        if len(prices) < 2:
            return 0.01
        returns = np.diff(prices) / prices[:-1]
        if len(returns) < window:
            return float(np.std(returns))
        return float(np.std(returns[-window:]))

    def _stacking_meta_learner(self, predictions, accuracies, confidences, weights):
        try:
            preds = np.asarray(predictions, dtype=float)
            accs = np.asarray(accuracies, dtype=float)
            confs = np.asarray(confidences, dtype=float)
            w = np.asarray(weights, dtype=float)

            base_acc = float(np.average(accs, weights=w))

            current_price = float(preds[0]) * 0.95
            directional_agreement = float(np.mean(np.sign(preds - current_price)))

            stacking_boost = 1 + 0.15 * abs(directional_agreement)
            return float(min(100.0, base_acc * stacking_boost))
        except Exception:
            return float(np.mean(accuracies) if len(accuracies) > 0 else 0.0)

    def predict_ensemble(self, prices, df, days_ahead=1):
        predictions, accuracies, confidences = [], [], []

        models = [
            ('LSTM', self.predict_lstm(prices, days_ahead)),
            ('GRU', self.predict_gru(prices, days_ahead)),
            ('Prophet', self.predict_prophet(df, days_ahead)),
            ('ARIMA', self.predict_arima(prices, days_ahead))
        ]

        prices = np.asarray(prices, dtype=float)
        if len(prices) == 0:
            return None, 0
        current_price = float(prices[-1])

        for _, result in models:
            pred, acc = result
            if pred is not None:
                pred_val = float(pred)
                acc_val = float(acc)
                error_bound = current_price * (1 - acc_val / 100.0) * 0.02
                _ = error_bound  # unused but kept for clarity
                confidence = acc_val * (1 - abs(pred_val - current_price) / (current_price * 0.05))
                predictions.append(pred_val)
                accuracies.append(acc_val)
                confidences.append(max(0.0, confidence))

        if len(predictions) < 2:
            if len(predictions) == 1:
                return float(predictions[0]), float(accuracies[0])
            return None, 0

        predictions = np.asarray(predictions, dtype=float)
        accuracies = np.asarray(accuracies, dtype=float)
        confidences = np.asarray(confidences, dtype=float)

        recent_perf_weights = accuracies ** 2
        if confidences.sum() > 0:
            confidence_weights = confidences / confidences.sum()
        else:
            confidence_weights = np.ones_like(confidences) / len(confidences)
        inverse_variance_weights = 1.0 / (accuracies + 1e-8)

        vol_factor = self._estimate_volatility(prices[-30:])
        if vol_factor > 0.02:
            weights = 0.4 * recent_perf_weights + 0.3 * confidence_weights + 0.3 * inverse_variance_weights
        else:
            weights = 0.5 * confidence_weights + 0.5 * recent_perf_weights

        weights = weights / weights.sum()

        ensemble_pred = float(np.average(predictions, weights=weights))

        pred_std = float(np.std(predictions))
        disagreement_threshold = current_price * 0.015

        if pred_std > disagreement_threshold:
            ensemble_pred = float(np.median(predictions) + 0.1 * (ensemble_pred - current_price))

        ensemble_acc = self._stacking_meta_learner(predictions, accuracies, confidences, weights)
        signal_strength = float(min(100.0, ensemble_acc * (1 - pred_std / max(current_price, 1e-8))))

        return ensemble_pred, signal_strength


class TerminalPricePredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Price Predictor - Terminal View")
        self.root.geometry("1500x850")

        self.colors = {
            'bg': '#000000',
            'panel_bg': '#111111',
            'primary': '#F4F725',
            'accent_blue': '#33A1FF',
            'accent_green': '#34E265',
            'accent_orange': '#FB8B1E',
            'danger': '#FF433D',
            'text': '#E0E0E0',
            'muted': '#7A7A7A'
        }

        self.root.configure(bg=self.colors['bg'])
        self.predictor = PredictionModels(60)
        self.current_data = None
        self.current_currency = ""

        if not YF_AVAILABLE:
            messagebox.showerror(
                "Missing Library",
                "yfinance library not installed!\n\nPlease run: pip install yfinance"
            )
            self.root.destroy()
            return

        self.configure_style()
        self.create_widgets()

    def fmt_price(self, value):
        if value is None:
            return "--"
        ccy = self.current_currency or ""
        try:
            value = float(value)
        except Exception:
            return "--"
        return f"{ccy} {value:.2f}" if ccy else f"{value:.2f}"

    def configure_style(self):
        style = ttk.Style()
        style.theme_use('clam')

        style.configure(
            '.',
            background=self.colors['bg'],
            foreground=self.colors['text'],
            fieldbackground=self.colors['bg']
        )

        style.configure(
            'TCombobox',
            foreground=self.colors['text'],
            fieldbackground='#050505',
            background='#050505',
            arrowcolor=self.colors['primary'],
            bordercolor='#333333'
        )
        style.map(
            'TCombobox',
            fieldbackground=[('readonly', '#050505')],
            foreground=[('readonly', self.colors['text'])]
        )

        style.configure(
            'Status.TLabel',
            background=self.colors['bg'],
            foreground=self.colors['accent_green'],
            font=('Consolas', 9)
        )

    def create_widgets(self):
        control_frame = tk.Frame(self.root, bg=self.colors['panel_bg'], pady=8)
        control_frame.pack(fill=tk.X, padx=8, pady=(6, 3))

        tk.Label(
            control_frame,
            text="SYM",
            bg=self.colors['panel_bg'],
            fg=self.colors['primary'],
            font=('Consolas', 11, 'bold')
        ).pack(side=tk.LEFT, padx=(8, 5))

        self.symbol_entry = tk.Entry(
            control_frame,
            font=('Consolas', 11),
            width=10,
            bg='#050505',
            fg=self.colors['text'],
            insertbackground=self.colors['primary'],
            relief=tk.FLAT
        )
        self.symbol_entry.insert(0, "AAPL")
        self.symbol_entry.pack(side=tk.LEFT, padx=4)

        tk.Label(
            control_frame,
            text="TYPE",
            bg=self.colors['panel_bg'],
            fg=self.colors['primary'],
            font=('Consolas', 11, 'bold')
        ).pack(side=tk.LEFT, padx=(16, 4))

        self.asset_var = tk.StringVar(value="Stock")
        type_cb = ttk.Combobox(
            control_frame,
            textvariable=self.asset_var,
            values=["Stock", "Crypto"],
            width=10,
            state='readonly',
            font=('Consolas', 10)
        )
        type_cb.pack(side=tk.LEFT, padx=4)

        tk.Label(
            control_frame,
            text="MODEL",
            bg=self.colors['panel_bg'],
            fg=self.colors['primary'],
            font=('Consolas', 11, 'bold')
        ).pack(side=tk.LEFT, padx=(16, 4))

        models_available = []
        if TF_AVAILABLE:
            models_available.extend(["LSTM", "GRU"])
        if PROPHET_AVAILABLE:
            models_available.append("Prophet")
        if ARIMA_AVAILABLE:
            models_available.append("ARIMA")
        if len(models_available) > 1:
            models_available.append("Ensemble")
        if not models_available:
            models_available = ["No Models"]

        self.model_var = tk.StringVar(value=models_available[0])
        model_cb = ttk.Combobox(
            control_frame,
            textvariable=self.model_var,
            values=models_available,
            width=12,
            state='readonly',
            font=('Consolas', 10)
        )
        model_cb.pack(side=tk.LEFT, padx=4)

        self.predict_btn = tk.Button(
            control_frame,
            text="GO",
            command=self.start_prediction,
            font=('Consolas', 11, 'bold'),
            bg=self.colors['accent_blue'],
            fg='#000000',
            padx=18,
            pady=3,
            cursor='hand2',
            activebackground='#0056ff',
            activeforeground='#000000',
            relief=tk.FLAT
        )
        self.predict_btn.pack(side=tk.LEFT, padx=18)

        self.status_label = tk.Label(
            control_frame,
            text="READY",
            bg=self.colors['panel_bg'],
            fg=self.colors['accent_green'],
            font=('Consolas', 10, 'bold')
        )
        self.status_label.pack(side=tk.LEFT, padx=10)

        main_container = tk.Frame(self.root, bg=self.colors['bg'])
        main_container.pack(fill=tk.BOTH, expand=True, padx=8, pady=(2, 6))

        left_col = tk.Frame(main_container, bg=self.colors['bg'], width=430)
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 6))
        left_col.pack_propagate(False)

        desc_frame = tk.Frame(left_col, bg=self.colors['panel_bg'], bd=1, relief=tk.SOLID)
        desc_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 6))

        tk.Label(
            desc_frame,
            text="ASSET INFO",
            bg=self.colors['panel_bg'],
            fg=self.colors['primary'],
            font=('Consolas', 11, 'bold'),
            anchor='w'
        ).pack(fill=tk.X, padx=10, pady=(8, 4))

        self.desc_text = scrolledtext.ScrolledText(
            desc_frame,
            height=10,
            bg='#050505',
            fg=self.colors['text'],
            font=('Consolas', 9),
            relief=tk.FLAT,
            padx=8,
            pady=6,
            wrap='word',
            state='disabled',
            insertbackground=self.colors['primary']
        )
        self.desc_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        stats_frame = tk.Frame(left_col, bg=self.colors['bg'])
        stats_frame.pack(fill=tk.X)

        self.stat_labels = {}
        stats = [
            ("Current Price", self.colors['accent_blue']),
            ("24h Change", self.colors['accent_orange']),
            ("Predicted (1D)", self.colors['accent_green']),
            ("Accuracy", '#9C27B0')
        ]

        for i, (stat, color) in enumerate(stats):
            row = i // 2
            col = i % 2
            card = tk.Frame(
                stats_frame,
                bg=self.colors['panel_bg'],
                bd=1,
                relief=tk.SOLID
            )
            card.grid(row=row, column=col, padx=3, pady=3, sticky='nsew')
            stats_frame.grid_columnconfigure(col, weight=1)

            tk.Label(
                card,
                text=stat,
                bg=self.colors['panel_bg'],
                fg=self.colors['muted'],
                font=('Consolas', 9)
            ).pack(pady=(4, 1))

            label = tk.Label(
                card,
                text="--",
                bg=self.colors['panel_bg'],
                fg=color,
                font=('Consolas', 14, 'bold')
            )
            label.pack(pady=(0, 4))
            self.stat_labels[stat] = label

        pred_frame = tk.Frame(left_col, bg=self.colors['panel_bg'], bd=1, relief=tk.SOLID)
        pred_frame.pack(fill=tk.X, pady=(6, 0))

        tk.Label(
            pred_frame,
            text="TARGETS",
            bg=self.colors['panel_bg'],
            fg=self.colors['primary'],
            font=('Consolas', 11, 'bold')
        ).pack(pady=(6, 2))

        pred_grid = tk.Frame(pred_frame, bg=self.colors['panel_bg'])
        pred_grid.pack(fill=tk.X, padx=8, pady=(0, 6))

        self.pred_labels = {}
        pred_times = ["1 Day", "1 Week", "1 Month", "3 Months"]
        for i, time in enumerate(pred_times):
            row = i // 2
            col = i % 2
            card = tk.Frame(pred_grid, bg='#050505', relief=tk.FLAT)
            card.grid(row=row, column=col, padx=3, pady=3, sticky='nsew')
            pred_grid.grid_columnconfigure(col, weight=1)

            tk.Label(
                card,
                text=time,
                bg='#050505',
                fg=self.colors['muted'],
                font=('Consolas', 9)
            ).pack(pady=(4, 1))

            label = tk.Label(
                card,
                text="--",
                bg='#050505',
                fg=self.colors['accent_green'],
                font=('Consolas', 12, 'bold')
            )
            label.pack(pady=(0, 4))
            self.pred_labels[time] = label

        right_col = tk.Frame(main_container, bg=self.colors['panel_bg'], bd=1, relief=tk.SOLID)
        right_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tk.Label(
            right_col,
            text="PRICE & FORECAST",
            bg=self.colors['panel_bg'],
            fg=self.colors['primary'],
            font=('Consolas', 11, 'bold'),
            anchor='w'
        ).pack(fill=tk.X, padx=10, pady=(6, 2))

        self.fig = Figure(figsize=(8, 5), facecolor=self.colors['bg'], dpi=90)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('#000000')

        self.canvas = FigureCanvasTkAgg(self.fig, master=right_col)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        bottom_frame = tk.Frame(self.root, bg=self.colors['panel_bg'], bd=1, relief=tk.SOLID)
        bottom_frame.pack(fill=tk.X, padx=8, pady=(0, 6))

        tk.Label(
            bottom_frame,
            text="SUMMARY",
            bg=self.colors['panel_bg'],
            fg=self.colors['primary'],
            font=('Consolas', 10, 'bold')
        ).pack(anchor='w', padx=10, pady=(4, 1))

        self.summary_text = tk.Text(
            bottom_frame,
            height=3,
            bg='#050505',
            fg=self.colors['text'],
            font=('Consolas', 9),
            relief=tk.FLAT,
            padx=10,
            pady=6,
            state='disabled',
            wrap='word',
            insertbackground=self.colors['primary']
        )
        self.summary_text.pack(fill=tk.X, padx=8, pady=(0, 6))

        self.status_bar = ttk.Label(
            self.root,
            text="READY",
            style='Status.TLabel',
            anchor=tk.W
        )
        self.status_bar.pack(fill=tk.X, padx=8, pady=(0, 4))

    # ========== PREDICTION FLOW ==========
    def start_prediction(self):
        self.predict_btn.config(state=tk.DISABLED, text="...")
        self.status_label.config(text="FETCHING", fg=self.colors['accent_orange'])
        self.status_bar.config(text="FETCHING DATA")
        thread = threading.Thread(target=self.run_prediction, daemon=True)
        thread.start()

    def run_prediction(self):
        try:
            symbol = self.symbol_entry.get().strip().upper()
            asset_type = self.asset_var.get()
            model_type = self.model_var.get()

            if not symbol:
                self.root.after(0, lambda: messagebox.showerror("Error", "Please enter a symbol"))
                return

            self.root.after(0, lambda: self.status_label.config(text="LOADING", fg=self.colors['accent_blue']))

            ticker = yf.Ticker(symbol) if asset_type == "Stock" else yf.Ticker(f"{symbol}-USD")
            df = ticker.history(period="1y")
            info = ticker.info

            if df.empty:
                self.root.after(0, lambda: messagebox.showerror("Error", f"No data for {symbol}"))
                return

            currency = None
            try:
                meta = getattr(df, "history_metadata", None) or getattr(ticker, "history_metadata", None)
                if meta and isinstance(meta, dict):
                    currency = meta.get("currency")
            except Exception:
                pass

            if not currency and isinstance(info, dict):
                currency = info.get("currency")
            if not currency:
                currency = ""

            self.current_currency = currency

            self.current_data = df
            prices = df['Close'].values
            if len(prices) < 2:
                self.root.after(0, lambda: messagebox.showerror("Error", "Not enough price data"))
                return

            current_price = float(prices[-1])
            prev_price = float(prices[-2])
            change_pct = ((current_price - prev_price) / prev_price) * 100

            self.root.after(0, lambda: self.update_description(symbol, info, asset_type))

            self.root.after(0, lambda:
                self.stat_labels["Current Price"].config(
                    text=self.fmt_price(current_price)
                )
            )
            self.root.after(0, lambda:
                self.stat_labels["24h Change"].config(
                    text=f"{change_pct:+.2f}%",
                    fg=self.colors['accent_green'] if change_pct >= 0 else self.colors['danger']
                )
            )

            self.root.after(0, lambda: self.status_label.config(text="PREDICTING", fg=self.colors['accent_blue']))
            self.root.after(0, lambda: self.status_bar.config(text="RUNNING MODEL"))

            prediction_days = [1, 7, 30, 90]
            predictions = {}

            for days in prediction_days:
                if model_type == "LSTM":
                    pred, acc = self.predictor.predict_lstm(prices, days)
                elif model_type == "GRU":
                    pred, acc = self.predictor.predict_gru(prices, days)
                elif model_type == "Prophet":
                    pred, acc = self.predictor.predict_prophet(df, days)
                elif model_type == "ARIMA":
                    pred, acc = self.predictor.predict_arima(prices, days)
                else:
                    pred, acc = self.predictor.predict_ensemble(prices, df, days)
                predictions[days] = (pred, acc)

            self.root.after(0, lambda: self.update_ui(symbol, predictions, df, current_price))
            self.root.after(0, lambda: self.status_label.config(text="DONE", fg=self.colors['accent_green']))
            self.root.after(0, lambda: self.status_bar.config(text="COMPLETED"))

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            self.root.after(0, lambda: messagebox.showerror("Prediction Failed", error_msg))
            self.root.after(0, lambda: self.status_label.config(text="ERROR", fg=self.colors['danger']))
            self.root.after(0, lambda: self.status_bar.config(text="ERROR"))
        finally:
            self.root.after(0, lambda: self.predict_btn.config(state=tk.NORMAL, text="GO"))

    # ========== UI UPDATES ==========
    def update_description(self, symbol, info, asset_type):
        self.desc_text.config(state='normal')
        self.desc_text.delete('1.0', tk.END)

        lines = []
        lines.append(f"SYM : {symbol}\n")
        lines.append(f"TYPE: {asset_type}\n")
        if self.current_currency:
            lines.append(f"CCY : {self.current_currency}\n")
        lines.append("-" * 52 + "\n\n")

        if isinstance(info, dict) and info:
            name = info.get('longName') or info.get('shortName') or symbol
            lines.append(f"NAME: {name}\n\n")

            if asset_type == "Stock":
                sector = info.get('sector', 'N/A')
                industry = info.get('industry', 'N/A')
                country = info.get('country', 'N/A')
                lines.append(f"SECTOR : {sector}\n")
                lines.append(f"INDSTRY: {industry}\n")
                lines.append(f"CNTRY  : {country}\n\n")

            summary = info.get('longBusinessSummary') or info.get('description', 'No description available.')
            lines.append(str(summary) + "\n")
        else:
            lines.append("No additional information available.\n")

        self.desc_text.insert('1.0', ''.join(lines))
        self.desc_text.config(state='disabled')

    def update_ui(self, symbol, predictions, df, current_price):
        card_mapping = {1: "1 Day", 7: "1 Week", 30: "1 Month", 90: "3 Months"}

        for days, label in card_mapping.items():
            if days in predictions and predictions[days][0] is not None:
                pred_price = predictions[days][0]
                self.pred_labels[label].config(text=self.fmt_price(pred_price))

        if 1 in predictions and predictions[1][1] is not None:
            accuracy = predictions[1][1]
            self.stat_labels["Accuracy"].config(text=f"{accuracy:.1f}%")

        if 1 in predictions and predictions[1][0] is not None:
            pred_price = predictions[1][0]
            self.stat_labels["Predicted (1D)"].config(text=self.fmt_price(pred_price))

        self.plot_chart(symbol, df, predictions)
        self.update_summary(current_price, predictions, card_mapping)

    def update_summary(self, current_price, predictions, card_mapping):
        self.summary_text.config(state='normal')
        self.summary_text.delete('1.0', tk.END)

        parts = [f"SPOT {self.fmt_price(current_price)}  |  "]
        for days, label in card_mapping.items():
            if days in predictions and predictions[days][0] is not None:
                pred_price = predictions[days][0]
                try:
                    change_pct = ((float(pred_price) - float(current_price)) / float(current_price)) * 100
                except Exception:
                    continue
                arrow = "↑" if change_pct >= 0 else "↓"
                parts.append(
                    f"{label}: {self.fmt_price(pred_price)} ({arrow}{abs(change_pct):.2f}%)  |  "
                )

        self.summary_text.insert('1.0', ''.join(parts))
        self.summary_text.config(state='disabled')

    def plot_chart(self, symbol, df, predictions):
        self.ax.clear()
        recent_df = df.tail(180).copy()  # Longer context window

        # --- 1) Plot historical close price ---
        self.ax.plot(
            recent_df.index,
            recent_df['Close'],
            label='HIST',
            color=self.colors['accent_blue'],
            linewidth=1.8,
            marker='',
        )

        # --- 2) Build forecast points on time axis ---
        last_date = df.index[-1]
        last_close = float(df['Close'].iloc[-1])
        pred_days = [1, 7, 30, 90]
        pred_dates = []
        pred_prices = []

        for d in pred_days:
            if d in predictions and predictions[d][0] is not None:
                pred_dates.append(last_date + timedelta(days=d))
                pred_prices.append(float(predictions[d][0]))

        # --- 3) Plot forecast path + confidence-style band ---
        if pred_prices:
            x_full = [last_date] + pred_dates
            y_full = [last_close] + pred_prices

            # Forecast line
            self.ax.plot(
                x_full,
                y_full,
                label='FORECAST',
                color=self.colors['accent_green'],
                linewidth=2.2,
                marker='o',
                markersize=4,
                linestyle='--'
            )

            # Subtle forecast cone (±2%)
            y_lower = [y * 0.98 for y in y_full]
            y_upper = [y * 1.02 for y in y_full]
            self.ax.fill_between(
                x_full,
                y_lower,
                y_upper,
                color=self.colors['accent_green'],
                alpha=0.10,
                linewidth=0
            )

            # Vertical marker at forecast horizon end
            self.ax.axvline(
                x_full[-1],
                color=self.colors['muted'],
                linestyle=':',
                linewidth=1
            )

        # --- 4) Axis, grid, and legend styling ---
        self.ax.set_title(
            f"{symbol}  |  {self.model_var.get()} FORECAST",
            fontsize=12,
            fontweight='bold',
            loc='left',
            pad=10
        )

        self.ax.set_xlabel('DATE', fontsize=9, labelpad=6)
        ccy = self.current_currency or ""
        self.ax.set_ylabel(f"PRICE ({ccy})" if ccy else "PRICE", fontsize=9, labelpad=6)

        # Tight x‑limits around visible data
        self.ax.set_xlim(recent_df.index.min(), (last_date + timedelta(days=95)))

        # Light grid
        self.ax.grid(
            True,
            which='major',
            axis='both',
            linestyle='--',
            linewidth=0.6,
            alpha=0.35
        )

        # Legend with semi‑transparent box
        leg = self.ax.legend(
            loc='upper left',
            fontsize=8,
            frameon=True
        )
        leg.get_frame().set_edgecolor('#333333')
        leg.get_frame().set_alpha(0.7)

        # Remove top/right spines for cleaner look
        for spine in ['top', 'right']:
            self.ax.spines[spine].set_visible(False)

        self.fig.tight_layout()
        self.fig.autofmt_xdate()
        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = TerminalPricePredictorApp(root)
    root.mainloop()
