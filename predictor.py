import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import threading
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings('ignore')

try:
    import yfinance as yf

    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False
    print("ERROR: yfinance not installed. Run: pip install yfinance")

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    import tensorflow as tf

    tf.get_logger().setLevel('ERROR')
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    from prophet import Prophet
    import logging

    logging.getLogger('prophet').setLevel(logging.ERROR)
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

try:
    from statsmodels.tsa.arima.model import ARIMA

    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False

plt.style.use('seaborn-v0_8-whitegrid')


class PredictionModels:
    def __init__(self, seq_length=60):
        self.seq_length = seq_length
        self.scaler = MinMaxScaler()

    def prepare_data(self, prices):
        try:
            scaled_data = self.scaler.fit_transform(prices.reshape(-1, 1))
            X, y = [], []
            for i in range(self.seq_length, len(scaled_data)):
                X.append(scaled_data[i - self.seq_length:i, 0])
                y.append(scaled_data[i, 0])
            X, y = np.array(X), np.array(y)
            X = X.reshape((X.shape[0], X.shape[1], 1))
            return X, y, scaled_data
        except Exception as e:
            print(f"Data preparation error: {e}")
            return None, None, None

    def predict_lstm(self, prices, days_ahead=1):
        if not TF_AVAILABLE or len(prices) < self.seq_length + 30:
            return None, 0
        try:
            X, y, scaled_data = self.prepare_data(prices)
            if X is None:
                return None, 0
            split = int(0.8 * len(X))
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]
            model = Sequential([
                LSTM(64, return_sequences=True, input_shape=(X.shape[1], 1)),
                Dropout(0.2),
                LSTM(32),
                Dropout(0.2),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')
            early_stop = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
            model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1,
                      callbacks=[early_stop], verbose=0)
            predictions = []
            current_seq = scaled_data[-self.seq_length:].reshape(1, self.seq_length, 1)
            for _ in range(days_ahead):
                pred = model.predict(current_seq, verbose=0)
                predictions.append(pred[0, 0])
                current_seq = np.append(current_seq[:, 1:, :], pred.reshape(1, 1, 1), axis=1)
            test_pred = model.predict(X_test, verbose=0)
            mape = np.mean(np.abs((y_test - test_pred.flatten()) / (y_test + 1e-8))) * 100
            accuracy = max(0, min(100, 100 - mape))
            predictions = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
            return predictions[-1][0], accuracy
        except Exception as e:
            print(f"LSTM error: {e}")
            return None, 0

    def predict_gru(self, prices, days_ahead=1):
        if not TF_AVAILABLE or len(prices) < self.seq_length + 30:
            return None, 0
        try:
            X, y, scaled_data = self.prepare_data(prices)
            if X is None:
                return None, 0
            split = int(0.8 * len(X))
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]
            model = Sequential([
                GRU(64, return_sequences=True, input_shape=(X.shape[1], 1)),
                Dropout(0.2),
                GRU(32),
                Dropout(0.2),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')
            early_stop = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
            model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1,
                      callbacks=[early_stop], verbose=0)
            predictions = []
            current_seq = scaled_data[-self.seq_length:].reshape(1, self.seq_length, 1)
            for _ in range(days_ahead):
                pred = model.predict(current_seq, verbose=0)
                predictions.append(pred[0, 0])
                current_seq = np.append(current_seq[:, 1:, :], pred.reshape(1, 1, 1), axis=1)
            test_pred = model.predict(X_test, verbose=0)
            mape = np.mean(np.abs((y_test - test_pred.flatten()) / (y_test + 1e-8))) * 100
            accuracy = max(0, min(100, 100 - mape))
            predictions = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
            return predictions[-1][0], accuracy
        except Exception as e:
            print(f"GRU error: {e}")
            return None, 0

    def predict_prophet(self, df, days_ahead=1):
        if not PROPHET_AVAILABLE or len(df) < 30:
            return None, 0
        try:
            prophet_df = pd.DataFrame({'ds': df.index, 'y': df['Close'].values})
            model = Prophet(daily_seasonality=True, yearly_seasonality=True,
                            changepoint_prior_scale=0.05)
            model.fit(prophet_df)
            future = model.make_future_dataframe(periods=days_ahead)
            forecast = model.predict(future)
            train_size = int(0.8 * len(prophet_df))
            model_test = Prophet(changepoint_prior_scale=0.05)
            model_test.fit(prophet_df[:train_size])
            test_forecast = model_test.predict(prophet_df[train_size:][['ds']])
            actual = prophet_df[train_size:]['y'].values
            predicted = test_forecast['yhat'].values
            mape = np.mean(np.abs((actual - predicted) / (actual + 1e-8))) * 100
            accuracy = max(0, min(100, 100 - mape))
            return forecast['yhat'].iloc[-1], accuracy
        except Exception as e:
            print(f"Prophet error: {e}")
            return None, 0

    def predict_arima(self, prices, days_ahead=1):
        if not ARIMA_AVAILABLE or len(prices) < 50:
            return None, 0
        try:
            model = ARIMA(prices, order=(5, 1, 2))
            fitted = model.fit()
            forecast = fitted.forecast(steps=days_ahead)
            train_size = int(0.8 * len(prices))
            model_test = ARIMA(prices[:train_size], order=(5, 1, 2))
            fitted_test = model_test.fit()
            predictions = fitted_test.forecast(steps=len(prices) - train_size)
            actual = prices[train_size:]
            mape = np.mean(np.abs((actual - predictions) / (actual + 1e-8))) * 100
            accuracy = max(0, min(100, 100 - mape))
            return forecast.iloc[-1] if hasattr(forecast, 'iloc') else forecast[-1], accuracy
        except Exception as e:
            print(f"ARIMA error: {e}")
            return None, 0

    def predict_ensemble(self, prices, df, days_ahead=1):
        predictions, accuracies = [], []
        lstm_pred, lstm_acc = self.predict_lstm(prices, days_ahead)
        if lstm_pred is not None:
            predictions.append(lstm_pred)
            accuracies.append(lstm_acc)
        gru_pred, gru_acc = self.predict_gru(prices, days_ahead)
        if gru_pred is not None:
            predictions.append(gru_pred)
            accuracies.append(gru_acc)
        prophet_pred, prophet_acc = self.predict_prophet(df, days_ahead)
        if prophet_pred is not None:
            predictions.append(prophet_pred)
            accuracies.append(prophet_acc)
        arima_pred, arima_acc = self.predict_arima(prices, days_ahead)
        if arima_pred is not None:
            predictions.append(arima_pred)
            accuracies.append(arima_acc)
        if not predictions:
            return None, 0
        weights = np.array(accuracies) / sum(accuracies)
        ensemble_pred = np.average(predictions, weights=weights)
        ensemble_acc = np.mean(accuracies)
        return ensemble_pred, ensemble_acc


class SimplePricePredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Price Predictor")
        self.root.configure(bg='#f5f5f5')
        self.root.geometry("1400x800")
        self.predictor = PredictionModels(60)
        self.current_data = None

        if not YF_AVAILABLE:
            messagebox.showerror("Missing Library",
                                 "yfinance library not installed!\n\n"
                                 "Please run: pip install yfinance")
            self.root.destroy()
            return
        self.create_widgets()

    def create_widgets(self):
        # Top control bar
        control_frame = tk.Frame(self.root, bg='#ffffff', pady=12, relief=tk.FLAT)
        control_frame.pack(fill=tk.X, padx=15, pady=(10, 5))

        tk.Label(control_frame, text="Symbol:", bg='#ffffff', fg='#1a2027',
                 font=('Segoe UI', 11, 'bold')).pack(side=tk.LEFT, padx=(10, 5))
        self.symbol_entry = tk.Entry(control_frame, font=('Segoe UI', 11), width=12, bg='#f4f6fa')
        self.symbol_entry.insert(0, "AAPL")
        self.symbol_entry.pack(side=tk.LEFT, padx=3)

        tk.Label(control_frame, text="Type:", bg='#ffffff', fg='#1a2027',
                 font=('Segoe UI', 11, 'bold')).pack(side=tk.LEFT, padx=(15, 5))
        self.asset_var = tk.StringVar(value="Stock")
        ttk.Combobox(control_frame, textvariable=self.asset_var,
                     values=["Stock", "Crypto"], width=10, state='readonly',
                     font=('Segoe UI', 10)).pack(side=tk.LEFT, padx=3)

        tk.Label(control_frame, text="Model:", bg='#ffffff', fg='#1a2027',
                 font=('Segoe UI', 11, 'bold')).pack(side=tk.LEFT, padx=(15, 5))
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
        ttk.Combobox(control_frame, textvariable=self.model_var,
                     values=models_available, width=12, state='readonly',
                     font=('Segoe UI', 10)).pack(side=tk.LEFT, padx=3)

        self.predict_btn = tk.Button(control_frame, text="🔮 Predict",
                                     command=self.start_prediction, font=('Segoe UI', 11, 'bold'),
                                     bg='#2196F3', fg='white', padx=18, pady=5,
                                     cursor='hand2', activebackground='#1976d2')
        self.predict_btn.pack(side=tk.LEFT, padx=15)

        self.status_label = tk.Label(control_frame, text="Ready", bg='#ffffff',
                                     fg='#009688', font=('Segoe UI', 10, 'bold'))
        self.status_label.pack(side=tk.LEFT, padx=8)

        # Main container (2 columns)
        main_container = tk.Frame(self.root, bg='#f5f5f5')
        main_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=5)

        # LEFT COLUMN: Stock Info + Stats
        left_col = tk.Frame(main_container, bg='#f5f5f5', width=450)
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 8))
        left_col.pack_propagate(False)

        # Stock Description
        desc_frame = tk.Frame(left_col, bg='#ffffff', relief=tk.RIDGE, bd=1)
        desc_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        tk.Label(desc_frame, text="Asset Information", bg='#ffffff', fg='#263238',
                 font=('Segoe UI', 12, 'bold'), anchor='w').pack(fill=tk.X, padx=12, pady=(10, 5))

        self.desc_text = scrolledtext.ScrolledText(desc_frame, height=10, bg='#f8f9fa',
                                                   fg='#1a2027', font=('Segoe UI', 10),
                                                   relief=tk.FLAT, padx=10, pady=8,
                                                   wrap='word', state='disabled')
        self.desc_text.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 10))

        # Stats cards
        stats_frame = tk.Frame(left_col, bg='#f5f5f5')
        stats_frame.pack(fill=tk.X)

        self.stat_labels = {}
        stats = [("Current Price", '#1976d2'), ("24h Change", '#fb8c00'),
                 ("Predicted (1D)", '#388e3c'), ("Accuracy", '#7c388e')]

        for i, (stat, color) in enumerate(stats):
            row = i // 2
            col = i % 2
            card = tk.Frame(stats_frame, bg='#ffffff', relief=tk.RIDGE, bd=1)
            card.grid(row=row, column=col, padx=4, pady=4, sticky='nsew')
            stats_frame.grid_columnconfigure(col, weight=1)

            tk.Label(card, text=stat, bg='#ffffff', fg='#888888',
                     font=('Segoe UI', 9)).pack(pady=(6, 1))
            label = tk.Label(card, text="--", bg='#ffffff', fg=color,
                             font=('Segoe UI', 16, 'bold'))
            label.pack(pady=(1, 6))
            self.stat_labels[stat] = label

        # Prediction cards
        pred_frame = tk.Frame(left_col, bg='#ffffff', relief=tk.RIDGE, bd=1)
        pred_frame.pack(fill=tk.X, pady=(8, 0))

        tk.Label(pred_frame, text="Future Targets", bg='#ffffff', fg='#263238',
                 font=('Segoe UI', 11, 'bold')).pack(pady=(8, 5))

        pred_grid = tk.Frame(pred_frame, bg='#ffffff')
        pred_grid.pack(fill=tk.X, padx=10, pady=(0, 8))

        self.pred_labels = {}
        pred_times = ["1 Day", "1 Week", "1 Month", "3 Months"]

        for i, time in enumerate(pred_times):
            row = i // 2
            col = i % 2
            card = tk.Frame(pred_grid, bg='#f4f6fa', relief=tk.FLAT)
            card.grid(row=row, column=col, padx=3, pady=3, sticky='nsew')
            pred_grid.grid_columnconfigure(col, weight=1)

            tk.Label(card, text=time, bg='#f4f6fa', fg='#7b7b7b',
                     font=('Segoe UI', 9)).pack(pady=(5, 1))
            label = tk.Label(card, text="--", bg='#f4f6fa', fg='#388e3c',
                             font=('Segoe UI', 13, 'bold'))
            label.pack(pady=(1, 5))
            self.pred_labels[time] = label

        # RIGHT COLUMN: Chart
        right_col = tk.Frame(main_container, bg='#ffffff', relief=tk.RIDGE, bd=1)
        right_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tk.Label(right_col, text="Price Chart & Predictions", bg='#ffffff', fg='#263238',
                 font=('Segoe UI', 12, 'bold'), anchor='w').pack(fill=tk.X, padx=12, pady=(10, 5))

        self.fig = Figure(figsize=(8, 5), facecolor='#ffffff', dpi=90)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('#ffffff')

        self.canvas = FigureCanvasTkAgg(self.fig, master=right_col)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # BOTTOM: Summary
        summary_frame = tk.Frame(self.root, bg='#ffffff', relief=tk.RIDGE, bd=1)
        summary_frame.pack(fill=tk.X, padx=15, pady=(5, 10))

        tk.Label(summary_frame, text="Prediction Summary", bg='#ffffff', fg='#263238',
                 font=('Segoe UI', 11, 'bold')).pack(pady=(8, 3))

        self.summary_text = tk.Text(summary_frame, height=4, bg='#f8f9fa', fg='#1a2027',
                                    font=('Courier New', 10), relief=tk.FLAT, padx=12, pady=8,
                                    state='disabled', wrap='word')
        self.summary_text.pack(fill=tk.X, padx=12, pady=(0, 8))

    def start_prediction(self):
        self.predict_btn.config(state=tk.DISABLED, text="⏳ Working...")
        self.status_label.config(text="Fetching data...", fg='#fb8c00')
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

            self.root.after(0, lambda: self.status_label.config(text="Loading data..."))
            ticker = yf.Ticker(symbol) if asset_type == "Stock" else yf.Ticker(f"{symbol}-USD")
            df = ticker.history(period="1y")
            info = ticker.info

            if df.empty:
                self.root.after(0, lambda: messagebox.showerror("Error", f"No data for {symbol}"))
                return

            self.current_data = df
            prices = df['Close'].values
            current_price = prices[-1]
            prev_price = prices[-2]
            change_pct = ((current_price - prev_price) / prev_price) * 100

            # Update stock description
            self.root.after(0, lambda: self.update_description(symbol, info, asset_type))

            self.root.after(0, lambda: self.stat_labels["Current Price"].config(text=f"${current_price:.2f}"))
            self.root.after(0, lambda: self.stat_labels["24h Change"].config(
                text=f"{change_pct:+.2f}%", fg="#388e3c" if change_pct >= 0 else "#c62828"
            ))

            self.root.after(0, lambda: self.status_label.config(text="Predicting...", fg="#1976d2"))

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
            self.root.after(0, lambda: self.status_label.config(text="✅ Complete!", fg='#388e3c'))

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            self.root.after(0, lambda: messagebox.showerror("Prediction Failed", error_msg))
            self.root.after(0, lambda: self.status_label.config(text="❌ Error", fg='#b71c1c'))
        finally:
            self.root.after(0, lambda: self.predict_btn.config(state=tk.NORMAL, text="🔮 Predict"))

    def update_description(self, symbol, info, asset_type):
        """Display stock/crypto description."""
        self.desc_text.config(state='normal')
        self.desc_text.delete('1.0', tk.END)

        desc_lines = []
        desc_lines.append(f"Symbol: {symbol}\n")
        desc_lines.append(f"Type: {asset_type}\n")
        desc_lines.append("-" * 50 + "\n\n")

        if info:
            name = info.get('longName') or info.get('shortName') or symbol
            desc_lines.append(f"Name: {name}\n\n")

            if asset_type == "Stock":
                sector = info.get('sector', 'N/A')
                industry = info.get('industry', 'N/A')
                country = info.get('country', 'N/A')
                desc_lines.append(f"Sector: {sector}\n")
                desc_lines.append(f"Industry: {industry}\n")
                desc_lines.append(f"Country: {country}\n\n")

            summary = info.get('longBusinessSummary') or info.get('description', 'No description available.')
            desc_lines.append(f"{summary}\n")
        else:
            desc_lines.append("No additional information available.\n")

        self.desc_text.insert('1.0', ''.join(desc_lines))
        self.desc_text.config(state='disabled')

    def update_ui(self, symbol, predictions, df, current_price):
        card_mapping = {1: "1 Day", 7: "1 Week", 30: "1 Month", 90: "3 Months"}

        for days, label in card_mapping.items():
            if days in predictions and predictions[days][0] is not None:
                pred_price = predictions[days][0]
                self.pred_labels[label].config(text=f"${pred_price:.2f}")

        if 1 in predictions and predictions[1][1] is not None:
            accuracy = predictions[1][1]
            self.stat_labels["Accuracy"].config(text=f"{accuracy:.1f}%")

        if 1 in predictions and predictions[1][0] is not None:
            pred_price = predictions[1][0]
            self.stat_labels["Predicted (1D)"].config(text=f"${pred_price:.2f}")

        self.plot_chart(symbol, df, predictions)
        self.update_summary(current_price, predictions, card_mapping)

    def update_summary(self, current_price, predictions, card_mapping):
        """Display prediction summary."""
        self.summary_text.config(state='normal')
        self.summary_text.delete('1.0', tk.END)

        summary_lines = []
        summary_lines.append(f"Current: ${current_price:.2f}  |  ")

        for days, label in card_mapping.items():
            if days in predictions and predictions[days][0] is not None:
                pred_price = predictions[days][0]
                change_pct = ((pred_price - current_price) / current_price) * 100
                arrow = "↑" if change_pct >= 0 else "↓"
                summary_lines.append(f"{label}: ${pred_price:.2f} ({arrow}{abs(change_pct):.2f}%)  |  ")

        self.summary_text.insert('1.0', ''.join(summary_lines))
        self.summary_text.config(state='disabled')

    def plot_chart(self, symbol, df, predictions):
        self.ax.clear()
        recent_df = df.tail(60)

        self.ax.plot(recent_df.index, recent_df['Close'],
                     label='Historical', color='#1976d2', linewidth=2, marker='o',
                     markersize=3, alpha=0.8)

        last_date = df.index[-1]
        pred_days = [1, 7, 30, 90]
        pred_dates = [last_date + timedelta(days=d) for d in pred_days]
        pred_prices = [predictions[d][0] for d in pred_days
                       if d in predictions and predictions[d][0] is not None]

        if pred_prices:
            xx = [df.index[-1]] + pred_dates[:len(pred_prices)]
            yy = [df['Close'].iloc[-1]] + pred_prices
            self.ax.plot(xx, yy, label='Predicted', color='#388e3c', linewidth=2,
                         marker='s', markersize=5, linestyle="--", alpha=0.9)
            self.ax.fill_between(xx, [y * 0.98 for y in yy], [y * 1.02 for y in yy],
                                 color="#81c784", alpha=0.12)

        self.ax.set_title(f"{symbol} - {self.model_var.get()} Model",
                          fontsize=13, color='#263238', fontweight='bold', pad=10)
        self.ax.set_xlabel('Date', fontsize=10, color='#333333')
        self.ax.set_ylabel('Price ($)', fontsize=10, color='#333333')
        self.ax.grid(which='both', color='#e3e7ef', linestyle='-', linewidth=0.8, alpha=0.6)
        self.ax.legend(loc='upper left', fontsize=9, frameon=True, facecolor='white')
        plt.setp(self.ax.get_xticklabels(), fontsize=8, color="#29374f")
        plt.setp(self.ax.get_yticklabels(), fontsize=8, color="#29374f")
        self.fig.tight_layout()
        self.fig.autofmt_xdate()
        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = SimplePricePredictorApp(root)
    root.mainloop()
