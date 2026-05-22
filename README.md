# Price Prediction with Advanced ML Models

A comprehensive Python desktop application for predicting asset prices using advanced machine learning and time-series models.

## Overview

This application leverages multiple time-series forecasting models including LSTM, GRU, and ARIMA to predict stock and cryptocurrency prices. It features a modern Tkinter-based GUI with real-time data visualization and multi-horizon forecasting capabilities.

## Features

- **Multiple ML Models**: Implementation of LSTM, GRU, and ARIMA models for time-series prediction
- **Real-time Data Integration**: Live price data fetching using yfinance API
- **Interactive Visualizations**: Dynamic charts showing historical and predicted price trends
- **Ensemble Predictions**: Combined model predictions for enhanced accuracy
- **Data Preprocessing**: Automated data cleaning, scaling, and feature engineering
- **Custom Neural Networks**: Early stopping and MAPE-based model evaluation
- **Multi-threaded Architecture**: Responsive UI with background data processing
- **Multi-horizon Forecasts**: Short-term and long-term price predictions

## Tech Stack

- **Language**: Python 3.8+
- **GUI Framework**: Tkinter
- **Deep Learning**: TensorFlow, Keras
- **Time Series**: Prophet, Statsmodels, pmdarima
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib
- **Financial Data**: yfinance

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository
```bash
git clone https://github.com/abhang-a1/PRICE-PREDICTION-WITH-ADVANCED-ML-MODELS.git
cd PRICE-PREDICTION-WITH-ADVANCED-ML-MODELS
```

2. Install required dependencies
```bash
pip install -r requirements.txt
```

3. Run the application
```bash
python predictor.py
```

## Usage

1. Launch the application
2. Enter the ticker symbol for the asset you want to analyze
3. Select the prediction model (LSTM, GRU, ARIMA, or Ensemble)
4. Choose the forecast horizon
5. Click "Predict" to generate forecasts
6. View interactive visualizations and prediction statistics

## Project Structure

```
├── predictor.py          # Main application file
├── config.py            # Configuration settings
├── requirements.txt     # Project dependencies
└── README.md           # Project documentation
```

## Models

### LSTM (Long Short-Term Memory)
Deep learning model designed for sequence prediction with memory cells that can learn long-term dependencies.

### GRU (Gated Recurrent Unit)
Simplified variant of LSTM with fewer parameters, offering faster training while maintaining prediction accuracy.

### ARIMA (AutoRegressive Integrated Moving Average)
Statistical model for time-series forecasting based on autoregression and moving averages.

### Ensemble
Combines predictions from multiple models using weighted averaging for improved accuracy and robustness.

## Performance Metrics

- Mean Absolute Percentage Error (MAPE)
- Root Mean Square Error (RMSE)
- Mean Absolute Error (MAE)
- R-squared Score

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Author

Abhang A1

## Acknowledgments

- yfinance for providing free financial data API
- TensorFlow and Keras teams for deep learning frameworks
- Prophet and Statsmodels for time-series analysis tools
