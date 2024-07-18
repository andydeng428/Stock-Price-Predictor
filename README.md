# Stock-Price-Predictor

This project uses Long Short-Term Memory (LSTM) neural networks to predict future stock prices based on historical data. The LSTM model is implemented using TensorFlow and Keras, and data is visualized using Matplotlib.

# Overview #

## Features 
* Data Fetching: Retrieves historical stock price data using Yahoo Finance.
* Data Preprocessing: Scales and prepares the data for LSTM input.
* LSTM Model: Builds and trains an LSTM model to forecast stock prices.
* Evaluation: Measures the error to evaluate model performance and visualizes the results.
* Future Prediction: Predicts stock prices for a specified number of future days.

## Technologies Used
* TensorFlow/Keras: For building and training the LSTM model.
* Matplotlib: For data visualization.
* yfinance: For fetching historical stock data.
* scikit-learn: For data scaling and evaluation metrics.

![image](https://github.com/user-attachments/assets/166aee21-c1fb-46a5-b0a0-7b495b0af893)

# Getting Started

Use the following command to install the required packages:
```
pip install numpy pandas matplotlib yfinance tensorflow scikit-learn
```
Use the command-line interface to specify parameters and execute the script:
```
python stock_prediction.py --ticker <TICKER> --start_date <START_DATE> --end_date <END_DATE> --n_steps <N_STEPS> --epochs <EPOCHS> --batch_size <BATCH_SIZE> --future_days <FUTURE_DAYS>
```
Arguments:
* --ticker: Stock ticker symbol (e.g., NVDA).
* --start_date: Start date for historical data in YYYY-MM-DD format.
* --end_date: End date for historical data in YYYY-MM-DD format.
* --n_steps: Number of time steps for LSTM input (default: 50).
* --epochs: Number of epochs for training the model (default: 50).
* --batch_size: Batch size for training (default: 32).
* --future_days: Number of future days to predict (default: 10).
