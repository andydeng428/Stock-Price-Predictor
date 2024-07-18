import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import argparse
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Function to fetch the stock data
def fetch_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data['Close'].values.reshape(-1, 1)  # Only use close prices

# Prepare the data for LSTM model
def prepare_data(data, n_steps):
    X, y = [], []
    for i in range(n_steps, len(data)):
        X.append(data[i-n_steps:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

# Function to evaluate the model
def evaluate_model(model, X_test, y_test, scaler):
    predicted_stock_price = model.predict(X_test)
    predicted_stock_price = scaler.inverse_transform(predicted_stock_price)
    real_stock_price = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    mse = mean_squared_error(real_stock_price, predicted_stock_price)
    print(f'Mean Squared Error: {mse}')
    
    plt.plot(real_stock_price, color='red', label='Real Stock Price')
    plt.plot(predicted_stock_price, color='blue', label='Predicted Stock Price')
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

# Function to plot training loss
def plot_training_history(history):
    plt.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Function to predict future stock prices
def predict_future_prices(model, data, n_steps, n_days, scaler):
    future_prices = []
    input_seq = data[-n_steps:]
    
    for _ in range(n_days):
        input_seq = input_seq.reshape(1, n_steps, 1)
        predicted_price = model.predict(input_seq)
        future_prices.append(predicted_price[0, 0])
        input_seq = np.append(input_seq[:, 1:, :], predicted_price.reshape(1, 1, 1), axis=1)
    
    future_prices = scaler.inverse_transform(np.array(future_prices).reshape(-1, 1))
    return future_prices

# Main function
def main(ticker, start_date, end_date, n_steps, epochs, batch_size, future_days):
    # Fetch data
    data = fetch_stock_data(ticker, start_date, end_date)
    
    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    
    # Prepare data
    X, y = prepare_data(data_scaled, n_steps)
    X = X.reshape(X.shape[0], X.shape[1], 1)  # Reshape for LSTM [samples, time steps, features]
    
    # Split into training and testing datasets
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Build the LSTM model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(n_steps, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train the model
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    
    # Evaluate the model
    evaluate_model(model, X_test, y_test, scaler)
    
    # Plot training history
    plot_training_history(history)
    
    # Predict future prices
    future_prices = predict_future_prices(model, data_scaled, n_steps, future_days, scaler)
    print(f'Predicted future prices for the next {future_days} days: {future_prices.flatten()}')
    
    plt.plot(data, color='blue', label='Historical Prices')
    future_dates = pd.date_range(start=end_date, periods=future_days + 1)[1:]
    plt.plot(future_dates, future_prices, color='green', label='Future Prices')
    plt.title('Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stock Price Prediction using LSTM')
    parser.add_argument('--ticker', type=str, required=True, help='Stock ticker symbol')
    parser.add_argument('--start_date', type=str, required=True, help='Start date for historical data (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, required=True, help='End date for historical data (YYYY-MM-DD)')
    parser.add_argument('--n_steps', type=int, default=50, help='Number of steps for LSTM input')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--future_days', type=int, default=10, help='Number of days to predict into the future')
    args = parser.parse_args()
    
    main(args.ticker, args.start_date, args.end_date, args.n_steps, args.epochs, args.batch_size, args.future_days)
