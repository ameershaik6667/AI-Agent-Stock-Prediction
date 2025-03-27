import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yfinance as yf
from crewai import Agent, Crew, Task
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Define the technical indicators

def extract_stock_symbol(user_input):
    """Extract stock symbol from user input."""
    words = user_input.split()
    for word in words:
        if word.isupper() and len(word) <= 5:
            return {"symbol": word}
    return None

def calculate_indicators(df):
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['RSI'] = compute_rsi(df['Close'], window=14)
    df['MACD'], df['Signal_Line'] = compute_macd(df['Close'])
    df['Upper_BB'], df['Lower_BB'] = compute_bollinger_bands(df['Close'])
    df.dropna(inplace=True)
    return df

def compute_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_macd(series, short=12, long=26, signal=9):
    short_ema = series.ewm(span=short, adjust=False).mean()
    long_ema = series.ewm(span=long, adjust=False).mean()
    macd = short_ema - long_ema
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def compute_bollinger_bands(series, window=20, num_std=2):
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    upper_band = rolling_mean + (num_std * rolling_std)
    lower_band = rolling_mean - (num_std * rolling_std)
    return upper_band, lower_band

# Download and prepare data
def get_stock_data(ticker, years=5):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)
    df = yf.download(ticker, start=start_date, end=end_date)
    df.reset_index(inplace=True)
    df = calculate_indicators(df)
    return df

# Prepare dataset for RNN
def prepare_data(df, lookback=6):
    features = ['Close', 'MA_10', 'RSI', 'MACD', 'Signal_Line', 'Upper_BB', 'Lower_BB']
    scaler = MinMaxScaler(feature_range=(-1, 1))
    df_scaled = scaler.fit_transform(df[features])
    x_data, y_data = [], []
    for i in range(lookback, len(df_scaled)):
        x_data.append(df_scaled[i - lookback:i])
        y_data.append(df_scaled[i, 0])
    return np.array(x_data), np.array(y_data), scaler

# Define RNN model
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size=20, num_layers=2):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), 20).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

# Training function
def train_model(model, train_loader, val_loader, epochs=50, lr=5e-4):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(x_batch)
            loss = loss_fn(output, y_batch.view(-1, 1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch+1}, Training Loss: {avg_loss:.5f}')
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'best_model.pth')

# Prediction function
def predict_next_day(model, last_days, scaler):
    model.eval()
    with torch.no_grad():
        last_days = torch.tensor(last_days, dtype=torch.float32).unsqueeze(0).to(device)
        prediction = model(last_days).cpu().numpy()

        # Ensure prediction is 2D
        prediction = prediction.reshape(1, -1)

        # Create a correctly shaped dummy input
        dummy_input = np.zeros((1, scaler.n_features_in_))
        dummy_input[:, 0] = prediction

        # Apply inverse transformation
        inverse_pred = scaler.inverse_transform(dummy_input)[:, 0]

        return inverse_pred[0]

# Backtesting function
def backtest(model, test_data, actual_prices, scaler):
    predictions = []
    for i in range(len(test_data)):
        pred = predict_next_day(model, test_data[i], scaler)
        predictions.append(pred)
    return predictions

# Forward test function for future predictions
def forward_test(model, ticker, lookback=6):
    df = get_stock_data(ticker)
    x_data, _, scaler = prepare_data(df, lookback)
    last_days = x_data[-1]
    return predict_next_day(model, last_days, scaler)

# Generate a structured prompt with stock data
def generate_prompt(ticker, model, scaler, lookback=6):
    df = get_stock_data(ticker)
    x_data, _, scaler = prepare_data(df, lookback)
    last_days = x_data[-1]
    predicted_price = predict_next_day(model, last_days, scaler)

    # Ensure we extract single numerical values
    latest_data = df.iloc[-1]
    ma_10 = latest_data['MA_10'].item()
    rsi = latest_data['RSI'].item()
    macd = latest_data['MACD'].item()
    signal_line = latest_data['Signal_Line'].item()
    upper_bb = latest_data['Upper_BB'].item()
    lower_bb = latest_data['Lower_BB'].item()
    close_price = latest_data['Close'].item()

    prompt = f"""
    Stock Analysis for {ticker}:

    - **Current Closing Price:** {close_price:.2f}
    - **Predicted Next-Day Price:** {predicted_price:.2f}

    **Technical Indicators:**
    - Moving Average (10-day): {ma_10:.2f}
    - Relative Strength Index (RSI): {rsi:.2f}
    - MACD: {macd:.2f}
    - Signal Line: {signal_line:.2f}
    - Bollinger Bands:
      - Upper Band: {upper_bb:.2f}
      - Lower Band: {lower_bb:.2f}
    """
    return prompt