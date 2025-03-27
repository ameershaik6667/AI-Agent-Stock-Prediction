import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from UI.main import (RNNModel, backtest, generate_prompt, get_stock_data,
                     predict_next_day, prepare_data)


# Load trained model
def load_model(model_path, input_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RNNModel(input_size=input_size).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model, device

# Run backtest
def run_backtest(model, ticker, lookback=6):
    stock_data = get_stock_data(ticker)
    x_data, _, scaler = prepare_data(stock_data, lookback)

    test_size = int(0.2 * len(x_data))  # 20% of data for testing
    test_data = x_data[-test_size:]  
    actual_prices = stock_data['Close'].values[-test_size:]

    # Generate predictions
    backtest_predictions = backtest(model, test_data, actual_prices, scaler)

    # Print Backtesting Results
    print("\nBacktesting Results (Actual vs Predicted):")
    for i in range(len(backtest_predictions)):
        actual = float(actual_prices[i])
        predicted = float(backtest_predictions[i])
        print(f"Day {i+1}: Actual: {actual:.2f}, Predicted: {predicted:.2f}")

# Run forward test
def run_forward_test(model, ticker, lookback=6):
    df = get_stock_data(ticker)
    x_data, _, scaler = prepare_data(df, lookback)
    last_days = x_data[-1]
    last_days = last_days.astype(np.float32)
    predicted_price = predict_next_day(model, last_days, scaler)
    print(f"\nPredicted Next-Day Price for {ticker}: {predicted_price:.2f}")
    return predicted_price

# Main execution
if __name__ == "__main__":
    ticker = "AAPL"
    model_path = "best_model.pth"

    # Load sample data to get input size
    stock_data = get_stock_data(ticker)
    x_data, _, _ = prepare_data(stock_data)
    input_size = x_data.shape[2]

    # Load trained model
    model, device = load_model(model_path, input_size)

    # Run Backtesting
    run_backtest(model, ticker)

    # Run Forward Testing
    predicted_price = run_forward_test(model, ticker)

    # Generate and print structured prompt
    prompt = generate_prompt(ticker, model, None)
    print(prompt)
