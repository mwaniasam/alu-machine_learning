# Time Series Forecasting - BTC Price Prediction

## Description
This project uses Recurrent Neural Networks (RNNs) to forecast Bitcoin (BTC)
closing prices. Given the past 24 hours of BTC data, the model predicts the
closing price of the following hour.

## Requirements
- Python 3.5
- numpy 1.15
- tensorflow 2.4.1
- pandas 1.1.5

## Datasets
- `coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv`
- `bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv`

Each row represents a 60-second time window with:
- Timestamp (Unix time)
- Open price (USD)
- High price (USD)
- Low price (USD)
- Close price (USD)
- Volume in BTC
- Volume in Currency (USD)
- Weighted average price (USD)

## Files
| File | Description |
|------|-------------|
| `preprocess_data.py` | Cleans, resamples, and normalizes raw BTC data |
| `forecast_btc.py` | Builds, trains, and evaluates the LSTM model |

## Usage

### Step 1: Preprocess the data
```bash
python3 preprocess_data.py
```
This will generate:
- `close_prices.npy` — normalized close prices (hourly)
- `close_min.npy` — min values used for normalization
- `close_max.npy` — max values used for normalization

### Step 2: Train and evaluate the model
```bash
python3 forecast_btc.py
```

## Preprocessing Steps
1. Load both coinbase and bitstamp CSV datasets
2. Forward-fill missing values
3. Resample 60-second data to 1-hour intervals
4. Merge datasets (coinbase preferred, bitstamp fills gaps)
5. Extract only the Close price column
6. Normalize using min-max scaling to range [0, 1]

## Model Architecture
- **Input:** 24 hourly close prices (sliding window)
- **Layer 1:** LSTM (64 units, return sequences)
- **Dropout:** 0.2
- **Layer 2:** LSTM (32 units)
- **Dropout:** 0.2
- **Dense:** 16 units, ReLU activation
- **Output:** 1 unit (predicted close price)
- **Loss:** Mean Squared Error (MSE)
- **Optimizer:** Adam (lr=0.001)

## Results
The model is evaluated on a held-out validation set using MSE and MAE metrics.
Training uses early stopping to prevent overfitting.

## Author
Holberton School - Machine Learning Track
