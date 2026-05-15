#!/usr/bin/env python3
"""
Preprocesses BTC data from coinbase and bitstamp datasets
for time series forecasting.
"""
import numpy as np
import pandas as pd


def load_and_clean(filepath):
    """
    Loads a CSV file and performs initial cleaning.

    Args:
        filepath: str - path to the CSV file

    Returns:
        pandas.DataFrame - cleaned dataframe
    """
    df = pd.read_csv(filepath)

    # Drop rows where all values are NaN
    df = df.dropna(how='all')

    # Forward fill missing values (carry last known value forward)
    df = df.fillna(method='ffill')

    # Drop any remaining NaN rows
    df = df.dropna()

    return df


def resample_to_hourly(df):
    """
    Resamples 60-second data to 1-hour intervals.

    Args:
        df: pandas.DataFrame - dataframe with Unix timestamp column

    Returns:
        pandas.DataFrame - hourly resampled dataframe
    """
    # Convert Unix timestamp to datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
    df = df.set_index('Timestamp')

    # Resample to 1 hour using appropriate aggregations
    df_hourly = df.resample('1H').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume_(BTC)': 'sum',
        'Volume_(Currency)': 'sum',
        'Weighted_Price': 'mean'
    })

    # Drop any NaN rows after resampling
    df_hourly = df_hourly.dropna()

    return df_hourly


def normalize(df):
    """
    Normalizes the dataframe using min-max scaling.

    Args:
        df: pandas.DataFrame - dataframe to normalize

    Returns:
        tuple: (normalized dataframe, min series, max series)
    """
    df_min = df.min()
    df_max = df.max()
    df_norm = (df - df_min) / (df_max - df_min)

    return df_norm, df_min, df_max


def preprocess():
    """
    Main preprocessing function.
    Loads coinbase and bitstamp datasets, cleans, resamples to hourly,
    merges, normalizes, and saves the result as a numpy file.
    """
    print("Loading coinbase data...")
    coinbase = load_and_clean(
        'coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv')

    print("Loading bitstamp data...")
    bitstamp = load_and_clean(
        'bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv')

    print("Resampling coinbase to hourly...")
    coinbase_hourly = resample_to_hourly(coinbase)

    print("Resampling bitstamp to hourly...")
    bitstamp_hourly = resample_to_hourly(bitstamp)

    # Use coinbase where available, fill gaps with bitstamp
    print("Merging datasets...")
    combined = coinbase_hourly.combine_first(bitstamp_hourly)
    combined = combined.dropna()

    # Only use Close price for forecasting
    close_prices = combined[['Close']]

    print("Normalizing data...")
    close_norm, close_min, close_max = normalize(close_prices)

    # Save preprocessed data
    print("Saving preprocessed data...")
    np.save('close_prices.npy', close_norm.values)
    np.save('close_min.npy', close_min.values)
    np.save('close_max.npy', close_max.values)

    print("Preprocessing complete.")
    print("Shape of preprocessed data:", close_norm.shape)
    print("Date range: {} to {}".format(
        combined.index.min(), combined.index.max()))


if __name__ == '__main__':
    preprocess()
