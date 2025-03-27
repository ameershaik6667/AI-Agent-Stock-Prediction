import pandas as pd
import numpy as np

class ADXIndicator:
    def __init__(self, period=14, smoothing_method="SMA"):
        """
        Initialize the ADXIndicator with a given period and smoothing method.
        
        Parameters:
            period (int): The time period for the ADX calculation.
            smoothing_method (str): The smoothing method to use ("SMA" or "EMA").
        """
        self.period = period
        self.smoothing_method = smoothing_method.upper()  # Convert to uppercase for consistency

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the Average Directional Index (ADX) along with the directional indicators.

        This function computes the following:
         - True Range (TR) as the maximum of:
             * (High - Low)
             * Absolute(High - Previous Close)
             * Absolute(Low - Previous Close)
         - Positive and negative directional movements (+DM and -DM)
         - Smoothing of TR, +DM, and -DM using the chosen smoothing method (SMA or EMA)
         - Directional Indicators (+DI and -DI) as percentages
         - DX, which is the directional movement index
         - ADX as the rolling average of DX

        Parameters:
            data (pd.DataFrame): Historical stock data with columns 'High', 'Low', and 'Close'.

        Returns:
            pd.DataFrame: DataFrame with additional columns for +DM, -DM, TR, smoothed values,
                          +DI, -DI, DX, and ADX.
        """
        # Create a copy of the input DataFrame to avoid modifying the original data
        df = data.copy()

        # Extract the required columns
        high = df['High']
        low = df['Low']
        close = df['Close']

        # --- Calculate True Range (TR) ---
        # Difference between High and Low
        df['H-L'] = high - low
        # Absolute difference between High and previous Close
        df['H-PC'] = (high - close.shift(1)).abs()
        # Absolute difference between Low and previous Close
        df['L-PC'] = (low - close.shift(1)).abs()
        # True Range is the maximum of the three values calculated above
        df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)

        # --- Calculate Directional Movements ---
        # Difference between current and previous High prices
        diff_high = high.diff()
        # Difference between current and previous Low prices
        diff_low = low.diff()

        # Calculate +DM (positive directional movement)
        # Only count diff_high if it is greater than the absolute diff_low and positive; otherwise 0
        plus_dm = np.where((diff_high > diff_low.abs()) & (diff_high > 0), diff_high, 0)
        # Calculate -DM (negative directional movement)
        # Only count the absolute diff_low if it is greater than diff_high and positive; otherwise 0
        minus_dm = np.where((diff_low.abs() > diff_high) & (diff_low.abs() > 0), diff_low.abs(), 0)

        # Add the directional movement columns to the DataFrame
        df['+DM'] = plus_dm
        df['-DM'] = minus_dm

        # --- Smooth the TR and directional movements ---
        if self.smoothing_method == "EMA":
            # Use Exponential Moving Average for smoothing
            df['TR_smooth'] = df['TR'].ewm(span=self.period, min_periods=self.period, adjust=False).mean()
            df['+DM_smooth'] = df['+DM'].ewm(span=self.period, min_periods=self.period, adjust=False).mean()
            df['-DM_smooth'] = df['-DM'].ewm(span=self.period, min_periods=self.period, adjust=False).mean()
        else:
            # Default to Simple Moving Average by summing over the period
            df['TR_smooth'] = df['TR'].rolling(window=self.period, min_periods=self.period).sum()
            df['+DM_smooth'] = df['+DM'].rolling(window=self.period, min_periods=self.period).sum()
            df['-DM_smooth'] = df['-DM'].rolling(window=self.period, min_periods=self.period).sum()

        # --- Calculate Directional Indicators (+DI and -DI) ---
        df['+DI'] = 100 * (df['+DM_smooth'] / df['TR_smooth'])
        df['-DI'] = 100 * (df['-DM_smooth'] / df['TR_smooth'])

        # --- Calculate DX ---
        # DX measures the absolute difference between +DI and -DI as a percentage of their sum
        df['DX'] = 100 * (abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI']))

        # --- Calculate ADX ---
        # ADX is the rolling average of DX over the specified period
        df['ADX'] = df['DX'].rolling(window=self.period, min_periods=self.period).mean()

        # Return the DataFrame with the newly added columns
        return df
