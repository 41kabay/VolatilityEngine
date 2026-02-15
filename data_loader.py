"""
Data loader for S&P 500 historical data.
Uses yfinance to download OHLCV data and compute realized volatility.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path


class SP500DataLoader:
    """Load and preprocess S&P 500 data for volatility analysis."""
    
    def __init__(self, ticker="^GSPC", data_dir="data"):
        """
        Initialize data loader.
        
        Args:
            ticker: Yahoo Finance ticker symbol (^GSPC for S&P 500)
            data_dir: Directory to save/load data
        """
        self.ticker = ticker
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
    def download_data(self, start_date=None, end_date=None, interval="1h"):
        """
        Download S&P 500 data from Yahoo Finance.
        
        Args:
            start_date: Start date (string or datetime)
            end_date: End date (string or datetime)
            interval: Data interval (1m, 5m, 15m, 1h, 1d)
            
        Returns:
            DataFrame with OHLCV data
        """
        if start_date is None:
            # Default: 6 months of data
            start_date = datetime.now() - timedelta(days=180)
        if end_date is None:
            end_date = datetime.now()
            
        print(f"Downloading {self.ticker} data from {start_date} to {end_date}")
        print(f"Interval: {interval}")
        
        # Download data
        df = yf.download(
            self.ticker,
            start=start_date,
            end=end_date,
            interval=interval,
            progress=True
        )
        
        if df.empty:
            raise ValueError("No data downloaded. Check ticker and date range.")
        
        # Clean column names
        df.columns = [col.lower() if isinstance(col, str) else col for col in df.columns]
        
        print(f"Downloaded {len(df)} rows")
        
        # Save to CSV
        filename = f"{self.ticker.replace('^', '')}_{interval}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
        filepath = self.data_dir / filename
        df.to_csv(filepath)
        print(f"Saved to {filepath}")
        
        return df
    
    def compute_returns(self, df, price_col='close'):
        """
        Compute log returns.
        
        Args:
            df: DataFrame with price data
            price_col: Column name for prices
            
        Returns:
            DataFrame with returns column added
        """
        df = df.copy()
        df['returns'] = np.log(df[price_col] / df[price_col].shift(1))
        return df
    
    def compute_realized_volatility(self, df, windows=[5, 10, 20, 50]):
        """
        Compute realized volatility over multiple windows.
        
        Realized volatility = std(returns) * sqrt(periods_per_year)
        For hourly data: 252 trading days * 6.5 hours ≈ 1638 periods/year
        
        Args:
            df: DataFrame with returns
            windows: List of window sizes for rolling volatility
            
        Returns:
            DataFrame with volatility columns added
        """
        df = df.copy()
        
        # Annualization factor (hourly data)
        annual_factor = np.sqrt(252 * 6.5)  # Trading hours per year
        
        for window in windows:
            col_name = f'realized_vol_{window}'
            df[col_name] = df['returns'].rolling(window=window).std() * annual_factor
            
        # Also compute forward volatility (target variable)
        for window in windows:
            col_name = f'forward_vol_{window}'
            # Forward-looking realized vol (what we want to predict)
            df[col_name] = df['returns'].shift(-window).rolling(window=window).std() * annual_factor
            
        return df
    
    def add_volume_features(self, df):
        """
        Add volume-based features.
        
        Args:
            df: DataFrame with volume data
            
        Returns:
            DataFrame with volume features added
        """
        df = df.copy()
        
        # Volume change
        df['volume_change'] = df['volume'].pct_change()
        
        # Volume moving averages
        df['volume_ma_5'] = df['volume'].rolling(window=5).mean()
        df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma_20']
        
        return df
    
    def add_price_features(self, df):
        """
        Add price-based technical features.
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            DataFrame with price features added
        """
        df = df.copy()
        
        # High-Low range (intraday volatility proxy)
        df['hl_range'] = (df['high'] - df['low']) / df['close']
        
        # Close-Open (directional move)
        df['co_change'] = (df['close'] - df['open']) / df['open']
        
        # Moving averages
        for window in [5, 20, 50]:
            df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
            df[f'price_to_sma_{window}'] = df['close'] / df[f'sma_{window}']
        
        return df
    
    def prepare_dataset(self, df, target_window=5):
        """
        Prepare final dataset with all features and target variable.
        
        Args:
            df: Raw DataFrame
            target_window: Window for forward volatility prediction
            
        Returns:
            Clean DataFrame ready for modeling
        """
        df = df.copy()
        
        # Compute all features
        df = self.compute_returns(df)
        df = self.compute_realized_volatility(df)
        df = self.add_volume_features(df)
        df = self.add_price_features(df)
        
        # Set target variable
        df['target'] = df[f'forward_vol_{target_window}']
        
        # Drop NaN values
        df = df.dropna()
        
        print(f"Dataset prepared: {len(df)} samples, {len(df.columns)} features")
        
        return df
    
    def load_cached_data(self, filename):
        """Load previously saved data."""
        filepath = self.data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        return pd.read_csv(filepath, index_col=0, parse_dates=True)


def main():
    """Example usage."""
    loader = SP500DataLoader()
    
    # Download 6 months of hourly data
    df = loader.download_data(interval="1h")
    
    # Prepare dataset
    df_prepared = loader.prepare_dataset(df, target_window=5)
    
    print("\nDataset info:")
    print(df_prepared.info())
    print("\nFirst few rows:")
    print(df_prepared.head())
    print("\nTarget variable statistics:")
    print(df_prepared['target'].describe())
    
    # Save prepared dataset
    df_prepared.to_csv(loader.data_dir / "sp500_prepared.csv")
    print("\nPrepared dataset saved to data/sp500_prepared.csv")


if __name__ == "__main__":
    main()
