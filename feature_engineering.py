"""
Feature engineering for volatility forecasting.
Creates lagged features, technical indicators, and temporal features.
"""

import pandas as pd
import numpy as np
from datetime import datetime


class VolatilityFeatureEngineer:
    """Create features for volatility prediction."""
    
    def __init__(self):
        self.feature_names = []
        
    def create_lagged_features(self, df, columns, lags=[1, 2, 3, 5, 10]):
        """
        Create lagged versions of specified columns.
        
        Args:
            df: DataFrame
            columns: List of column names to lag
            lags: List of lag periods
            
        Returns:
            DataFrame with lagged features
        """
        df = df.copy()
        
        for col in columns:
            if col not in df.columns:
                continue
                
            for lag in lags:
                new_col = f'{col}_lag{lag}'
                df[new_col] = df[col].shift(lag)
                self.feature_names.append(new_col)
        
        return df
    
    def create_rolling_features(self, df, columns, windows=[5, 10, 20]):
        """
        Create rolling statistics (mean, std, min, max).
        
        Args:
            df: DataFrame
            columns: Columns to compute rolling stats on
            windows: Window sizes
            
        Returns:
            DataFrame with rolling features
        """
        df = df.copy()
        
        for col in columns:
            if col not in df.columns:
                continue
                
            for window in windows:
                # Rolling mean
                new_col = f'{col}_ma{window}'
                df[new_col] = df[col].rolling(window=window).mean()
                self.feature_names.append(new_col)
                
                # Rolling std
                new_col = f'{col}_std{window}'
                df[new_col] = df[col].rolling(window=window).std()
                self.feature_names.append(new_col)
                
                # Rolling min/max
                new_col = f'{col}_min{window}'
                df[new_col] = df[col].rolling(window=window).min()
                self.feature_names.append(new_col)
                
                new_col = f'{col}_max{window}'
                df[new_col] = df[col].rolling(window=window).max()
                self.feature_names.append(new_col)
        
        return df
    
    def create_return_features(self, df):
        """
        Create return-based features.
        
        Args:
            df: DataFrame with 'returns' column
            
        Returns:
            DataFrame with return features
        """
        df = df.copy()
        
        if 'returns' not in df.columns:
            return df
        
        # Squared returns (ARCH effect)
        df['returns_squared'] = df['returns'] ** 2
        self.feature_names.append('returns_squared')
        
        # Absolute returns
        df['returns_abs'] = df['returns'].abs()
        self.feature_names.append('returns_abs')
        
        # Sign of returns
        df['returns_sign'] = np.sign(df['returns'])
        self.feature_names.append('returns_sign')
        
        # Rolling skewness and kurtosis
        for window in [20, 50]:
            df[f'returns_skew{window}'] = df['returns'].rolling(window=window).skew()
            self.feature_names.append(f'returns_skew{window}')
            
            df[f'returns_kurt{window}'] = df['returns'].rolling(window=window).kurt()
            self.feature_names.append(f'returns_kurt{window}')
        
        # Realized range (high-low volatility)
        if 'hl_range' in df.columns:
            df['hl_range_ma5'] = df['hl_range'].rolling(window=5).mean()
            self.feature_names.append('hl_range_ma5')
        
        return df
    
    def create_volatility_features(self, df):
        """
        Create volatility-specific features.
        
        Args:
            df: DataFrame with realized_vol columns
            
        Returns:
            DataFrame with volatility features
        """
        df = df.copy()
        
        # Get all realized vol columns
        vol_cols = [col for col in df.columns if 'realized_vol' in col]
        
        for vol_col in vol_cols:
            # Volatility changes
            new_col = f'{vol_col}_change'
            df[new_col] = df[vol_col].pct_change()
            self.feature_names.append(new_col)
            
            # Volatility momentum
            new_col = f'{vol_col}_momentum'
            df[new_col] = df[vol_col] - df[vol_col].shift(5)
            self.feature_names.append(new_col)
        
        # Volatility spread (short vs long term)
        if 'realized_vol_5' in df.columns and 'realized_vol_20' in df.columns:
            df['vol_spread'] = df['realized_vol_5'] - df['realized_vol_20']
            self.feature_names.append('vol_spread')
        
        return df
    
    def create_temporal_features(self, df):
        """
        Create time-based features.
        
        Args:
            df: DataFrame with datetime index
            
        Returns:
            DataFrame with temporal features
        """
        df = df.copy()
        
        # Hour of day (for intraday data)
        df['hour'] = df.index.hour
        self.feature_names.append('hour')
        
        # Day of week (Monday = 0, Friday = 4)
        df['day_of_week'] = df.index.dayofweek
        self.feature_names.append('day_of_week')
        
        # Is Monday/Friday (higher volatility)
        df['is_monday'] = (df.index.dayofweek == 0).astype(int)
        df['is_friday'] = (df.index.dayofweek == 4).astype(int)
        self.feature_names.extend(['is_monday', 'is_friday'])
        
        # Market open/close hours (higher volatility)
        df['is_market_open'] = ((df.index.hour >= 9) & (df.index.hour <= 10)).astype(int)
        df['is_market_close'] = ((df.index.hour >= 15) & (df.index.hour <= 16)).astype(int)
        self.feature_names.extend(['is_market_open', 'is_market_close'])
        
        return df
    
    def create_all_features(self, df):
        """
        Create all features at once.
        
        Args:
            df: Prepared DataFrame from data_loader
            
        Returns:
            DataFrame with all features
        """
        print("Creating features...")
        
        # Lagged features (key predictors)
        lag_cols = ['returns', 'returns_squared', 'realized_vol_5', 'realized_vol_10']
        df = self.create_lagged_features(df, lag_cols, lags=[1, 2, 3, 5, 10])
        
        # Rolling features
        roll_cols = ['returns', 'volume_ratio']
        df = self.create_rolling_features(df, roll_cols, windows=[5, 10, 20])
        
        # Return features
        df = self.create_return_features(df)
        
        # Volatility features
        df = self.create_volatility_features(df)
        
        # Temporal features
        df = self.create_temporal_features(df)
        
        print(f"Created {len(self.feature_names)} features")
        
        return df
    
    def get_feature_names(self):
        """Return list of created feature names."""
        return self.feature_names


def create_train_test_split(df, test_size=0.2):
    """
    Split data into train/test sets (time-based, no shuffle).
    
    Args:
        df: DataFrame with features and target
        test_size: Proportion for test set
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    # Drop rows with NaN
    df = df.dropna()
    
    # Split point
    split_idx = int(len(df) * (1 - test_size))
    
    # Feature columns (everything except target)
    feature_cols = [col for col in df.columns if col not in ['target', 'forward_vol_5']]
    
    # Split
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    X_train = train_df[feature_cols]
    y_train = train_df['target']
    X_test = test_df[feature_cols]
    y_test = test_df['target']
    
    print(f"Train set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Features: {len(feature_cols)}")
    
    return X_train, X_test, y_train, y_test


def main():
    """Example usage."""
    from data_loader import SP500DataLoader
    
    # Load data
    loader = SP500DataLoader()
    df = loader.load_cached_data("sp500_prepared.csv")
    
    # Create features
    engineer = VolatilityFeatureEngineer()
    df_features = engineer.create_all_features(df)
    
    print("\nFeature engineering complete!")
    print(f"Original columns: {len(df.columns)}")
    print(f"Final columns: {len(df_features.columns)}")
    print(f"\nFeature names:")
    for i, feat in enumerate(engineer.get_feature_names()[:20], 1):
        print(f"  {i}. {feat}")
    print(f"  ... and {len(engineer.get_feature_names()) - 20} more")
    
    # Create train/test split
    X_train, X_test, y_train, y_test = create_train_test_split(df_features)
    
    print("\nData ready for modeling!")


if __name__ == "__main__":
    main()
