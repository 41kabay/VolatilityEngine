"""
Stochastic volatility models: GARCH(1,1) and related approaches.
These models capture volatility clustering and mean reversion.
"""

import numpy as np
import pandas as pd
from arch import arch_model
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


class GARCHModel:
    """
    GARCH(1,1) model for volatility forecasting.
    
    Model: σ²ₜ = ω + α·r²ₜ₋₁ + β·σ²ₜ₋₁
    
    Where:
    - σ²ₜ: conditional variance at time t
    - r²ₜ₋₁: squared return (shock/news term)
    - ω: constant term (long-run variance level)
    - α: ARCH coefficient (impact of recent shocks)
    - β: GARCH coefficient (persistence of volatility)
    
    Interpretation:
    - α + β close to 1: high persistence (volatility clustering)
    - α > 0: positive shocks increase volatility
    - β > α: past volatility matters more than recent shocks
    """
    
    def __init__(self, p=1, q=1):
        """
        Initialize GARCH model.
        
        Args:
            p: Order of GARCH terms (usually 1)
            q: Order of ARCH terms (usually 1)
        """
        self.p = p
        self.q = q
        self.model = None
        self.fitted_model = None
        self.params = None
        
    def fit(self, returns, scale=100):
        """
        Fit GARCH model to returns data.
        
        Args:
            returns: Series of returns
            scale: Scale returns by this factor (helps numerical stability)
            
        Returns:
            Fitted model results
        """
        # Scale returns for numerical stability
        returns_scaled = returns * scale
        
        # Create and fit model
        self.model = arch_model(
            returns_scaled, 
            vol='Garch', 
            p=self.p, 
            q=self.q,
            rescale=False
        )
        
        try:
            self.fitted_model = self.model.fit(disp='off', show_warning=False)
            self.params = self.fitted_model.params
            
            # Unscale parameters
            self.omega = self.params['omega'] / (scale ** 2)
            self.alpha = self.params['alpha[1]']
            self.beta = self.params['beta[1]']
            
            print("GARCH(1,1) fitted successfully!")
            print(f"ω (omega) = {self.omega:.6f}")
            print(f"α (alpha) = {self.alpha:.4f} - news impact")
            print(f"β (beta)  = {self.beta:.4f} - persistence")
            print(f"α + β     = {self.alpha + self.beta:.4f} - total persistence")
            
            return self.fitted_model
            
        except Exception as e:
            print(f"GARCH fitting failed: {e}")
            return None
    
    def forecast(self, horizon=1):
        """
        Forecast volatility for next periods.
        
        Args:
            horizon: Number of periods ahead to forecast
            
        Returns:
            Forecasted variance values
        """
        if self.fitted_model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        forecasts = self.fitted_model.forecast(horizon=horizon)
        return forecasts.variance.values[-1, :]
    
    def get_conditional_volatility(self):
        """
        Get conditional volatility (fitted values).
        
        Returns:
            Series of conditional volatility
        """
        if self.fitted_model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return self.fitted_model.conditional_volatility
    
    def rolling_forecast(self, returns, train_size=500, refit_freq=50):
        """
        Perform rolling window forecasts.
        
        Args:
            returns: Full returns series
            train_size: Initial training window size
            refit_freq: How often to refit model (in periods)
            
        Returns:
            DataFrame with forecasts and actuals
        """
        n = len(returns)
        forecasts = []
        actuals = []
        
        for i in range(train_size, n):
            # Training data
            train_returns = returns.iloc[i-train_size:i]
            
            # Refit if needed
            if (i - train_size) % refit_freq == 0:
                self.fit(train_returns)
            
            # Forecast next period
            forecast = self.forecast(horizon=1)[0]
            forecasts.append(np.sqrt(forecast))  # Convert variance to volatility
            
            # Actual (next period realized vol - simplified as abs return)
            actual = np.abs(returns.iloc[i])
            actuals.append(actual)
        
        results = pd.DataFrame({
            'forecast': forecasts,
            'actual': actuals
        }, index=returns.index[train_size:])
        
        return results


class SimpleStochasticVol:
    """
    Simplified stochastic volatility model inspired by Heston.
    
    Instead of full SDE solution, we use mean-reverting process:
    σₜ = σₜ₋₁ + κ(θ - σₜ₋₁) + ε
    
    Where:
    - κ: speed of mean reversion
    - θ: long-run mean volatility
    - ε: random shock
    """
    
    def __init__(self):
        self.kappa = None  # Mean reversion speed
        self.theta = None  # Long-run mean
        self.fitted = False
        
    def fit(self, realized_vol):
        """
        Fit mean-reverting parameters to realized volatility.
        
        Args:
            realized_vol: Series of realized volatility
        """
        # Remove NaN
        vol = realized_vol.dropna()
        
        # Long-run mean
        self.theta = vol.mean()
        
        # Estimate mean reversion speed using AR(1) regression
        # Δσₜ = κ(θ - σₜ₋₁) + ε
        # Rearranged: σₜ = (1-κ)σₜ₋₁ + κθ + ε
        
        vol_lag = vol.shift(1).dropna()
        vol_current = vol[1:]
        
        # Simple OLS: current vol = a + b * lagged vol
        from scipy.stats import linregress
        slope, intercept, _, _, _ = linregress(vol_lag, vol_current)
        
        # Extract parameters
        self.kappa = 1 - slope
        
        # Constrain kappa to be positive (mean-reverting)
        if self.kappa < 0:
            self.kappa = 0.01
        
        self.fitted = True
        
        print(f"Stochastic Vol Model fitted:")
        print(f"κ (kappa) = {self.kappa:.4f} - mean reversion speed")
        print(f"θ (theta) = {self.theta:.4f} - long-run mean volatility")
        print(f"Half-life = {np.log(2)/self.kappa:.2f} periods")
        
    def forecast(self, current_vol, horizon=1):
        """
        Forecast volatility using mean reversion.
        
        Args:
            current_vol: Current volatility level
            horizon: Periods ahead
            
        Returns:
            Forecasted volatility
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Mean reversion formula
        forecast = self.theta + (current_vol - self.theta) * np.exp(-self.kappa * horizon)
        return forecast


def compare_volatility_models(returns, realized_vol):
    """
    Compare GARCH vs simple stochastic vol model.
    
    Args:
        returns: Returns series
        realized_vol: Realized volatility series
        
    Returns:
        Dictionary with model results
    """
    print("=" * 60)
    print("FITTING GARCH(1,1) MODEL")
    print("=" * 60)
    
    garch = GARCHModel()
    garch.fit(returns)
    
    print("\n" + "=" * 60)
    print("FITTING SIMPLE STOCHASTIC VOL MODEL")
    print("=" * 60)
    
    stoch_vol = SimpleStochasticVol()
    stoch_vol.fit(realized_vol)
    
    return {
        'garch': garch,
        'stoch_vol': stoch_vol
    }


def main():
    """Example usage with synthetic data."""
    np.random.seed(42)
    
    # Generate synthetic returns with volatility clustering
    n = 1000
    returns = np.random.randn(n) * 0.01
    
    # Add GARCH-like structure
    for i in range(1, n):
        vol = 0.01 + 0.3 * returns[i-1]**2 + 0.6 * (returns[i-1] * 0.01)**2
        returns[i] = np.random.randn() * np.sqrt(vol)
    
    returns = pd.Series(returns, index=pd.date_range('2023-01-01', periods=n, freq='h'))
    
    # Compute realized vol
    realized_vol = returns.rolling(window=20).std() * np.sqrt(252 * 6.5)
    
    print("Testing stochastic volatility models on synthetic data")
    print(f"Data shape: {len(returns)} observations\n")
    
    # Fit models
    models = compare_volatility_models(returns, realized_vol)
    
    print("\n" + "=" * 60)
    print("MODELS FITTED SUCCESSFULLY!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Use these models as features for ML")
    print("2. Compare forecasts against ML predictions")
    print("3. Evaluate on out-of-sample data")


if __name__ == "__main__":
    main()
