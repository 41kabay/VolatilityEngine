"""
Machine Learning models for volatility forecasting.
Implements ensemble of LightGBM, Random Forest, and Linear models.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import joblib
from pathlib import Path
import json


class VolatilityMLEnsemble:
    """
    Ensemble of ML models for volatility forecasting.
    
    Models included:
    1. LightGBM - gradient boosting (fast, accurate)
    2. Random Forest - robust ensemble
    3. Ridge Regression - linear baseline
    """
    
    def __init__(self, models_dir="models"):
        """Initialize ensemble."""
        self.models = {}
        self.weights = {}
        self.scaler = StandardScaler()
        self.feature_importance = {}
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
    def _initialize_models(self):
        """Create model instances."""
        self.models = {
            'lightgbm': lgb.LGBMRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=7,
                num_leaves=31,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            ),
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=20,
                min_samples_leaf=10,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            ),
            'ridge': Ridge(
                alpha=1.0,
                random_state=42
            )
        }
        
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train all models in the ensemble.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
        """
        print("=" * 60)
        print("TRAINING ML ENSEMBLE")
        print("=" * 60)
        
        self._initialize_models()
        
        # Scale features for linear model
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            if name == 'ridge':
                # Use scaled features for linear model
                model.fit(X_train_scaled, y_train)
            else:
                # Tree-based models don't need scaling
                model.fit(X_train, y_train)
            
            # Training performance
            if name == 'ridge':
                train_pred = model.predict(X_train_scaled)
            else:
                train_pred = model.predict(X_train)
            
            train_mse = mean_squared_error(y_train, train_pred)
            train_mae = mean_absolute_error(y_train, train_pred)
            
            print(f"  Train MSE: {train_mse:.6f}")
            print(f"  Train MAE: {train_mae:.6f}")
            
            # Feature importance (tree-based only)
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = model.feature_importances_
        
        # Determine ensemble weights based on validation performance
        if X_val is not None and y_val is not None:
            print("\nComputing ensemble weights based on validation set...")
            self._compute_weights(X_val, y_val)
        else:
            # Equal weights
            self.weights = {name: 1.0 / len(self.models) for name in self.models.keys()}
            print(f"\nUsing equal weights: {self.weights}")
        
        print("\n" + "=" * 60)
        print("ENSEMBLE TRAINING COMPLETE!")
        print("=" * 60)
    
    def _compute_weights(self, X_val, y_val):
        """
        Compute optimal ensemble weights based on validation performance.
        
        Uses inverse MSE as weights (better models get higher weight).
        """
        val_errors = {}
        
        for name, model in self.models.items():
            if name == 'ridge':
                X_val_scaled = self.scaler.transform(X_val)
                val_pred = model.predict(X_val_scaled)
            else:
                val_pred = model.predict(X_val)
            
            val_mse = mean_squared_error(y_val, val_pred)
            val_errors[name] = val_mse
            
            print(f"{name:15s} - Val MSE: {val_mse:.6f}")
        
        # Compute weights (inverse MSE)
        inverse_errors = {name: 1.0 / mse for name, mse in val_errors.items()}
        total = sum(inverse_errors.values())
        self.weights = {name: inv_err / total for name, inv_err in inverse_errors.items()}
        
        print(f"\nOptimal weights: {self.weights}")
    
    def predict(self, X):
        """
        Make ensemble prediction.
        
        Args:
            X: Features
            
        Returns:
            Ensemble predictions
        """
        predictions = {}
        
        for name, model in self.models.items():
            if name == 'ridge':
                X_scaled = self.scaler.transform(X)
                predictions[name] = model.predict(X_scaled)
            else:
                predictions[name] = model.predict(X)
        
        # Weighted average
        ensemble_pred = np.zeros(len(X))
        for name, pred in predictions.items():
            ensemble_pred += pred * self.weights[name]
        
        return ensemble_pred
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate ensemble on test set.
        
        Args:
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary with metrics
        """
        print("\n" + "=" * 60)
        print("EVALUATING ON TEST SET")
        print("=" * 60)
        
        # Individual model predictions
        individual_results = {}
        
        for name, model in self.models.items():
            if name == 'ridge':
                X_test_scaled = self.scaler.transform(X_test)
                pred = model.predict(X_test_scaled)
            else:
                pred = model.predict(X_test)
            
            mse = mean_squared_error(y_test, pred)
            mae = mean_absolute_error(y_test, pred)
            r2 = r2_score(y_test, pred)
            
            individual_results[name] = {
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'rmse': np.sqrt(mse)
            }
            
            print(f"\n{name.upper()}")
            print(f"  MSE:  {mse:.6f}")
            print(f"  MAE:  {mae:.6f}")
            print(f"  RMSE: {np.sqrt(mse):.6f}")
            print(f"  R²:   {r2:.4f}")
        
        # Ensemble prediction
        ensemble_pred = self.predict(X_test)
        
        ensemble_mse = mean_squared_error(y_test, ensemble_pred)
        ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
        ensemble_r2 = r2_score(y_test, ensemble_pred)
        
        print(f"\nENSEMBLE")
        print(f"  MSE:  {ensemble_mse:.6f}")
        print(f"  MAE:  {ensemble_mae:.6f}")
        print(f"  RMSE: {np.sqrt(ensemble_mse):.6f}")
        print(f"  R²:   {ensemble_r2:.4f}")
        
        # Compare with baseline (always predict mean)
        baseline_pred = np.full(len(y_test), y_test.mean())
        baseline_mae = mean_absolute_error(y_test, baseline_pred)
        
        improvement = (1 - ensemble_mae / baseline_mae) * 100
        print(f"\nImprovement over baseline: {improvement:.1f}%")
        
        results = {
            'individual': individual_results,
            'ensemble': {
                'mse': ensemble_mse,
                'mae': ensemble_mae,
                'r2': ensemble_r2,
                'rmse': np.sqrt(ensemble_mse)
            },
            'baseline_mae': baseline_mae,
            'improvement_pct': improvement
        }
        
        return results
    
    def get_feature_importance(self, feature_names, top_n=20):
        """
        Get feature importance from tree-based models.
        
        Args:
            feature_names: List of feature names
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        if 'lightgbm' not in self.feature_importance:
            print("No feature importance available")
            return None
        
        # Use LightGBM importance
        importance = self.feature_importance['lightgbm']
        
        df_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False).head(top_n)
        
        return df_importance
    
    def save_models(self, prefix='volatility_ensemble'):
        """Save trained models."""
        for name, model in self.models.items():
            filepath = self.models_dir / f"{prefix}_{name}.pkl"
            joblib.dump(model, filepath)
        
        # Save scaler
        joblib.dump(self.scaler, self.models_dir / f"{prefix}_scaler.pkl")
        
        # Save weights and metadata
        metadata = {
            'weights': self.weights,
            'feature_importance': {k: v.tolist() for k, v in self.feature_importance.items()}
        }
        with open(self.models_dir / f"{prefix}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nModels saved to {self.models_dir}")
    
    def load_models(self, prefix='volatility_ensemble'):
        """Load trained models."""
        self.models = {}
        
        for name in ['lightgbm', 'random_forest', 'ridge']:
            filepath = self.models_dir / f"{prefix}_{name}.pkl"
            if filepath.exists():
                self.models[name] = joblib.load(filepath)
        
        # Load scaler
        scaler_path = self.models_dir / f"{prefix}_scaler.pkl"
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
        
        # Load metadata
        metadata_path = self.models_dir / f"{prefix}_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.weights = metadata['weights']
        
        print(f"Models loaded from {self.models_dir}")


def main():
    """Example usage."""
    from data_loader import SP500DataLoader
    from feature_engineering import VolatilityFeatureEngineer, create_train_test_split
    
    print("Loading data...")
    loader = SP500DataLoader()
    df = loader.load_cached_data("sp500_prepared.csv")
    
    print("\nCreating features...")
    engineer = VolatilityFeatureEngineer()
    df_features = engineer.create_all_features(df)
    
    print("\nSplitting data...")
    X_train, X_test, y_train, y_test = create_train_test_split(df_features, test_size=0.2)
    
    # Further split train into train/val
    val_size = int(len(X_train) * 0.2)
    X_val = X_train[-val_size:]
    y_val = y_train[-val_size:]
    X_train = X_train[:-val_size]
    y_train = y_train[:-val_size]
    
    print(f"Final train size: {len(X_train)}")
    print(f"Validation size: {len(X_val)}")
    print(f"Test size: {len(X_test)}")
    
    # Train ensemble
    ensemble = VolatilityMLEnsemble()
    ensemble.fit(X_train, y_train, X_val, y_val)
    
    # Evaluate
    results = ensemble.evaluate(X_test, y_test)
    
    # Feature importance
    print("\n" + "=" * 60)
    print("TOP 15 MOST IMPORTANT FEATURES")
    print("=" * 60)
    importance_df = ensemble.get_feature_importance(X_train.columns, top_n=15)
    if importance_df is not None:
        print(importance_df.to_string(index=False))
    
    # Save models
    ensemble.save_models()
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
