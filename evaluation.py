"""
Evaluation and visualization for volatility forecasting models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pathlib import Path

# Set plotting style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)


class VolatilityEvaluator:
    """Evaluate and visualize volatility forecasting results."""
    
    def __init__(self, results_dir="results"):
        """Initialize evaluator."""
        self.results_dir = Path(results_dir)
        self.plots_dir = self.results_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
    def plot_predictions_vs_actual(self, y_true, y_pred, model_name="Model", save=True):
        """
        Plot predicted vs actual volatility.
        
        Args:
            y_true: True volatility values
            y_pred: Predicted volatility values
            model_name: Name for plot title
            save: Whether to save plot
        """
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Time series plot
        ax = axes[0]
        if isinstance(y_true, pd.Series):
            ax.plot(y_true.index, y_true.values, label='Actual', alpha=0.7, linewidth=1.5)
            ax.plot(y_true.index, y_pred, label='Predicted', alpha=0.7, linewidth=1.5)
        else:
            ax.plot(y_true, label='Actual', alpha=0.7)
            ax.plot(y_pred, label='Predicted', alpha=0.7)
        
        ax.set_title(f'{model_name}: Predicted vs Actual Volatility', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Volatility')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Scatter plot
        ax = axes[1]
        ax.scatter(y_true, y_pred, alpha=0.5, s=20)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction', linewidth=2)
        
        ax.set_title('Predicted vs Actual (Scatter)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Actual Volatility')
        ax.set_ylabel('Predicted Volatility')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add metrics text
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        corr = np.corrcoef(y_true, y_pred)[0, 1]
        
        metrics_text = f'MSE: {mse:.6f}\nMAE: {mae:.6f}\nCorr: {corr:.4f}'
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save:
            filename = f"{model_name.lower().replace(' ', '_')}_predictions.png"
            plt.savefig(self.plots_dir / filename, dpi=150, bbox_inches='tight')
            print(f"Saved plot: {filename}")
        
        plt.show()
        plt.close()
        
    def plot_residuals(self, y_true, y_pred, model_name="Model", save=True):
        """
        Plot residuals analysis.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name for plot title
            save: Whether to save plot
        """
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Residuals over time
        ax = axes[0]
        if isinstance(residuals, pd.Series):
            ax.plot(residuals.index, residuals.values, alpha=0.6)
        else:
            ax.plot(residuals, alpha=0.6)
        ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax.set_title(f'{model_name}: Residuals Over Time', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Residual (Actual - Predicted)')
        ax.grid(True, alpha=0.3)
        
        # Residuals distribution
        ax = axes[1]
        ax.hist(residuals, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax.set_title('Residuals Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Residual')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        mean_res = residuals.mean()
        std_res = residuals.std()
        stats_text = f'Mean: {mean_res:.6f}\nStd: {std_res:.6f}'
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        plt.tight_layout()
        
        if save:
            filename = f"{model_name.lower().replace(' ', '_')}_residuals.png"
            plt.savefig(self.plots_dir / filename, dpi=150, bbox_inches='tight')
            print(f"Saved plot: {filename}")
        
        plt.show()
        plt.close()
        
    def plot_feature_importance(self, importance_df, top_n=20, save=True):
        """
        Plot feature importance.
        
        Args:
            importance_df: DataFrame with 'feature' and 'importance' columns
            top_n: Number of top features to show
            save: Whether to save plot
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        top_features = importance_df.head(top_n).sort_values('importance')
        
        ax.barh(range(len(top_features)), top_features['importance'].values)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'].values)
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title(f'Top {top_n} Most Important Features', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.plots_dir / "feature_importance.png", dpi=150, bbox_inches='tight')
            print("Saved plot: feature_importance.png")
        
        plt.show()
        plt.close()
        
    def plot_model_comparison(self, results_dict, save=True):
        """
        Compare multiple models' performance.
        
        Args:
            results_dict: Dict with model names as keys and metrics dicts as values
            save: Whether to save plot
        """
        models = list(results_dict.keys())
        metrics = ['mae', 'rmse', 'r2']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            values = [results_dict[model].get(metric, 0) for model in models]
            
            bars = ax.bar(range(len(models)), values, alpha=0.7)
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels(models, rotation=45, ha='right')
            ax.set_ylabel(metric.upper())
            ax.set_title(f'Model Comparison: {metric.upper()}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for i, (bar, val) in enumerate(zip(bars, values)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.4f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.plots_dir / "model_comparison.png", dpi=150, bbox_inches='tight')
            print("Saved plot: model_comparison.png")
        
        plt.show()
        plt.close()
        
    def plot_volatility_regimes(self, y_true, y_pred, save=True):
        """
        Analyze prediction accuracy in different volatility regimes.
        
        Args:
            y_true: True volatility
            y_pred: Predicted volatility
            save: Whether to save plot
        """
        # Define volatility regimes (low, medium, high)
        terciles = np.percentile(y_true, [33, 67])
        
        regime = pd.cut(y_true, bins=[0, terciles[0], terciles[1], np.inf],
                       labels=['Low Vol', 'Medium Vol', 'High Vol'])
        
        # Calculate MAE by regime
        df = pd.DataFrame({
            'true': y_true,
            'pred': y_pred,
            'regime': regime
        })
        
        regime_mae = df.groupby('regime').apply(
            lambda x: mean_absolute_error(x['true'], x['pred'])
        )
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(range(len(regime_mae)), regime_mae.values, alpha=0.7)
        ax.set_xticks(range(len(regime_mae)))
        ax.set_xticklabels(regime_mae.index)
        ax.set_ylabel('MAE')
        ax.set_title('Prediction Accuracy by Volatility Regime', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, val in zip(bars, regime_mae.values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.5f}', ha='center', va='bottom', fontsize=11)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.plots_dir / "volatility_regimes.png", dpi=150, bbox_inches='tight')
            print("Saved plot: volatility_regimes.png")
        
        plt.show()
        plt.close()
        
    def create_evaluation_report(self, results, y_test, ensemble_pred, feature_importance):
        """
        Create comprehensive evaluation report with all visualizations.
        
        Args:
            results: Results dict from ensemble.evaluate()
            y_test: Test target values
            ensemble_pred: Ensemble predictions
            feature_importance: Feature importance DataFrame
        """
        print("\n" + "=" * 60)
        print("GENERATING EVALUATION REPORT")
        print("=" * 60)
        
        # 1. Predictions vs actual
        self.plot_predictions_vs_actual(y_test, ensemble_pred, "ML Ensemble")
        
        # 2. Residuals analysis
        self.plot_residuals(y_test, ensemble_pred, "ML Ensemble")
        
        # 3. Feature importance
        if feature_importance is not None:
            self.plot_feature_importance(feature_importance, top_n=20)
        
        # 4. Model comparison
        models_results = {
            'LightGBM': results['individual']['lightgbm'],
            'Random Forest': results['individual']['random_forest'],
            'Ridge': results['individual']['ridge'],
            'Ensemble': results['ensemble']
        }
        self.plot_model_comparison(models_results)
        
        # 5. Volatility regimes
        self.plot_volatility_regimes(y_test, ensemble_pred)
        
        print("\n" + "=" * 60)
        print("EVALUATION REPORT COMPLETE!")
        print(f"All plots saved to: {self.plots_dir}")
        print("=" * 60)


def main():
    """Example usage - full evaluation pipeline."""
    from data_loader import SP500DataLoader
    from feature_engineering import VolatilityFeatureEngineer, create_train_test_split
    from ml_models import VolatilityMLEnsemble
    
    print("Loading data...")
    loader = SP500DataLoader()
    df = loader.load_cached_data("sp500_prepared.csv")
    
    print("Creating features...")
    engineer = VolatilityFeatureEngineer()
    df_features = engineer.create_all_features(df)
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = create_train_test_split(df_features, test_size=0.2)
    
    print("Loading trained ensemble...")
    ensemble = VolatilityMLEnsemble()
    ensemble.load_models()
    
    print("Evaluating...")
    results = ensemble.evaluate(X_test, y_test)
    
    print("Getting predictions...")
    ensemble_pred = ensemble.predict(X_test)
    
    print("Getting feature importance...")
    feature_importance = ensemble.get_feature_importance(X_train.columns, top_n=20)
    
    print("Creating evaluation report...")
    evaluator = VolatilityEvaluator()
    evaluator.create_evaluation_report(results, y_test, ensemble_pred, feature_importance)
    
    print("\n✓ Evaluation complete!")


if __name__ == "__main__":
    main()
