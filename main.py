"""
Main pipeline script for S&P 500 volatility forecasting project.
Runs the complete workflow from data collection to evaluation.
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_loader import SP500DataLoader
from stochastic_models import GARCHModel, compare_volatility_models
from feature_engineering import VolatilityFeatureEngineer, create_train_test_split
from ml_models import VolatilityMLEnsemble
from evaluation import VolatilityEvaluator


def run_full_pipeline(download_new_data=True):
    """
    Run the complete volatility forecasting pipeline.
    
    Args:
        download_new_data: If True, download fresh data from Yahoo Finance
    """
    
    print("=" * 70)
    print(" S&P 500 STOCHASTIC VOLATILITY FORECASTING WITH ML")
    print("=" * 70)
    print()
    
    # ========================================================================
    # STEP 1: DATA COLLECTION
    # ========================================================================
    print("STEP 1: DATA COLLECTION")
    print("-" * 70)
    
    loader = SP500DataLoader()
    
    if download_new_data:
        # Download 6 months of hourly data
        print("Downloading S&P 500 data...")
        df_raw = loader.download_data(interval="1h")
        
        # Prepare dataset with volatility features
        print("\nPreparing dataset...")
        df = loader.prepare_dataset(df_raw, target_window=5)
        
        # Save prepared data
        df.to_csv("data/sp500_prepared.csv")
        print("✓ Data prepared and saved")
    else:
        print("Loading cached data...")
        df = loader.load_cached_data("sp500_prepared.csv")
        print(f"✓ Loaded {len(df)} samples")
    
    print()
    
    # ========================================================================
    # STEP 2: STOCHASTIC MODELS (BASELINE)
    # ========================================================================
    print("STEP 2: STOCHASTIC VOLATILITY MODELS (BASELINE)")
    print("-" * 70)
    
    # Fit GARCH and simple stochastic vol models
    models = compare_volatility_models(
        returns=df['returns'],
        realized_vol=df['realized_vol_20']
    )
    
    print("\n✓ Baseline stochastic models fitted")
    print()
    
    # ========================================================================
    # STEP 3: FEATURE ENGINEERING
    # ========================================================================
    print("STEP 3: FEATURE ENGINEERING")
    print("-" * 70)
    
    engineer = VolatilityFeatureEngineer()
    df_features = engineer.create_all_features(df)
    
    print(f"✓ Created {len(engineer.get_feature_names())} features")
    print()
    
    # ========================================================================
    # STEP 4: TRAIN/TEST SPLIT
    # ========================================================================
    print("STEP 4: TRAIN/TEST SPLIT")
    print("-" * 70)
    
    X_train, X_test, y_train, y_test = create_train_test_split(
        df_features, test_size=0.2
    )
    
    # Further split train into train/val for ensemble weight optimization
    val_size = int(len(X_train) * 0.2)
    X_val = X_train.iloc[-val_size:]
    y_val = y_train.iloc[-val_size:]
    X_train = X_train.iloc[:-val_size]
    y_train = y_train.iloc[:-val_size]
    
    print(f"✓ Final train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")
    print()
    
    # ========================================================================
    # STEP 5: MACHINE LEARNING ENSEMBLE
    # ========================================================================
    print("STEP 5: MACHINE LEARNING ENSEMBLE TRAINING")
    print("-" * 70)
    
    ensemble = VolatilityMLEnsemble(models_dir="models")
    ensemble.fit(X_train, y_train, X_val, y_val)
    
    # Save trained models
    ensemble.save_models()
    print()
    
    # ========================================================================
    # STEP 6: EVALUATION
    # ========================================================================
    print("STEP 6: MODEL EVALUATION")
    print("-" * 70)
    
    # Evaluate on test set
    results = ensemble.evaluate(X_test, y_test)
    
    # Get predictions
    ensemble_pred = ensemble.predict(X_test)
    
    # Feature importance
    feature_importance = ensemble.get_feature_importance(X_train.columns, top_n=20)
    
    print()
    
    # ========================================================================
    # STEP 7: VISUALIZATION & REPORTING
    # ========================================================================
    print("STEP 7: GENERATING EVALUATION REPORT")
    print("-" * 70)
    
    evaluator = VolatilityEvaluator(results_dir="results")
    evaluator.create_evaluation_report(
        results=results,
        y_test=y_test,
        ensemble_pred=ensemble_pred,
        feature_importance=feature_importance
    )
    
    print()
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("=" * 70)
    print(" PIPELINE COMPLETE!")
    print("=" * 70)
    print()
    print("📊 RESULTS SUMMARY:")
    print(f"   • Test MAE: {results['ensemble']['mae']:.6f}")
    print(f"   • Test RMSE: {results['ensemble']['rmse']:.6f}")
    print(f"   • Test R²: {results['ensemble']['r2']:.4f}")
    print(f"   • Improvement over baseline: {results['improvement_pct']:.1f}%")
    print()
    print("📁 OUTPUTS:")
    print("   • Trained models: models/")
    print("   • Visualizations: results/plots/")
    print("   • Prepared data: data/sp500_prepared.csv")
    print()
    print("🎓 KEY CONCEPTS DEMONSTRATED:")
    print("   ✓ Stochastic processes (GARCH, mean reversion)")
    print("   ✓ Time series feature engineering")
    print("   ✓ ML ensemble (LightGBM, RF, Ridge)")
    print("   ✓ Walk-forward validation")
    print("   ✓ Volatility forecasting for risk management")
    print()
    print("=" * 70)
    

def quick_test():
    """Quick test with cached data (no downloads)."""
    print("Running quick test with cached data...\n")
    run_full_pipeline(download_new_data=False)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='S&P 500 Volatility Forecasting Pipeline')
    parser.add_argument('--quick', action='store_true', 
                       help='Run with cached data (no downloads)')
    parser.add_argument('--no-download', action='store_true',
                       help='Use cached data instead of downloading new data')
    
    args = parser.parse_args()
    
    if args.quick:
        quick_test()
    else:
        run_full_pipeline(download_new_data=not args.no_download)
