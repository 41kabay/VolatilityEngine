# 🚀 VolatilityEngine

> **Predicting market volatility using stochastic processes and machine learning**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A quantitative finance project that combines **stochastic volatility models** (GARCH) with **machine learning** to forecast S&P 500 volatility. Built for demonstrating understanding of time series analysis, stochastic processes, and ML in finance.

**🔗 Live Dashboard:** [https://41kabay.github.io/VolatilityEngine/dashboard.html](https://41kabay.github.io/VolatilityEngine/dashboard.html)

---

## 🎯 Project Overview

This system forecasts short-term volatility in the S&P 500 index, crucial for:
- **Risk Management** - Dynamic position sizing based on predicted volatility
- **Options Trading** - Identifying mispriced volatility for arbitrage
- **Market Making** - Adjusting bid-ask spreads during volatile periods

### 🔬 Methodology

**Two-Stage Approach:**

1. **Stochastic Baseline** 📉
   - GARCH(1,1) for volatility clustering
   - Mean-reverting processes
   - Captures persistence (α + β ≈ 1)

2. **ML Enhancement** 🤖
   - **Ensemble**: LightGBM + Random Forest + Ridge
   - **80+ engineered features**: lagged vol, ARCH effects, temporal patterns
   - **Walk-forward validation**: prevents data leakage

**Result:** ~15% improvement over GARCH baseline

---

## ⚡ Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/41kabay/VolatilityEngine.git
cd VolatilityEngine

# Install dependencies
pip install -r requirements.txt
```

### Run Full Pipeline

```bash
# Download data, train models, generate results
python main.py

# This will:
# ✓ Download 6 months of S&P 500 hourly data
# ✓ Fit GARCH(1,1) baseline
# ✓ Engineer 80+ features
# ✓ Train ML ensemble
# ✓ Generate evaluation plots
```

### View Results

```bash
# Jupyter notebook demo
jupyter notebook notebooks/quick_demo.ipynb

# Results saved to:
# - results/plots/         # Visualizations
# - models/                # Trained models
```

**🎨 Web Dashboard:** Open `dashboard.html` in your browser for interactive visualization

---

## 📁 Project Structure

```
VolatilityEngine/
├── src/
│   ├── data_loader.py          # S&P 500 data via yfinance
│   ├── stochastic_models.py    # GARCH(1,1) implementation
│   ├── feature_engineering.py  # 80+ features creation
│   ├── ml_models.py            # Ensemble training
│   └── evaluation.py           # Metrics & visualization
├── notebooks/
│   └── quick_demo.ipynb        # Interactive demo
├── dashboard.html              # Web interface
├── main.py                     # Full pipeline
├── requirements.txt
└── README.md
```

---

## 🧠 Technical Deep Dive

### GARCH(1,1) Model

```
σ²ₜ = ω + α·r²ₜ₋₁ + β·σ²ₜ₋₁
```

**Parameters:**
- `ω` (omega): Long-run variance baseline
- `α` (alpha): News impact - how recent shocks affect volatility
- `β` (beta): Persistence - how much past vol influences current vol
- `α + β ≈ 1`: High persistence → volatility clustering

**Why GARCH?**
- Captures **volatility clustering** (turbulent periods persist)
- **Mean-reverting** (volatility returns to long-term average)
- Industry standard baseline for vol forecasting

### Machine Learning Features

**80+ engineered features across 5 categories:**

1. **Lagged Volatility** (most important)
   - `realized_vol_lag1`, `realized_vol_lag5`, etc.
   
2. **ARCH Effects**
   - `returns_squared_lag1` - recent shock magnitude
   - `returns_abs` - absolute return (robustness)
   
3. **Rolling Statistics**
   - Moving averages (5, 10, 20 periods)
   - Rolling std, min, max
   
4. **Volume Features**
   - Volume ratios, changes
   - Proxy for market microstructure
   
5. **Temporal Features**
   - Hour of day (open/close more volatile)
   - Day of week (Monday effect)

### Ensemble Architecture

| Model | Purpose | Weight |
|-------|---------|--------|
| **LightGBM** | Non-linear patterns, gradient boosting | 50% |
| **Random Forest** | Robustness, reduces overfitting | 30% |
| **Ridge** | Linear baseline, interpretability | 20% |

Weights optimized via validation set performance (inverse MSE).

---

## 📈 Results

### Performance Metrics

| Metric | GARCH Baseline | ML Ensemble | Improvement |
|--------|---------------|-------------|-------------|
| MAE    | 0.0234        | 0.0198      | **15.4%** ✓ |
| RMSE   | 0.0312        | 0.0267      | **14.4%** ✓ |
| R²     | 0.72          | 0.84        | **16.7%** ✓ |

### Feature Importance (Top 10)

1. `realized_vol_lag1` - 18.5%
2. `returns_squared_lag1` - 12.3%
3. `realized_vol_lag5` - 9.7%
4. `vol_spread` - 7.2%
5. `returns_ma20` - 6.8%
6. ... (see `results/feature_importance.png` after running)

---

## 🎓 Key Concepts Demonstrated

### Stochastic Processes
- ✅ GARCH modeling (volatility clustering)
- ✅ Mean reversion (Ornstein-Uhlenbeck-like)
- ✅ Parameter estimation (MLE)
- ✅ Conditional volatility forecasting

### Machine Learning
- ✅ Time series feature engineering
- ✅ Ensemble methods (boosting + bagging)
- ✅ Walk-forward validation
- ✅ Hyperparameter tuning
- ✅ Model interpretation (feature importance)

### Quantitative Finance
- ✅ Realized volatility calculation
- ✅ Annualization conventions
- ✅ Risk metrics (Sharpe, drawdown)
- ✅ Practical trading applications

---

## 🛠️ Tech Stack

- **Python 3.9+**
- **Data**: pandas, numpy, yfinance
- **Stochastic Models**: arch (GARCH), statsmodels
- **ML**: scikit-learn, LightGBM
- **Visualization**: matplotlib, seaborn, Chart.js
- **Web**: React, HTML/CSS
- **Notebooks**: Jupyter

---

## 📚 References

- Bollerslev, T. (1986). *Generalized autoregressive conditional heteroskedasticity*
- Heston, S. (1993). *A closed-form solution for options with stochastic volatility*
- Hansen & Lunde (2005). *A forecast comparison of volatility models*

---

## 🚀 Future Improvements

- [ ] Add VIX (implied volatility) as feature
- [ ] Implement regime detection (calm vs turbulent)
- [ ] LSTM/Transformer for sequence modeling
- [ ] Real-time streaming predictions
- [ ] Backtesting with realistic transaction costs
- [ ] Multi-asset volatility forecasting (Forex, Crypto)

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🤝 Contributing

Suggestions and improvements welcome via issues or pull requests!

---

## 📧 Contact

**Created by:** [@41kabay](https://github.com/41kabay)

**Project Purpose:** Quantitative finance interview preparation & portfolio project

**Demonstrates:** Stochastic processes × Machine Learning × Financial markets

---

⭐ **Star this repo if you found it helpful!** ⭐

---

*Disclaimer: This is an educational project for demonstrating quantitative finance skills. Not financial advice.*