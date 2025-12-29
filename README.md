# HMM Market Regime Classifier 📈

A Hidden Markov Model implementation for identifying financial market regimes (bull, bear, and sideways markets) from time series data. This project explores how unsupervised learning can be applied to financial markets to recognize different market conditions automatically.

## Project Overview

This project came from wanting to explore quantitative machine learning through a beginner-friendly lens. Instead of jumping straight into complex neural networks, I wanted to understand how classical probabilistic models like Hidden Markov Models (HMMs) can capture underlying patterns in financial markets.

The goal is straightforward: can we teach a model to recognize when the market is in a bullish uptrend, a bearish downturn, or somewhere in between? By treating market regimes as hidden states that generate observable price movements, HMMs provide a natural framework for this kind of classification task.

The project implements a complete pipeline from raw financial data to an interactive dashboard, demonstrating how machine learning concepts translate into practical applications. It's structured to be educational while still producing meaningful results that could inform trading strategies or risk management decisions.

## Demo

The interactive Streamlit dashboard provides a comprehensive view of the model's performance and insights. Below is an interactive regime analyzer that starts with day 0 at 1/14/2010 as well as an analysis of state transition patterns:


To explore the full dashboard yourself, see the [How to Run](#how-to-run) section below.

## System Architecture

### Core Pipeline

The project follows a standard machine learning pipeline adapted for time series and probabilistic modeling:

**Data Processing** → **Feature Engineering** → **Discretization** → **HMM Training** → **Inference** → **Evaluation**

The workflow starts with raw S&P 500 price data. After computing technical indicators and log returns, the continuous data is discretized into categorical observations that the HMM can process. The Baum-Welch EM algorithm learns the transition probabilities between hidden states and emission probabilities for observations. Once trained, the Viterbi algorithm finds the most likely sequence of market regimes, which are then mapped to bull/bear classifications.

### Model Components

**Hidden Markov Model (HMM)**
- **Hidden States**: Represent different market regimes (typically 5 states capturing bull, bear, and intermediate conditions)
- **Observations**: Discretized price movements and volatility indicators
- **Transition Matrix**: Models how likely the market is to switch from one regime to another
- **Emission Matrix**: Defines what price patterns we expect to see in each regime

**Training Process**
- Baum-Welch Expectation-Maximization algorithm iteratively refines model parameters
- Forward-Backward algorithm computes state probabilities during training
- Structured initialization helps the model learn distinct regimes (e.g., low volatility bull markets vs high volatility bear markets)

**Inference**
- Viterbi algorithm finds the most likely sequence of hidden states
- State-to-regime mapping using correlation analysis or majority voting
- Classification threshold tuning to balance precision and recall

**Evaluation**
- Standard classification metrics (accuracy, precision, recall, F1)
- Confusion matrix analysis
- State interpretation based on actual market labels
- Transition pattern analysis to understand regime dynamics


## Data Source & Preprocessing

**Dataset**: S&P 500 historical data from Kaggle ([easily searchable as "S&P 500 stock data"](https://www.kaggle.com/datasets))

The dataset contains daily OHLCV (Open, High, Low, Close, Volume) data for the S&P 500 index over several years. This is a standard, publicly available dataset perfect for exploring financial time series analysis. The starting date for this dataset is around 2010. 

**Preprocessing Steps**:

- **Log Returns Calculation**: Convert raw prices to log returns to make the data stationary and comparable across different price levels
- **Feature Engineering**: Compute technical indicators including:
  - High-Low spread (volatility proxy)
  - Moving averages
  - Additional features via the `feature_engineering` module
- **Normalization**: Standardize features to zero mean and unit variance for stable discretization
- **Regime Labeling**: Create ground truth labels by smoothing log returns over a 5-day window and thresholding at zero (positive = bull, negative = bear)
- **Discretization**: Convert continuous features to categorical observations using equal-frequency binning (20 bins) to preserve distribution characteristics
- **Train/Test Split**: 80/20 split preserving temporal order (no shuffling, as this is time series data)

The preprocessing pipeline is designed to be transparent and reproducible, with all transformations clearly documented in the codebase.

## How to Run

### Prerequisites

- Python 3.8+
- Virtual environment (recommended)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd DSU_HMM

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

Key dependencies include PyTorch for tensor operations, pandas/numpy for data manipulation, Streamlit for the dashboard, and scikit-learn for evaluation metrics and discretization utilities.

### Running the Project

**1. Train the Model**

```bash
python -m src.models.hyperparameter_test
```

This trains the HMM with optimized hyperparameters (5 states, 20 observation bins, equal-frequency discretization). The trained model is saved to `results/optimized_hmm_classification_model.pt`.

**2. Prepare Dashboard Data**

```bash
python web_app/prepare_data.py
```

This loads the trained model, runs inference on the full dataset, and generates the JSON file needed by the dashboard.

**3. Launch the Dashboard**

```bash
streamlit run web_app/app.py
```

The dashboard will open in your browser at `http://localhost:8501`, displaying interactive visualizations of model performance, state analysis, and time series with regime classifications.

**4. Generate Analysis Reports (Optional)**

```bash
python -m src.models.regime_analysis
```

This creates additional visualizations and analysis reports in the `results/visualizations/` directory, including transition matrix heatmaps and regime persistence statistics.

## Project Structure

```
DSU_HMM/
├── src/                        # Core library code
│   ├── models/                 # HMM model implementations
│   │   ├── hmm_model.py       # Main HMM class with Viterbi and Baum-Welch
│   │   ├── base_hmm.py        # Abstract base class
│   │   ├── discrete_hmm.py    # Discrete HMM implementation
│   │   ├── trainer.py         # Training logic
│   │   ├── utils.py           # Utility functions and regime mapping
│   │   ├── hyperparameter_test.py # Model training script
│   │   └── regime_analysis.py # Regime transition analysis script
│   └── data/                   # Data processing
│       ├── data_processor.py  # Data loader and preprocessing
│       └── feature_engineering.py # Technical indicators
├── web_app/                    # Streamlit dashboard
│   ├── app.py                 # Dashboard application
│   └── prepare_data.py        # Dashboard data preparation
├── results/                    # Model outputs
│   ├── MODEL_METRICS.md
│   ├── optimized_hmm_classification_model.pt
│   ├── structured_emission_model_results.json
│   └── visualizations/        # Generated visualizations
├── data/                       # Raw data files
│   └── financial_data.csv
├── requirements.txt
├── pyproject.toml
└── README.md
```

The structure separates core model code (`src/`), application code (`web_app/`), outputs (`results/`), and raw data (`data/`), making it easy to navigate and understand the project organization.

## Future Improvements

## Future Improvements

There's definitely room to expand this project in some interesting directions. I'd like to experiment with more sophisticated HMM variants like hierarchical models that could capture nested market regimes, or time-varying transition probabilities that adapt as market conditions evolve. On the data side, incorporating alternative sources like sentiment analysis or options flow data could add another dimension beyond just price movements. The feature engineering could go deeper, as multi-timeframe analysis or cross-asset correlations might reveal patterns the current model misses. Eventually it'd be ideal to build this into something more practical, like a simple API endpoint for real-time regime classification or even a basic portfolio optimization tool that adjusts allocation based on detected regimes. The foundation is solid enough that these extensions feel achievable without starting from scratch.