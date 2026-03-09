"""
Regime Detection Module
Uses Hidden Markov Model with 4 states to classify market regimes
Features: returns, volatility, volume ratio, ATR, ADX
"""

import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

import config
from features import get_hmm_feature_columns


class RegimeDetector:
    """
    Hidden Markov Model for market regime detection

    States are learned unsupervised and named based on their statistical properties:
    - Low volatility / ranging
    - Trending up
    - Trending down
    - High volatility / crisis
    """

    def __init__(self, n_states: int = None):
        self.n_states = n_states or config.HMM_N_STATES
        self.model = None
        self.scaler = StandardScaler()
        self.regime_names = {}
        self.regime_stats = {}
        self.feature_columns = get_hmm_feature_columns()

    def _prepare_data(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare features for HMM"""
        features = df[self.feature_columns].copy()
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.ffill().bfill().fillna(0)
        return features.values

    def fit(self, df: pd.DataFrame) -> "RegimeDetector":
        """
        Train the HMM on historical data

        Args:
            df: DataFrame with OHLCV and computed features

        Returns:
            self
        """
        print(f"Training HMM with {self.n_states} states...")

        # Prepare data
        X = self._prepare_data(df)
        X_scaled = self.scaler.fit_transform(X)

        # Initialize and train HMM
        self.model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type=config.HMM_COVARIANCE_TYPE,
            n_iter=config.HMM_N_ITER,
            random_state=42
        )

        self.model.fit(X_scaled)

        # Get regime assignments
        regimes = self.model.predict(X_scaled)

        # Analyze and name each regime
        self._analyze_regimes(df, regimes)

        print(f"✅ HMM trained, log-likelihood: {self.model.score(X_scaled):.2f}")

        return self

    def _analyze_regimes(self, df: pd.DataFrame, regimes: np.ndarray):
        """
        Analyze each regime's statistical properties and assign names
        """
        df = df.copy()
        df["regime"] = regimes

        self.regime_stats = {}

        for regime in range(self.n_states):
            regime_df = df[df["regime"] == regime]

            if len(regime_df) == 0:
                continue

            stats = {
                "count": len(regime_df),
                "pct": len(regime_df) / len(df) * 100,
                "mean_return": regime_df["return_1h"].mean() if "return_1h" in regime_df else 0,
                "volatility": regime_df["volatility_24h"].mean() if "volatility_24h" in regime_df else 0,
                "mean_volume_ratio": regime_df["volume_ratio"].mean() if "volume_ratio" in regime_df else 1,
                "mean_atr": regime_df["atr"].mean() if "atr" in regime_df else 0,
                "mean_adx": regime_df["adx"].mean() if "adx" in regime_df else 0,
            }

            self.regime_stats[regime] = stats

        # Name regimes based on characteristics
        self._name_regimes()

    def _name_regimes(self):
        """Assign descriptive names to each regime based on statistics"""
        if not self.regime_stats:
            return

        # Sort by volatility
        sorted_by_vol = sorted(
            self.regime_stats.items(),
            key=lambda x: x[1]["volatility"]
        )

        # Sort by return
        sorted_by_ret = sorted(
            self.regime_stats.items(),
            key=lambda x: x[1]["mean_return"]
        )

        # Assign names
        for regime, stats in self.regime_stats.items():
            vol = stats["volatility"]
            ret = stats["mean_return"]
            adx = stats["mean_adx"]

            # Determine volatility level
            vol_rank = [r for r, _ in sorted_by_vol].index(regime)

            if vol_rank == 0:
                vol_desc = "Low Vol"
            elif vol_rank == self.n_states - 1:
                vol_desc = "High Vol"
            else:
                vol_desc = "Med Vol"

            # Determine trend
            if ret > 0.001:
                trend_desc = "Bullish"
            elif ret < -0.001:
                trend_desc = "Bearish"
            else:
                trend_desc = "Neutral"

            # Determine if trending or ranging
            if adx > 25:
                market_type = "Trending"
            else:
                market_type = "Ranging"

            self.regime_names[regime] = f"{vol_desc} {trend_desc} ({market_type})"

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict regime for each row in dataframe

        Returns:
            Array of regime labels (0 to n_states-1)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        X = self._prepare_data(df)
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict regime probabilities for each row

        Returns:
            Array of shape (n_samples, n_states) with probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        X = self._prepare_data(df)
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def get_current_regime(self, df: pd.DataFrame) -> Tuple[int, str, Dict]:
        """
        Get the current regime (last row)

        Returns:
            (regime_id, regime_name, regime_stats)
        """
        regimes = self.predict(df)
        current = regimes[-1]
        return (
            current,
            self.regime_names.get(current, f"Regime {current}"),
            self.regime_stats.get(current, {})
        )

    def get_transition_matrix(self) -> np.ndarray:
        """Get the transition probability matrix"""
        if self.model is None:
            return np.array([])
        return self.model.transmat_

    def save(self, path: str = None):
        """Save the trained model"""
        if path is None:
            path = config.HMM_MODEL_PATH

        data = {
            "model": self.model,
            "scaler": self.scaler,
            "regime_names": self.regime_names,
            "regime_stats": self.regime_stats,
            "feature_columns": self.feature_columns,
            "n_states": self.n_states
        }
        joblib.dump(data, path)
        print(f"✅ HMM model saved to {path}")

    def load(self, path: str = None) -> "RegimeDetector":
        """Load a trained model"""
        if path is None:
            path = config.HMM_MODEL_PATH

        data = joblib.load(path)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.regime_names = data["regime_names"]
        self.regime_stats = data["regime_stats"]
        self.feature_columns = data["feature_columns"]
        self.n_states = data["n_states"]

        print(f"✅ HMM model loaded from {path}")
        return self

    def print_regime_summary(self):
        """Print summary of all regimes"""
        print("\n" + "=" * 60)
        print("REGIME SUMMARY")
        print("=" * 60)

        for regime in range(self.n_states):
            stats = self.regime_stats.get(regime, {})
            name = self.regime_names.get(regime, f"Regime {regime}")

            print(f"\n📊 Regime {regime}: {name}")
            print(f"   Occurrence: {stats.get('count', 0):,} candles ({stats.get('pct', 0):.1f}%)")
            print(f"   Mean Return: {stats.get('mean_return', 0)*100:.3f}%")
            print(f"   Volatility: {stats.get('volatility', 0)*100:.3f}%")
            print(f"   Avg Volume Ratio: {stats.get('mean_volume_ratio', 1):.2f}x")
            print(f"   Avg ADX: {stats.get('mean_adx', 0):.1f}")

        # Print transition matrix
        trans = self.get_transition_matrix()
        if trans.size > 0:
            print("\n📈 Transition Probabilities:")
            print("   From\\To  " + "  ".join([f"R{i}" for i in range(self.n_states)]))
            for i in range(self.n_states):
                probs = "  ".join([f"{p:.2f}" for p in trans[i]])
                print(f"   R{i}      {probs}")


def add_regime_to_dataframe(df: pd.DataFrame, detector: RegimeDetector) -> pd.DataFrame:
    """Add regime labels to dataframe"""
    df = df.copy()
    df["regime"] = detector.predict(df)
    return df


if __name__ == "__main__":
    print("Testing regime detection module...")

    # Create sample data
    np.random.seed(42)
    n = 1000
    dates = pd.date_range(start="2024-01-01", periods=n, freq="1h")

    # Simulate different market regimes
    close = [40000]
    regime_true = []
    for i in range(n - 1):
        # Switch regimes periodically
        if i % 200 < 50:  # Low vol ranging
            change = np.random.randn() * 50
            regime_true.append(0)
        elif i % 200 < 100:  # Trending up
            change = np.random.randn() * 100 + 30
            regime_true.append(1)
        elif i % 200 < 150:  # High vol
            change = np.random.randn() * 200
            regime_true.append(2)
        else:  # Trending down
            change = np.random.randn() * 100 - 30
            regime_true.append(3)
        close.append(close[-1] + change)
    regime_true.append(regime_true[-1])

    df = pd.DataFrame({
        "timestamp": dates,
        "open": close,
        "high": [c + abs(np.random.randn() * 100) for c in close],
        "low": [c - abs(np.random.randn() * 100) for c in close],
        "close": close,
        "volume": [1000000 * (1 + np.random.rand()) for _ in range(n)]
    })

    # Compute features
    from features import compute_all_features
    df = compute_all_features(df, include_sentiment=False)

    # Train HMM
    detector = RegimeDetector(n_states=4)
    detector.fit(df)

    # Print summary
    detector.print_regime_summary()

    # Add regime to dataframe
    df = add_regime_to_dataframe(df, detector)
    print(f"\nRegime distribution:\n{df['regime'].value_counts().sort_index()}")

    # Test save/load
    detector.save("models/test_hmm.pkl")
    detector2 = RegimeDetector()
    detector2.load("models/test_hmm.pkl")
    print("\n✅ Save/load test passed")

    import os
    os.remove("models/test_hmm.pkl")
