"""
LSTM Model Module
One LSTM per regime for predicting positive 4h returns
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from typing import Dict, List, Tuple, Optional
import joblib
import os

import config
from features import get_feature_columns


class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series data"""

    def __init__(self, X: np.ndarray, y: np.ndarray, sequence_length: int):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.X) - self.sequence_length

    def __getitem__(self, idx):
        return (
            self.X[idx:idx + self.sequence_length],
            self.y[idx + self.sequence_length - 1]
        )


class LSTMModel(nn.Module):
    """LSTM model for binary classification (positive return prediction)"""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = None,
        num_layers: int = None,
        dropout: float = None
    ):
        super().__init__()

        self.hidden_size = hidden_size or config.LSTM_HIDDEN_SIZE
        self.num_layers = num_layers or config.LSTM_NUM_LAYERS
        dropout = dropout or config.LSTM_DROPOUT

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=dropout if self.num_layers > 1 else 0
        )

        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # LSTM output
        lstm_out, _ = self.lstm(x)

        # Take the last time step
        last_hidden = lstm_out[:, -1, :]

        # Fully connected layers
        output = self.fc(last_hidden)
        return output.squeeze(-1)  # Only squeeze last dimension to preserve batch


class RegimeLSTM:
    """
    LSTM model for a specific market regime
    Trains on candles from that regime only
    """

    def __init__(self, regime_id: int, feature_columns: List[str] = None):
        self.regime_id = regime_id
        self.feature_columns = feature_columns or get_feature_columns()
        self.model = None
        self.scaler = StandardScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.accuracy = 0.0
        self.auc = 0.0
        self.trained = False
        self.trained_columns = []  # Columns actually used in training

    def _prepare_data(
        self,
        df: pd.DataFrame,
        include_sentiment: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare features and target for training"""
        # Filter for this regime
        regime_df = df[df["regime"] == self.regime_id].copy()

        if len(regime_df) < config.LSTM_SEQUENCE_LENGTH + 10:
            return np.array([]), np.array([]), []

        # Select features
        cols = self.feature_columns.copy()
        if not include_sentiment:
            cols = [c for c in cols if c not in ["fng_value", "finbert_1h", "finbert_4h"]]

        available_cols = [c for c in cols if c in regime_df.columns]

        # Get features
        features = regime_df[available_cols].copy()
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.ffill().bfill().fillna(0)

        # Create target: positive return in next LSTM_PREDICTION_HORIZON candles
        regime_df["target"] = (
            regime_df["close"].shift(-config.LSTM_PREDICTION_HORIZON) > regime_df["close"]
        ).astype(float)

        # Remove rows without valid target
        valid_idx = ~regime_df["target"].isna()
        features = features[valid_idx].values
        target = regime_df.loc[valid_idx, "target"].values

        return features, target, available_cols

    def train(
        self,
        df: pd.DataFrame,
        include_sentiment: bool = True,
        epochs: int = None,
        batch_size: int = None,
        validation_split: float = 0.2
    ) -> Dict:
        """
        Train the LSTM model

        Returns:
            Training metrics dict
        """
        epochs = epochs or config.LSTM_EPOCHS
        batch_size = batch_size or config.LSTM_BATCH_SIZE

        X, y, used_cols = self._prepare_data(df, include_sentiment)

        if len(X) < config.LSTM_SEQUENCE_LENGTH + 10:
            print(f"⚠️ Not enough data for regime {self.regime_id}: {len(X)} samples")
            return {"accuracy": 0, "auc": 0, "samples": len(X)}

        # Store the columns used for training
        self.trained_columns = used_cols

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Split train/validation
        split_idx = int(len(X_scaled) * (1 - validation_split))
        X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Create datasets
        train_dataset = TimeSeriesDataset(X_train, y_train, config.LSTM_SEQUENCE_LENGTH)
        val_dataset = TimeSeriesDataset(X_val, y_val, config.LSTM_SEQUENCE_LENGTH)

        if len(train_dataset) < batch_size:
            batch_size = max(1, len(train_dataset))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Initialize model
        input_size = X_scaled.shape[1]
        self.model = LSTMModel(input_size=input_size).to(self.device)

        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=config.LSTM_LEARNING_RATE)

        # Training loop
        best_val_loss = float("inf")
        patience = 10
        patience_counter = 0

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # Validation
            self.model.eval()
            val_loss = 0
            val_preds = []
            val_targets = []

            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)

                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()

                    val_preds.extend(outputs.cpu().numpy())
                    val_targets.extend(batch_y.cpu().numpy())

            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"    Early stopping at epoch {epoch + 1}")
                    break

            if (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Calculate final metrics
        val_preds_binary = (np.array(val_preds) > 0.5).astype(int)
        self.accuracy = accuracy_score(val_targets, val_preds_binary)

        try:
            self.auc = roc_auc_score(val_targets, val_preds)
        except:
            self.auc = 0.5

        self.trained = True

        return {
            "accuracy": self.accuracy,
            "auc": self.auc,
            "samples": len(X),
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset)
        }

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict probability of positive return

        Returns:
            Array of probabilities
        """
        if not self.trained or self.model is None:
            return np.array([0.5] * len(df))

        # Use the exact columns that were used during training
        if self.trained_columns:
            available_cols = [c for c in self.trained_columns if c in df.columns]
        else:
            # Fallback to feature columns
            cols = self.feature_columns.copy()
            available_cols = [c for c in cols if c in df.columns]

        features = df[available_cols].copy()
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.ffill().bfill().fillna(0)

        # Ensure we have the right number of features
        n_features = self.scaler.n_features_in_
        if features.shape[1] != n_features:
            return np.array([0.5] * len(df))

        X = self.scaler.transform(features.values)

        # Create sequences
        predictions = []
        self.model.eval()

        with torch.no_grad():
            for i in range(config.LSTM_SEQUENCE_LENGTH, len(X) + 1):
                seq = X[i - config.LSTM_SEQUENCE_LENGTH:i]
                seq_tensor = torch.FloatTensor(seq).unsqueeze(0).to(self.device)
                pred = self.model(seq_tensor).cpu().numpy()[0]
                predictions.append(pred)

        # Pad the beginning with zeros
        padding = [0.5] * (len(df) - len(predictions))
        return np.array(padding + predictions)

    def save(self, path: str = None):
        """Save the model"""
        if path is None:
            path = config.LSTM_MODEL_PATH_TEMPLATE.format(self.regime_id)

        os.makedirs(os.path.dirname(path), exist_ok=True)

        data = {
            "model_state": self.model.state_dict() if self.model else None,
            "scaler": self.scaler,
            "feature_columns": self.feature_columns,
            "trained_columns": self.trained_columns,
            "accuracy": self.accuracy,
            "auc": self.auc,
            "trained": self.trained,
            "regime_id": self.regime_id
        }
        torch.save(data, path)
        print(f"✅ LSTM model for regime {self.regime_id} saved to {path}")

    def load(self, path: str = None) -> "RegimeLSTM":
        """Load a trained model"""
        if path is None:
            path = config.LSTM_MODEL_PATH_TEMPLATE.format(self.regime_id)

        data = torch.load(path, map_location=self.device, weights_only=False)

        self.scaler = data["scaler"]
        self.feature_columns = data["feature_columns"]
        self.trained_columns = data.get("trained_columns", [])
        self.accuracy = data["accuracy"]
        self.auc = data["auc"]
        self.trained = data["trained"]
        self.regime_id = data["regime_id"]

        if data["model_state"]:
            input_size = len(self.trained_columns) if self.trained_columns else len(self.feature_columns)
            self.model = LSTMModel(input_size=input_size).to(self.device)
            self.model.load_state_dict(data["model_state"])
            self.model.eval()

        print(f"✅ LSTM model for regime {self.regime_id} loaded (Acc: {self.accuracy:.2%}, AUC: {self.auc:.2f})")
        return self


class RegimeLSTMEnsemble:
    """Manages multiple LSTM models, one per regime"""

    def __init__(self, n_regimes: int = None):
        self.n_regimes = n_regimes or config.HMM_N_STATES
        self.models: Dict[int, RegimeLSTM] = {}

    def train_all(
        self,
        df: pd.DataFrame,
        include_sentiment: bool = True
    ) -> Dict:
        """
        Train LSTM for each regime

        Returns:
            Dict of metrics per regime
        """
        results = {}

        for regime in range(self.n_regimes):
            print(f"\n🔧 Training LSTM for Regime {regime}...")
            model = RegimeLSTM(regime_id=regime)
            metrics = model.train(df, include_sentiment=include_sentiment)
            self.models[regime] = model
            results[regime] = metrics
            print(f"   Accuracy: {metrics['accuracy']:.2%}, AUC: {metrics['auc']:.2f}, Samples: {metrics['samples']}")

        return results

    def predict(self, df: pd.DataFrame, regime: int) -> float:
        """Get prediction for current regime"""
        if regime not in self.models:
            return 0.5

        model = self.models[regime]
        if not model.trained:
            return 0.5

        probs = model.predict_proba(df)
        return probs[-1] if len(probs) > 0 else 0.5

    def save_all(self):
        """Save all models"""
        for regime, model in self.models.items():
            model.save()

    def load_all(self) -> "RegimeLSTMEnsemble":
        """Load all models"""
        for regime in range(self.n_regimes):
            path = config.LSTM_MODEL_PATH_TEMPLATE.format(regime)
            if os.path.exists(path):
                model = RegimeLSTM(regime_id=regime)
                model.load(path)
                self.models[regime] = model
        return self


if __name__ == "__main__":
    print("Testing LSTM model module...")

    # Create sample data with regime labels
    np.random.seed(42)
    n = 500

    df = pd.DataFrame({
        "timestamp": pd.date_range(start="2024-01-01", periods=n, freq="1h"),
        "close": 40000 + np.cumsum(np.random.randn(n) * 100),
        "return_1h": np.random.randn(n) * 0.01,
        "return_4h": np.random.randn(n) * 0.02,
        "return_24h": np.random.randn(n) * 0.05,
        "atr": np.random.rand(n) * 500,
        "volatility_24h": np.random.rand(n) * 0.02,
        "rsi": 30 + np.random.rand(n) * 40,
        "macd": np.random.randn(n) * 100,
        "macd_signal": np.random.randn(n) * 100,
        "stoch_k": np.random.rand(n) * 100,
        "stoch_d": np.random.rand(n) * 100,
        "adx": 10 + np.random.rand(n) * 30,
        "obv": np.cumsum(np.random.randn(n) * 1000000),
        "volume_ratio": 0.5 + np.random.rand(n),
        "funding_rate": np.random.randn(n) * 0.0001,
        "funding_8h_change": np.random.randn(n) * 0.0001,
        "funding_24h_cumulative": np.random.randn(n) * 0.0003,
        "fng_value": np.random.rand(n),
        "finbert_1h": np.random.randn(n) * 0.1,
        "finbert_4h": np.random.randn(n) * 0.1,
        "regime": np.random.randint(0, 4, n)
    })

    # Test single regime model
    print("\nTraining LSTM for regime 0...")
    model = RegimeLSTM(regime_id=0)
    metrics = model.train(df, epochs=5)
    print(f"Metrics: {metrics}")

    # Test prediction
    probs = model.predict_proba(df.tail(50))
    print(f"Last 5 predictions: {probs[-5:]}")

    # Test save/load
    model.save("models/test_lstm_0.pt")
    model2 = RegimeLSTM(regime_id=0)
    model2.load("models/test_lstm_0.pt")

    os.remove("models/test_lstm_0.pt")
    print("\n✅ LSTM module test passed")
