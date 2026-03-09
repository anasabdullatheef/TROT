"""
Order Flow Backtest
Uses leading indicators instead of lagging technical indicators

Features:
- Order flow imbalance (bid vs ask volume)
- Trade flow toxicity (VPIN)
- Funding rate momentum (8h, 24h, 72h rate of change)
- Liquidation cascade proximity
- Cross-exchange price divergence
- Spot vs futures basis

Removed:
- RSI, MACD, Stochastic, EMA, Bollinger Bands (useless lagging indicators)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
from typing import Dict, Tuple, List
from sklearn.metrics import roc_auc_score, accuracy_score
import torch

import config
from exchange import fetch_historical_ohlcv
from funding import get_historical_funding_rates, classify_funding_signal
from orderflow_features import compute_orderflow_features, get_orderflow_feature_columns
from regime import RegimeDetector, add_regime_to_dataframe
from models.lstm_model import RegimeLSTM, RegimeLSTMEnsemble, LSTMModel, TimeSeriesDataset
from risk import calculate_max_drawdown, calculate_sharpe_ratio, calculate_sortino_ratio
from logger import BacktestLogger
from torch.utils.data import DataLoader


# Constants
POSITION_SIZE_PCT = 0.02
TRADING_FEE = 0.001
SLIPPAGE = 0.0005
TOTAL_COST_PCT = TRADING_FEE + SLIPPAGE
MAX_DRAWDOWN_LIMIT = 0.20
TRAIN_MONTHS = 18
TEST_MONTHS = 6


def split_train_test(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Strict time-based train/test split"""
    df = df.sort_values("timestamp").reset_index(drop=True)

    total_days = (df["timestamp"].max() - df["timestamp"].min()).days
    train_days = int(total_days * TRAIN_MONTHS / (TRAIN_MONTHS + TEST_MONTHS))
    cutoff_date = df["timestamp"].min() + timedelta(days=train_days)

    train_df = df[df["timestamp"] < cutoff_date].copy()
    test_df = df[df["timestamp"] >= cutoff_date].copy()

    print(f"\n📅 Train/Test Split:")
    print(f"   Train: {train_df['timestamp'].min().date()} to {train_df['timestamp'].max().date()} ({len(train_df)} candles)")
    print(f"   Test:  {test_df['timestamp'].min().date()} to {test_df['timestamp'].max().date()} ({len(test_df)} candles)")

    return train_df, test_df


class OrderFlowLSTM:
    """LSTM trained on order flow features only"""

    def __init__(self, regime_id: int, feature_columns: List[str] = None):
        self.regime_id = regime_id
        self.feature_columns = feature_columns or get_orderflow_feature_columns()
        self.model = None
        self.scaler = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.accuracy = 0.0
        self.auc = 0.0
        self.trained = False
        self.trained_columns = []

    def train(
        self,
        df: pd.DataFrame,
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.2
    ) -> Dict:
        """Train LSTM on order flow features"""
        from sklearn.preprocessing import StandardScaler

        # Filter for this regime
        regime_df = df[df["regime"] == self.regime_id].copy()

        if len(regime_df) < config.LSTM_SEQUENCE_LENGTH + 10:
            print(f"   ⚠️ Not enough data for regime {self.regime_id}: {len(regime_df)} samples")
            return {"accuracy": 0, "auc": 0.5, "samples": len(regime_df)}

        # Select only order flow features (no technical indicators)
        available_cols = [c for c in self.feature_columns if c in regime_df.columns]
        self.trained_columns = available_cols

        features = regime_df[available_cols].copy()
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.ffill().bfill().fillna(0)

        # Create target: positive return in next 4 candles
        regime_df["target"] = (
            regime_df["close"].shift(-config.LSTM_PREDICTION_HORIZON) > regime_df["close"]
        ).astype(float)

        valid_idx = ~regime_df["target"].isna()
        X = features[valid_idx].values
        y = regime_df.loc[valid_idx, "target"].values

        if len(X) < config.LSTM_SEQUENCE_LENGTH + 10:
            return {"accuracy": 0, "auc": 0.5, "samples": len(X)}

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Split train/validation (time-based)
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

        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=config.LSTM_LEARNING_RATE)

        # Training loop
        best_val_loss = float("inf")
        patience = 10
        patience_counter = 0

        for epoch in range(epochs):
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

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

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
            "features": len(available_cols)
        }

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predict probability of positive return"""
        if not self.trained or self.model is None:
            return np.array([0.5] * len(df))

        available_cols = [c for c in self.trained_columns if c in df.columns]
        if len(available_cols) != len(self.trained_columns):
            return np.array([0.5] * len(df))

        features = df[available_cols].copy()
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.ffill().bfill().fillna(0)

        X = self.scaler.transform(features.values)

        predictions = []
        self.model.eval()

        with torch.no_grad():
            for i in range(config.LSTM_SEQUENCE_LENGTH, len(X) + 1):
                seq = X[i - config.LSTM_SEQUENCE_LENGTH:i]
                seq_tensor = torch.FloatTensor(seq).unsqueeze(0).to(self.device)
                pred = self.model(seq_tensor).cpu().numpy()[0]
                predictions.append(pred)

        padding = [0.5] * (len(df) - len(predictions))
        return np.array(padding + predictions)


def train_orderflow_models(train_df: pd.DataFrame) -> Tuple[RegimeDetector, Dict[int, OrderFlowLSTM]]:
    """Train HMM and LSTM models on order flow features"""
    print("\n" + "=" * 70)
    print("TRAINING MODELS (ORDER FLOW FEATURES ONLY)")
    print("=" * 70)

    # Train HMM (uses price data for regime detection)
    print("\n🔧 Training HMM Regime Detector...")
    regime_detector = RegimeDetector(n_states=config.HMM_N_STATES)
    regime_detector.fit(train_df)

    # Add regime labels
    train_df = add_regime_to_dataframe(train_df, regime_detector)

    # Train LSTM for each regime with order flow features
    print("\n🔧 Training LSTM Models with Order Flow Features...")
    print(f"   Features: {get_orderflow_feature_columns()}")

    lstm_models = {}
    results = {}

    for regime in range(config.HMM_N_STATES):
        print(f"\n   Regime {regime}:")
        model = OrderFlowLSTM(regime_id=regime)
        metrics = model.train(train_df)
        lstm_models[regime] = model
        results[regime] = metrics

        auc_status = "✅" if metrics['auc'] >= 0.60 else "⚠️"
        print(f"      {auc_status} AUC: {metrics['auc']:.3f} | Acc: {metrics['accuracy']:.2%} | Samples: {metrics['samples']}")

    return regime_detector, lstm_models


def run_orderflow_backtest(
    test_df: pd.DataFrame,
    regime_detector: RegimeDetector,
    lstm_models: Dict[int, OrderFlowLSTM],
    initial_capital: float = None
) -> Tuple[Dict, pd.DataFrame, BacktestLogger]:
    """Run backtest with order flow features"""
    if initial_capital is None:
        initial_capital = config.INITIAL_CAPITAL

    print("\n" + "=" * 70)
    print("RUNNING BACKTEST (ORDER FLOW FEATURES)")
    print("=" * 70)

    # Add regimes
    test_df = add_regime_to_dataframe(test_df, regime_detector)

    # Pre-compute predictions
    print("Pre-computing predictions...")
    predictions = {}
    for regime_id, model in lstm_models.items():
        if model.trained:
            preds = model.predict_proba(test_df)
            predictions[regime_id] = preds
        else:
            predictions[regime_id] = np.full(len(test_df), 0.5)

    # Map predictions
    test_df["lstm_pred"] = 0.5
    for i, (idx, row) in enumerate(test_df.iterrows()):
        regime = row.get("regime", 0)
        if regime in predictions and i < len(predictions[regime]):
            test_df.loc[idx, "lstm_pred"] = predictions[regime][i]

    # Trading simulation
    capital = initial_capital
    peak_capital = initial_capital
    position = None
    equity_curve = []
    logger = BacktestLogger()
    circuit_breaker_triggered = False

    print("Running simulation...")

    for i in range(config.LSTM_SEQUENCE_LENGTH, len(test_df)):
        row = test_df.iloc[i]
        ts = row["timestamp"]
        price = row["close"]
        atr = row.get("atr", price * 0.02) or price * 0.02
        funding_rate = row.get("funding_rate", 0) or 0
        regime = row.get("regime", 0)
        lstm_pred = row.get("lstm_pred", 0.5)

        if pd.isna(atr) or atr < 1:
            continue

        # Check circuit breaker
        current_dd = (peak_capital - capital) / peak_capital if peak_capital > 0 else 0
        if current_dd >= MAX_DRAWDOWN_LIMIT and not circuit_breaker_triggered:
            circuit_breaker_triggered = True
            print(f"\n⛔ CIRCUIT BREAKER at {ts}: DD {current_dd*100:.1f}%")
            if position:
                # Close position
                if position["type"] == "long":
                    pnl = (price * (1 - TOTAL_COST_PCT) - position["entry_price"]) * position["size"]
                else:
                    pnl = (position["entry_price"] - price * (1 + TOTAL_COST_PCT)) * position["size"]
                capital += pnl
                logger.log_trade({
                    "entry_time": position["entry_time"], "exit_time": ts,
                    "entry_price": position["entry_price"], "exit_price": price,
                    "position_type": position["type"], "size": position["size"],
                    "pnl": pnl, "pnl_pct": pnl / (position["entry_price"] * position["size"]),
                    "exit_reason": "Circuit breaker", "regime": position["regime"],
                    "win": pnl > 0
                })
                position = None

        if circuit_breaker_triggered:
            equity_curve.append({"timestamp": ts, "capital": capital, "regime": regime})
            continue

        # Check exit
        if position is not None:
            if position["type"] == "long":
                raw_pnl_pct = (price - position["entry_price"]) / position["entry_price"]
            else:
                raw_pnl_pct = (position["entry_price"] - price) / position["entry_price"]

            tp_pct = (atr * config.ATR_TAKE_PROFIT_MULTIPLIER) / position["entry_price"]
            sl_pct = (atr * config.ATR_STOP_LOSS_MULTIPLIER) / position["entry_price"]

            exit_reason = None
            if raw_pnl_pct >= tp_pct:
                exit_reason = f"TP: {raw_pnl_pct*100:.2f}%"
            elif raw_pnl_pct <= -sl_pct:
                exit_reason = f"SL: {raw_pnl_pct*100:.2f}%"

            if exit_reason:
                if position["type"] == "long":
                    exit_price = price * (1 - TOTAL_COST_PCT)
                    pnl = (exit_price - position["entry_price"]) * position["size"]
                else:
                    exit_price = price * (1 + TOTAL_COST_PCT)
                    pnl = (position["entry_price"] - exit_price) * position["size"]

                capital += pnl
                logger.log_trade({
                    "entry_time": position["entry_time"], "exit_time": ts,
                    "entry_price": position["entry_price"], "exit_price": exit_price,
                    "position_type": position["type"], "size": position["size"],
                    "pnl": pnl, "pnl_pct": pnl / (position["entry_price"] * position["size"]),
                    "exit_reason": exit_reason, "regime": position["regime"],
                    "win": pnl > 0
                })
                position = None

        # Check entry
        if position is None and not circuit_breaker_triggered:
            confidence = lstm_pred
            if confidence > 0.5:
                direction = "long"
            else:
                direction = "short"
                confidence = 1 - confidence

            # Funding boost
            funding_signal = classify_funding_signal(funding_rate)
            if funding_signal == direction:
                confidence = min(confidence + config.FUNDING_BOOST_WEIGHT, 1.0)

            if confidence >= config.LSTM_CONFIDENCE_THRESHOLD:
                position_value = capital * POSITION_SIZE_PCT

                if direction == "long":
                    entry_price = price * (1 + TOTAL_COST_PCT)
                else:
                    entry_price = price * (1 - TOTAL_COST_PCT)

                position = {
                    "type": direction,
                    "entry_price": entry_price,
                    "size": position_value / entry_price,
                    "entry_time": ts,
                    "regime": regime,
                    "confidence": confidence
                }

        # Track equity
        unrealized = 0
        if position:
            if position["type"] == "long":
                unrealized = (price - position["entry_price"]) * position["size"]
            else:
                unrealized = (position["entry_price"] - price) * position["size"]

        equity = capital + unrealized
        peak_capital = max(peak_capital, equity)
        equity_curve.append({"timestamp": ts, "capital": equity, "regime": regime})

        if i % 1000 == 0:
            print(f"  {i}/{len(test_df)} | Capital: ${equity:,.0f}")

    # Close remaining
    if position:
        last_price = test_df.iloc[-1]["close"]
        if position["type"] == "long":
            pnl = (last_price * (1 - TOTAL_COST_PCT) - position["entry_price"]) * position["size"]
        else:
            pnl = (position["entry_price"] - last_price * (1 + TOTAL_COST_PCT)) * position["size"]
        capital += pnl
        logger.log_trade({
            "entry_time": position["entry_time"], "exit_time": test_df.iloc[-1]["timestamp"],
            "entry_price": position["entry_price"], "exit_price": last_price,
            "position_type": position["type"], "size": position["size"],
            "pnl": pnl, "pnl_pct": pnl / (position["entry_price"] * position["size"]),
            "exit_reason": "End of backtest", "regime": position["regime"],
            "win": pnl > 0
        })

    # Stats
    equity_df = pd.DataFrame(equity_curve)
    if not equity_df.empty:
        running_max = equity_df["capital"].cummax()
        equity_df["drawdown"] = (equity_df["capital"] - running_max) / running_max
        equity_df["returns"] = equity_df["capital"].pct_change().fillna(0)

    stats = calculate_stats(initial_capital, capital, equity_df, logger, circuit_breaker_triggered)

    return stats, equity_df, logger


def calculate_stats(initial, final, equity_df, logger, circuit_breaker) -> Dict:
    """Calculate statistics"""
    trades = logger.trades
    total_pnl = final - initial
    return_pct = (final / initial - 1) * 100

    winning = [t for t in trades if t["pnl"] > 0]
    losing = [t for t in trades if t["pnl"] <= 0]
    gross_profit = sum(t["pnl"] for t in winning) if winning else 0
    gross_loss = abs(sum(t["pnl"] for t in losing)) if losing else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

    stats = {
        "initial_capital": initial,
        "final_capital": final,
        "total_pnl": total_pnl,
        "return_pct": return_pct,
        "total_trades": len(trades),
        "winning_trades": len(winning),
        "losing_trades": len(losing),
        "win_rate": len(winning) / len(trades) * 100 if trades else 0,
        "profit_factor": profit_factor,
        "circuit_breaker_triggered": circuit_breaker
    }

    if not equity_df.empty:
        stats["sharpe"] = calculate_sharpe_ratio(equity_df["returns"])
        stats["sortino"] = calculate_sortino_ratio(equity_df["returns"])
        stats["max_drawdown"] = calculate_max_drawdown(equity_df["capital"]) * 100

    return stats


def print_auc_report(results: Dict):
    """Print AUC results per regime"""
    print("\n" + "=" * 70)
    print("📊 MODEL AUC RESULTS (Order Flow Features)")
    print("=" * 70)

    all_above_60 = True
    for regime, metrics in sorted(results.items()):
        auc = metrics.get("auc", 0.5)
        acc = metrics.get("accuracy", 0)
        samples = metrics.get("samples", 0)

        if auc >= 0.60:
            status = "✅ GOOD"
        elif auc >= 0.55:
            status = "⚠️ MARGINAL"
            all_above_60 = False
        else:
            status = "❌ POOR"
            all_above_60 = False

        print(f"Regime {regime}: AUC={auc:.3f} Acc={acc:.2%} Samples={samples:,} {status}")

    print("\n" + "-" * 50)
    if all_above_60:
        print("✅ All regimes have AUC >= 0.60 - Worth pursuing further")
    else:
        print("⚠️ Some regimes have AUC < 0.60 - Strategy needs improvement")

    return all_above_60


def main():
    """Main execution"""
    print("🚀 ATLAS BOT - ORDER FLOW BACKTEST")
    print("=" * 70)
    print("\nFeatures Used (Leading Indicators):")
    print("  ✓ Funding rate momentum (8h, 24h, 72h)")
    print("  ✓ Order flow imbalance (estimated)")
    print("  ✓ Trade toxicity / VPIN (estimated)")
    print("  ✓ Liquidation cascade proximity")
    print("  ✓ Cross-exchange divergence")
    print("  ✓ Spot-futures basis")
    print("\nRemoved (Lagging Indicators):")
    print("  ✗ RSI, MACD, Stochastic, EMA, Bollinger Bands")

    # Fetch data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=config.LOOKBACK_DAYS)

    print(f"\nPeriod: {start_date.date()} to {end_date.date()}")

    print("\n" + "=" * 70)
    print("FETCHING DATA")
    print("=" * 70)

    ohlcv_df = fetch_historical_ohlcv(
        symbol=config.BINANCE_SYMBOL,
        start_date=start_date,
        end_date=end_date
    )
    print(f"✅ OHLCV: {len(ohlcv_df)} candles")

    funding_df = get_historical_funding_rates(
        start_time=start_date,
        end_time=end_date
    )
    print(f"✅ Funding: {len(funding_df)} records")

    # Compute order flow features
    print("\n" + "=" * 70)
    print("COMPUTING ORDER FLOW FEATURES")
    print("=" * 70)
    df = compute_orderflow_features(ohlcv_df, funding_df, include_sentiment=True)

    # We still need ATR for stop loss calculation
    df['atr'] = df['high'].rolling(14).max() - df['low'].rolling(14).min()

    print(f"✅ Features: {len(get_orderflow_feature_columns())} order flow features")

    # Split
    train_df, test_df = split_train_test(df)

    # Train models
    regime_detector, lstm_models = train_orderflow_models(train_df)

    # Collect AUC results
    results = {}
    for regime, model in lstm_models.items():
        if "regime" in train_df.columns:
            samples = len(train_df[train_df["regime"] == regime])
        else:
            samples = 0
        results[regime] = {"auc": model.auc, "accuracy": model.accuracy, "samples": samples}

    # Print AUC report
    auc_ok = print_auc_report(results)

    # Run backtest regardless
    stats, equity_df, logger = run_orderflow_backtest(test_df, regime_detector, lstm_models)

    # Print results
    print("\n" + "=" * 70)
    print("📊 BACKTEST RESULTS (6-Month Test Period)")
    print("=" * 70)

    annual_return = stats["return_pct"] * 2

    print(f"\n💰 PERFORMANCE")
    print(f"   Initial Capital:     ${stats['initial_capital']:,.2f}")
    print(f"   Final Capital:       ${stats['final_capital']:,.2f}")
    print(f"   Total Return:        {stats['return_pct']:+.2f}%")
    print(f"   Annualized Return:   {annual_return:+.2f}%")

    print(f"\n📈 RISK METRICS")
    print(f"   Sharpe Ratio:        {stats.get('sharpe', 0):.2f}")
    print(f"   Sortino Ratio:       {stats.get('sortino', 0):.2f}")
    print(f"   Max Drawdown:        {stats.get('max_drawdown', 0):.2f}%")
    print(f"   Profit Factor:       {stats['profit_factor']:.2f}")

    print(f"\n🎯 TRADE STATISTICS")
    print(f"   Total Trades:        {stats['total_trades']}")
    print(f"   Win Rate:            {stats['win_rate']:.1f}%")

    if stats.get("circuit_breaker_triggered"):
        print(f"\n⛔ Circuit breaker triggered (20% max DD)")

    # Save chart
    os.makedirs("data", exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(equity_df["timestamp"], equity_df["capital"], color="#00B5AD")
    ax.axhline(y=config.INITIAL_CAPITAL, color="gray", linestyle="--")
    ax.set_title(f"Order Flow Strategy - Return: {stats['return_pct']:.2f}%")
    ax.set_ylabel("Capital")
    ax.grid(True, alpha=0.3)
    plt.savefig("data/orderflow_backtest.png", dpi=150)
    plt.close()

    print(f"\n📊 Chart saved to data/orderflow_backtest.png")
    print("\n" + "=" * 70)
    print("✅ ORDER FLOW BACKTEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
