#!/usr/bin/env python3
"""
Tardis Backtest - Uses REAL order book and trade data from Tardis.dev

Features (all REAL, not simulated):
- Order book imbalance (bid vs ask volume in top 10 levels)
- VPIN (Volume-synchronized Probability of Informed Trading)
- Cross-exchange price divergence (Binance vs Bybit vs OKX)
- Funding rate momentum

This is the definitive test of whether order flow features have predictive power.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Tuple, List
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader

import config
from exchange import fetch_historical_ohlcv
from funding import get_historical_funding_rates
from orderflow_features import compute_hmm_features, calculate_funding_momentum
from regime import RegimeDetector, add_regime_to_dataframe
from models.lstm_model import LSTMModel, TimeSeriesDataset
from risk import calculate_max_drawdown, calculate_sharpe_ratio, calculate_sortino_ratio
from logger import BacktestLogger
from funding import classify_funding_signal
from tardis_download import load_and_process_tardis_data, get_available_data_info

# Constants
POSITION_SIZE_PCT = 0.02
TOTAL_COST_PCT = 0.0015
MAX_DRAWDOWN_LIMIT = 0.20
TRAIN_MONTHS = 18
TEST_MONTHS = 6
DATA_DIR = Path("data/tardis")


def get_tardis_feature_columns() -> List[str]:
    """Real order flow features from Tardis data"""
    return [
        # Real order book features
        'orderbook_imbalance',
        'spread',

        # Real trade flow features
        'vpin',
        'trade_imbalance',

        # Real cross-exchange divergence
        'max_divergence',

        # Funding rate momentum (from Binance API)
        'funding_rate',
        'funding_8h_momentum',
        'funding_24h_momentum',
        'funding_72h_momentum',
        'funding_zscore',
    ]


def load_tardis_features() -> pd.DataFrame:
    """Load pre-processed Tardis features"""
    processed_path = DATA_DIR / "processed_features.parquet"

    if processed_path.exists():
        print(f"📂 Loading processed Tardis data from {processed_path}")
        return pd.read_parquet(processed_path)

    print("⚠️ No processed Tardis data found")
    print("   Run: python tardis_download.py --download")
    print("   Then: python tardis_download.py --process")
    return pd.DataFrame()


def merge_tardis_with_ohlcv(ohlcv_df: pd.DataFrame, tardis_df: pd.DataFrame) -> pd.DataFrame:
    """Merge Tardis features with OHLCV data"""
    if tardis_df.empty:
        return ohlcv_df

    ohlcv_df = ohlcv_df.copy()
    ohlcv_df['timestamp'] = pd.to_datetime(ohlcv_df['timestamp'])

    tardis_df = tardis_df.copy()
    tardis_df['timestamp'] = pd.to_datetime(tardis_df['timestamp'])

    # Merge on timestamp (forward fill Tardis data to match OHLCV frequency)
    merged = pd.merge_asof(
        ohlcv_df.sort_values('timestamp'),
        tardis_df.sort_values('timestamp'),
        on='timestamp',
        direction='backward'
    )

    return merged


def split_train_test(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Strict time-based split"""
    df = df.sort_values("timestamp").reset_index(drop=True)
    total_days = (df["timestamp"].max() - df["timestamp"].min()).days
    train_days = int(total_days * TRAIN_MONTHS / (TRAIN_MONTHS + TEST_MONTHS))
    cutoff = df["timestamp"].min() + timedelta(days=train_days)

    train_df = df[df["timestamp"] < cutoff].copy()
    test_df = df[df["timestamp"] >= cutoff].copy()

    print(f"\n📅 Train/Test Split:")
    print(f"   Train: {train_df['timestamp'].min().date()} to {train_df['timestamp'].max().date()} ({len(train_df)})")
    print(f"   Test:  {test_df['timestamp'].min().date()} to {test_df['timestamp'].max().date()} ({len(test_df)})")

    return train_df, test_df


class TardisLSTM:
    """LSTM trained on real Tardis order flow features"""

    def __init__(self, regime_id: int, feature_columns: List[str] = None):
        self.regime_id = regime_id
        self.feature_columns = feature_columns or get_tardis_feature_columns()
        self.model = None
        self.scaler = StandardScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.accuracy = 0.0
        self.auc = 0.0
        self.trained = False
        self.trained_columns = []

    def train(self, df: pd.DataFrame, epochs: int = 50, batch_size: int = 32) -> Dict:
        """Train on real Tardis features"""
        regime_df = df[df["regime"] == self.regime_id].copy()

        if len(regime_df) < config.LSTM_SEQUENCE_LENGTH + 20:
            return {"accuracy": 0, "auc": 0.5, "samples": len(regime_df), "features": 0}

        # Get available features
        available = [c for c in self.feature_columns if c in regime_df.columns]
        self.trained_columns = available

        if len(available) < 3:
            print(f"   ⚠️ Only {len(available)} features available: {available}")
            return {"accuracy": 0, "auc": 0.5, "samples": len(regime_df), "features": len(available)}

        features = regime_df[available].copy()
        features = features.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)

        # Target: positive return in next 4 candles
        regime_df["target"] = (regime_df["close"].shift(-4) > regime_df["close"]).astype(float)
        valid = ~regime_df["target"].isna()

        X = features[valid].values
        y = regime_df.loc[valid, "target"].values

        if len(X) < config.LSTM_SEQUENCE_LENGTH + 20:
            return {"accuracy": 0, "auc": 0.5, "samples": len(X), "features": len(available)}

        # Scale and split
        X_scaled = self.scaler.fit_transform(X)
        split = int(len(X_scaled) * 0.8)
        X_train, X_val = X_scaled[:split], X_scaled[split:]
        y_train, y_val = y[:split], y[split:]

        train_ds = TimeSeriesDataset(X_train, y_train, config.LSTM_SEQUENCE_LENGTH)
        val_ds = TimeSeriesDataset(X_val, y_val, config.LSTM_SEQUENCE_LENGTH)

        if len(train_ds) < batch_size:
            batch_size = max(1, len(train_ds))

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)

        # Model
        self.model = LSTMModel(input_size=X_scaled.shape[1]).to(self.device)
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        best_loss = float("inf")
        patience = 0

        for epoch in range(epochs):
            self.model.train()
            for bx, by in train_loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(bx), by)
                loss.backward()
                optimizer.step()

            # Validate
            self.model.eval()
            val_loss = 0
            preds, targets = [], []
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(self.device), by.to(self.device)
                    out = self.model(bx)
                    val_loss += criterion(out, by).item()
                    preds.extend(out.cpu().numpy())
                    targets.extend(by.cpu().numpy())

            val_loss /= max(1, len(val_loader))
            if val_loss < best_loss:
                best_loss = val_loss
                patience = 0
            else:
                patience += 1
                if patience >= 10:
                    break

        # Metrics
        preds_bin = (np.array(preds) > 0.5).astype(int)
        self.accuracy = accuracy_score(targets, preds_bin)
        try:
            self.auc = roc_auc_score(targets, preds)
        except:
            self.auc = 0.5

        self.trained = True
        return {
            "accuracy": self.accuracy,
            "auc": self.auc,
            "samples": len(X),
            "features": len(available)
        }

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if not self.trained:
            return np.full(len(df), 0.5)

        available = [c for c in self.trained_columns if c in df.columns]
        if len(available) != len(self.trained_columns):
            return np.full(len(df), 0.5)

        features = df[available].replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)
        X = self.scaler.transform(features.values)

        preds = []
        self.model.eval()
        with torch.no_grad():
            for i in range(config.LSTM_SEQUENCE_LENGTH, len(X) + 1):
                seq = torch.FloatTensor(X[i - config.LSTM_SEQUENCE_LENGTH:i]).unsqueeze(0).to(self.device)
                preds.append(self.model(seq).cpu().numpy()[0])

        return np.array([0.5] * (len(df) - len(preds)) + preds)


def train_tardis_models(train_df: pd.DataFrame) -> Tuple[RegimeDetector, Dict, Dict]:
    """Train HMM and Tardis LSTM models"""
    print("\n" + "=" * 70)
    print("TRAINING MODELS (REAL TARDIS DATA)")
    print("=" * 70)

    # HMM
    print("\n🔧 Training HMM...")
    regime_detector = RegimeDetector(n_states=config.HMM_N_STATES)
    regime_detector.fit(train_df)
    train_df = add_regime_to_dataframe(train_df, regime_detector)

    # LSTM per regime
    print("\n🔧 Training LSTM with REAL order flow features...")
    features = get_tardis_feature_columns()
    available = [f for f in features if f in train_df.columns]
    print(f"   Available features ({len(available)}): {available}")

    models = {}
    results = {}

    for regime in range(config.HMM_N_STATES):
        model = TardisLSTM(regime_id=regime, feature_columns=available)
        metrics = model.train(train_df)
        models[regime] = model
        results[regime] = metrics

        status = "✅" if metrics["auc"] >= 0.60 else "⚠️" if metrics["auc"] >= 0.55 else "❌"
        print(f"   Regime {regime}: {status} AUC={metrics['auc']:.3f} Acc={metrics['accuracy']:.2%} "
              f"Samples={metrics['samples']} Features={metrics.get('features', 0)}")

    return regime_detector, models, results


def run_backtest(test_df, regime_detector, models) -> Tuple[Dict, pd.DataFrame]:
    """Run trading backtest"""
    print("\n" + "=" * 70)
    print("RUNNING BACKTEST")
    print("=" * 70)

    test_df = add_regime_to_dataframe(test_df, regime_detector)

    # Pre-compute predictions
    predictions = {}
    for regime, model in models.items():
        if model.trained:
            predictions[regime] = model.predict_proba(test_df)
        else:
            predictions[regime] = np.full(len(test_df), 0.5)

    test_df["lstm_pred"] = 0.5
    for i, (idx, row) in enumerate(test_df.iterrows()):
        regime = row.get("regime", 0)
        if regime in predictions and i < len(predictions[regime]):
            test_df.loc[idx, "lstm_pred"] = predictions[regime][i]

    # Simulation
    capital = config.INITIAL_CAPITAL
    peak = capital
    position = None
    equity = []
    logger = BacktestLogger()
    halted = False

    for i in range(config.LSTM_SEQUENCE_LENGTH, len(test_df)):
        row = test_df.iloc[i]
        ts, price = row["timestamp"], row["close"]
        atr = row.get("atr", price * 0.02) or price * 0.02
        regime = row.get("regime", 0)
        pred = row.get("lstm_pred", 0.5)
        funding = row.get("funding_rate", 0) or 0

        # Circuit breaker
        dd = (peak - capital) / peak if peak > 0 else 0
        if dd >= MAX_DRAWDOWN_LIMIT and not halted:
            halted = True
            if position:
                pnl = ((price - position["entry"]) if position["type"] == "long" else (position["entry"] - price)) * position["size"]
                pnl -= position["size"] * price * TOTAL_COST_PCT
                capital += pnl
                logger.log_trade({"pnl": pnl, "win": pnl > 0, "regime": regime,
                                  "entry_time": position["ts"], "exit_time": ts,
                                  "entry_price": position["entry"], "exit_price": price,
                                  "position_type": position["type"], "size": position["size"],
                                  "pnl_pct": pnl / (position["entry"] * position["size"]),
                                  "exit_reason": "Circuit breaker"})
                position = None

        if halted:
            equity.append({"timestamp": ts, "capital": capital})
            continue

        # Exit check
        if position:
            raw_pnl = ((price - position["entry"]) / position["entry"]) if position["type"] == "long" else ((position["entry"] - price) / position["entry"])
            tp = atr * 3 / position["entry"]
            sl = atr * 1.5 / position["entry"]

            if raw_pnl >= tp or raw_pnl <= -sl:
                pnl = raw_pnl * position["entry"] * position["size"]
                pnl -= position["size"] * price * TOTAL_COST_PCT
                capital += pnl
                logger.log_trade({"pnl": pnl, "win": pnl > 0, "regime": regime,
                                  "entry_time": position["ts"], "exit_time": ts,
                                  "entry_price": position["entry"], "exit_price": price,
                                  "position_type": position["type"], "size": position["size"],
                                  "pnl_pct": raw_pnl, "exit_reason": "TP" if raw_pnl >= tp else "SL"})
                position = None

        # Entry check
        if not position and not halted:
            conf = pred if pred > 0.5 else 1 - pred
            direction = "long" if pred > 0.5 else "short"

            if classify_funding_signal(funding) == direction:
                conf = min(conf + 0.20, 1.0)

            if conf >= config.LSTM_CONFIDENCE_THRESHOLD:
                entry = price * (1 + TOTAL_COST_PCT) if direction == "long" else price * (1 - TOTAL_COST_PCT)
                size = capital * POSITION_SIZE_PCT / entry
                position = {"type": direction, "entry": entry, "size": size, "ts": ts, "regime": regime}

        # Track
        unreal = 0
        if position:
            unreal = ((price - position["entry"]) if position["type"] == "long" else (position["entry"] - price)) * position["size"]
        eq = capital + unreal
        peak = max(peak, eq)
        equity.append({"timestamp": ts, "capital": eq})

    # Close final
    if position:
        last = test_df.iloc[-1]["close"]
        pnl = ((last - position["entry"]) if position["type"] == "long" else (position["entry"] - last)) * position["size"]
        pnl -= position["size"] * last * TOTAL_COST_PCT
        capital += pnl
        logger.log_trade({"pnl": pnl, "win": pnl > 0, "regime": position["regime"],
                          "entry_time": position["ts"], "exit_time": test_df.iloc[-1]["timestamp"],
                          "entry_price": position["entry"], "exit_price": last,
                          "position_type": position["type"], "size": position["size"],
                          "pnl_pct": pnl / (position["entry"] * position["size"]), "exit_reason": "EOB"})

    equity_df = pd.DataFrame(equity)
    if not equity_df.empty:
        equity_df["returns"] = equity_df["capital"].pct_change().fillna(0)

    # Stats
    trades = logger.trades
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    gp = sum(t["pnl"] for t in wins)
    gl = abs(sum(t["pnl"] for t in losses))

    stats = {
        "initial": config.INITIAL_CAPITAL,
        "final": capital,
        "return_pct": (capital / config.INITIAL_CAPITAL - 1) * 100,
        "trades": len(trades),
        "win_rate": len(wins) / len(trades) * 100 if trades else 0,
        "profit_factor": gp / gl if gl > 0 else 0,
        "sharpe": calculate_sharpe_ratio(equity_df["returns"]) if not equity_df.empty else 0,
        "max_dd": calculate_max_drawdown(equity_df["capital"]) * 100 if not equity_df.empty else 0,
        "halted": halted
    }

    return stats, equity_df


def main():
    """Main execution"""
    print("🚀 ATLAS BOT - TARDIS REAL DATA BACKTEST")
    print("=" * 70)

    # Check for Tardis data
    info = get_available_data_info()

    if not info:
        print("\n❌ No Tardis data available")
        print("\nTo download data:")
        print("  1. Set TARDIS_API_KEY in .env")
        print("  2. Run: python tardis_download.py --download")
        print("  3. Run: python tardis_download.py --process")
        print("  4. Run: python tardis_backtest.py")
        return

    # Load Tardis features
    tardis_df = load_tardis_features()

    if tardis_df.empty:
        print("\n❌ No processed Tardis features")
        print("   Run: python tardis_download.py --process")
        return

    print(f"\n✅ Tardis features loaded: {len(tardis_df)} records")
    print(f"   Columns: {list(tardis_df.columns)}")

    # Fetch OHLCV data
    end = datetime.now()
    start = end - timedelta(days=config.LOOKBACK_DAYS)

    print(f"\nPeriod: {start.date()} to {end.date()}")

    print("\n" + "=" * 70)
    print("FETCHING OHLCV & FUNDING DATA")
    print("=" * 70)

    ohlcv = fetch_historical_ohlcv(symbol=config.BINANCE_SYMBOL, start_date=start, end_date=end)
    print(f"✅ OHLCV: {len(ohlcv)} candles")

    funding = get_historical_funding_rates(start, end)
    print(f"✅ Funding: {len(funding)} records")

    # Merge Tardis with OHLCV
    print("\n" + "=" * 70)
    print("MERGING FEATURES")
    print("=" * 70)

    df = merge_tardis_with_ohlcv(ohlcv, tardis_df)
    print(f"✅ Merged: {len(df)} records")

    # Add HMM features
    df = compute_hmm_features(df)

    # Add funding momentum
    if not funding.empty:
        funding_mom = calculate_funding_momentum(funding)
        funding_mom["timestamp"] = pd.to_datetime(funding_mom["timestamp"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = pd.merge_asof(df.sort_values("timestamp"),
                           funding_mom.sort_values("timestamp"),
                           on="timestamp", direction="backward")

    # Split
    train_df, test_df = split_train_test(df)

    # Train
    regime_detector, models, results = train_tardis_models(train_df)

    # AUC Report
    print("\n" + "=" * 70)
    print("📊 TARDIS REAL DATA AUC RESULTS")
    print("=" * 70)

    all_good = True
    for regime, metrics in sorted(results.items()):
        auc = metrics["auc"]
        status = "✅ GOOD" if auc >= 0.60 else "⚠️ MARGINAL" if auc >= 0.55 else "❌ POOR"
        if auc < 0.60:
            all_good = False
        print(f"Regime {regime}: AUC={auc:.3f} Acc={metrics['accuracy']:.2%} Features={metrics.get('features', 0)} {status}")

    print("\n" + "-" * 50)
    if all_good:
        print("✅ All regimes AUC >= 0.60 - STRATEGY VIABLE!")
    else:
        print("⚠️ Some regimes AUC < 0.60 - May need more data or features")

    # Backtest
    stats, equity_df = run_backtest(test_df, regime_detector, models)

    print("\n" + "=" * 70)
    print("📊 BACKTEST RESULTS (6-Month Test)")
    print("=" * 70)
    print(f"Return: {stats['return_pct']:+.2f}% | Annualized: {stats['return_pct']*2:+.2f}%")
    print(f"Trades: {stats['trades']} | Win Rate: {stats['win_rate']:.1f}%")
    print(f"Sharpe: {stats['sharpe']:.2f} | Max DD: {stats['max_dd']:.2f}%")
    print(f"Profit Factor: {stats['profit_factor']:.2f}")
    if stats["halted"]:
        print("⛔ Circuit breaker triggered")

    # Save chart
    os.makedirs("data", exist_ok=True)
    plt.figure(figsize=(12, 5))
    plt.plot(equity_df["timestamp"], equity_df["capital"], color="#00B5AD")
    plt.axhline(config.INITIAL_CAPITAL, color="gray", linestyle="--")
    plt.title(f"Tardis Real Data Strategy - Return: {stats['return_pct']:.2f}%")
    plt.ylabel("Capital")
    plt.grid(True, alpha=0.3)
    plt.savefig("data/tardis_backtest.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n📊 Chart saved to data/tardis_backtest.png")


if __name__ == "__main__":
    main()
