"""
Self-Retraining Module
Fetches latest data, retrains models, replaces if better
Sends Telegram report
"""

import os
from datetime import datetime, timedelta
from typing import Dict, Tuple
import pandas as pd

import config
from exchange import fetch_historical_ohlcv
from funding import get_historical_funding_rates
from features import compute_all_features
from regime import RegimeDetector, add_regime_to_dataframe
from models.lstm_model import RegimeLSTMEnsemble
from database import Database
from telegram_alerts import alerts


def load_existing_models() -> Tuple[RegimeDetector, RegimeLSTMEnsemble]:
    """Load existing trained models"""
    regime_detector = None
    lstm_ensemble = None

    if os.path.exists(config.HMM_MODEL_PATH):
        try:
            regime_detector = RegimeDetector()
            regime_detector.load()
        except Exception as e:
            print(f"Could not load HMM: {e}")

    lstm_ensemble = RegimeLSTMEnsemble()
    lstm_ensemble.load_all()

    return regime_detector, lstm_ensemble


def get_model_performance(
    df: pd.DataFrame,
    regime_detector: RegimeDetector,
    lstm_ensemble: RegimeLSTMEnsemble
) -> Dict:
    """Evaluate model performance on validation data"""
    if regime_detector is None or not lstm_ensemble.models:
        return {"accuracy": 0, "auc": 0}

    # Use last 20% as validation
    val_start = int(len(df) * 0.8)
    val_df = df.iloc[val_start:].copy()

    # Add regime labels
    val_df = add_regime_to_dataframe(val_df, regime_detector)

    # Calculate accuracy per regime
    total_correct = 0
    total_samples = 0

    for regime, model in lstm_ensemble.models.items():
        if not model.trained:
            continue

        regime_df = val_df[val_df["regime"] == regime]
        if len(regime_df) < config.LSTM_SEQUENCE_LENGTH:
            continue

        # Get predictions
        probs = model.predict_proba(regime_df)
        predictions = (probs > 0.5).astype(int)

        # Get actual returns
        actual = (regime_df["close"].shift(-config.LSTM_PREDICTION_HORIZON) > regime_df["close"]).astype(int)

        # Count correct (skip first LSTM_SEQUENCE_LENGTH and last PREDICTION_HORIZON)
        valid_idx = slice(config.LSTM_SEQUENCE_LENGTH, -config.LSTM_PREDICTION_HORIZON)
        correct = (predictions[valid_idx] == actual.values[valid_idx]).sum()
        total_correct += correct
        total_samples += len(predictions[valid_idx])

    accuracy = total_correct / total_samples if total_samples > 0 else 0

    return {
        "accuracy": accuracy,
        "samples": total_samples
    }


def retrain_models(
    df: pd.DataFrame,
    include_sentiment: bool = False
) -> Tuple[RegimeDetector, RegimeLSTMEnsemble, Dict]:
    """Train new models on latest data"""
    print("🔧 Training new HMM Regime Detector...")
    regime_detector = RegimeDetector(n_states=config.HMM_N_STATES)
    regime_detector.fit(df)

    # Add regime labels
    df = add_regime_to_dataframe(df, regime_detector)

    print("🔧 Training new LSTM Models...")
    lstm_ensemble = RegimeLSTMEnsemble(n_regimes=config.HMM_N_STATES)
    lstm_results = lstm_ensemble.train_all(df, include_sentiment=include_sentiment)

    return regime_detector, lstm_ensemble, lstm_results


def run_retraining():
    """Main retraining routine"""
    print("=" * 70)
    print("🔄 ATLAS BOT - MODEL RETRAINING")
    print("=" * 70)
    print(f"Time: {datetime.now()}")

    db = Database()

    # Fetch latest data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=config.LOOKBACK_DAYS)

    print("\n📥 Fetching latest data...")
    ohlcv_df = fetch_historical_ohlcv(start_date=start_date, end_date=end_date)
    funding_df = get_historical_funding_rates(start_time=start_date, end_time=end_date)

    if ohlcv_df.empty:
        print("❌ Failed to fetch data")
        alerts.send_error("Retraining Failed", "Could not fetch OHLCV data")
        return

    # Save to database
    db.save_candles(ohlcv_df)
    db.save_funding_rates(funding_df)

    # Compute features
    print("\n🔧 Computing features...")
    df = compute_all_features(ohlcv_df, funding_df, include_sentiment=False)

    # Load existing models and evaluate
    print("\n📊 Evaluating existing models...")
    old_regime, old_lstm = load_existing_models()

    old_performance = {"accuracy": 0, "samples": 0}
    if old_regime is not None and old_lstm.models:
        old_performance = get_model_performance(df, old_regime, old_lstm)
        print(f"   Old model accuracy: {old_performance['accuracy']:.2%}")

    # Train new models
    print("\n🔧 Training new models...")
    new_regime, new_lstm, lstm_results = retrain_models(df, include_sentiment=False)

    # Evaluate new models
    print("\n📊 Evaluating new models...")
    new_performance = get_model_performance(df, new_regime, new_lstm)
    print(f"   New model accuracy: {new_performance['accuracy']:.2%}")

    # Compare and decide
    should_replace = new_performance["accuracy"] > old_performance["accuracy"]

    details = f"""
Old Accuracy: {old_performance['accuracy']:.2%}
New Accuracy: {new_performance['accuracy']:.2%}
Training samples: {new_performance.get('samples', 0)}

LSTM Results per Regime:
"""
    for regime, metrics in lstm_results.items():
        details += f"  Regime {regime}: Acc={metrics['accuracy']:.2%}, AUC={metrics['auc']:.2f}\n"

    if should_replace:
        print("\n✅ New models are better - replacing...")
        new_regime.save()
        new_lstm.save_all()

        # Save metadata
        db.save_model_metadata({
            "model_type": "hmm",
            "trained_at": datetime.now(),
            "train_start": start_date,
            "train_end": end_date,
            "accuracy": new_performance["accuracy"],
            "auc": 0,
            "notes": "Auto-retrained"
        })

        for regime, metrics in lstm_results.items():
            db.save_model_metadata({
                "model_type": f"lstm_regime_{regime}",
                "trained_at": datetime.now(),
                "train_start": start_date,
                "train_end": end_date,
                "accuracy": metrics["accuracy"],
                "auc": metrics["auc"],
                "notes": f"Samples: {metrics['samples']}"
            })
    else:
        print("\n⏸️ Old models are better or equal - keeping existing...")

    # Send Telegram report
    alerts.send_retrain_report(
        model_type="HMM + LSTM Ensemble",
        old_accuracy=old_performance["accuracy"],
        new_accuracy=new_performance["accuracy"],
        replaced=should_replace,
        details=details
    )

    db.close()

    print("\n" + "=" * 70)
    print("✅ RETRAINING COMPLETE")
    print("=" * 70)

    return {
        "old_accuracy": old_performance["accuracy"],
        "new_accuracy": new_performance["accuracy"],
        "replaced": should_replace
    }


if __name__ == "__main__":
    result = run_retraining()
    print(f"\nResult: {result}")
