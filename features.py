"""
Feature Engineering Module
Computes all features for ML models:
- Price returns (1h, 4h, 24h)
- Volatility (ATR, rolling std)
- Momentum (RSI, MACD, Stochastic)
- Volume (OBV, volume ratio)
- Funding (rate, change, cumulative)
- Sentiment (Fear & Greed, FinBERT news)
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple
from ta.volatility import AverageTrueRange
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, ADXIndicator
from ta.volume import OnBalanceVolumeIndicator
import warnings
warnings.filterwarnings("ignore")

import config

# FinBERT model (lazy loaded)
_finbert_model = None
_finbert_tokenizer = None


def get_finbert():
    """Lazy load FinBERT model"""
    global _finbert_model, _finbert_tokenizer

    if _finbert_model is None:
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch

            print("Loading FinBERT model...")
            model_name = "ProsusAI/finbert"
            _finbert_tokenizer = AutoTokenizer.from_pretrained(model_name)
            _finbert_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            _finbert_model.eval()
            print("✅ FinBERT loaded")
        except Exception as e:
            print(f"⚠️ Could not load FinBERT: {e}")
            return None, None

    return _finbert_model, _finbert_tokenizer


def compute_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute price-based features: returns for multiple periods"""
    df = df.copy()

    # Returns for different periods
    for period in config.RETURN_PERIODS:
        df[f"return_{period}h"] = df["close"].pct_change(periods=period)

    return df


def compute_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute volatility features: ATR, rolling standard deviation"""
    df = df.copy()

    # ATR
    atr = AverageTrueRange(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        window=config.ATR_PERIOD,
        fillna=False
    )
    df["atr"] = atr.average_true_range()

    # Rolling volatility (std of returns)
    df["volatility_24h"] = df["close"].pct_change().rolling(
        window=config.VOLATILITY_WINDOW
    ).std()

    return df


def compute_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute momentum features: RSI, MACD, Stochastic, ADX"""
    df = df.copy()

    # RSI
    rsi = RSIIndicator(close=df["close"], window=config.RSI_PERIOD, fillna=False)
    df["rsi"] = rsi.rsi()

    # MACD
    macd = MACD(
        close=df["close"],
        window_slow=config.MACD_SLOW,
        window_fast=config.MACD_FAST,
        window_sign=config.MACD_SIGNAL,
        fillna=False
    )
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    # Stochastic
    stoch = StochasticOscillator(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        window=config.STOCH_K,
        smooth_window=config.STOCH_D,
        fillna=False
    )
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    # ADX
    adx = ADXIndicator(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        window=config.RSI_PERIOD,
        fillna=False
    )
    df["adx"] = adx.adx()

    return df


def compute_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute volume features: OBV, volume ratio"""
    df = df.copy()

    # On-Balance Volume
    obv = OnBalanceVolumeIndicator(close=df["close"], volume=df["volume"], fillna=False)
    df["obv"] = obv.on_balance_volume()

    # Volume ratio vs moving average
    volume_ma = df["volume"].rolling(window=config.VOLUME_MA_PERIOD).mean()
    df["volume_ratio"] = df["volume"] / volume_ma

    return df


def fetch_fear_greed_index(limit: int = 30) -> pd.DataFrame:
    """
    Fetch Fear & Greed Index from alternative.me API
    Returns DataFrame with timestamp and value
    """
    try:
        response = requests.get(
            config.FNG_API_URL,
            params={"limit": limit, "format": "json"},
            timeout=10
        )
        response.raise_for_status()
        data = response.json()

        if "data" not in data:
            return pd.DataFrame()

        records = []
        for item in data["data"]:
            records.append({
                "timestamp": datetime.fromtimestamp(int(item["timestamp"])),
                "fng_value": int(item["value"]) / 100.0  # Normalize to 0-1
            })

        df = pd.DataFrame(records)
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    except Exception as e:
        print(f"Error fetching Fear & Greed Index: {e}")
        return pd.DataFrame()


def fetch_crypto_news(limit: int = 50) -> List[Dict]:
    """
    Fetch crypto news from CryptoPanic API
    Returns list of news items with title, timestamp, and source
    """
    try:
        params = {
            "auth_token": config.CRYPTOPANIC_API_KEY,
            "currencies": "BTC",
            "filter": "hot",
            "public": "true"
        }

        # Use public endpoint if no API key
        if not config.CRYPTOPANIC_API_KEY:
            url = "https://cryptopanic.com/api/free/v1/posts/"
            params = {"currencies": "BTC", "public": "true"}
        else:
            url = config.CRYPTOPANIC_API_URL

        response = requests.get(url, params=params, timeout=10)

        if response.status_code != 200:
            return []

        data = response.json()

        if "results" not in data:
            return []

        news = []
        for item in data["results"][:limit]:
            news.append({
                "title": item.get("title", ""),
                "timestamp": datetime.fromisoformat(
                    item.get("published_at", "").replace("Z", "+00:00")
                ) if item.get("published_at") else datetime.now(),
                "source": item.get("source", {}).get("title", "Unknown"),
                "url": item.get("url", "")
            })

        return news

    except Exception as e:
        print(f"Error fetching crypto news: {e}")
        return []


def analyze_sentiment_finbert(texts: List[str]) -> List[float]:
    """
    Analyze sentiment of texts using FinBERT
    Returns list of sentiment scores (-1 to 1)
    """
    model, tokenizer = get_finbert()

    if model is None or not texts:
        return [0.0] * len(texts)

    try:
        import torch

        sentiments = []
        for text in texts:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )

            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)

                # FinBERT outputs: negative, neutral, positive
                neg, neu, pos = probs[0].tolist()

                # Convert to single score: -1 (negative) to +1 (positive)
                sentiment = pos - neg
                sentiments.append(sentiment)

        return sentiments

    except Exception as e:
        print(f"Error in FinBERT analysis: {e}")
        return [0.0] * len(texts)


def compute_sentiment_features(
    df: pd.DataFrame,
    news_cache: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Compute sentiment features:
    - Fear & Greed Index value
    - FinBERT 1h weighted sentiment
    - FinBERT 4h weighted sentiment
    """
    df = df.copy()

    # Initialize sentiment columns
    df["fng_value"] = np.nan
    df["finbert_1h"] = 0.0
    df["finbert_4h"] = 0.0

    # Fetch Fear & Greed Index
    fng_df = fetch_fear_greed_index(limit=100)

    if not fng_df.empty:
        # Merge with candle data (FNG is daily, forward fill)
        fng_df["date"] = fng_df["timestamp"].dt.date
        df["date"] = df["timestamp"].dt.date

        fng_daily = fng_df.drop_duplicates(subset=["date"], keep="last")
        df = df.merge(
            fng_daily[["date", "fng_value"]],
            on="date",
            how="left",
            suffixes=("", "_fng")
        )
        if "fng_value_fng" in df.columns:
            df["fng_value"] = df["fng_value_fng"]
            df = df.drop(columns=["fng_value_fng"])

        df["fng_value"] = df["fng_value"].ffill()
        df = df.drop(columns=["date"])

    # Fetch and analyze news for FinBERT sentiment
    # This is expensive, so in backtest we may skip or use cached values
    if news_cache is not None:
        # Use cached sentiment scores
        for i, row in df.iterrows():
            ts = row["timestamp"]
            if ts in news_cache:
                df.loc[i, "finbert_1h"] = news_cache[ts].get("finbert_1h", 0.0)
                df.loc[i, "finbert_4h"] = news_cache[ts].get("finbert_4h", 0.0)

    return df


def compute_all_features(
    df: pd.DataFrame,
    funding_df: pd.DataFrame = None,
    include_sentiment: bool = True,
    news_cache: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Compute all features for the dataset

    Args:
        df: OHLCV DataFrame
        funding_df: Funding rate DataFrame
        include_sentiment: Whether to fetch and compute sentiment features
        news_cache: Optional cache of pre-computed sentiment scores

    Returns:
        DataFrame with all features
    """
    df = df.copy()

    print("Computing price features...")
    df = compute_price_features(df)

    print("Computing volatility features...")
    df = compute_volatility_features(df)

    print("Computing momentum features...")
    df = compute_momentum_features(df)

    print("Computing volume features...")
    df = compute_volume_features(df)

    # Add funding features if provided
    if funding_df is not None and not funding_df.empty:
        print("Computing funding features...")
        from funding import compute_funding_features
        df = compute_funding_features(df, funding_df)
    else:
        df["funding_rate"] = np.nan
        df["funding_8h_change"] = np.nan
        df["funding_24h_cumulative"] = np.nan

    # Add sentiment features
    if include_sentiment:
        print("Computing sentiment features...")
        df = compute_sentiment_features(df, news_cache)
    else:
        df["fng_value"] = np.nan
        df["finbert_1h"] = 0.0
        df["finbert_4h"] = 0.0

    return df


def get_feature_columns() -> List[str]:
    """Return list of all feature column names"""
    return [
        # Price
        "return_1h", "return_4h", "return_24h",
        # Volatility
        "atr", "volatility_24h",
        # Momentum
        "rsi", "macd", "macd_signal", "stoch_k", "stoch_d", "adx",
        # Volume
        "obv", "volume_ratio",
        # Funding
        "funding_rate", "funding_8h_change", "funding_24h_cumulative",
        # Sentiment
        "fng_value", "finbert_1h", "finbert_4h"
    ]


def get_hmm_feature_columns() -> List[str]:
    """Return feature columns used for HMM regime detection"""
    return ["return_1h", "volatility_24h", "volume_ratio", "atr", "adx"]


def prepare_features_for_model(
    df: pd.DataFrame,
    feature_columns: List[str] = None,
    fill_method: str = "ffill"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare features for ML model:
    - Select feature columns
    - Handle missing values
    - Return features and target

    Returns:
        (features_df, target_df)
    """
    if feature_columns is None:
        feature_columns = get_feature_columns()

    df = df.copy()

    # Create target: positive 4h return
    df["target"] = (df["close"].shift(-config.LSTM_PREDICTION_HORIZON) > df["close"]).astype(int)

    # Select features
    available_cols = [c for c in feature_columns if c in df.columns]
    features = df[available_cols].copy()

    # Fill missing values
    if fill_method == "ffill":
        features = features.ffill().bfill()
    elif fill_method == "zero":
        features = features.fillna(0)

    target = df[["target"]].copy()

    return features, target


if __name__ == "__main__":
    print("Testing features module...")

    # Create sample data
    dates = pd.date_range(start="2024-01-01", periods=100, freq="1h")
    np.random.seed(42)
    close = 40000 + np.cumsum(np.random.randn(100) * 100)

    df = pd.DataFrame({
        "timestamp": dates,
        "open": close - np.random.rand(100) * 50,
        "high": close + np.random.rand(100) * 100,
        "low": close - np.random.rand(100) * 100,
        "close": close,
        "volume": np.random.rand(100) * 1000000
    })

    # Compute features
    df = compute_all_features(df, include_sentiment=False)

    print(f"\nFeatures computed: {len(get_feature_columns())} features")
    print(f"Sample features:\n{df[get_feature_columns()].tail()}")

    # Test Fear & Greed
    print("\nFetching Fear & Greed Index...")
    fng = fetch_fear_greed_index(limit=5)
    print(fng)

    # Test news fetch
    print("\nFetching crypto news...")
    news = fetch_crypto_news(limit=3)
    for n in news:
        print(f"  - {n['title'][:60]}...")
