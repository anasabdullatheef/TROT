"""
Database Module
SQLite storage for candles, features, trades, and model metadata
"""

import os
from datetime import datetime
from typing import List, Dict, Optional
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import config

Base = declarative_base()


class Candle(Base):
    """OHLCV candle data"""
    __tablename__ = "candles"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, unique=True, index=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)


class FundingRate(Base):
    """Historical funding rates"""
    __tablename__ = "funding_rates"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, unique=True, index=True)
    rate = Column(Float)


class Feature(Base):
    """Computed features for each candle"""
    __tablename__ = "features"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, unique=True, index=True)
    # Price features
    return_1h = Column(Float)
    return_4h = Column(Float)
    return_24h = Column(Float)
    # Volatility
    atr = Column(Float)
    volatility_24h = Column(Float)
    # Momentum
    rsi = Column(Float)
    macd = Column(Float)
    macd_signal = Column(Float)
    stoch_k = Column(Float)
    stoch_d = Column(Float)
    # Volume
    obv = Column(Float)
    volume_ratio = Column(Float)
    # Funding
    funding_rate = Column(Float)
    funding_8h_change = Column(Float)
    funding_24h_cumulative = Column(Float)
    # Sentiment
    fng_value = Column(Float)
    finbert_1h = Column(Float)
    finbert_4h = Column(Float)
    # Regime
    regime = Column(Integer)
    # ADX
    adx = Column(Float)


class Trade(Base):
    """Trade history"""
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True)
    entry_time = Column(DateTime)
    exit_time = Column(DateTime)
    direction = Column(String(10))
    entry_price = Column(Float)
    exit_price = Column(Float)
    size = Column(Float)
    pnl = Column(Float)
    pnl_pct = Column(Float)
    exit_reason = Column(String(50))
    regime = Column(Integer)
    confidence = Column(Float)
    funding_boost = Column(Boolean)
    sentiment_score = Column(Float)


class ModelMetadata(Base):
    """Model training metadata"""
    __tablename__ = "model_metadata"

    id = Column(Integer, primary_key=True)
    model_type = Column(String(20))  # 'hmm' or 'lstm_regime_X'
    trained_at = Column(DateTime)
    train_start = Column(DateTime)
    train_end = Column(DateTime)
    accuracy = Column(Float)
    auc = Column(Float)
    notes = Column(Text)


class Sentiment(Base):
    """Raw sentiment data"""
    __tablename__ = "sentiment"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, index=True)
    source = Column(String(20))  # 'fng' or 'finbert'
    value = Column(Float)
    raw_text = Column(Text)


class Database:
    """Database manager"""

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = config.DATABASE_PATH

        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()

    def save_candles(self, df: pd.DataFrame):
        """Save candles to database"""
        for _, row in df.iterrows():
            existing = self.session.query(Candle).filter_by(timestamp=row["timestamp"]).first()
            if existing:
                continue
            candle = Candle(
                timestamp=row["timestamp"],
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row["volume"]
            )
            self.session.add(candle)
        self.session.commit()

    def get_candles(self, start: datetime = None, end: datetime = None) -> pd.DataFrame:
        """Get candles from database"""
        query = self.session.query(Candle)
        if start:
            query = query.filter(Candle.timestamp >= start)
        if end:
            query = query.filter(Candle.timestamp <= end)
        query = query.order_by(Candle.timestamp)

        rows = query.all()
        if not rows:
            return pd.DataFrame()

        return pd.DataFrame([{
            "timestamp": r.timestamp,
            "open": r.open,
            "high": r.high,
            "low": r.low,
            "close": r.close,
            "volume": r.volume
        } for r in rows])

    def save_funding_rates(self, df: pd.DataFrame):
        """Save funding rates to database"""
        for _, row in df.iterrows():
            existing = self.session.query(FundingRate).filter_by(timestamp=row["timestamp"]).first()
            if existing:
                continue
            fr = FundingRate(timestamp=row["timestamp"], rate=row["funding_rate"])
            self.session.add(fr)
        self.session.commit()

    def get_funding_rates(self, start: datetime = None, end: datetime = None) -> pd.DataFrame:
        """Get funding rates from database"""
        query = self.session.query(FundingRate)
        if start:
            query = query.filter(FundingRate.timestamp >= start)
        if end:
            query = query.filter(FundingRate.timestamp <= end)
        query = query.order_by(FundingRate.timestamp)

        rows = query.all()
        if not rows:
            return pd.DataFrame()

        return pd.DataFrame([{
            "timestamp": r.timestamp,
            "funding_rate": r.rate
        } for r in rows])

    def save_features(self, df: pd.DataFrame):
        """Save computed features to database"""
        for _, row in df.iterrows():
            existing = self.session.query(Feature).filter_by(timestamp=row["timestamp"]).first()

            feature_data = {
                "timestamp": row["timestamp"],
                "return_1h": row.get("return_1h"),
                "return_4h": row.get("return_4h"),
                "return_24h": row.get("return_24h"),
                "atr": row.get("atr"),
                "volatility_24h": row.get("volatility_24h"),
                "rsi": row.get("rsi"),
                "macd": row.get("macd"),
                "macd_signal": row.get("macd_signal"),
                "stoch_k": row.get("stoch_k"),
                "stoch_d": row.get("stoch_d"),
                "obv": row.get("obv"),
                "volume_ratio": row.get("volume_ratio"),
                "funding_rate": row.get("funding_rate"),
                "funding_8h_change": row.get("funding_8h_change"),
                "funding_24h_cumulative": row.get("funding_24h_cumulative"),
                "fng_value": row.get("fng_value"),
                "finbert_1h": row.get("finbert_1h"),
                "finbert_4h": row.get("finbert_4h"),
                "regime": row.get("regime"),
                "adx": row.get("adx")
            }

            if existing:
                for key, value in feature_data.items():
                    if key != "timestamp" and value is not None:
                        setattr(existing, key, value)
            else:
                self.session.add(Feature(**feature_data))

        self.session.commit()

    def get_features(self, start: datetime = None, end: datetime = None) -> pd.DataFrame:
        """Get features from database"""
        query = self.session.query(Feature)
        if start:
            query = query.filter(Feature.timestamp >= start)
        if end:
            query = query.filter(Feature.timestamp <= end)
        query = query.order_by(Feature.timestamp)

        rows = query.all()
        if not rows:
            return pd.DataFrame()

        return pd.DataFrame([{
            "timestamp": r.timestamp,
            "return_1h": r.return_1h,
            "return_4h": r.return_4h,
            "return_24h": r.return_24h,
            "atr": r.atr,
            "volatility_24h": r.volatility_24h,
            "rsi": r.rsi,
            "macd": r.macd,
            "macd_signal": r.macd_signal,
            "stoch_k": r.stoch_k,
            "stoch_d": r.stoch_d,
            "obv": r.obv,
            "volume_ratio": r.volume_ratio,
            "funding_rate": r.funding_rate,
            "funding_8h_change": r.funding_8h_change,
            "funding_24h_cumulative": r.funding_24h_cumulative,
            "fng_value": r.fng_value,
            "finbert_1h": r.finbert_1h,
            "finbert_4h": r.finbert_4h,
            "regime": r.regime,
            "adx": r.adx
        } for r in rows])

    def save_trade(self, trade_data: Dict):
        """Save a trade to database"""
        trade = Trade(**trade_data)
        self.session.add(trade)
        self.session.commit()
        return trade.id

    def get_trades(self, start: datetime = None, end: datetime = None) -> pd.DataFrame:
        """Get trades from database"""
        query = self.session.query(Trade)
        if start:
            query = query.filter(Trade.entry_time >= start)
        if end:
            query = query.filter(Trade.entry_time <= end)
        query = query.order_by(Trade.entry_time)

        rows = query.all()
        if not rows:
            return pd.DataFrame()

        return pd.DataFrame([{
            "entry_time": r.entry_time,
            "exit_time": r.exit_time,
            "direction": r.direction,
            "entry_price": r.entry_price,
            "exit_price": r.exit_price,
            "size": r.size,
            "pnl": r.pnl,
            "pnl_pct": r.pnl_pct,
            "exit_reason": r.exit_reason,
            "regime": r.regime,
            "confidence": r.confidence,
            "funding_boost": r.funding_boost,
            "sentiment_score": r.sentiment_score
        } for r in rows])

    def save_model_metadata(self, metadata: Dict):
        """Save model training metadata"""
        model = ModelMetadata(**metadata)
        self.session.add(model)
        self.session.commit()

    def get_latest_model_metadata(self, model_type: str) -> Optional[Dict]:
        """Get latest model metadata"""
        result = self.session.query(ModelMetadata).filter_by(
            model_type=model_type
        ).order_by(ModelMetadata.trained_at.desc()).first()

        if result:
            return {
                "model_type": result.model_type,
                "trained_at": result.trained_at,
                "train_start": result.train_start,
                "train_end": result.train_end,
                "accuracy": result.accuracy,
                "auc": result.auc,
                "notes": result.notes
            }
        return None

    def save_sentiment(self, timestamp: datetime, source: str, value: float, raw_text: str = None):
        """Save sentiment data"""
        sentiment = Sentiment(
            timestamp=timestamp,
            source=source,
            value=value,
            raw_text=raw_text
        )
        self.session.add(sentiment)
        self.session.commit()

    def close(self):
        """Close database connection"""
        self.session.close()


if __name__ == "__main__":
    # Test database
    db = Database("data/test_atlas.db")
    print("Database initialized successfully")

    # Test candle storage
    test_df = pd.DataFrame([{
        "timestamp": datetime.now(),
        "open": 50000,
        "high": 50100,
        "low": 49900,
        "close": 50050,
        "volume": 1000
    }])
    db.save_candles(test_df)
    print("Test candle saved")

    candles = db.get_candles()
    print(f"Retrieved {len(candles)} candles")

    db.close()
    os.remove("data/test_atlas.db")
    print("Test complete")
