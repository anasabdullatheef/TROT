"""
Order Flow Features Module
Leading indicators that predict price movement, not lag it

Features:
1. Order flow imbalance (bid vs ask volume in top 10 levels)
2. Trade flow toxicity (VPIN - buy vs sell initiated volume)
3. Funding rate momentum (8h, 24h, 72h rate of change)
4. Liquidation cascade proximity
5. Cross-exchange price divergence (Binance vs Bybit vs OKX)
6. Spot vs futures basis (premium/discount)
"""

import numpy as np
import pandas as pd
import ccxt
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import config


# Initialize exchanges
def get_exchanges() -> Dict[str, ccxt.Exchange]:
    """Initialize multiple exchanges for cross-exchange data"""
    exchanges = {}

    # Binance Futures
    exchanges['binance'] = ccxt.binanceusdm({
        'enableRateLimit': True,
        'options': {'defaultType': 'future'}
    })

    # Binance Spot
    exchanges['binance_spot'] = ccxt.binance({
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'}
    })

    # Bybit
    exchanges['bybit'] = ccxt.bybit({
        'enableRateLimit': True,
        'options': {'defaultType': 'linear'}
    })

    # OKX
    exchanges['okx'] = ccxt.okx({
        'enableRateLimit': True,
        'options': {'defaultType': 'swap'}
    })

    return exchanges


EXCHANGES = None

def init_exchanges():
    global EXCHANGES
    if EXCHANGES is None:
        EXCHANGES = get_exchanges()
    return EXCHANGES


# =============================================================================
# 1. ORDER FLOW IMBALANCE
# =============================================================================

def fetch_order_book_imbalance(symbol: str = "BTC/USDT:USDT", levels: int = 10) -> Dict:
    """
    Fetch order book and calculate bid/ask imbalance

    Imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
    Positive = more buying pressure
    Negative = more selling pressure
    """
    exchanges = init_exchanges()

    try:
        orderbook = exchanges['binance'].fetch_order_book(symbol, limit=levels)

        bid_volume = sum([bid[1] for bid in orderbook['bids'][:levels]])
        ask_volume = sum([ask[1] for ask in orderbook['asks'][:levels]])

        total_volume = bid_volume + ask_volume
        if total_volume == 0:
            imbalance = 0
        else:
            imbalance = (bid_volume - ask_volume) / total_volume

        # Also calculate volume-weighted average prices
        bid_vwap = sum([b[0] * b[1] for b in orderbook['bids'][:levels]]) / bid_volume if bid_volume > 0 else 0
        ask_vwap = sum([a[0] * a[1] for a in orderbook['asks'][:levels]]) / ask_volume if ask_volume > 0 else 0

        spread = (ask_vwap - bid_vwap) / bid_vwap if bid_vwap > 0 else 0

        return {
            'orderbook_imbalance': imbalance,
            'bid_volume': bid_volume,
            'ask_volume': ask_volume,
            'spread': spread,
            'timestamp': datetime.now()
        }
    except Exception as e:
        print(f"Error fetching order book: {e}")
        return {
            'orderbook_imbalance': 0,
            'bid_volume': 0,
            'ask_volume': 0,
            'spread': 0,
            'timestamp': datetime.now()
        }


# =============================================================================
# 2. TRADE FLOW TOXICITY (VPIN)
# =============================================================================

def fetch_recent_trades(symbol: str = "BTCUSDT", limit: int = 1000) -> pd.DataFrame:
    """Fetch recent aggregated trades from Binance"""
    url = f"https://fapi.binance.com/fapi/v1/aggTrades"
    params = {
        'symbol': symbol,
        'limit': limit
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()

        if isinstance(data, list):
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['T'], unit='ms')
            df['price'] = df['p'].astype(float)
            df['quantity'] = df['q'].astype(float)
            df['is_buyer_maker'] = df['m']  # True = sell-initiated, False = buy-initiated
            return df[['timestamp', 'price', 'quantity', 'is_buyer_maker']]
    except Exception as e:
        print(f"Error fetching trades: {e}")

    return pd.DataFrame()


def calculate_vpin(trades_df: pd.DataFrame, bucket_size: float = None) -> float:
    """
    Calculate Volume-synchronized Probability of Informed Trading (VPIN)

    VPIN = |Buy Volume - Sell Volume| / Total Volume
    High VPIN = high toxicity = informed trading = price move imminent
    """
    if trades_df.empty:
        return 0.5

    # Buy-initiated = seller is maker (is_buyer_maker = True means SELL order hit the book)
    # Confusingly named: is_buyer_maker=True means the buyer was the maker, so trade was sell-initiated
    sell_volume = trades_df[trades_df['is_buyer_maker'] == True]['quantity'].sum()
    buy_volume = trades_df[trades_df['is_buyer_maker'] == False]['quantity'].sum()

    total_volume = buy_volume + sell_volume
    if total_volume == 0:
        return 0.5

    # VPIN as buy pressure ratio (0-1)
    # High value = more buy-initiated trades = bullish
    vpin = buy_volume / total_volume

    return vpin


def calculate_trade_toxicity(trades_df: pd.DataFrame) -> Dict:
    """
    Calculate trade flow toxicity metrics
    """
    if trades_df.empty:
        return {
            'vpin': 0.5,
            'buy_volume': 0,
            'sell_volume': 0,
            'trade_imbalance': 0,
            'avg_trade_size': 0
        }

    sell_volume = trades_df[trades_df['is_buyer_maker'] == True]['quantity'].sum()
    buy_volume = trades_df[trades_df['is_buyer_maker'] == False]['quantity'].sum()
    total_volume = buy_volume + sell_volume

    vpin = buy_volume / total_volume if total_volume > 0 else 0.5
    imbalance = (buy_volume - sell_volume) / total_volume if total_volume > 0 else 0

    return {
        'vpin': vpin,
        'buy_volume': buy_volume,
        'sell_volume': sell_volume,
        'trade_imbalance': imbalance,
        'avg_trade_size': trades_df['quantity'].mean()
    }


# =============================================================================
# 3. FUNDING RATE MOMENTUM
# =============================================================================

def calculate_funding_momentum(funding_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate funding rate momentum over multiple timeframes
    Acceleration matters more than level
    """
    if funding_df.empty:
        return funding_df

    df = funding_df.copy()
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Funding rate is every 8 hours, so:
    # 8h momentum = 1 period change
    # 24h momentum = 3 period change
    # 72h momentum = 9 period change

    df['funding_8h_momentum'] = df['funding_rate'].diff(1)
    df['funding_24h_momentum'] = df['funding_rate'].diff(3)
    df['funding_72h_momentum'] = df['funding_rate'].diff(9)

    # Acceleration (momentum of momentum)
    df['funding_acceleration'] = df['funding_8h_momentum'].diff(1)

    # Z-score of current funding vs recent history
    rolling_mean = df['funding_rate'].rolling(window=21).mean()  # ~7 days
    rolling_std = df['funding_rate'].rolling(window=21).std()
    df['funding_zscore'] = (df['funding_rate'] - rolling_mean) / (rolling_std + 1e-10)

    return df


# =============================================================================
# 4. LIQUIDATION CASCADE PROXIMITY
# =============================================================================

def fetch_liquidations(symbol: str = "BTCUSDT") -> pd.DataFrame:
    """
    Fetch recent liquidation data
    Note: Binance doesn't provide historical liquidations via public API
    This fetches from forceOrder endpoint (real-time only)
    For backtesting, we'll estimate from price volatility
    """
    url = "https://fapi.binance.com/fapi/v1/forceOrders"
    params = {
        'symbol': symbol,
        'limit': 100
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()

        if isinstance(data, list) and len(data) > 0:
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['time'], unit='ms')
            df['price'] = df['price'].astype(float)
            df['quantity'] = df['origQty'].astype(float)
            df['side'] = df['side']
            return df[['timestamp', 'price', 'quantity', 'side']]
    except Exception as e:
        # This endpoint requires authentication or may be rate limited
        pass

    return pd.DataFrame()


def estimate_liquidation_levels(df: pd.DataFrame, leverage: int = 10) -> Dict:
    """
    Estimate major liquidation clusters based on recent price action

    Assumption: Traders enter at local highs/lows with leverage
    Liquidation price ≈ Entry price ± (Entry price / leverage)
    """
    if df.empty or 'close' not in df.columns:
        return {'long_liq_distance': 0, 'short_liq_distance': 0, 'liq_risk': 0}

    current_price = df['close'].iloc[-1]

    # Find recent local maxima (potential short entries) and minima (potential long entries)
    recent = df.tail(72)  # Last 3 days

    # Local highs - potential short entry points
    local_highs = recent['close'].rolling(5, center=True).max()
    major_high = local_highs.max()

    # Local lows - potential long entry points
    local_lows = recent['close'].rolling(5, center=True).min()
    major_low = local_lows.min()

    # Estimated liquidation levels
    # Shorts entered at highs get liquidated if price rises to: high * (1 + 1/leverage)
    short_liq_level = major_high * (1 + 1/leverage)
    short_liq_distance = (short_liq_level - current_price) / current_price

    # Longs entered at lows get liquidated if price falls to: low * (1 - 1/leverage)
    long_liq_level = major_low * (1 - 1/leverage)
    long_liq_distance = (current_price - long_liq_level) / current_price

    # Liquidation risk score (how close are we to cascade)
    # Lower distance = higher risk
    min_distance = min(abs(short_liq_distance), abs(long_liq_distance))
    liq_risk = 1 / (1 + min_distance * 10)  # Normalize to 0-1

    return {
        'long_liq_distance': long_liq_distance,
        'short_liq_distance': short_liq_distance,
        'liq_risk': liq_risk,
        'nearest_long_liq': long_liq_level,
        'nearest_short_liq': short_liq_level
    }


# =============================================================================
# 5. CROSS-EXCHANGE PRICE DIVERGENCE
# =============================================================================

def fetch_multi_exchange_prices() -> Dict:
    """
    Fetch BTC price from multiple exchanges simultaneously
    Divergence > 0.05% often precedes reversion
    """
    exchanges = init_exchanges()
    prices = {}

    symbols = {
        'binance': 'BTC/USDT:USDT',
        'bybit': 'BTC/USDT:USDT',
        'okx': 'BTC/USDT:USDT'
    }

    def fetch_price(exchange_name: str, symbol: str):
        try:
            ticker = exchanges[exchange_name].fetch_ticker(symbol)
            return exchange_name, ticker['last']
        except Exception as e:
            return exchange_name, None

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(fetch_price, name, sym) for name, sym in symbols.items()]
        for future in as_completed(futures):
            name, price = future.result()
            if price:
                prices[name] = price

    return prices


def calculate_cross_exchange_divergence(prices: Dict) -> Dict:
    """
    Calculate price divergence across exchanges
    """
    if len(prices) < 2:
        return {
            'max_divergence': 0,
            'binance_vs_bybit': 0,
            'binance_vs_okx': 0,
            'divergence_direction': 0
        }

    binance = prices.get('binance', 0)
    bybit = prices.get('bybit', 0)
    okx = prices.get('okx', 0)

    divergences = []

    if binance and bybit:
        div = (binance - bybit) / binance
        divergences.append(('binance_vs_bybit', div))

    if binance and okx:
        div = (binance - okx) / binance
        divergences.append(('binance_vs_okx', div))

    if bybit and okx:
        div = (bybit - okx) / bybit
        divergences.append(('bybit_vs_okx', div))

    max_div = max([abs(d[1]) for d in divergences]) if divergences else 0

    # Direction: positive = binance higher (expect binance to fall or others to rise)
    avg_div = np.mean([d[1] for d in divergences]) if divergences else 0

    result = {
        'max_divergence': max_div,
        'avg_divergence': avg_div,
        'divergence_direction': 1 if avg_div > 0 else -1 if avg_div < 0 else 0
    }

    for name, div in divergences:
        result[name] = div

    return result


# =============================================================================
# 6. SPOT VS FUTURES BASIS
# =============================================================================

def fetch_spot_futures_basis() -> Dict:
    """
    Calculate basis between spot and futures prices
    Basis = (Futures - Spot) / Spot

    Positive basis (contango) = futures premium = bullish sentiment
    Negative basis (backwardation) = futures discount = bearish sentiment
    Unusual basis often precedes corrections
    """
    exchanges = init_exchanges()

    try:
        # Fetch spot price
        spot_ticker = exchanges['binance_spot'].fetch_ticker('BTC/USDT')
        spot_price = spot_ticker['last']

        # Fetch futures price
        futures_ticker = exchanges['binance'].fetch_ticker('BTC/USDT:USDT')
        futures_price = futures_ticker['last']

        basis = (futures_price - spot_price) / spot_price

        # Annualized basis (assuming perpetual, approximate)
        # This is a rough approximation
        annualized_basis = basis * 365 * 3  # Assume 8-hour funding cycle

        return {
            'spot_price': spot_price,
            'futures_price': futures_price,
            'basis': basis,
            'basis_bps': basis * 10000,  # Basis points
            'annualized_basis': annualized_basis,
            'timestamp': datetime.now()
        }
    except Exception as e:
        print(f"Error fetching basis: {e}")
        return {
            'spot_price': 0,
            'futures_price': 0,
            'basis': 0,
            'basis_bps': 0,
            'annualized_basis': 0,
            'timestamp': datetime.now()
        }


# =============================================================================
# HISTORICAL DATA SIMULATION FOR BACKTESTING
# =============================================================================

def simulate_orderflow_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate order flow features for backtesting based on available data

    Since we don't have historical order book/trade data, we estimate from:
    - Volume patterns
    - Price momentum
    - Volatility

    These are ESTIMATES and should be replaced with real data for live trading
    """
    df = df.copy()

    # Estimate order book imbalance from volume and price direction
    # If price went up on high volume, likely bid-heavy order book
    df['price_change'] = df['close'].pct_change()
    df['volume_zscore'] = (df['volume'] - df['volume'].rolling(24).mean()) / (df['volume'].rolling(24).std() + 1e-10)

    # Simulated order book imbalance
    # Positive price change + high volume = bid heavy
    df['orderbook_imbalance_est'] = np.tanh(df['price_change'] * 100 * (1 + df['volume_zscore']))

    # Simulated VPIN (trade toxicity)
    # Use volume acceleration as proxy for informed trading
    df['volume_acceleration'] = df['volume'].pct_change().pct_change()
    df['vpin_est'] = 0.5 + np.tanh(df['volume_acceleration'] * 10) * 0.3

    # Trade imbalance estimate
    df['trade_imbalance_est'] = df['orderbook_imbalance_est'] * 0.8  # Correlated but dampened

    return df


def simulate_liquidation_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate liquidation proximity based on price levels and volatility
    """
    df = df.copy()

    # Rolling high/low for liquidation level estimation
    df['rolling_high_72h'] = df['high'].rolling(72).max()
    df['rolling_low_72h'] = df['low'].rolling(72).min()

    # Estimate liquidation distances (assuming 10x leverage typical)
    leverage = 10

    # Distance to short liquidation (price rising towards short squeeze)
    df['short_liq_level'] = df['rolling_high_72h'] * (1 + 1/leverage)
    df['short_liq_distance'] = (df['short_liq_level'] - df['close']) / df['close']

    # Distance to long liquidation (price falling towards long cascade)
    df['long_liq_level'] = df['rolling_low_72h'] * (1 - 1/leverage)
    df['long_liq_distance'] = (df['close'] - df['long_liq_level']) / df['close']

    # Liquidation risk score
    df['min_liq_distance'] = df[['short_liq_distance', 'long_liq_distance']].abs().min(axis=1)
    df['liq_risk'] = 1 / (1 + df['min_liq_distance'] * 10)

    # Which side is closer to liquidation (-1 = longs at risk, +1 = shorts at risk)
    df['liq_direction'] = np.where(
        df['long_liq_distance'].abs() < df['short_liq_distance'].abs(),
        -1,  # Longs closer to liquidation
        1    # Shorts closer to liquidation
    )

    return df


def simulate_cross_exchange_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate cross-exchange divergence

    In reality, divergences are small and mean-reverting
    We simulate based on volatility and momentum
    """
    df = df.copy()

    # Simulate divergence as noise that increases with volatility
    volatility = df['close'].pct_change().rolling(24).std()

    # Random divergence scaled by volatility
    np.random.seed(42)  # Reproducible
    noise = np.random.randn(len(df)) * 0.0005  # Base divergence ~0.05%
    df['cross_exchange_divergence'] = noise * (1 + volatility * 100)

    # Divergence direction (-1 to 1)
    df['divergence_direction'] = np.sign(df['cross_exchange_divergence'])

    return df


def simulate_basis_features(df: pd.DataFrame, funding_df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate spot-futures basis from funding rate

    Basis and funding rate are related:
    High funding = contango = positive basis
    Low/negative funding = backwardation = negative basis
    """
    df = df.copy()

    # Check if funding_rate already exists (from earlier merge)
    if 'funding_rate' not in df.columns:
        # Merge funding data
        if not funding_df.empty and 'funding_rate' in funding_df.columns:
            funding_df = funding_df.copy()
            funding_df['timestamp'] = pd.to_datetime(funding_df['timestamp'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Forward fill funding rate to all candles
            merged = pd.merge_asof(
                df.sort_values('timestamp'),
                funding_df[['timestamp', 'funding_rate']].sort_values('timestamp'),
                on='timestamp',
                direction='backward',
                suffixes=('', '_funding')
            )

            if 'funding_rate' in merged.columns:
                df = merged.copy()
            elif 'funding_rate_funding' in merged.columns:
                df = merged.copy()
                df['funding_rate'] = df['funding_rate_funding']
        else:
            df['funding_rate'] = 0

    # Ensure funding_rate exists
    if 'funding_rate' not in df.columns:
        df['funding_rate'] = 0

    # Estimate basis from funding rate
    # Basis ≈ funding_rate * 3 (8h funding → 24h basis approximation)
    df['basis_est'] = df['funding_rate'].fillna(0) * 3
    df['basis_bps'] = df['basis_est'] * 10000

    # Basis momentum
    df['basis_momentum'] = df['basis_est'].diff(8)  # 8-hour change

    # Unusual basis (z-score)
    basis_mean = df['basis_est'].rolling(72).mean()
    basis_std = df['basis_est'].rolling(72).std()
    df['basis_zscore'] = (df['basis_est'] - basis_mean) / (basis_std + 1e-10)

    return df


# =============================================================================
# MAIN FEATURE COMPUTATION
# =============================================================================

def compute_hmm_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute basic features needed for HMM regime detection
    These are separate from LSTM features
    """
    from ta.volatility import AverageTrueRange
    from ta.trend import ADXIndicator

    df = df.copy()

    # Return
    df['return_1h'] = df['close'].pct_change(periods=1)

    # Volatility
    df['volatility_24h'] = df['close'].pct_change().rolling(window=24).std()

    # ATR
    atr = AverageTrueRange(
        high=df['high'],
        low=df['low'],
        close=df['close'],
        window=14,
        fillna=False
    )
    df['atr'] = atr.average_true_range()

    # Volume ratio
    df['volume_ma'] = df['volume'].rolling(window=24).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_ma'] + 1e-10)

    # ADX
    adx = ADXIndicator(
        high=df['high'],
        low=df['low'],
        close=df['close'],
        window=14,
        fillna=False
    )
    df['adx'] = adx.adx()

    return df


def compute_orderflow_features(
    ohlcv_df: pd.DataFrame,
    funding_df: pd.DataFrame,
    include_sentiment: bool = True
) -> pd.DataFrame:
    """
    Compute all order flow based features

    For backtesting: uses simulated/estimated features
    For live trading: fetches real-time data
    """
    print("Computing order flow features...")

    df = ohlcv_df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # First compute HMM features (needed for regime detection)
    print("  Computing HMM features (for regime detection)...")
    df = compute_hmm_features(df)

    # 1. Funding rate momentum (from actual funding data)
    print("  Computing funding rate momentum...")
    if not funding_df.empty:
        funding_momentum = calculate_funding_momentum(funding_df)

        # Merge funding features
        funding_momentum['timestamp'] = pd.to_datetime(funding_momentum['timestamp'])
        df = pd.merge_asof(
            df.sort_values('timestamp'),
            funding_momentum[[
                'timestamp', 'funding_rate',
                'funding_8h_momentum', 'funding_24h_momentum', 'funding_72h_momentum',
                'funding_acceleration', 'funding_zscore'
            ]].sort_values('timestamp'),
            on='timestamp',
            direction='backward'
        )
    else:
        df['funding_rate'] = 0
        df['funding_8h_momentum'] = 0
        df['funding_24h_momentum'] = 0
        df['funding_72h_momentum'] = 0
        df['funding_acceleration'] = 0
        df['funding_zscore'] = 0

    # 2. Simulated order book imbalance
    print("  Computing order flow imbalance (estimated)...")
    df = simulate_orderflow_features(df)

    # 3. Simulated liquidation features
    print("  Computing liquidation proximity (estimated)...")
    df = simulate_liquidation_features(df)

    # 4. Simulated cross-exchange divergence
    print("  Computing cross-exchange divergence (estimated)...")
    df = simulate_cross_exchange_features(df)

    # 5. Basis features
    print("  Computing spot-futures basis (estimated)...")
    df = simulate_basis_features(df, funding_df)

    # 6. Sentiment features (if included)
    if include_sentiment:
        print("  Adding sentiment features...")
        # Fear & Greed - use placeholder for backtesting
        df['fng_value'] = 0.5  # Neutral placeholder

        # FinBERT - use placeholder for backtesting
        df['finbert_1h'] = 0
        df['finbert_4h'] = 0

    # Clean up
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.ffill().bfill().fillna(0)

    print(f"✅ Order flow features computed: {len(df)} rows")

    return df


def get_orderflow_feature_columns() -> List[str]:
    """Return list of order flow feature column names"""
    return [
        # Funding rate features
        'funding_rate',
        'funding_8h_momentum',
        'funding_24h_momentum',
        'funding_72h_momentum',
        'funding_acceleration',
        'funding_zscore',

        # Order flow estimates
        'orderbook_imbalance_est',
        'vpin_est',
        'trade_imbalance_est',

        # Liquidation features
        'short_liq_distance',
        'long_liq_distance',
        'liq_risk',
        'liq_direction',

        # Cross-exchange features
        'cross_exchange_divergence',
        'divergence_direction',

        # Basis features
        'basis_est',
        'basis_bps',
        'basis_momentum',
        'basis_zscore',

        # Sentiment (if included)
        'fng_value',
        'finbert_1h',
        'finbert_4h'
    ]


if __name__ == "__main__":
    print("Testing order flow features module...")

    # Test real-time data fetching
    print("\n1. Testing order book imbalance...")
    imbalance = fetch_order_book_imbalance()
    print(f"   Imbalance: {imbalance}")

    print("\n2. Testing trade toxicity...")
    trades = fetch_recent_trades(limit=100)
    if not trades.empty:
        toxicity = calculate_trade_toxicity(trades)
        print(f"   VPIN: {toxicity['vpin']:.3f}")

    print("\n3. Testing multi-exchange prices...")
    prices = fetch_multi_exchange_prices()
    print(f"   Prices: {prices}")

    divergence = calculate_cross_exchange_divergence(prices)
    print(f"   Divergence: {divergence}")

    print("\n4. Testing basis...")
    basis = fetch_spot_futures_basis()
    print(f"   Basis: {basis}")

    print("\n✅ Order flow features test complete")
