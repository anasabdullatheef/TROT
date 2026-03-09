#!/usr/bin/env python3
"""
Tardis.dev Data Downloader
Downloads historical tick-level order book and trade data for BTC/USDT

Data types downloaded:
- Order book snapshots (L2) - for order book imbalance
- Trades - for VPIN calculation
- Book snapshots from multiple exchanges - for cross-exchange divergence

Requires TARDIS_API_KEY environment variable
"""

import os
import asyncio
import gzip
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from pathlib import Path

# Tardis imports
from tardis_dev import datasets, get_exchange_details

# Configuration
TARDIS_API_KEY = os.getenv("TARDIS_API_KEY", "")
DATA_DIR = Path("data/tardis")
SYMBOL = "BTCUSDT"

# Exchanges for cross-exchange divergence
EXCHANGES = {
    "binance-futures": "BTCUSDT",
    "bybit": "BTCUSDT",
    "okex-swap": "BTC-USDT-SWAP"
}


def get_date_range(years: int = 2) -> tuple:
    """Get date range for download"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * years)
    return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")


async def download_exchange_data(
    exchange: str,
    symbol: str,
    data_types: List[str],
    from_date: str,
    to_date: str
):
    """Download data for a single exchange"""

    output_dir = DATA_DIR / exchange
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📥 Downloading {exchange} {symbol}")
    print(f"   Data types: {data_types}")
    print(f"   Period: {from_date} to {to_date}")

    try:
        await datasets.download(
            exchange=exchange,
            data_types=data_types,
            from_date=from_date,
            to_date=to_date,
            symbols=[symbol],
            api_key=TARDIS_API_KEY,
            download_dir=str(output_dir)
        )
        print(f"   ✅ Download complete")
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


async def download_all_data(from_date: str, to_date: str):
    """Download data from all exchanges"""

    print("=" * 70)
    print("TARDIS DATA DOWNLOAD")
    print("=" * 70)

    if not TARDIS_API_KEY:
        print("❌ TARDIS_API_KEY not set")
        print("   Set it in .env or export TARDIS_API_KEY=your_key")
        return False

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Data types to download
    # - book_snapshot_25 or incremental_book_L2 for order book
    # - trades for trade flow
    data_types = [
        "trades",
        "book_snapshot_25"  # Top 25 levels every 100ms
    ]

    tasks = []
    for exchange, symbol in EXCHANGES.items():
        task = download_exchange_data(
            exchange=exchange,
            symbol=symbol,
            data_types=data_types,
            from_date=from_date,
            to_date=to_date
        )
        tasks.append(task)

    results = await asyncio.gather(*tasks, return_exceptions=True)

    success = sum(1 for r in results if r is True)
    print(f"\n✅ Downloaded {success}/{len(EXCHANGES)} exchanges")

    return success > 0


def parse_trades_file(filepath: Path) -> pd.DataFrame:
    """Parse a trades CSV file"""
    try:
        if filepath.suffix == '.gz':
            df = pd.read_csv(filepath, compression='gzip')
        else:
            df = pd.read_csv(filepath)

        # Standardize columns
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='us')
        df['price'] = df['price'].astype(float)
        df['amount'] = df['amount'].astype(float)

        # side: 'buy' or 'sell' (taker side)
        if 'side' in df.columns:
            df['is_buy'] = df['side'] == 'buy'

        return df
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return pd.DataFrame()


def parse_orderbook_file(filepath: Path) -> pd.DataFrame:
    """Parse an order book snapshot file"""
    try:
        if filepath.suffix == '.gz':
            df = pd.read_csv(filepath, compression='gzip')
        else:
            df = pd.read_csv(filepath)

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='us')

        return df
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return pd.DataFrame()


def compute_vpin_from_trades(trades_df: pd.DataFrame, bucket_minutes: int = 60) -> pd.DataFrame:
    """
    Compute VPIN from trade data

    VPIN = |Buy Volume - Sell Volume| / Total Volume per bucket
    """
    if trades_df.empty or 'is_buy' not in trades_df.columns:
        return pd.DataFrame()

    df = trades_df.copy()
    df['bucket'] = df['timestamp'].dt.floor(f'{bucket_minutes}min')

    # Aggregate by bucket
    grouped = df.groupby('bucket').agg({
        'amount': 'sum',
        'is_buy': lambda x: (x * df.loc[x.index, 'amount']).sum()  # Buy volume
    }).rename(columns={'amount': 'total_volume', 'is_buy': 'buy_volume'})

    grouped['sell_volume'] = grouped['total_volume'] - grouped['buy_volume']
    grouped['vpin'] = grouped['buy_volume'] / grouped['total_volume']
    grouped['trade_imbalance'] = (grouped['buy_volume'] - grouped['sell_volume']) / grouped['total_volume']

    grouped = grouped.reset_index()
    grouped = grouped.rename(columns={'bucket': 'timestamp'})

    return grouped[['timestamp', 'total_volume', 'buy_volume', 'sell_volume', 'vpin', 'trade_imbalance']]


def compute_orderbook_imbalance(ob_df: pd.DataFrame, levels: int = 10) -> pd.DataFrame:
    """
    Compute order book imbalance from snapshot data

    Imbalance = (Bid Volume - Ask Volume) / (Bid Volume + Ask Volume)
    """
    if ob_df.empty:
        return pd.DataFrame()

    results = []

    # Group by timestamp
    for ts, group in ob_df.groupby('timestamp'):
        bids = group[group['side'] == 'bid'].nlargest(levels, 'price')
        asks = group[group['side'] == 'ask'].nsmallest(levels, 'price')

        bid_volume = bids['amount'].sum()
        ask_volume = asks['amount'].sum()
        total = bid_volume + ask_volume

        imbalance = (bid_volume - ask_volume) / total if total > 0 else 0

        best_bid = bids['price'].max() if len(bids) > 0 else 0
        best_ask = asks['price'].min() if len(asks) > 0 else 0
        spread = (best_ask - best_bid) / best_bid if best_bid > 0 else 0

        results.append({
            'timestamp': ts,
            'bid_volume': bid_volume,
            'ask_volume': ask_volume,
            'orderbook_imbalance': imbalance,
            'spread': spread,
            'mid_price': (best_bid + best_ask) / 2 if best_bid > 0 else 0
        })

    return pd.DataFrame(results)


def compute_cross_exchange_divergence(prices: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Compute price divergence across exchanges

    Uses mid prices from order book snapshots
    """
    # Resample all to common frequency and merge
    dfs = []
    for exchange, df in prices.items():
        if df.empty or 'mid_price' not in df.columns:
            continue

        temp = df[['timestamp', 'mid_price']].copy()
        temp = temp.set_index('timestamp').resample('1min').last()
        temp = temp.rename(columns={'mid_price': f'price_{exchange}'})
        dfs.append(temp)

    if len(dfs) < 2:
        return pd.DataFrame()

    # Merge all
    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.join(df, how='outer')

    merged = merged.ffill().dropna()

    # Calculate divergences
    price_cols = [c for c in merged.columns if c.startswith('price_')]

    if 'price_binance-futures' in price_cols:
        base = merged['price_binance-futures']

        for col in price_cols:
            if col != 'price_binance-futures':
                exchange = col.replace('price_', '')
                merged[f'div_{exchange}'] = (base - merged[col]) / base

    # Max divergence
    div_cols = [c for c in merged.columns if c.startswith('div_')]
    if div_cols:
        merged['max_divergence'] = merged[div_cols].abs().max(axis=1)

    return merged.reset_index()


def load_and_process_tardis_data(from_date: str, to_date: str) -> pd.DataFrame:
    """
    Load all Tardis data and compute features

    Returns DataFrame with:
    - timestamp
    - vpin, trade_imbalance
    - orderbook_imbalance, spread
    - max_divergence
    """
    print("\n" + "=" * 70)
    print("PROCESSING TARDIS DATA")
    print("=" * 70)

    all_trades = []
    all_orderbooks = {}

    for exchange, symbol in EXCHANGES.items():
        exchange_dir = DATA_DIR / exchange

        if not exchange_dir.exists():
            print(f"⚠️ No data for {exchange}")
            continue

        print(f"\n📂 Processing {exchange}...")

        # Load trades
        trade_files = list(exchange_dir.glob(f"*trades*.csv*"))
        for f in trade_files[:10]:  # Limit for memory
            df = parse_trades_file(f)
            if not df.empty:
                df['exchange'] = exchange
                all_trades.append(df)

        # Load order book
        ob_files = list(exchange_dir.glob(f"*book*.csv*"))
        ob_dfs = []
        for f in ob_files[:10]:  # Limit for memory
            df = parse_orderbook_file(f)
            if not df.empty:
                ob_dfs.append(df)

        if ob_dfs:
            all_orderbooks[exchange] = pd.concat(ob_dfs, ignore_index=True)

    # Compute features
    features = pd.DataFrame()

    # VPIN from trades
    if all_trades:
        print("\n📊 Computing VPIN...")
        trades_df = pd.concat(all_trades, ignore_index=True)
        # Use only Binance for primary VPIN
        binance_trades = trades_df[trades_df['exchange'] == 'binance-futures']
        vpin_df = compute_vpin_from_trades(binance_trades)
        if not vpin_df.empty:
            features = vpin_df
            print(f"   ✅ VPIN computed: {len(vpin_df)} records")

    # Order book imbalance
    if 'binance-futures' in all_orderbooks:
        print("\n📊 Computing order book imbalance...")
        ob_df = all_orderbooks['binance-futures']
        imbalance_df = compute_orderbook_imbalance(ob_df)
        if not imbalance_df.empty:
            imbalance_df = imbalance_df.set_index('timestamp').resample('1H').mean().reset_index()
            if features.empty:
                features = imbalance_df
            else:
                features = features.merge(imbalance_df, on='timestamp', how='outer')
            print(f"   ✅ Imbalance computed: {len(imbalance_df)} records")

    # Cross-exchange divergence
    if len(all_orderbooks) >= 2:
        print("\n📊 Computing cross-exchange divergence...")
        # First compute mid prices for each exchange
        ob_with_mid = {}
        for exchange, ob_df in all_orderbooks.items():
            mid_df = compute_orderbook_imbalance(ob_df)
            if not mid_df.empty:
                ob_with_mid[exchange] = mid_df

        div_df = compute_cross_exchange_divergence(ob_with_mid)
        if not div_df.empty:
            div_df = div_df.set_index('timestamp').resample('1H').mean().reset_index()
            if features.empty:
                features = div_df
            else:
                features = features.merge(div_df, on='timestamp', how='outer')
            print(f"   ✅ Divergence computed: {len(div_df)} records")

    # Clean up
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.ffill().bfill()

    print(f"\n✅ Total features: {len(features)} records")
    print(f"   Columns: {list(features.columns)}")

    return features


def get_available_data_info():
    """Check what data is available locally"""
    print("\n" + "=" * 70)
    print("AVAILABLE TARDIS DATA")
    print("=" * 70)

    if not DATA_DIR.exists():
        print("❌ No data directory found")
        print(f"   Run download first: python tardis_download.py --download")
        return {}

    info = {}
    for exchange_dir in DATA_DIR.iterdir():
        if exchange_dir.is_dir():
            files = list(exchange_dir.glob("*.csv*"))
            total_size = sum(f.stat().st_size for f in files) / (1024**3)  # GB

            # Get date range from filenames
            dates = []
            for f in files:
                try:
                    # Format: exchange_datatype_YYYY-MM-DD_symbol.csv.gz
                    parts = f.stem.replace('.csv', '').split('_')
                    for p in parts:
                        if len(p) == 10 and p.count('-') == 2:
                            dates.append(p)
                except:
                    pass

            info[exchange_dir.name] = {
                'files': len(files),
                'size_gb': total_size,
                'dates': sorted(set(dates)) if dates else []
            }

            print(f"\n📂 {exchange_dir.name}")
            print(f"   Files: {len(files)}")
            print(f"   Size: {total_size:.2f} GB")
            if dates:
                print(f"   Dates: {min(dates)} to {max(dates)}")

    return info


async def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Tardis.dev Data Manager')
    parser.add_argument('--download', action='store_true', help='Download data')
    parser.add_argument('--info', action='store_true', help='Show available data')
    parser.add_argument('--process', action='store_true', help='Process and compute features')
    parser.add_argument('--years', type=int, default=2, help='Years of data to download')

    args = parser.parse_args()

    if args.info:
        get_available_data_info()
        return

    if args.download:
        from_date, to_date = get_date_range(args.years)
        print(f"Date range: {from_date} to {to_date}")
        await download_all_data(from_date, to_date)
        return

    if args.process:
        from_date, to_date = get_date_range(args.years)
        features = load_and_process_tardis_data(from_date, to_date)

        # Save processed features
        output_path = DATA_DIR / "processed_features.parquet"
        features.to_parquet(output_path)
        print(f"\n✅ Saved to {output_path}")
        return

    # Default: show help
    parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
