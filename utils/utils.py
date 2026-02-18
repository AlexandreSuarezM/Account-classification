from google.cloud import bigquery
from google.oauth2 import service_account
import os 
from dotenv import load_dotenv 
from datetime import datetime, timezone, timedelta
from functools import lru_cache
import requests
import pandas as pd
import numpy as np
import time

load_dotenv ()

PROJECT_ID = os.getenv("PROJECT_ID")

def bigquery(query):
    credentials = service_account.Credentials.from_service_account_file(filename='credentials/insights-credentials.json')
    client = bigquery.Client(
        credentials = credentials, 
        project = PROJECT_ID)

    query_job = client.query(query)
    rows = query_job.result() 
    results = [dict(row) for row in rows]
    return results


def calculate_fee_rate(df):
    """
    Calculate fee rates with proper handling of zero divisions and apply tolerance filtering.
    
    Parameters:
    df (pandas.DataFrame): DataFrame with 'fee' and 'amount' columns
    tolerance_threshold (float): Maximum allowed fee rate (1.01 = 1% tolerance)
    min_amount (float): Minimum amount to consider for fee rate calculation
    
    Returns:
    pandas.DataFrame: Cleaned DataFrame with fee_rate column
    """
    
    df = df.copy()
    
    # Handle different scenarios for fee rate calculation
    conditions = [
        (df['amount'] == 0) & (df['fee_computed'] == 0),   
        (df['amount'] == 0) & (df['fee_computed'] > 0),
        df['amount'] >= 0             
    ]
    
    choices = [
        0,                                      
        np.inf,          
        df['fee_computed'] / df['amount']                    
    ]
    
    # Calculate fee rate using numpy select for vectorized conditions
    df['fee_rate'] = np.select(conditions, choices, default=np.inf)
    
    # Replace any remaining NaN or inf values
    df['fee_rate'] = df['fee_rate'].replace([np.inf, -np.inf, np.nan], np.inf)
    
    return df

# Cache to store asset prices by asset_id (single avg price per asset)
_price_cache = {}

# Supported asset IDs
SUPPORTED_ASSETS = {
    'usdc': 31566704,       'hog': 3178895177,      'ora': 1284444444,      'talgo': 2537013734,
    'vote': 452399768,      'gobtc': 386192725,     'xusd': 760037151,      'tiny': 2200000000,
    'alpha': 2726252423,    'opul': 287867876,      'finite': 400593267,    'niko': 1265975021,
    'pow': 2994233666,      'xalgo': 1134696561,    'coop': 796425061,      'vest': 700965019,
    'goeth': 386195940,     'compx': 1732165149,    'usdt': 312769,         'sparky': 3054226103
}

SUPPORTED_ASSET_IDS = set(SUPPORTED_ASSETS.values())
# Special case asset IDs
ASSET_KEEP_NATIVE = 971381860  # Keep native amount
ASSET_USE_USDC = 971384592     # Use USDC multiplier
WC_ASSETS = [127746157, 127745593, 127746786, 3145862805]
ASSET_USE_USDC_SCALED = 849191641  # Use USDC * 0.01484306067
USDC_SCALING_FACTOR = 0.01484306067

def date_to_unix_timestamp(start_date_str, end_date_str):
    """
    Converts start and end dates in YYYY-MM-DD format to Unix timestamps.

    Args:
        start_date_str: Start date string in YYYY-MM-DD format.
        end_date_str: End date string in YYYY-MM-DD format.

    Returns:
        A tuple containing the start and end Unix timestamps.
    """
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc) + timedelta(days=1) - timedelta(seconds=1)

    start_timestamp = int(start_date.timestamp())
    end_timestamp = int(end_date.timestamp())

    return start_timestamp, end_timestamp


def get_avg_close_price(asset_id, start_date, end_date, max_retries=3, base_delay=1):
    """
    Fetches the average close price for an asset over a date range.
    
    Args:
        asset_id: The asset ID
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds between retries (exponential backoff)

    Returns:
        float: Average close price over the period, or None if not available
    """
    start_unix, end_unix = date_to_unix_timestamp(start_date, end_date)

    price_feed = f'https://indexer.vestige.fi/assets/{asset_id}/candles?network_id=0&interval=86400&start={start_unix}&end={end_unix}&denominating_asset_id=0&volume_in_denominating_asset=false'

    for attempt in range(max_retries):
        try:
            response = requests.get(price_feed, timeout=10)
            
            if response.status_code == 429:
                # Rate limited - exponential backoff
                wait_time = base_delay * (2 ** attempt)
                print(f"Rate limited for asset {asset_id}. Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                time.sleep(wait_time)
                continue
            
            response.raise_for_status()
            data = response.json()
            
            if not data or len(data) == 0:
                return None
            
            # Calculate average close price
            close_prices = [candle.get('close') for candle in data if candle.get('close') is not None]
            
            if not close_prices:
                return None
            
            avg_price = sum(close_prices) / len(close_prices)
            return avg_price
        
        except requests.HTTPError as e:
            if response.status_code == 429 and attempt < max_retries - 1:
                continue
            print(f"HTTP Error fetching price for asset {asset_id}: {e}")
            return None
        except (requests.RequestException, KeyError, IndexError, TypeError) as e:
            print(f"Error fetching price for asset {asset_id}: {e}")
            return None
    
    print(f"Max retries exceeded for asset {asset_id}")
    return None


def get_cached_avg_price(asset_id):
    """
    Gets average close price for an asset, using cache to avoid duplicate API calls.
    
    Args:
        asset_id: The asset ID
    
    Returns:
        float: Average close price, or None if not available
    """
    if asset_id in _price_cache:
        return _price_cache[asset_id]
    
    return None  # Return None if not in cache (should be pre-populated)


def fetch_all_supported_assets(start_date, end_date, delay_between_requests=0.5):
    """
    Fetches average prices for all supported assets over the date range.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        delay_between_requests: Delay in seconds between requests
    
    Returns:
        dict: Mapping of asset_id to average price
    """
    print(f"Fetching average prices for {len(SUPPORTED_ASSETS)} assets from {start_date} to {end_date}...")
    
    for idx, (name, asset_id) in enumerate(SUPPORTED_ASSETS.items(), 1):
        print(f"[{idx}/{len(SUPPORTED_ASSETS)}] Fetching {name} (ID: {asset_id})...")
        
        avg_price = get_avg_close_price(asset_id, start_date, end_date)
        _price_cache[asset_id] = avg_price
        
        if avg_price:
            print(f"{name}: {avg_price:.6f} ALGO")
        else:
            print(f"{name}: No data available")
        
        # Rate limiting
        if idx < len(SUPPORTED_ASSETS):
            time.sleep(delay_between_requests)
    
    print(f"\nFetched prices for {sum(1 for v in _price_cache.values() if v is not None)}/{len(SUPPORTED_ASSETS)} assets")
    return _price_cache.copy()


def clear_price_cache():
    """Clears the price cache."""
    _price_cache.clear()
