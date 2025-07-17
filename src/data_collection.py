"""
Data collection module for fetching and caching stock data from Alpaca.
"""
import os
import sys
import json
import logging
from datetime import datetime, timedelta, timezone
import pytz
from typing import Dict, List, Optional, Union, Any
import pandas as pd

# Add parent directory to the path to make imports work properly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.api_client import RateLimitedAlpacaClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StockDataCollector:
    """
    Class for collecting stock data from Alpaca API and managing cached data.
    Handles the API limitations and provides efficient data retrieval.
    """
    
    def __init__(self, cache_dir: str = 'data'):
        """
        Initialize the data collector.
        
        Args:
            cache_dir: Directory to store cached data
        """
        self.client = RateLimitedAlpacaClient()
        self.cache_dir = cache_dir
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        logger.info(f"Stock data collector initialized with cache directory: {cache_dir}")
    
    def _get_cache_filename(self, symbols: Union[str, List[str]], timeframe: str, 
                          start_date: datetime, end_date: datetime) -> str:
        """
        Generate a cache filename based on the query parameters.
        
        Args:
            symbols: Stock symbols
            timeframe: Bar timeframe
            start_date: Start date
            end_date: End date
            
        Returns:
            Cache filename
        """
        if isinstance(symbols, list):
            # For multiple symbols, use the count and first symbol
            symbols_str = f"{len(symbols)}symbols_{symbols[0]}"
        else:
            symbols_str = symbols
            
        start_str = start_date.strftime('%Y%m%d')
        end_str = end_date.strftime('%Y%m%d')
        
        return os.path.join(
            self.cache_dir, 
            f"{symbols_str}_{timeframe}_{start_str}_{end_str}.csv"
        )
    
    def get_historical_data(
        self,
        symbols: Union[str, List[str]],
        timeframe: str = '1D',
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        use_cache: bool = True,
        force_refresh: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        Get historical bar data for given symbols with caching support.
        
        Args:
            symbols: A symbol or list of symbols
            timeframe: The timeframe for the bars ('1Min', '5Min', '15Min', '1H', '1D', etc.)
            start_date: Start date (defaults to 2 years ago)
            end_date: End date (defaults to today)
            use_cache: Whether to use cached data if available
            force_refresh: Whether to force refresh the data from API
            
        Returns:
            Dictionary with DataFrame for each symbol
        """
        # Set default dates if not provided
        if end_date is None:
            end_date = datetime.now(timezone.utc)
        elif end_date.tzinfo is None:
            # Convert naive datetime to timezone-aware
            end_date = end_date.replace(tzinfo=timezone.utc)
        
        if start_date is None:
            # Default to 2 years of data
            start_date = end_date - timedelta(days=2*365)
        elif start_date.tzinfo is None:
            # Convert naive datetime to timezone-aware
            start_date = start_date.replace(tzinfo=timezone.utc)
        
        # Convert single symbol to list for consistent handling
        if isinstance(symbols, str):
            symbols_list = [symbols]
        else:
            symbols_list = symbols
        
        # Check cache first if enabled
        cache_file = self._get_cache_filename(symbols, timeframe, start_date, end_date)
        
        if use_cache and os.path.exists(cache_file) and not force_refresh:
            logger.info(f"Loading cached data from {cache_file}")
            try:
                # Load all symbols from the cache file
                all_data = {}
                df = pd.read_csv(cache_file)
                
                # Group by symbol
                for symbol in df['symbol'].unique():
                    symbol_data = df[df['symbol'] == symbol].copy()
                    symbol_data['timestamp'] = pd.to_datetime(symbol_data['timestamp'], utc=True)
                    symbol_data.set_index('timestamp', inplace=True)
                    all_data[symbol] = symbol_data
                
                return all_data
            except Exception as e:
                logger.warning(f"Error loading cached data: {e}. Fetching from API instead.")
        
        # Fetch data from API
        logger.info(f"Fetching {timeframe} data for {len(symbols_list)} symbols from {start_date} to {end_date}")
        
        try:
            bars_dict = self.client.get_bars(
                symbols=symbols_list,
                timeframe=timeframe,
                start=start_date,
                end=end_date
            )
            
            # Convert to DataFrames
            result = {}
            all_data_rows = []
            
            for symbol, bars in bars_dict.items():
                if not bars:
                    logger.warning(f"No data returned for {symbol}")
                    continue
                
                # Convert to DataFrame
                df = pd.DataFrame(bars)
                df['symbol'] = symbol
                
                # Convert timestamp to datetime with timezone information
                df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
                
                # Add to all data for caching
                all_data_rows.append(df)
                
                # Set timestamp as index for returned data
                df_indexed = df.set_index('timestamp')
                result[symbol] = df_indexed
            
            # Cache the data if we have any results
            if all_data_rows and use_cache:
                all_data_df = pd.concat(all_data_rows)
                all_data_df.to_csv(cache_file, index=False)
                logger.info(f"Cached data to {cache_file}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            raise
    
    def get_tradable_symbols(self, min_price: float = 5.0, max_symbols: int = 100) -> List[str]:
        """
        Get a list of tradable stock symbols.
        
        Args:
            min_price: Minimum price filter
            max_symbols: Maximum number of symbols to return
            
        Returns:
            List of stock symbols
        """
        logger.info(f"Fetching tradable symbols with min_price={min_price}")
        
        # Check cache first
        cache_file = os.path.join(self.cache_dir, 'tradable_symbols.json')
        
        if os.path.exists(cache_file):
            # Check if cache is less than 1 day old
            cache_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
            cache_time = cache_time.replace(tzinfo=timezone.utc)  # Make timezone-aware
            now = datetime.now(timezone.utc)
            
            if cache_time > now - timedelta(days=1):
                logger.info(f"Loading tradable symbols from cache {cache_file}")
                with open(cache_file, 'r') as f:
                    return json.load(f)
        
        try:
            # Fetch active assets
            assets = self.client.list_assets(status='active', asset_class='us_equity')
            
            # Filter for tradable and minimum price
            tradable_symbols = []
            
            for asset in assets:
                if asset['tradable']:
                    symbol = asset['symbol']
                    
                    try:
                        # Get latest quote to check price
                        quote = self.client.get_latest_trade(symbol)
                        price = quote['p']
                        
                        if price >= min_price:
                            tradable_symbols.append(symbol)
                            
                            # Stop if we have enough symbols
                            if len(tradable_symbols) >= max_symbols:
                                break
                    except Exception as e:
                        logger.warning(f"Error getting quote for {symbol}: {e}")
                        continue
            
            # Cache the results
            with open(cache_file, 'w') as f:
                json.dump(tradable_symbols, f)
            
            logger.info(f"Found {len(tradable_symbols)} tradable symbols")
            return tradable_symbols
            
        except Exception as e:
            logger.error(f"Error fetching tradable symbols: {e}")
            raise
    
    def get_market_hours(self, date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Get market hours for a specific date.
        
        Args:
            date: The date to check (defaults to today)
            
        Returns:
            Dictionary with market hours information
        """
        if date is None:
            date = datetime.now(timezone.utc)
        elif date.tzinfo is None:
            # Convert naive datetime to timezone-aware
            date = date.replace(tzinfo=timezone.utc)
        
        start_date = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = start_date + timedelta(days=1)
        
        try:
            calendar = self.client.get_calendar(start=start_date, end=end_date)
            
            if not calendar:
                return {'is_open': False}
            
            return {
                'is_open': True,
                'open_time': calendar[0]['open'],
                'close_time': calendar[0]['close']
            }
            
        except Exception as e:
            logger.error(f"Error fetching market hours: {e}")
            raise
