"""
API client for Alpaca with rate limiting to handle the 200 calls/minute restriction.
"""
import os
import time
import logging
from datetime import datetime, timedelta, timezone
import pytz
from typing import Dict, List, Optional, Union, Any
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import REST

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RateLimitedAlpacaClient:
    """
    A wrapper around the Alpaca API client that implements rate limiting
    to stay within the 200 calls/minute limit.
    """
    
    def __init__(self, max_calls_per_minute: int = 190):
        """
        Initialize the Alpaca client with rate limiting.
        
        Args:
            max_calls_per_minute: Maximum number of API calls per minute (default: 190, 
                                  slightly under the 200 limit to provide a safety buffer)
        """
        load_dotenv()
        
        # Try to get credentials from Streamlit secrets first, then fall back to env vars
        try:
            import streamlit as st
            if 'alpaca' in st.secrets:
                self.api_key = st.secrets.alpaca.api_key_id
                self.api_secret = st.secrets.alpaca.api_secret_key
                self.base_url = st.secrets.alpaca.api_base_url
                self.data_url = st.secrets.alpaca.data_url
                logger.info("Using Streamlit secrets for Alpaca API credentials")
            else:
                raise ImportError("Streamlit secrets not available")
        except (ImportError, AttributeError):
            # Fall back to environment variables
            self.api_key = os.getenv('ALPACA_API_KEY_ID')
            self.api_secret = os.getenv('ALPACA_API_SECRET_KEY')
            self.base_url = os.getenv('ALPACA_API_BASE_URL')
            self.data_url = os.getenv('ALPACA_DATA_URL', 'https://data.alpaca.markets/v2')
            logger.info("Using environment variables for Alpaca API credentials")
        
        if not all([self.api_key, self.api_secret, self.base_url]):
            raise ValueError("API credentials not found. Please set credentials either in Streamlit secrets "
                            "(.streamlit/secrets.toml) or in environment variables (.env file).")
        
        # Create Alpaca API client
        self.api = tradeapi.REST(
            key_id=self.api_key,
            secret_key=self.api_secret,
            base_url=self.base_url,
            # Don't specify api_version if it's already in the base_url
            # api_version='v2'
        )
        
        # Rate limiting parameters
        self.max_calls_per_minute = max_calls_per_minute
        self.call_timestamps: List[float] = []
        
        logger.info(f"Alpaca client initialized with rate limiting: {max_calls_per_minute} calls/minute")
    
    def _check_rate_limit(self) -> None:
        """
        Check if we're exceeding the rate limit and sleep if necessary.
        """
        current_time = time.time()
        
        # Remove timestamps older than 1 minute
        one_minute_ago = current_time - 60
        self.call_timestamps = [ts for ts in self.call_timestamps if ts > one_minute_ago]
        
        # If we've reached the limit, sleep until we can make another call
        if len(self.call_timestamps) >= self.max_calls_per_minute:
            oldest_timestamp = min(self.call_timestamps)
            sleep_time = 60 - (current_time - oldest_timestamp)
            
            if sleep_time > 0:
                logger.warning(f"Rate limit reached. Sleeping for {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
        
        # Add the current timestamp
        self.call_timestamps.append(time.time())
    
    def get_account(self) -> Dict[str, Any]:
        """Get Alpaca account information."""
        self._check_rate_limit()
        return self.api.get_account()._raw
    
    def get_bars(
        self, 
        symbols: Union[str, List[str]], 
        timeframe: str,
        start: datetime, 
        end: Optional[datetime] = None,
        limit: Optional[int] = None,
        adjustment: str = 'raw'
    ) -> Dict[str, Any]:
        """
        Get historical bar data for given symbols.
        
        Args:
            symbols: A symbol or list of symbols
            timeframe: The timeframe for the bars ('1Min', '5Min', '15Min', '1H', '1D', etc.)
            start: Start datetime
            end: End datetime (defaults to now)
            limit: Maximum number of bars to return
            adjustment: Adjustments to apply ('raw', 'split', 'dividend', 'all')
            
        Returns:
            Dictionary with bar data for each symbol
        """
        self._check_rate_limit()
        
        if end is None:
            end = datetime.now(timezone.utc)
        elif end.tzinfo is None:
            # Convert naive datetime to UTC
            end = end.replace(tzinfo=timezone.utc)
            
        if start.tzinfo is None:
            # Convert naive datetime to UTC
            start = start.replace(tzinfo=timezone.utc)
        
        if isinstance(symbols, str):
            symbols = [symbols]
        
        logger.info(f"Fetching {timeframe} bars for {len(symbols)} symbols: {', '.join(symbols[:5])}" + 
                   (f"... and {len(symbols) - 5} more" if len(symbols) > 5 else ""))
        
        try:
            # Process one symbol at a time to avoid API formatting issues
            all_bars = {}
            
            # Format dates as YYYY-MM-DD which Alpaca accepts
            start_str = start.strftime('%Y-%m-%d')
            end_str = end.strftime('%Y-%m-%d')
            
            for symbol in symbols:
                self._check_rate_limit()  # Check rate limit before each symbol
                
                # Get bars for this symbol
                bars = self.api.get_bars(
                    symbol,
                    timeframe,
                    start=start_str,
                    end=end_str,
                    limit=limit,
                    adjustment=adjustment,
                    feed='iex'  # Explicitly use IEX feed for free tier data
                )
                
                # Process the bars
                if symbol not in all_bars:
                    all_bars[symbol] = []
                
                for bar in bars:
                    all_bars[symbol].append({
                        'timestamp': bar.t,
                        'open': bar.o,
                        'high': bar.h,
                        'low': bar.l,
                        'close': bar.c,
                        'volume': bar.v
                    })
            
            return all_bars
            
        except Exception as e:
            logger.error(f"Error fetching bars: {e}")
            raise
    
    def get_latest_trade(self, symbol: str) -> Dict[str, Any]:
        """
        Get the latest trade for a symbol.
        
        Args:
            symbol: The stock symbol
            
        Returns:
            Trade information
        """
        self._check_rate_limit()
        return self.api.get_latest_trade(symbol)._raw
    
    def get_latest_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Get the latest quote for a symbol.
        
        Args:
            symbol: The stock symbol
            
        Returns:
            Quote information
        """
        self._check_rate_limit()
        return self.api.get_latest_quote(symbol)._raw
    
    def list_assets(self, status: str = 'active', asset_class: str = 'us_equity') -> List[Dict[str, Any]]:
        """
        List assets available for trading.
        
        Args:
            status: Asset status ('active', 'inactive')
            asset_class: Asset class ('us_equity', 'crypto')
            
        Returns:
            List of assets
        """
        self._check_rate_limit()
        assets = self.api.list_assets(status=status, asset_class=asset_class)
        return [asset._raw for asset in assets]
    
    def get_calendar(self, start: datetime, end: datetime) -> List[Dict[str, Any]]:
        """
        Get market calendar.
        
        Args:
            start: Start date
            end: End date
            
        Returns:
            List of market days
        """
        self._check_rate_limit()
        calendar = self.api.get_calendar(
            start=start.strftime('%Y-%m-%d'),
            end=end.strftime('%Y-%m-%d')
        )
        return [day._raw for day in calendar]
