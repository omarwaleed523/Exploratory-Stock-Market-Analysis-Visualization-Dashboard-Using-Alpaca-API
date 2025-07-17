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
        logger.info("Initializing Alpaca API client...")
        
        # Detailed credential diagnostics
        self.api_key = None
        self.api_secret = None
        self.base_url = None
        self.data_url = None
        
        # Try to get credentials from Streamlit secrets first
        streamlit_creds_found = False
        try:
            import streamlit as st
            if hasattr(st, 'secrets'):
                logger.info("Streamlit is available and secrets object exists")
                
                if 'alpaca' in st.secrets:
                    logger.info("'alpaca' section found in Streamlit secrets")
                    
                    # Check each required key
                    missing_keys = []
                    for key in ['api_key_id', 'api_secret_key', 'api_base_url']:
                        if key not in st.secrets.alpaca:
                            missing_keys.append(key)
                    
                    if missing_keys:
                        logger.warning(f"Missing keys in Streamlit secrets: {', '.join(missing_keys)}")
                    else:
                        # All keys present, set credentials
                        self.api_key = st.secrets.alpaca.api_key_id
                        self.api_secret = st.secrets.alpaca.api_secret_key
                        self.base_url = st.secrets.alpaca.api_base_url
                        self.data_url = st.secrets.alpaca.get('data_url', 'https://data.alpaca.markets')
                        streamlit_creds_found = True
                        logger.info("Successfully loaded credentials from Streamlit secrets")
                else:
                    logger.warning("'alpaca' section not found in Streamlit secrets")
                    available_sections = list(st.secrets.keys()) if hasattr(st.secrets, 'keys') else []
                    logger.info(f"Available sections in secrets: {available_sections}")
            else:
                logger.warning("Streamlit secrets attribute not found")
        except ImportError:
            logger.info("Streamlit module not available")
        except Exception as e:
            logger.warning(f"Error accessing Streamlit secrets: {str(e)}")
        
        # Fall back to environment variables if Streamlit secrets not available
        if not streamlit_creds_found:
            logger.info("Falling back to environment variables")
            self.api_key = os.getenv('ALPACA_API_KEY_ID')
            self.api_secret = os.getenv('ALPACA_API_SECRET_KEY')
            self.base_url = os.getenv('ALPACA_API_BASE_URL')
            self.data_url = os.getenv('ALPACA_DATA_URL', 'https://data.alpaca.markets')
            
            # Log what we found
            logger.info(f"API Key from env: {'Found' if self.api_key else 'Not found'}")
            logger.info(f"API Secret from env: {'Found' if self.api_secret else 'Not found'}")
            logger.info(f"Base URL from env: {'Found' if self.base_url else 'Not found'}")
            logger.info(f"Data URL from env: {'Found' if self.data_url else 'Using default'}")
        
        # Check if we have the minimum required credentials
        if not all([self.api_key, self.api_secret, self.base_url]):
            missing = []
            if not self.api_key: missing.append("API Key")
            if not self.api_secret: missing.append("API Secret")
            if not self.base_url: missing.append("Base URL")
            
            error_msg = f"Missing required Alpaca API credentials: {', '.join(missing)}. " + \
                        "Please set credentials either in Streamlit secrets " + \
                        "(.streamlit/secrets.toml) or in environment variables (.env file)."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Log successful initialization
        logger.info(f"Using API Base URL: {self.base_url}")
        logger.info(f"Using Data URL: {self.data_url}")
        
        # Create Alpaca API client
        try:
            self.api = tradeapi.REST(
                key_id=self.api_key,
                secret_key=self.api_secret,
                base_url=self.base_url,
                # Don't specify api_version if it's already in the base_url
            )
            logger.info("Successfully created Alpaca REST client")
            
            # Perform a simple API call to validate credentials
            try:
                _ = self.api.get_account()
                logger.info("✅ API credentials validated successfully")
            except Exception as e:
                logger.warning(f"⚠️ API credential validation failed: {str(e)}")
                logger.warning("The client is initialized but API credentials may not be valid")
                # We don't raise here since the credentials might still work for data endpoints
        except Exception as e:
            logger.error(f"Failed to create Alpaca REST client: {str(e)}")
            raise ValueError(f"Failed to initialize Alpaca API client: {str(e)}")
        
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
        
        # Format dates as YYYY-MM-DD which Alpaca accepts
        start_str = start.strftime('%Y-%m-%d')
        end_str = end.strftime('%Y-%m-%d')
        
        # Process one symbol at a time to avoid API formatting issues
        all_bars = {}
        errors = {}
        
        for symbol in symbols:
            self._check_rate_limit()  # Check rate limit before each symbol
            
            try:
                # Get bars for this symbol
                logger.debug(f"Requesting {timeframe} bars for {symbol} from {start_str} to {end_str}")
                
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
                if len(bars) == 0:
                    logger.warning(f"No data returned for {symbol} in timeframe {timeframe}")
                    all_bars[symbol] = []
                    continue
                
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
                
                logger.info(f"Successfully retrieved {len(all_bars[symbol])} bars for {symbol}")
                
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Error fetching bars for {symbol}: {error_msg}")
                errors[symbol] = error_msg
                all_bars[symbol] = []  # Empty list for this symbol
        
        # Check if we got any data
        total_bars = sum(len(bars) for bars in all_bars.values())
        if total_bars == 0:
            if errors:
                error_details = "; ".join(f"{symbol}: {error}" for symbol, error in errors.items())
                raise ValueError(f"Failed to retrieve any data. Errors: {error_details}")
            else:
                raise ValueError(f"No data available for the selected symbols and timeframe")
        
        # If some symbols had errors but others succeeded, log a warning
        if errors and total_bars > 0:
            logger.warning(f"Retrieved {total_bars} bars total, but {len(errors)} symbols had errors")
        
        return all_bars
    
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
