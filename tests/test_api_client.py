"""
Unit tests for the API client and data collection modules.
"""
import unittest
import os
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import pandas as pd

# Import modules to test
from src.api_client import RateLimitedAlpacaClient
from src.data_collection import StockDataCollector

class TestRateLimitedAlpacaClient(unittest.TestCase):
    """
    Unit tests for the RateLimitedAlpacaClient class.
    """
    
    @patch('src.api_client.tradeapi.REST')
    @patch('src.api_client.load_dotenv')
    def setUp(self, mock_load_dotenv, mock_rest):
        """Set up test environment."""
        # Mock environment variables
        os.environ['ALPACA_API_KEY_ID'] = 'test_key'
        os.environ['ALPACA_API_SECRET_KEY'] = 'test_secret'
        os.environ['ALPACA_API_BASE_URL'] = 'https://test.alpaca.markets'
        
        # Create client
        self.client = RateLimitedAlpacaClient(max_calls_per_minute=10)
        
        # Reset the REST mock
        mock_rest.reset_mock()
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove environment variables
        for key in ['ALPACA_API_KEY_ID', 'ALPACA_API_SECRET_KEY', 'ALPACA_API_BASE_URL']:
            if key in os.environ:
                del os.environ[key]
    
    def test_init(self):
        """Test client initialization."""
        self.assertEqual(self.client.max_calls_per_minute, 10)
        self.assertEqual(self.client.api_key, 'test_key')
        self.assertEqual(self.client.api_secret, 'test_secret')
        self.assertEqual(self.client.base_url, 'https://test.alpaca.markets')
    
    @patch('src.api_client.time.sleep')
    @patch('src.api_client.time.time')
    def test_rate_limiting(self, mock_time, mock_sleep):
        """Test rate limiting functionality."""
        # Set up time mock to return increasing values
        mock_time.side_effect = [1000 + i for i in range(20)]
        
        # Mock API call
        self.client.api.get_account = MagicMock(return_value=MagicMock(_raw={'id': 'test_account'}))
        
        # Make multiple API calls
        for _ in range(15):
            self.client.get_account()
        
        # Check if sleep was called after the rate limit was reached
        self.assertTrue(mock_sleep.called)
        self.assertEqual(mock_sleep.call_count, 5)  # Called after 10th request
    
    @patch('src.api_client.time.sleep')
    @patch('src.api_client.time.time')
    def test_get_bars(self, mock_time, mock_sleep):
        """Test get_bars method."""
        # Set up time mock
        mock_time.return_value = 1000
        
        # Mock API response
        mock_bar = MagicMock()
        mock_bar.symbol = 'AAPL'
        mock_bar.t = '2023-01-01T00:00:00Z'
        mock_bar.o = 150.0
        mock_bar.h = 155.0
        mock_bar.l = 148.0
        mock_bar.c = 152.0
        mock_bar.v = 1000000
        
        self.client.api.get_bars = MagicMock(return_value=[mock_bar])
        
        # Call method
        start = datetime(2023, 1, 1)
        end = datetime(2023, 1, 31)
        result = self.client.get_bars('AAPL', '1D', start, end)
        
        # Check result
        self.assertIn('AAPL', result)
        self.assertEqual(len(result['AAPL']), 1)
        self.assertEqual(result['AAPL'][0]['open'], 150.0)

class TestStockDataCollector(unittest.TestCase):
    """
    Unit tests for the StockDataCollector class.
    """
    
    @patch('src.data_collection.RateLimitedAlpacaClient')
    def setUp(self, mock_client_class):
        """Set up test environment."""
        # Mock client instance
        self.mock_client = MagicMock()
        mock_client_class.return_value = self.mock_client
        
        # Create temporary cache directory
        self.cache_dir = 'test_cache'
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Create collector
        self.collector = StockDataCollector(cache_dir=self.cache_dir)
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove test cache directory
        for file in os.listdir(self.cache_dir):
            os.remove(os.path.join(self.cache_dir, file))
        os.rmdir(self.cache_dir)
    
    def test_get_historical_data_no_cache(self):
        """Test getting historical data without cache."""
        # Mock API response
        mock_bars = {
            'AAPL': [
                {
                    'timestamp': '2023-01-01T00:00:00Z',
                    'open': 150.0,
                    'high': 155.0,
                    'low': 148.0,
                    'close': 152.0,
                    'volume': 1000000
                }
            ]
        }
        self.mock_client.get_bars.return_value = mock_bars
        
        # Call method
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 31)
        result = self.collector.get_historical_data(
            symbols='AAPL',
            timeframe='1D',
            start_date=start_date,
            end_date=end_date,
            use_cache=False
        )
        
        # Check result
        self.assertIn('AAPL', result)
        self.assertEqual(len(result['AAPL']), 1)
        self.assertEqual(result['AAPL']['open'].iloc[0], 150.0)
    
    def test_get_tradable_symbols(self):
        """Test getting tradable symbols."""
        # Mock API responses
        self.mock_client.list_assets.return_value = [
            {'symbol': 'AAPL', 'tradable': True},
            {'symbol': 'MSFT', 'tradable': True},
            {'symbol': 'GOOG', 'tradable': False}
        ]
        
        self.mock_client.get_latest_trade.return_value = {'p': 150.0}
        
        # Call method
        result = self.collector.get_tradable_symbols(min_price=100.0, max_symbols=2)
        
        # Check result
        self.assertEqual(len(result), 2)
        self.assertIn('AAPL', result)
        self.assertIn('MSFT', result)
        self.assertNotIn('GOOG', result)

if __name__ == '__main__':
    unittest.main()
