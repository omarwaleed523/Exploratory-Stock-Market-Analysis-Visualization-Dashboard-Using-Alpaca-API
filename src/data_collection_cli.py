"""
Main module for data collection from Alpaca API.
This script can be run directly to fetch and cache stock data.
"""
import os
import sys
import logging
import argparse
from datetime import datetime, timedelta, timezone
import pytz
from typing import List, Optional
from dotenv import load_dotenv

# Add parent directory to the path to make imports work properly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_collection import StockDataCollector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """
    Main function to collect stock data from Alpaca API.
    """
    # Load environment variables
    load_dotenv()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Collect stock market data from Alpaca API')
    
    parser.add_argument(
        '--symbols', 
        type=str, 
        default="AAPL,MSFT,GOOGL,AMZN,META",
        help='Comma-separated list of stock symbols'
    )
    
    parser.add_argument(
        '--timeframe', 
        type=str, 
        default="1D",
        choices=["1Min", "5Min", "15Min", "1H", "1D"],
        help='Timeframe for bars'
    )
    
    parser.add_argument(
        '--days', 
        type=int, 
        default=365,
        help='Number of days of historical data to fetch'
    )
    
    parser.add_argument(
        '--force-refresh', 
        action='store_true',
        help='Force refresh data from API (ignore cache)'
    )
    
    args = parser.parse_args()
    
    # Parse symbols
    symbols = [symbol.strip().upper() for symbol in args.symbols.split(",")]
    
    logger.info(f"Fetching data for symbols: {symbols}")
    logger.info(f"Timeframe: {args.timeframe}")
    logger.info(f"Days: {args.days}")
    logger.info(f"Force refresh: {args.force_refresh}")
    
    # Initialize data collector
    data_collector = StockDataCollector()
    
    # Calculate date range
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=args.days)
    
    # Fetch data
    try:
        data = data_collector.get_historical_data(
            symbols=symbols,
            timeframe=args.timeframe,
            start_date=start_date,
            end_date=end_date,
            use_cache=True,
            force_refresh=args.force_refresh
        )
        
        # Print summary
        for symbol, df in data.items():
            logger.info(f"{symbol}: {len(df)} bars from {df.index.min()} to {df.index.max()}")
        
        logger.info("Data collection complete")
        
    except Exception as e:
        logger.error(f"Error collecting data: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
