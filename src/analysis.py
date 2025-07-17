"""
Analysis module with functions for stock market EDA and indicators.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime, timedelta, timezone
import pytz
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StockAnalyzer:
    """Class for analyzing stock data with various technical indicators and statistics."""
    
    @staticmethod
    def calculate_returns(df: pd.DataFrame, periods: List[int] = [1, 5, 20, 60]) -> pd.DataFrame:
        """
        Calculate returns over different periods.
        
        Args:
            df: DataFrame with stock data
            periods: List of periods for return calculation
            
        Returns:
            DataFrame with added return columns
        """
        result = df.copy()
        
        # Calculate daily returns
        result['daily_return'] = result['close'].pct_change()
        
        # Calculate returns for specified periods
        for period in periods:
            result[f'return_{period}d'] = result['close'].pct_change(period)
        
        return result
    
    @staticmethod
    def calculate_volatility(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """
        Calculate rolling volatility.
        
        Args:
            df: DataFrame with stock data
            window: Window size for volatility calculation
            
        Returns:
            DataFrame with added volatility column
        """
        result = df.copy()
        
        # Calculate daily returns if not already present
        if 'daily_return' not in result.columns:
            result['daily_return'] = result['close'].pct_change()
        
        # Calculate rolling volatility (annualized)
        result[f'volatility_{window}d'] = result['daily_return'].rolling(window=window).std() * np.sqrt(252)
        
        return result
    
    @staticmethod
    def calculate_moving_averages(
        df: pd.DataFrame, 
        windows: List[int] = [10, 20, 50, 200]
    ) -> pd.DataFrame:
        """
        Calculate moving averages for different windows.
        
        Args:
            df: DataFrame with stock data
            windows: List of window sizes
            
        Returns:
            DataFrame with added moving average columns
        """
        result = df.copy()
        
        for window in windows:
            result[f'ma_{window}'] = result['close'].rolling(window=window).mean()
            
            # Calculate percent difference from price to MA
            result[f'ma_{window}_diff'] = (result['close'] - result[f'ma_{window}']) / result[f'ma_{window}'] * 100
        
        return result
    
    @staticmethod
    def calculate_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            df: DataFrame with stock data
            window: Window size for RSI calculation
            
        Returns:
            DataFrame with added RSI column
        """
        result = df.copy()
        
        # Calculate daily returns if not already present
        if 'daily_return' not in result.columns:
            result['daily_return'] = result['close'].pct_change()
        
        # Calculate up and down moves
        result['up_move'] = result['daily_return'].apply(lambda x: x if x > 0 else 0)
        result['down_move'] = result['daily_return'].apply(lambda x: abs(x) if x < 0 else 0)
        
        # Calculate rolling averages of up and down moves
        result['avg_up'] = result['up_move'].rolling(window=window).mean()
        result['avg_down'] = result['down_move'].rolling(window=window).mean()
        
        # Calculate relative strength and RSI
        result['rs'] = result['avg_up'] / result['avg_down']
        result[f'rsi_{window}'] = 100 - (100 / (1 + result['rs']))
        
        # Drop intermediate columns
        result = result.drop(['up_move', 'down_move', 'avg_up', 'avg_down', 'rs'], axis=1)
        
        return result
    
    @staticmethod
    def calculate_bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
        """
        Calculate Bollinger Bands.
        
        Args:
            df: DataFrame with stock data
            window: Window size for moving average
            num_std: Number of standard deviations for bands
            
        Returns:
            DataFrame with added Bollinger Bands columns
        """
        result = df.copy()
        
        # Calculate rolling mean and standard deviation
        result[f'bb_ma_{window}'] = result['close'].rolling(window=window).mean()
        result[f'bb_std_{window}'] = result['close'].rolling(window=window).std()
        
        # Calculate upper and lower bands
        result[f'bb_upper_{window}'] = result[f'bb_ma_{window}'] + (result[f'bb_std_{window}'] * num_std)
        result[f'bb_lower_{window}'] = result[f'bb_ma_{window}'] - (result[f'bb_std_{window}'] * num_std)
        
        # Calculate bandwidth and %B
        result[f'bb_width_{window}'] = (result[f'bb_upper_{window}'] - result[f'bb_lower_{window}']) / result[f'bb_ma_{window}']
        result[f'bb_pct_b_{window}'] = (result['close'] - result[f'bb_lower_{window}']) / (result[f'bb_upper_{window}'] - result[f'bb_lower_{window}'])
        
        # Drop intermediate column
        result = result.drop([f'bb_std_{window}'], axis=1)
        
        return result
    
    @staticmethod
    def calculate_macd(df: pd.DataFrame, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> pd.DataFrame:
        """
        Calculate Moving Average Convergence Divergence (MACD).
        
        Args:
            df: DataFrame with stock data
            fast_period: Period for fast EMA
            slow_period: Period for slow EMA
            signal_period: Period for signal line
            
        Returns:
            DataFrame with added MACD columns
        """
        result = df.copy()
        
        # Calculate fast and slow EMAs
        result['ema_fast'] = result['close'].ewm(span=fast_period, adjust=False).mean()
        result['ema_slow'] = result['close'].ewm(span=slow_period, adjust=False).mean()
        
        # Calculate MACD and signal line
        result['macd'] = result['ema_fast'] - result['ema_slow']
        result['macd_signal'] = result['macd'].ewm(span=signal_period, adjust=False).mean()
        result['macd_histogram'] = result['macd'] - result['macd_signal']
        
        # Drop intermediate columns
        result = result.drop(['ema_fast', 'ema_slow'], axis=1)
        
        return result
    
    @staticmethod
    def calculate_correlation_matrix(
        data: Dict[str, pd.DataFrame], 
        column: str = 'daily_return',
        min_periods: int = 30
    ) -> pd.DataFrame:
        """
        Calculate correlation matrix between multiple stocks.
        
        Args:
            data: Dictionary of DataFrames with stock data
            column: Column to use for correlation calculation
            min_periods: Minimum number of periods required
            
        Returns:
            Correlation matrix DataFrame
        """
        # Create a dictionary to store the specified column for each symbol
        series_dict = {}
        
        for symbol, df in data.items():
            if column in df.columns:
                series_dict[symbol] = df[column]
        
        if not series_dict:
            logger.warning(f"No data found with column '{column}' for correlation calculation")
            return pd.DataFrame()
        
        # Create a DataFrame with all series and calculate correlation matrix
        combined_df = pd.DataFrame(series_dict)
        correlation_matrix = combined_df.corr(min_periods=min_periods)
        
        return correlation_matrix
    
    @staticmethod
    def calculate_beta(
        stock_df: pd.DataFrame, 
        market_df: pd.DataFrame,
        return_col: str = 'daily_return',
        window: int = 252
    ) -> pd.DataFrame:
        """
        Calculate rolling beta against a market index.
        
        Args:
            stock_df: DataFrame with stock data
            market_df: DataFrame with market index data
            return_col: Column with returns
            window: Window size for beta calculation
            
        Returns:
            DataFrame with added beta column
        """
        result = stock_df.copy()
        
        # Ensure both DataFrames have the return column
        if return_col not in result.columns:
            result[return_col] = result['close'].pct_change()
        
        if return_col not in market_df.columns:
            market_df = market_df.copy()
            market_df[return_col] = market_df['close'].pct_change()
        
        # Align the data
        aligned_data = pd.concat([result[return_col], market_df[return_col]], axis=1, join='inner')
        aligned_data.columns = ['stock_return', 'market_return']
        
        # Calculate rolling beta
        def rolling_beta(data):
            x = data['market_return'].values.reshape(-1, 1)
            y = data['stock_return'].values
            model = LinearRegression().fit(x, y)
            return model.coef_[0]
        
        result['beta'] = aligned_data.rolling(window=window).apply(rolling_beta, raw=False)
        
        return result
    
    @staticmethod
    def calculate_sharpe_ratio(
        df: pd.DataFrame, 
        return_col: str = 'daily_return',
        risk_free_rate: float = 0.01,
        window: int = 252
    ) -> pd.DataFrame:
        """
        Calculate rolling Sharpe ratio.
        
        Args:
            df: DataFrame with stock data
            return_col: Column with returns
            risk_free_rate: Annualized risk-free rate
            window: Window size for calculation
            
        Returns:
            DataFrame with added Sharpe ratio column
        """
        result = df.copy()
        
        # Ensure the DataFrame has the return column
        if return_col not in result.columns:
            result[return_col] = result['close'].pct_change()
        
        # Convert annual risk-free rate to daily
        daily_rf = (1 + risk_free_rate) ** (1/252) - 1
        
        # Calculate excess return
        result['excess_return'] = result[return_col] - daily_rf
        
        # Calculate rolling Sharpe ratio (annualized)
        result['sharpe_ratio'] = (
            result['excess_return'].rolling(window=window).mean() / 
            result['excess_return'].rolling(window=window).std()
        ) * np.sqrt(252)
        
        # Drop intermediate column
        result = result.drop(['excess_return'], axis=1)
        
        return result
    
    @staticmethod
    def test_stationarity(series: pd.Series) -> Dict[str, Any]:
        """
        Test for stationarity using Augmented Dickey-Fuller test.
        
        Args:
            series: Time series to test
            
        Returns:
            Dictionary with test results
        """
        result = {}
        
        # Drop NA values
        series = series.dropna()
        
        if len(series) < 20:
            logger.warning("Not enough data points for stationarity test")
            return {"error": "Insufficient data"}
        
        # Perform ADF test
        adf_test = adfuller(series)
        
        result = {
            'adf_statistic': adf_test[0],
            'p_value': adf_test[1],
            'critical_values': adf_test[4],
            'is_stationary': adf_test[1] < 0.05  # True if p-value < 0.05
        }
        
        return result
    
    @staticmethod
    def calculate_value_at_risk(
        df: pd.DataFrame, 
        return_col: str = 'daily_return',
        confidence_level: float = 0.95,
        window: int = 252
    ) -> pd.DataFrame:
        """
        Calculate rolling Value at Risk (VaR).
        
        Args:
            df: DataFrame with stock data
            return_col: Column with returns
            confidence_level: Confidence level for VaR
            window: Window size for calculation
            
        Returns:
            DataFrame with added VaR column
        """
        result = df.copy()
        
        # Ensure the DataFrame has the return column
        if return_col not in result.columns:
            result[return_col] = result['close'].pct_change()
        
        # Calculate rolling VaR
        alpha = 1 - confidence_level
        result[f'var_{int(confidence_level*100)}'] = result[return_col].rolling(window=window).quantile(alpha)
        
        return result
    
    @staticmethod
    def perform_complete_analysis(
        df: pd.DataFrame, 
        market_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Perform a complete analysis with all indicators.
        
        Args:
            df: DataFrame with stock data
            market_df: Optional DataFrame with market index data for beta calculation
            
        Returns:
            DataFrame with all analysis indicators
        """
        result = df.copy()
        
        # Calculate returns
        result = StockAnalyzer.calculate_returns(result)
        
        # Calculate volatility
        result = StockAnalyzer.calculate_volatility(result)
        
        # Calculate moving averages
        result = StockAnalyzer.calculate_moving_averages(result)
        
        # Calculate RSI
        result = StockAnalyzer.calculate_rsi(result)
        
        # Calculate Bollinger Bands
        result = StockAnalyzer.calculate_bollinger_bands(result)
        
        # Calculate MACD
        result = StockAnalyzer.calculate_macd(result)
        
        # Calculate Sharpe ratio
        result = StockAnalyzer.calculate_sharpe_ratio(result)
        
        # Calculate Value at Risk
        result = StockAnalyzer.calculate_value_at_risk(result)
        
        # Calculate beta if market data is provided
        if market_df is not None:
            result = StockAnalyzer.calculate_beta(result, market_df)
        
        return result
