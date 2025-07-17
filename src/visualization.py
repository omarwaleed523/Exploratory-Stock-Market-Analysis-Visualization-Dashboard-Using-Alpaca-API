"""
Visualization module for creating interactive stock market visualizations.
"""
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime, timedelta, timezone
import pytz

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StockVisualizer:
    """Class for creating visualizations of stock data."""
    
    @staticmethod
    def create_price_chart(
        df: pd.DataFrame, 
        title: str = 'Stock Price',
        include_volume: bool = True,
        ma_periods: List[int] = [50, 200]
    ) -> go.Figure:
        """
        Create an interactive price chart with moving averages and volume.
        
        Args:
            df: DataFrame with stock data
            title: Chart title
            include_volume: Whether to include volume subplot
            ma_periods: Moving average periods to include
            
        Returns:
            Plotly figure
        """
        # Create subplot structure
        if include_volume:
            fig = make_subplots(
                rows=2, 
                cols=1, 
                shared_xaxes=True,
                vertical_spacing=0.1,
                row_heights=[0.8, 0.2]
            )
        else:
            fig = go.Figure()
        
        # Add price candlestick
        candlestick = go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price'
        )
        
        if include_volume:
            fig.add_trace(candlestick, row=1, col=1)
        else:
            fig.add_trace(candlestick)
        
        # Add moving averages
        for period in ma_periods:
            ma_col = f'ma_{period}'
            
            # Calculate MA if not in DataFrame
            if ma_col not in df.columns:
                df[ma_col] = df['close'].rolling(window=period).mean()
            
            ma_trace = go.Scatter(
                x=df.index,
                y=df[ma_col],
                mode='lines',
                name=f'{period}-day MA',
                line=dict(width=1.5)
            )
            
            if include_volume:
                fig.add_trace(ma_trace, row=1, col=1)
            else:
                fig.add_trace(ma_trace)
        
        # Add volume subplot
        if include_volume:
            volume_colors = np.where(df['close'] >= df['open'], 'green', 'red')
            
            volume_trace = go.Bar(
                x=df.index,
                y=df['volume'],
                name='Volume',
                marker=dict(color=volume_colors, opacity=0.5)
            )
            
            fig.add_trace(volume_trace, row=2, col=1)
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Price',
            xaxis_rangeslider_visible=False,
            height=600,
            template='plotly_white',
            legend=dict(orientation='h', y=1.02)
        )
        
        if include_volume:
            fig.update_yaxes(title_text='Volume', row=2, col=1)
        
        return fig
    
    @staticmethod
    def create_returns_heatmap(data: Dict[str, pd.DataFrame]) -> go.Figure:
        """
        Create a heatmap of correlations between stock returns.
        
        Args:
            data: Dictionary of DataFrames with stock data
            
        Returns:
            Plotly figure
        """
        # Extract daily returns for each stock
        returns_dict = {}
        
        for symbol, df in data.items():
            if 'daily_return' not in df.columns:
                df = df.copy()
                df['daily_return'] = df['close'].pct_change()
            
            returns_dict[symbol] = df['daily_return']
        
        # Create correlation matrix
        corr_df = pd.DataFrame(returns_dict).corr()
        
        # Create heatmap
        fig = px.imshow(
            corr_df,
            x=corr_df.columns,
            y=corr_df.columns,
            color_continuous_scale='RdBu_r',
            zmin=-1, 
            zmax=1,
            title='Correlation of Stock Returns'
        )
        
        fig.update_layout(
            height=600,
            width=800,
            template='plotly_white'
        )
        
        return fig
    
    @staticmethod
    def create_volatility_chart(
        data: Dict[str, pd.DataFrame],
        window: int = 20
    ) -> go.Figure:
        """
        Create a chart comparing volatility across multiple stocks.
        
        Args:
            data: Dictionary of DataFrames with stock data
            window: Window size for volatility calculation
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        for symbol, df in data.items():
            vol_col = f'volatility_{window}d'
            
            # Calculate volatility if not in DataFrame
            if vol_col not in df.columns:
                if 'daily_return' not in df.columns:
                    df = df.copy()
                    df['daily_return'] = df['close'].pct_change()
                
                df[vol_col] = df['daily_return'].rolling(window=window).std() * np.sqrt(252)
            
            # Add volatility trace
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[vol_col],
                    mode='lines',
                    name=symbol
                )
            )
        
        # Update layout
        fig.update_layout(
            title=f'{window}-day Rolling Volatility',
            xaxis_title='Date',
            yaxis_title='Annualized Volatility',
            height=500,
            template='plotly_white',
            legend=dict(orientation='h', y=1.02)
        )
        
        return fig
    
    @staticmethod
    def create_technical_dashboard(df: pd.DataFrame, symbol: str) -> go.Figure:
        """
        Create a comprehensive technical analysis dashboard.
        
        Args:
            df: DataFrame with stock data and technical indicators
            symbol: Stock symbol
            
        Returns:
            Plotly figure
        """
        # Create subplot structure
        fig = make_subplots(
            rows=4, 
            cols=1, 
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.4, 0.2, 0.2, 0.2],
            subplot_titles=(
                f'{symbol} Price and Bollinger Bands',
                'Volume',
                'RSI (14)',
                'MACD'
            )
        )
        
        # 1. Price with Bollinger Bands
        # Add price candlestick
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        # Add Bollinger Bands if available
        if 'bb_ma_20' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['bb_ma_20'],
                    line=dict(color='blue', width=0.8),
                    name='MA (20)'
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['bb_upper_20'],
                    line=dict(color='gray', width=0.5),
                    name='Upper BB'
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['bb_lower_20'],
                    line=dict(color='gray', width=0.5),
                    name='Lower BB',
                    fill='tonexty'
                ),
                row=1, col=1
            )
        
        # 2. Volume
        colors = np.where(df['close'] >= df['open'], 'green', 'red')
        
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['volume'],
                marker_color=colors,
                name='Volume',
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # 3. RSI
        if 'rsi_14' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['rsi_14'],
                    line=dict(color='purple', width=1),
                    name='RSI (14)'
                ),
                row=3, col=1
            )
            
            # Add RSI zones
            fig.add_shape(
                type="rect",
                xref="x3",
                yref="y3",
                x0=df.index[0],
                y0=70,
                x1=df.index[-1],
                y1=100,
                fillcolor="red",
                opacity=0.1,
                layer="below",
                line_width=0,
            )
            
            fig.add_shape(
                type="rect",
                xref="x3",
                yref="y3",
                x0=df.index[0],
                y0=30,
                x1=df.index[-1],
                y1=0,
                fillcolor="green",
                opacity=0.1,
                layer="below",
                line_width=0,
            )
            
            fig.add_shape(
                type="line",
                xref="x3",
                yref="y3",
                x0=df.index[0],
                y0=70,
                x1=df.index[-1],
                y1=70,
                line=dict(color="red", width=1, dash="dash"),
            )
            
            fig.add_shape(
                type="line",
                xref="x3",
                yref="y3",
                x0=df.index[0],
                y0=30,
                x1=df.index[-1],
                y1=30,
                line=dict(color="green", width=1, dash="dash"),
            )
        
        # 4. MACD
        if all(col in df.columns for col in ['macd', 'macd_signal']):
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['macd'],
                    line=dict(color='blue', width=1.5),
                    name='MACD'
                ),
                row=4, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['macd_signal'],
                    line=dict(color='red', width=1),
                    name='Signal'
                ),
                row=4, col=1
            )
            
            # Add MACD histogram
            colors = np.where(df['macd'] >= df['macd_signal'], 'green', 'red')
            
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['macd_histogram'],
                    marker_color=colors,
                    name='Histogram',
                    opacity=0.7
                ),
                row=4, col=1
            )
        
        # Update layout
        fig.update_layout(
            title=f'{symbol} Technical Analysis Dashboard',
            height=900,
            width=1000,
            template='plotly_white',
            showlegend=True,
            legend=dict(orientation='h', y=1.02),
            xaxis_rangeslider_visible=False,
        )
        
        # Update y-axis titles
        fig.update_yaxes(title_text='Price', row=1, col=1)
        fig.update_yaxes(title_text='Volume', row=2, col=1)
        
        if 'rsi_14' in df.columns:
            fig.update_yaxes(title_text='RSI', row=3, col=1)
            fig.update_yaxes(range=[0, 100], row=3, col=1)
        
        if 'macd' in df.columns:
            fig.update_yaxes(title_text='MACD', row=4, col=1)
        
        return fig
    
    @staticmethod
    def create_performance_comparison(
        data: Dict[str, pd.DataFrame],
        start_date: Optional[datetime] = None
    ) -> go.Figure:
        """
        Create a performance comparison chart across multiple stocks.
        
        Args:
            data: Dictionary of DataFrames with stock data
            start_date: Start date for comparison (defaults to earliest common date)
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Find common date range if start_date not specified
        if start_date is None:
            start_dates = [df.index.min() for df in data.values()]
            start_date = max(start_dates)
        elif start_date.tzinfo is None:
            # Convert naive datetime to timezone-aware
            start_date = start_date.replace(tzinfo=timezone.utc)
        
        for symbol, df in data.items():
            # Filter data from start date
            df_filtered = df[df.index >= start_date].copy()
            
            if len(df_filtered) == 0:
                logger.warning(f"No data available for {symbol} after {start_date}")
                continue
            
            # Calculate normalized price (start = 100)
            first_price = df_filtered['close'].iloc[0]
            df_filtered['normalized'] = df_filtered['close'] / first_price * 100
            
            # Add trace
            fig.add_trace(
                go.Scatter(
                    x=df_filtered.index,
                    y=df_filtered['normalized'],
                    mode='lines',
                    name=symbol
                )
            )
        
        # Update layout
        fig.update_layout(
            title='Relative Performance Comparison (Base = 100)',
            xaxis_title='Date',
            yaxis_title='Normalized Price',
            height=500,
            template='plotly_white',
            legend=dict(orientation='h', y=1.02)
        )
        
        return fig
    
    @staticmethod
    def create_seasonal_analysis(df: pd.DataFrame, symbol: str) -> go.Figure:
        """
        Create a seasonal analysis visualization.
        
        Args:
            df: DataFrame with stock data
            symbol: Stock symbol
            
        Returns:
            Plotly figure
        """
        # Extract year and month
        df_copy = df.copy()
        df_copy['year'] = df_copy.index.year
        df_copy['month'] = df_copy.index.month
        
        # Calculate monthly returns
        monthly_returns = df_copy.groupby(['year', 'month'])['close'].apply(
            lambda x: (x.iloc[-1] / x.iloc[0] - 1) * 100
        ).reset_index()
        
        # Calculate average return by month
        avg_monthly_returns = monthly_returns.groupby('month')['close'].mean().reset_index()
        
        # Create heatmap data
        pivot_table = pd.pivot_table(
            monthly_returns, 
            values='close', 
            index='year', 
            columns='month', 
            fill_value=0
        )
        
        # Replace month numbers with names
        month_names = {
            1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
            7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
        }
        pivot_table.columns = [month_names[m] for m in pivot_table.columns]
        
        # Create subplots
        fig = make_subplots(
            rows=2, 
            cols=1,
            row_heights=[0.7, 0.3],
            subplot_titles=(
                f'{symbol} Monthly Returns Heatmap (%)',
                'Average Monthly Returns (%)'
            )
        )
        
        # Add heatmap
        heatmap = go.Heatmap(
            z=pivot_table.values,
            x=pivot_table.columns,
            y=pivot_table.index,
            colorscale='RdBu_r',
            colorbar=dict(title='Return %'),
            zmid=0
        )
        
        fig.add_trace(heatmap, row=1, col=1)
        
        # Add average monthly returns bar chart
        avg_monthly_returns['month_name'] = avg_monthly_returns['month'].map(month_names)
        
        bar = go.Bar(
            x=avg_monthly_returns['month_name'],
            y=avg_monthly_returns['close'],
            marker=dict(
                color=avg_monthly_returns['close'],
                colorscale='RdBu_r',
                cmid=0
            )
        )
        
        fig.add_trace(bar, row=2, col=1)
        
        # Update layout
        fig.update_layout(
            height=800,
            width=1000,
            template='plotly_white',
            title=f'{symbol} Seasonal Analysis',
            xaxis2=dict(title='Month'),
            yaxis2=dict(title='Avg Return (%)'),
        )
        
        return fig
