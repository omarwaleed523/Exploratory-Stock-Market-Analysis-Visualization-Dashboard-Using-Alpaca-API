"""
Streamlit application for interactive stock market analysis dashboard.
"""
import os
import sys
import logging
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime, timedelta, timezone
import pytz
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to the path to make imports work properly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import project modules - try both relative and absolute imports to ensure compatibility
try:
    # For deployment environment
    from api_client import RateLimitedAlpacaClient
    from data_collection import StockDataCollector
    from analysis import StockAnalyzer
    from visualization import StockVisualizer
    logger.info("Using relative imports (deployment mode)")
except ImportError:
    # For local development
    from src.api_client import RateLimitedAlpacaClient
    from src.data_collection import StockDataCollector
    from src.analysis import StockAnalyzer
    from src.visualization import StockVisualizer
    logger.info("Using absolute imports (local development mode)")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize services
try:
    # Initialize API client
    api_client = RateLimitedAlpacaClient()
    data_collector = StockDataCollector()
    api_initialized = True
except Exception as e:
    logger.error(f"Failed to initialize API client: {e}")
    api_initialized = False

# App configuration
st.set_page_config(
    page_title="Stock Market EDA Dashboard",
    page_icon="📈",
    layout="wide"
)

# App title and description
st.title("📊 Stock Market EDA Dashboard")

# Check if API client initialized successfully
if not api_initialized:
    st.error("⚠️ Failed to initialize Alpaca API client. Please check your credentials.")
    
    # Provide troubleshooting information
    st.markdown("""
    ### Troubleshooting
    
    The application could not connect to the Alpaca API. This could be due to:
    
    1. **Missing credentials** - Make sure you have set up your API credentials either:
       - In the `.streamlit/secrets.toml` file (for local development)
       - In the Streamlit Cloud dashboard (for deployed app)
       
    2. **Incorrect credentials** - Verify that your API key and secret are correct
    
    3. **Network issues** - Check your internet connection
    
    #### How to set up credentials
    
    For local development:
    - Create a `.streamlit/secrets.toml` file with:
    ```toml
    [alpaca]
    api_key_id = "YOUR_API_KEY"
    api_secret_key = "YOUR_API_SECRET"
    api_base_url = "https://paper-api.alpaca.markets"
    data_url = "https://data.alpaca.markets"
    ```
    
    For Streamlit Cloud deployment:
    - Go to your app dashboard
    - Click on "Secrets"
    - Add the same configuration as above
    """)
    
    # Show a button to run diagnostic
    if st.button("Run Diagnostic Test"):
        st.info("Running diagnostic test to check credential availability...")
        try:
            import streamlit as st_check
            
            if hasattr(st_check, 'secrets'):
                st.success("✅ Streamlit secrets are available")
                
                if 'alpaca' in st_check.secrets:
                    st.success("✅ 'alpaca' section found in secrets")
                    
                    # Check each required key
                    alpaca_keys = ['api_key_id', 'api_secret_key', 'api_base_url', 'data_url']
                    missing_keys = []
                    
                    for key in alpaca_keys:
                        if key in st_check.secrets.alpaca:
                            st.success(f"✅ '{key}' is set in secrets")
                        else:
                            st.error(f"❌ '{key}' is missing in secrets")
                            missing_keys.append(key)
                else:
                    st.error("❌ 'alpaca' section not found in secrets")
            else:
                st.error("❌ Streamlit secrets are not available")
                
            # Check environment variables
            st.subheader("Environment Variables")
            env_vars = {
                'ALPACA_API_KEY_ID': os.getenv('ALPACA_API_KEY_ID'),
                'ALPACA_API_SECRET_KEY': os.getenv('ALPACA_API_SECRET_KEY'),
                'ALPACA_API_BASE_URL': os.getenv('ALPACA_API_BASE_URL'),
                'ALPACA_DATA_URL': os.getenv('ALPACA_DATA_URL')
            }
            
            for key, value in env_vars.items():
                if value:
                    st.success(f"✅ '{key}' is set in environment variables")
                else:
                    st.warning(f"⚠️ '{key}' is not set in environment variables")
                    
        except Exception as diag_error:
            st.error(f"Diagnostic error: {str(diag_error)}")
    
    # Stop rendering the rest of the app
    st.stop()

st.markdown("""
This dashboard provides exploratory data analysis on stock market data using the Alpaca API.
Select stocks, timeframes, and analysis tools to visualize market trends and patterns.
""")

# Sidebar configuration
st.sidebar.header("Dashboard Controls")

# Date range selection
st.sidebar.subheader("Select Date Range")
end_date = datetime.now(timezone.utc)
start_date = end_date - timedelta(days=365)  # Default to 1 year

start_date_input = st.sidebar.date_input(
    "Start Date",
    value=start_date.date(),
    max_value=end_date.date() - timedelta(days=1)
)

end_date_input = st.sidebar.date_input(
    "End Date",
    value=end_date.date(),
    min_value=start_date_input,
    max_value=end_date.date()
)

# Convert to datetime with UTC timezone
start_datetime = datetime.combine(start_date_input, datetime.min.time(), tzinfo=timezone.utc)
end_datetime = datetime.combine(end_date_input, datetime.min.time(), tzinfo=timezone.utc)

# Stock selection
st.sidebar.subheader("Select Stocks")

# Option to enter custom symbols
custom_symbols = st.sidebar.text_input(
    "Enter stock symbols (comma-separated)",
    value="AAPL,MSFT,GOOGL,AMZN,META"
)

symbols = [symbol.strip().upper() for symbol in custom_symbols.split(",") if symbol.strip()]

# Check if valid symbols were entered
if not symbols:
    st.warning("Please enter at least one valid stock symbol.")
    st.stop()

# Timeframe selection
st.sidebar.subheader("Select Timeframe")
timeframe = st.sidebar.selectbox(
    "Bar Timeframe",
    options=["1D", "1H", "15Min"],
    index=0
)

# Analysis options
st.sidebar.subheader("Analysis Options")
show_moving_averages = st.sidebar.checkbox("Show Moving Averages", value=True)
show_bollinger_bands = st.sidebar.checkbox("Show Bollinger Bands", value=False)
show_rsi = st.sidebar.checkbox("Show RSI", value=True)
show_macd = st.sidebar.checkbox("Show MACD", value=True)

# Add a fetch data button
fetch_data = st.sidebar.button("Fetch Data")

# Main content
if fetch_data:
    # Show loading message
    with st.spinner("Fetching stock data..."):
        try:
            # Fetch data for all symbols
            data = data_collector.get_historical_data(
                symbols=symbols,
                timeframe=timeframe,
                start_date=start_datetime,
                end_date=end_datetime,
                use_cache=True
            )
            
            # Check if we got data for all symbols
            missing_symbols = [symbol for symbol in symbols if symbol not in data]
            if missing_symbols:
                st.warning(f"No data available for the following symbols: {', '.join(missing_symbols)}")
                
            # Filter symbols with data
            symbols_with_data = [symbol for symbol in symbols if symbol in data]
            
            if not symbols_with_data:
                st.error("No data available for any of the selected symbols.")
                st.stop()
            
            # Perform analysis for each symbol
            analyzed_data = {}
            for symbol in symbols_with_data:
                df = data[symbol]
                
                # Basic analysis for all symbols
                df = StockAnalyzer.calculate_returns(df)
                df = StockAnalyzer.calculate_volatility(df)
                
                # Add additional indicators based on user selection
                if show_moving_averages:
                    df = StockAnalyzer.calculate_moving_averages(df)
                
                if show_bollinger_bands:
                    df = StockAnalyzer.calculate_bollinger_bands(df)
                
                if show_rsi:
                    df = StockAnalyzer.calculate_rsi(df)
                
                if show_macd:
                    df = StockAnalyzer.calculate_macd(df)
                
                analyzed_data[symbol] = df
            
            # Create tabs for different views
            tab_overview, tab_individual, tab_comparison, tab_correlation, tab_seasonal = st.tabs([
                "Overview", 
                "Individual Analysis", 
                "Performance Comparison",
                "Correlation Analysis",
                "Seasonal Analysis"
            ])
            
            # 1. Overview Tab
            with tab_overview:
                st.header("Market Overview")
                
                # Summary metrics
                st.subheader("Summary Metrics")
                
                metrics_cols = st.columns(len(symbols_with_data))
                
                for i, symbol in enumerate(symbols_with_data):
                    df = analyzed_data[symbol]
                    last_price = df['close'].iloc[-1]
                    daily_change = df['daily_return'].iloc[-1] * 100 if 'daily_return' in df.columns else 0
                    
                    metrics_cols[i].metric(
                        label=symbol,
                        value=f"${last_price:.2f}",
                        delta=f"{daily_change:.2f}%"
                    )
                
                # Volatility comparison
                st.subheader("Volatility Comparison")
                volatility_fig = StockVisualizer.create_volatility_chart(analyzed_data)
                st.plotly_chart(volatility_fig, use_container_width=True)
                
                # Performance comparison
                st.subheader("Relative Performance")
                perf_fig = StockVisualizer.create_performance_comparison(analyzed_data)
                st.plotly_chart(perf_fig, use_container_width=True)
            
            # 2. Individual Analysis Tab
            with tab_individual:
                st.header("Individual Stock Analysis")
                
                # Symbol selection
                selected_symbol = st.selectbox(
                    "Select a stock for detailed analysis",
                    options=symbols_with_data
                )
                
                if selected_symbol:
                    # Get data for selected symbol
                    selected_data = analyzed_data[selected_symbol]
                    
                    # Create tabs for different chart types
                    chart_tabs = st.tabs(["Price Chart", "Technical Dashboard", "Statistics"])
                    
                    # Price chart
                    with chart_tabs[0]:
                        price_fig = StockVisualizer.create_price_chart(
                            selected_data,
                            title=f"{selected_symbol} Price Chart",
                            include_volume=True
                        )
                        st.plotly_chart(price_fig, use_container_width=True)
                    
                    # Technical dashboard
                    with chart_tabs[1]:
                        if show_bollinger_bands or show_rsi or show_macd:
                            tech_fig = StockVisualizer.create_technical_dashboard(
                                selected_data,
                                symbol=selected_symbol
                            )
                            st.plotly_chart(tech_fig, use_container_width=True)
                        else:
                            st.info("Enable indicators in the sidebar to view the technical dashboard.")
                    
                    # Statistics
                    with chart_tabs[2]:
                        st.subheader(f"{selected_symbol} Statistics")
                        
                        # Calculate statistics
                        stats = {
                            "Current Price": selected_data['close'].iloc[-1],
                            "Daily Change %": selected_data['daily_return'].iloc[-1] * 100,
                            "52-Week High": selected_data['high'].max(),
                            "52-Week Low": selected_data['low'].min(),
                            "Avg. Daily Volume": selected_data['volume'].mean(),
                            "20-Day Volatility": selected_data['volatility_20d'].iloc[-1] * 100 if 'volatility_20d' in selected_data.columns else None,
                            "Sharpe Ratio": selected_data['sharpe_ratio'].iloc[-1] if 'sharpe_ratio' in selected_data.columns else None,
                        }
                        
                        # Display in columns
                        stat_cols = st.columns(3)
                        
                        for i, (label, value) in enumerate(stats.items()):
                            if value is not None:
                                if "%" in label:
                                    stat_cols[i % 3].metric(label, f"{value:.2f}%")
                                elif "Price" in label or "High" in label or "Low" in label:
                                    stat_cols[i % 3].metric(label, f"${value:.2f}")
                                elif "Volume" in label:
                                    stat_cols[i % 3].metric(label, f"{value:,.0f}")
                                else:
                                    stat_cols[i % 3].metric(label, f"{value:.4f}")
                        
                        # Returns table
                        st.subheader("Returns Analysis")
                        
                        if 'return_5d' in selected_data.columns:
                            returns_df = pd.DataFrame({
                                "1-Day Return": [selected_data['daily_return'].iloc[-1] * 100],
                                "5-Day Return": [selected_data['return_5d'].iloc[-1] * 100],
                                "20-Day Return": [selected_data['return_20d'].iloc[-1] * 100],
                                "60-Day Return": [selected_data['return_60d'].iloc[-1] * 100]
                            }).T.reset_index()
                            
                            returns_df.columns = ["Period", "Return (%)"]
                            st.dataframe(returns_df, use_container_width=True)
            
            # 3. Performance Comparison Tab
            with tab_comparison:
                st.header("Performance Comparison")
                
                # Time period selection
                period_options = {
                    "1 Month": 30,
                    "3 Months": 90,
                    "6 Months": 180,
                    "1 Year": 365,
                    "YTD": (datetime.now() - datetime(datetime.now().year, 1, 1)).days,
                    "All": (end_datetime - start_datetime).days
                }
                
                selected_period = st.selectbox(
                    "Select time period",
                    options=list(period_options.keys())
                )
                
                days = period_options[selected_period]
                period_start = end_datetime - timedelta(days=days)
                
                if selected_period == "All":
                    period_start = start_datetime
                
                # Ensure period_start is timezone-aware
                if period_start.tzinfo is None:
                    period_start = period_start.replace(tzinfo=timezone.utc)
                
                # Generate performance comparison
                perf_fig = StockVisualizer.create_performance_comparison(
                    analyzed_data,
                    start_date=period_start
                )
                st.plotly_chart(perf_fig, use_container_width=True)
                
                # Returns table
                st.subheader("Returns Comparison")
                
                returns_data = {}
                for symbol, df in analyzed_data.items():
                    # Ensure period_start is timezone-aware
                    if period_start.tzinfo is None:
                        period_start = period_start.replace(tzinfo=timezone.utc)
                        
                    filtered_df = df[df.index >= period_start]
                    if len(filtered_df) > 0:
                        first_price = filtered_df['close'].iloc[0]
                        last_price = filtered_df['close'].iloc[-1]
                        total_return = (last_price / first_price - 1) * 100
                        returns_data[symbol] = total_return
                
                # Convert to DataFrame and sort
                returns_df = pd.DataFrame({
                    "Symbol": list(returns_data.keys()),
                    f"Return (%) - {selected_period}": list(returns_data.values())
                })
                
                returns_df = returns_df.sort_values(by=f"Return (%) - {selected_period}", ascending=False)
                
                # Format returns
                returns_df[f"Return (%) - {selected_period}"] = returns_df[f"Return (%) - {selected_period}"].apply(
                    lambda x: f"{x:.2f}%"
                )
                
                st.dataframe(returns_df, use_container_width=True)
            
            # 4. Correlation Tab
            with tab_correlation:
                st.header("Correlation Analysis")
                
                # Create correlation heatmap
                corr_fig = StockVisualizer.create_returns_heatmap(analyzed_data)
                st.plotly_chart(corr_fig, use_container_width=True)
                
                # Explanation
                st.markdown("""
                **Understanding the correlation heatmap:**
                - Values close to 1 (dark blue) indicate strong positive correlation (stocks move together)
                - Values close to -1 (dark red) indicate strong negative correlation (stocks move in opposite directions)
                - Values close to 0 (white) indicate little to no correlation
                
                Correlations are calculated based on daily returns.
                """)
            
            # 5. Seasonal Analysis Tab
            with tab_seasonal:
                st.header("Seasonal Analysis")
                
                # Symbol selection
                selected_symbol_seasonal = st.selectbox(
                    "Select a stock for seasonal analysis",
                    options=symbols_with_data,
                    key="seasonal_select"
                )
                
                if selected_symbol_seasonal:
                    # Get data for selected symbol
                    selected_data_seasonal = analyzed_data[selected_symbol_seasonal]
                    
                    # Create seasonal visualization
                    seasonal_fig = StockVisualizer.create_seasonal_analysis(
                        selected_data_seasonal,
                        symbol=selected_symbol_seasonal
                    )
                    st.plotly_chart(seasonal_fig, use_container_width=True)
                    
                    # Explanation
                    st.markdown("""
                    **Interpreting the seasonal analysis:**
                    - The heatmap shows monthly returns for each year
                    - Blue indicates positive returns, red indicates negative returns
                    - The bar chart shows the average return for each month across all years
                    - This can help identify seasonal patterns in stock performance
                    """)
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            logger.exception("Error in dashboard")
else:
    # Display placeholder when no data is loaded
    st.info("Click 'Fetch Data' in the sidebar to load stock data and generate visualizations.")
    
    # Display documentation
    st.markdown("""
    ## Dashboard Features
    
    ### Overview Tab
    Provides a high-level summary of all selected stocks, including:
    - Current prices and daily changes
    - Volatility comparison
    - Relative performance chart
    
    ### Individual Analysis Tab
    Detailed analysis of a single stock:
    - Interactive price chart with volume
    - Technical indicators (Moving Averages, Bollinger Bands, RSI, MACD)
    - Key statistics and returns
    
    ### Performance Comparison Tab
    Compare performance across different stocks:
    - Normalized price chart
    - Returns comparison table
    
    ### Correlation Analysis Tab
    Analyze relationships between stocks:
    - Correlation heatmap of daily returns
    
    ### Seasonal Analysis Tab
    Identify seasonal patterns in stock performance:
    - Monthly returns heatmap
    - Average monthly returns
    
    ---
    
    ## Data Limitations
    
    This dashboard uses Alpaca's API with the following limitations:
    - Limited to 200 API requests per minute
    - Free tier data is sourced only from IEX (Investors Exchange)
    - IEX accounts for ~2% of total market volume
    - Data may be relatively sparse with fewer quotes
    
    To mitigate these limitations, the dashboard implements:
    - API rate limiting
    - Data caching
    - Efficient batch requests
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Powered by Alpaca API")
st.sidebar.markdown("© 2025 Stock Market EDA Project")

if __name__ == "__main__":
    # This allows the app to be run with streamlit run app.py
    pass
