# Alpaca Stock Market EDA & Visualization Dashboard

## Overview
This project performs exploratory data analysis on stock market data using the Alpaca API, with interactive visualizations and technical analysis. It's designed to work within Alpaca's free tier API limitations while providing robust analysis capabilities.

## Features
- Historical data retrieval with rate-limiting controls (respects 200 API calls/min limit)
- Comprehensive stock market EDA (trends, volatility, correlations)
- Technical indicators (RSI, MACD, Bollinger Bands, Moving Averages)
- Risk metrics and performance comparison
- Interactive visualizations using Plotly
- Timezone-aware datetime handling

## API Limitations Handled
- Rate limiting (200 REST API requests per minute)
- Exclusive use of IEX data source (free tier requirement)
- Proper error handling for API limitations
- Data caching to minimize API calls

## Setup and Installation
1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up your Alpaca API credentials in a `.env` file:
   ```
   ALPACA_API_KEY_ID=your_key_here
   ALPACA_API_SECRET_KEY=your_secret_here
   ALPACA_API_BASE_URL=https://paper-api.alpaca.markets
   ALPACA_DATA_URL=https://data.alpaca.markets
   ```
4. Run the application using one of the methods below

## Deployment
When deploying the application to a hosting service:

1. **DO NOT** commit your `.env` file to version control
2. Set the following environment variables in your deployment platform's settings:
   - `ALPACA_API_KEY_ID`
   - `ALPACA_API_SECRET_KEY`
   - `ALPACA_API_BASE_URL`
   - `ALPACA_DATA_URL`

Different platforms handle this differently:
- **Heroku**: Use "Config Vars" in the settings
- **Streamlit Cloud**: Use Secrets Management
- **AWS/Azure**: Use environment variables in your service configuration
- **Docker**: Pass environment variables in your docker-compose file or run command

Without these environment variables, you'll get the error: `API credentials not found`.

## Running the Application
- **Streamlit Dashboard**: Run `streamlit run src/app.py`
- **Data Collection**: Run `python src/data_collection_cli.py`
- **Python API**: Import modules from `src/` in your own scripts

## Project Structure
- `src/`: Python source code
  - `api_client.py`: Alpaca API client with rate limiting
  - `data_collection.py`: Data ingestion with caching
  - `analysis.py`: Technical indicators and analysis functions
  - `visualization.py`: Visualization components
  - `app.py`: Streamlit dashboard application
  - `data_collection_cli.py`: Command-line interface for data collection
- `.env`: API credentials (not tracked in git)

## Technical Features
- **Rate-Limited API Client**: Custom client that respects Alpaca's rate limits
- **Timezone-Aware Analysis**: Prevents datetime comparison errors
- **Technical Indicators**: RSI, MACD, Bollinger Bands, Moving Averages
- **Risk Metrics**: Volatility, Sharpe Ratio, Value at Risk
- **Interactive Visualizations**: Price charts, correlation heatmaps, performance comparisons
- **Network Analysis**: Correlation network visualization
- **Seasonal Analysis**: Pattern detection in longer timeframes

## Requirements
- Python 3.10+
- Core: pandas, numpy, alpaca-trade-api, python-dotenv
- Visualization: matplotlib, seaborn, plotly
- Analysis: scikit-learn, statsmodels, networkx
- UI: streamlit

## License
MIT
