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
3. Set up your Alpaca API credentials using one of these methods:
   
   **Option 1: Using .env file (for local development)**
   Create a `.env` file in the project root:
   ```
   ALPACA_API_KEY_ID=your_key_here
   ALPACA_API_SECRET_KEY=your_secret_here
   ALPACA_API_BASE_URL=https://paper-api.alpaca.markets
   ALPACA_DATA_URL=https://data.alpaca.markets
   ```
   
   **Option 2: Using Streamlit secrets (for Streamlit Cloud)**
   Create a `.streamlit/secrets.toml` file:
   ```toml
   [alpaca]
   api_key_id = "your_key_here"
   api_secret_key = "your_secret_here"
   api_base_url = "https://paper-api.alpaca.markets"
   data_url = "https://data.alpaca.markets"
   ```

4. Run the application using one of the methods below

## Deployment
The application supports deployment to Streamlit Cloud and other platforms.

### Streamlit Cloud Deployment
1. Fork this repository to your GitHub account
2. Create a new app in [Streamlit Cloud](https://share.streamlit.io/)
3. Connect it to your GitHub repository
4. Set the main file path to `src/app.py`
5. In the Streamlit Cloud dashboard, add your secrets:
   - Go to "Advanced Settings" > "Secrets"
   - Add the following configuration in TOML format:
   ```toml
   [alpaca]
   api_key_id = "your_key_here"
   api_secret_key = "your_secret_here"
   api_base_url = "https://paper-api.alpaca.markets"
   data_url = "https://data.alpaca.markets"
   ```
6. Deploy the app

### Other Deployment Options
When deploying to other platforms:

1. **DO NOT** commit your `.env` or `.streamlit/secrets.toml` files to version control
2. Set the environment variables according to your platform:
   - **Heroku**: Use "Config Vars" in settings
   - **AWS/Azure**: Use environment variables in service configuration
   - **Docker**: Pass environment variables in docker-compose or run command

### Troubleshooting Deployment
If you see credential errors after deployment:

1. Verify your API credentials are correctly set up in your deployment platform
2. Check that the format matches what the application expects
3. For Streamlit Cloud, ensure the secrets section is named `[alpaca]` (case sensitive)
4. Run the diagnostic script `streamlit run debug.py` locally to verify credential detection

Without proper credentials, you'll see the error: `Missing required Alpaca API credentials`.

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
- `.env`: API credentials for local development (not tracked in git)
- `.streamlit/secrets.toml`: Streamlit Cloud credentials (not tracked in git)
- `debug.py`: Diagnostic tool for checking credential configuration
- `setup.sh`, `Procfile`, `runtime.txt`: Deployment helper files
- `requirements.txt`: Python dependencies

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
