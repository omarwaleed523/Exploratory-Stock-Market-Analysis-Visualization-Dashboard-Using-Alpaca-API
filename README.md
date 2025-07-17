# Alpaca Stock Market EDA & Visualization Dashboard

## Overview
This project performs exploratory data analysis on stock market data using the Alpaca API, with interactive visualizations and potential predictive analytics. The project is designed to work within Alpaca's API limitations.

## Features
- Historical data retrieval with rate-limiting controls (200 API calls/min limit)
- Comprehensive stock market EDA (trends, volatility, moving averages)
- Interactive visualizations
- Optional time-series predictive modeling

## API Limitations Handled
- Rate limiting (200 REST API requests per minute)
- IEX data source restrictions (free tier)
- Limited market coverage (~2% of total market volume)

## Setup and Installation
1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up your Alpaca API credentials in a `.env` file (see `.env.example`)
4. Run the application using one of the methods below

## Running the Application
- **Jupyter Notebooks**: Open the notebooks in the `notebooks/` directory
- **Streamlit Dashboard**: Run `streamlit run src/app.py`
- **Data Collection Only**: Run `python src/data_collection.py`

## Project Structure
- `src/`: Python source code
  - `api_client.py`: Alpaca API client with rate limiting
  - `data_collection.py`: Data ingestion from Alpaca
  - `analysis.py`: Data analysis functions
  - `visualization.py`: Visualization components
  - `app.py`: Streamlit application
- `notebooks/`: Jupyter notebooks for EDA
- `data/`: Cached data storage
- `.env`: API credentials (not tracked in git)

## Requirements
- Python 3.8+
- pandas, numpy, matplotlib, seaborn
- alpaca-trade-api
- streamlit
- scikit-learn (for predictive models)
- plotly (for interactive visualizations)

## License
MIT
