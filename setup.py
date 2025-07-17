"""
Setup file for the Alpaca Stock Market EDA project.
"""
from setuptools import setup, find_packages

setup(
    name="alpaca_stock_eda",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.1.0",
        "numpy>=1.26.0",
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0",
        "alpaca-trade-api>=3.0.0",
        "python-dotenv>=1.0.0",
        "streamlit>=1.30.0",
        "scikit-learn>=1.3.0",
        "plotly>=5.18.0",
        "statsmodels>=0.14.0",
        "jupyter>=1.0.0",
    ],
    python_requires=">=3.8",
)
