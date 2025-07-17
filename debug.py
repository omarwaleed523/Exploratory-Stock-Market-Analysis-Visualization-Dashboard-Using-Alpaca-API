"""
Diagnostic script to debug Streamlit secrets and environment variables.
Run this with streamlit run debug.py to see what credentials are available.
"""
import os
import sys
import logging
import streamlit as st
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Set up Streamlit page
st.set_page_config(
    page_title="API Credentials Diagnostic",
    page_icon="🔍",
    layout="wide"
)

st.title("API Credentials Diagnostic")
st.write("This page checks what credentials are available in your environment.")

# Check Streamlit secrets
st.header("Streamlit Secrets")
if hasattr(st, 'secrets'):
    st.write("✅ Streamlit secrets are available")
    
    if 'alpaca' in st.secrets:
        st.write("✅ 'alpaca' section found in secrets")
        
        # Check each required key
        alpaca_keys = ['api_key_id', 'api_secret_key', 'api_base_url', 'data_url']
        for key in alpaca_keys:
            if key in st.secrets.alpaca:
                masked_value = "✅ Set"
                if key == 'api_base_url' or key == 'data_url':
                    masked_value = st.secrets.alpaca[key]
                st.write(f"  • {key}: {masked_value}")
            else:
                st.write(f"  • {key}: ❌ Not found")
    else:
        st.write("❌ 'alpaca' section not found in secrets")
        st.write("Available sections:")
        for section in st.secrets:
            st.write(f"  • {section}")
else:
    st.write("❌ Streamlit secrets are not available")

# Check environment variables
st.header("Environment Variables")
env_vars = {
    'ALPACA_API_KEY_ID': os.getenv('ALPACA_API_KEY_ID'),
    'ALPACA_API_SECRET_KEY': os.getenv('ALPACA_API_SECRET_KEY'),
    'ALPACA_API_BASE_URL': os.getenv('ALPACA_API_BASE_URL'),
    'ALPACA_DATA_URL': os.getenv('ALPACA_DATA_URL')
}

for key, value in env_vars.items():
    if value:
        masked_value = "✅ Set"
        if 'URL' in key:
            masked_value = value
        st.write(f"  • {key}: {masked_value}")
    else:
        st.write(f"  • {key}: ❌ Not set")

# System information
st.header("System Information")
st.write(f"Python version: {sys.version}")
st.write(f"Current working directory: {os.getcwd()}")
st.write(f"Files in current directory: {os.listdir('.')}")

if os.path.exists('.streamlit'):
    st.write(f"Files in .streamlit directory: {os.listdir('.streamlit')}")
else:
    st.write("❌ .streamlit directory not found")

# Show recommendation
st.header("Recommendations")
st.write("""
If you're deploying to Streamlit Cloud:
1. Make sure you've added the secrets in the Streamlit Cloud dashboard
2. The format should be exactly as shown in your local `.streamlit/secrets.toml` file
3. Check that the section name is 'alpaca' and the keys match exactly
""")

# Show exact format needed
st.code("""
# This is what your secrets.toml should look like
[alpaca]
api_key_id = "YOUR_API_KEY"
api_secret_key = "YOUR_API_SECRET"
api_base_url = "https://paper-api.alpaca.markets"
data_url = "https://data.alpaca.markets"
""", language="toml")
