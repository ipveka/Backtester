import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from pathlib import Path
import logging
import csv
from tqdm import tqdm
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the project root directory
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'data')

# Make sure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, 'inflation'), exist_ok=True)

# Sample ETF database - in a production environment, this should be loaded from a file or database
DEFAULT_ETF_DATA = [
    # US Equities
    {"Ticker": "SPY", "Name": "SPDR S&P 500 ETF Trust", "Category": "Equity", "Region": "US", "Description": "Tracks the S&P 500 Index"},
    {"Ticker": "QQQ", "Name": "Invesco QQQ Trust", "Category": "Equity", "Region": "US", "Description": "Tracks the Nasdaq-100 Index"},
    {"Ticker": "IWM", "Name": "iShares Russell 2000 ETF", "Category": "Equity", "Region": "US", "Description": "Tracks the Russell 2000 Index"},
    {"Ticker": "VTI", "Name": "Vanguard Total Stock Market ETF", "Category": "Equity", "Region": "US", "Description": "Tracks the CRSP US Total Market Index"},
    {"Ticker": "VOO", "Name": "Vanguard S&P 500 ETF", "Category": "Equity", "Region": "US", "Description": "Tracks the S&P 500 Index"},
    
    # US Bonds
    {"Ticker": "AGG", "Name": "iShares Core U.S. Aggregate Bond ETF", "Category": "Bond", "Region": "US", "Description": "Tracks the Bloomberg US Aggregate Bond Index"},
    {"Ticker": "BND", "Name": "Vanguard Total Bond Market ETF", "Category": "Bond", "Region": "US", "Description": "Tracks the Bloomberg U.S. Aggregate Float Adjusted Index"},
    {"Ticker": "TLT", "Name": "iShares 20+ Year Treasury Bond ETF", "Category": "Bond", "Region": "US", "Description": "Tracks the ICE U.S. Treasury 20+ Year Bond Index"},
    
    # US Other
    {"Ticker": "GLD", "Name": "SPDR Gold Shares", "Category": "Commodity", "Region": "US", "Description": "Tracks the price of gold bullion"},
    {"Ticker": "VNQ", "Name": "Vanguard Real Estate ETF", "Category": "Real Estate", "Region": "US", "Description": "Tracks the MSCI US Investable Market Real Estate 25/50 Index"},
    
    # European Equities
    {"Ticker": "VGK", "Name": "Vanguard FTSE Europe ETF", "Category": "Equity", "Region": "EU", "Description": "Tracks the FTSE Developed Europe All Cap Index"},
    {"Ticker": "IEUR", "Name": "iShares Core MSCI Europe ETF", "Category": "Equity", "Region": "EU", "Description": "Tracks the MSCI Europe Investable Market Index"},
    {"Ticker": "EZU", "Name": "iShares MSCI Eurozone ETF", "Category": "Equity", "Region": "EU", "Description": "Tracks the MSCI EMU Index"},
    {"Ticker": "HEZU", "Name": "iShares Currency Hedged MSCI Eurozone ETF", "Category": "Equity", "Region": "EU", "Description": "Currency-hedged version of EZU"},
    
    # European Bonds
    {"Ticker": "IEAC", "Name": "iShares Core € Corp Bond UCITS ETF", "Category": "Bond", "Region": "EU", "Description": "Tracks the Bloomberg Euro Corporate Bond Index"},
    {"Ticker": "IBTE", "Name": "iShares $ Treasury Bond 1-3yr UCITS ETF", "Category": "Bond", "Region": "EU", "Description": "Tracks short-term US Treasury bonds"},
    
    # European Other
    {"Ticker": "PHAU", "Name": "Invesco Physical Gold ETC", "Category": "Commodity", "Region": "EU", "Description": "Tracks the spot gold price"},
    {"Ticker": "TRET", "Name": "VanEck Vectors European Real Estate UCITS ETF", "Category": "Real Estate", "Region": "EU", "Description": "Tracks European real estate companies"}
]

# Dictionary mapping benchmark names to tickers
BENCHMARK_TICKERS = {
    "S&P 500 (SPY)": "SPY",
    "MSCI World (URTH)": "URTH",
    "MSCI Europe (IEUR)": "IEUR",
    "None": None
}


def save_etf_metadata():
    """Save the ETF metadata to a CSV file."""
    etf_metadata_path = os.path.join(DATA_DIR, 'etf_metadata.csv')
    
    with open(etf_metadata_path, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["Ticker", "Name", "Category", "Region", "Description"])
        writer.writeheader()
        writer.writerows(DEFAULT_ETF_DATA)
    
    logger.info(f"ETF metadata saved to {etf_metadata_path}")


def get_available_etfs():
    """
    Get the list of available ETFs with their metadata.
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with ETF metadata
    """
    etf_metadata_path = os.path.join(DATA_DIR, 'etf_metadata.csv')
    
    # If the ETF metadata file doesn't exist, create it
    if not os.path.exists(etf_metadata_path):
        save_etf_metadata()
    
    # Load the ETF metadata
    try:
        etf_data = pd.read_csv(etf_metadata_path)
    except Exception as e:
        logger.error(f"Error reading ETF metadata: {e}")
        # Fallback to default data
        etf_data = pd.DataFrame(DEFAULT_ETF_DATA)
    
    return etf_data


def load_etf_data(tickers, start_date, end_date):
    """
    Load historical ETF price data for a list of tickers.
    
    Parameters:
    -----------
    tickers : list
        List of ETF ticker symbols
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with historical price data (indexed by date, with tickers as columns)
    """
    logger.info(f"Loading data for {len(tickers)} ETFs from {start_date} to {end_date}")
    
    # Check if tickers is empty
    if not tickers:
        logger.warning("No tickers provided. Returning empty DataFrame.")
        return pd.DataFrame()
    
    # Adjust start date to include a buffer period (for calculating returns)
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    buffer_start = (start_dt - timedelta(days=10)).strftime('%Y-%m-%d')
    
    # Initialize an empty DataFrame for the results
    all_data = pd.DataFrame()
    
    # Track failed tickers
    failed_tickers = []
    
    # Try to download actual data first - fixing the 'str object is not callable' error
    for ticker in tqdm(tickers, desc="Downloading ETF data"):
        try:
            ticker_obj = yf.Ticker(ticker)
            # Get historical data
            data = ticker_obj.history(
                start=buffer_start,
                end=end_date,
                auto_adjust=False
            )
            
            if data.empty:
                logger.warning(f"No data available for {ticker}")
                failed_tickers.append(ticker)
                continue
            
            # Extract the Close price if Adj Close is not available
            price_col = 'Close'
            if 'Adj Close' in data.columns:
                price_col = 'Adj Close'
            
            ticker_data = data[price_col].rename(ticker)
            
            # Add to the result DataFrame
            if all_data.empty:
                all_data = pd.DataFrame(ticker_data)
            else:
                all_data = all_data.join(ticker_data, how='outer')
            
            # Avoid hitting rate limits
            time.sleep(0.1)
        
        except Exception as e:
            logger.error(f"Error downloading data for {ticker}: {e}")
            failed_tickers.append(ticker)
    
    # If no real data was loaded, generate dummy data
    if all_data.empty:
        logger.warning("No actual ETF data could be loaded. Using dummy data instead.")
        all_data = load_dummy_data(tickers, start_date, end_date)
    else:
        # Filter to the requested date range
        try:
            all_data = all_data.loc[start_date:end_date]
        except:
            logger.warning("Error filtering data to requested date range.")
        
        # Fill missing values with forward fill followed by backward fill
        all_data = all_data.ffill().bfill()
    
    if failed_tickers and len(failed_tickers) < len(tickers):
        logger.warning(f"Failed to download data for {len(failed_tickers)} tickers: {failed_tickers}")
    
    logger.info(f"Successfully loaded data for {len(tickers) - len(failed_tickers)} ETFs")
    
    return all_data


def load_dummy_data(tickers, start_date, end_date):
    """
    Generate dummy data for testing when real data can't be loaded.
    
    Parameters:
    -----------
    tickers : list
        List of ETF ticker symbols
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with simulated price data
    """
    logger.info(f"Generating dummy data for {len(tickers)} ETFs")
    
    # Create date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')
    
    # Create dummy data
    data = {}
    np.random.seed(42)  # For reproducibility
    
    for ticker in tickers:
        # Generate random returns with a slight positive drift
        drift = 0.0002  # Small positive drift
        volatility = 0.01  # Typical daily volatility
        
        if "Bond" in ticker or "AGG" in ticker or "BND" in ticker or "TLT" in ticker:
            drift = 0.0001  # Lower drift for bonds
            volatility = 0.003  # Lower volatility for bonds
        
        # Generate returns
        returns = np.random.normal(drift, volatility, size=len(date_range))
        
        # Convert to prices
        initial_price = 100  # Starting price
        prices = initial_price * np.cumprod(1 + returns)
        
        data[ticker] = prices
    
    # Create DataFrame
    df = pd.DataFrame(data, index=date_range)
    
    return df


def load_inflation_data(country, start_date, end_date):
    """
    Load historical inflation data for a specific country.
    
    Parameters:
    -----------
    country : str
        Country code (e.g., 'US', 'UK', etc.)
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with historical inflation data
    """
    logger.info(f"Loading inflation data for {country} from {start_date} to {end_date}")
    
    # Convert string dates to datetime objects for consistent comparison
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    # Path to the inflation data file
    inflation_path = os.path.join(DATA_DIR, 'inflation', f'{country.lower()}_inflation.csv')
    
    # If the file exists, load it
    if os.path.exists(inflation_path):
        try:
            inflation_data = pd.read_csv(inflation_path, index_col=0, parse_dates=True)
            
            # Ensure index has no timezone information
            if inflation_data.index.tz is not None:
                inflation_data.index = inflation_data.index.tz_localize(None)
            
            # Filter to the requested date range - using datetime objects
            mask = (inflation_data.index >= start_dt) & (inflation_data.index <= end_dt)
            inflation_data = inflation_data.loc[mask]
            
            if not inflation_data.empty:
                logger.info(f"Successfully loaded inflation data from file")
                return inflation_data
        
        except Exception as e:
            logger.error(f"Error reading inflation data from file: {e}")
    
    # Create dummy inflation data
    logger.warning("Using dummy inflation data")
    
    # Create date range - fixing the deprecation warning
    date_range = pd.date_range(start=start_dt, end=end_dt, freq='ME')
    
    # Create dummy data (around 2-3% annual inflation)
    np.random.seed(42)  # For reproducibility
    dummy_inflation = pd.Series(
        np.random.normal(0.0025, 0.001, len(date_range)),  # Monthly values (2-3% annual)
        index=date_range,
        name='inflation_rate'
    )
    
    # Save the dummy data for future use
    os.makedirs(os.path.dirname(inflation_path), exist_ok=True)
    dummy_inflation.to_csv(inflation_path)
    
    return pd.DataFrame(dummy_inflation)