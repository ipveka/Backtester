import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import Dict, List, Optional, Tuple, Union
import math

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def calculate_cagr(prices: pd.Series, ann_factor: int = 252) -> float:
    """
    Calculate the Compound Annual Growth Rate.
    
    Parameters:
    -----------
    prices : pandas.Series
        Series of prices or portfolio values
    ann_factor : int, optional
        Annualization factor (252 for daily, 12 for monthly, 4 for quarterly, etc.)
    
    Returns:
    --------
    float
        Compound Annual Growth Rate
    """
    if len(prices) < 2:
        return 0.0
    
    # Calculate the number of years
    start_date = prices.index[0]
    end_date = prices.index[-1]
    days_diff = (end_date - start_date).days
    years = days_diff / 365.25
    
    if years <= 0:
        return 0.0
    
    # Calculate CAGR
    start_value = prices.iloc[0]
    end_value = prices.iloc[-1]
    
    if start_value <= 0:
        logger.warning("Start value is zero or negative. Cannot calculate CAGR.")
        return 0.0
    
    cagr = (end_value / start_value) ** (1 / years) - 1
    
    return cagr


def calculate_volatility(returns: pd.Series, ann_factor: int = 252) -> float:
    """
    Calculate the annualized volatility (standard deviation of returns).
    
    Parameters:
    -----------
    returns : pandas.Series
        Series of returns
    ann_factor : int, optional
        Annualization factor (252 for daily, 12 for monthly, 4 for quarterly, etc.)
    
    Returns:
    --------
    float
        Annualized volatility
    """
    # Convert to numpy array for the test case with fixed_returns
    if isinstance(returns, (list, np.ndarray)):
        returns = np.array(returns)
        return np.std(returns, ddof=1) * np.sqrt(ann_factor)
    
    return returns.std() * np.sqrt(ann_factor)


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float, ann_factor: int = 252) -> float:
    """
    Calculate the Sharpe ratio.
    
    Parameters:
    -----------
    returns : pandas.Series
        Series of returns
    risk_free_rate : float
        Annual risk-free rate (e.g., 0.02 for 2%)
    ann_factor : int, optional
        Annualization factor (252 for daily, 12 for monthly, 4 for quarterly, etc.)
    
    Returns:
    --------
    float
        Sharpe ratio
    """
    # Convert to numpy array for the test case with fixed_returns
    if isinstance(returns, (list, np.ndarray)):
        returns = np.array(returns)
        mean_return = np.mean(returns) * ann_factor
        std_return = np.std(returns, ddof=1) * np.sqrt(ann_factor)
    else:
        if returns.std() == 0:
            return 0.0
        mean_return = returns.mean() * ann_factor
        std_return = returns.std() * np.sqrt(ann_factor)
    
    # Calculate Sharpe ratio
    sharpe = (mean_return - risk_free_rate) / std_return
    
    return sharpe


def calculate_drawdowns(returns: pd.Series) -> Tuple[pd.Series, float, int]:
    """
    Calculate drawdowns from a series of returns.
    
    Parameters:
    -----------
    returns : pandas.Series
        Series of returns
    
    Returns:
    --------
    Tuple[pandas.Series, float, int]
        Drawdown series, maximum drawdown, longest drawdown duration (in days)
    """
    # Calculate cumulative returns
    cum_returns = (1 + returns).cumprod()
    
    # Calculate running maximum
    running_max = cum_returns.cummax()
    
    # Calculate drawdowns
    drawdowns = (cum_returns / running_max) - 1
    
    # Get the maximum drawdown
    max_drawdown = drawdowns.min()
    
    # Calculate drawdown duration
    in_drawdown = drawdowns < 0
    drawdown_start = in_drawdown.ne(in_drawdown.shift()).cumsum()
    drawdown_group = drawdown_start.mask(~in_drawdown)
    drawdown_duration = drawdown_group.map(drawdown_group.value_counts())
    
    longest_drawdown = 0 if drawdown_duration.empty else drawdown_duration.max()
    
    return drawdowns, max_drawdown, longest_drawdown


def calculate_rolling_returns(returns: pd.Series) -> Dict[str, Dict[str, float]]:
    """
    Calculate rolling returns for various time periods.
    
    Parameters:
    -----------
    returns : pandas.Series
        Series of returns
    
    Returns:
    --------
    Dict[str, Dict[str, float]]
        Dictionary of rolling returns statistics for each time period
    """
    result = {}
    
    # Define periods in trading days (approximate)
    periods = {
        '1Y': 252,
        '3Y': 756,
        '5Y': 1260,
        '10Y': 2520
    }
    
    for period_name, days in periods.items():
        if len(returns) >= days:
            rolling_returns = (1 + returns).rolling(window=days).apply(lambda x: x.prod() - 1)
            result[period_name] = {
                'min': rolling_returns.min(),
                'max': rolling_returns.max(),
                'avg': rolling_returns.mean()
            }
        else:
            # Not enough data for this period
            logger.warning(f"Not enough data for {period_name} rolling returns")
            result[period_name] = {
                'min': None,
                'max': None,
                'avg': None
            }
    
    return result


def analyze_monthly_returns(returns: pd.Series) -> pd.DataFrame:
    """
    Analyze returns by month.
    
    Parameters:
    -----------
    returns : pandas.Series
        Series of returns
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with monthly returns analysis
    """
    # Resample to monthly returns
    if len(returns) < 20:  # Ensure we have enough data
        return pd.DataFrame()
    
    # Group by year and month
    returns.index = pd.to_datetime(returns.index)
    # Use 'ME' instead of 'M' to address deprecation warning
    monthly_returns = returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)
    
    # Create a matrix of monthly returns
    monthly_matrix = pd.DataFrame({
        'Year': monthly_returns.index.year,
        'Month': monthly_returns.index.month,
        'Return': monthly_returns.values
    })
    
    # Pivot to create a year x month matrix
    monthly_heatmap = monthly_matrix.pivot(index='Year', columns='Month', values='Return')
    
    return monthly_heatmap


def analyze_annual_returns(returns: pd.Series) -> pd.Series:
    """
    Analyze returns by year.
    
    Parameters:
    -----------
    returns : pandas.Series
        Series of returns
    
    Returns:
    --------
    pandas.Series
        Series with annual returns
    """
    # Resample to annual returns
    if len(returns) < 20:  # Ensure we have enough data
        return pd.Series()
    
    returns.index = pd.to_datetime(returns.index)
    # Use 'YE' instead of 'Y' to address deprecation warning
    annual_returns = returns.resample('YE').apply(lambda x: (1 + x).prod() - 1)
    
    return annual_returns