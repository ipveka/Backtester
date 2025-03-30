import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import Dict, Union, Any
from scipy import stats

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def calculate_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Calculate the Value at Risk (VaR) at a specified confidence level.
    
    Parameters:
    -----------
    returns : pandas.Series
        Series of returns
    confidence : float, optional
        Confidence level (e.g., 0.95 for 95% confidence)
    
    Returns:
    --------
    float
        Value at Risk (as a positive number)
    """
    var = np.percentile(returns, 100 * (1 - confidence))
    return abs(var)  # Return as a positive number for easier interpretation


def calculate_expected_shortfall(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Calculate the Expected Shortfall (Conditional VaR) at a specified confidence level.
    
    Parameters:
    -----------
    returns : pandas.Series
        Series of returns
    confidence : float, optional
        Confidence level (e.g., 0.95 for 95% confidence)
    
    Returns:
    --------
    float
        Expected Shortfall (as a positive number)
    """
    var = np.percentile(returns, 100 * (1 - confidence))
    expected_shortfall = returns[returns <= var].mean()
    return abs(expected_shortfall)  # Return as a positive number for easier interpretation


def analyze_worst_periods(returns: pd.Series) -> Dict[str, Dict[str, Any]]:
    """
    Analyze the worst performing periods of various lengths.
    
    Parameters:
    -----------
    returns : pandas.Series
        Series of returns
    
    Returns:
    --------
    Dict[str, Dict[str, Any]]
        Dictionary with worst day, week, month, and year information
    """
    result = {}
    
    # Ensure we have a datetime index
    if not isinstance(returns.index, pd.DatetimeIndex):
        returns.index = pd.to_datetime(returns.index)
    
    # Worst day (already in daily format)
    if len(returns) > 0:
        worst_day_idx = returns.idxmin()
        result['day'] = {
            'return': returns.loc[worst_day_idx],
            'date': worst_day_idx
        }
    else:
        result['day'] = {
            'return': 0,
            'date': None
        }
    
    # Worst week
    try:
        weekly_returns = returns.resample('W').apply(lambda x: (1 + x).prod() - 1)
        worst_week_idx = weekly_returns.idxmin()
        result['week'] = {
            'return': weekly_returns.loc[worst_week_idx],
            'date': worst_week_idx
        }
    except Exception as e:
        logger.error(f"Error calculating worst week: {e}")
        result['week'] = {
            'return': 0,
            'date': None
        }
    
    # Worst month - using 'ME' instead of 'M' for month end frequency
    try:
        monthly_returns = returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)
        worst_month_idx = monthly_returns.idxmin()
        result['month'] = {
            'return': monthly_returns.loc[worst_month_idx],
            'date': worst_month_idx
        }
    except Exception as e:
        logger.error(f"Error calculating worst month: {e}")
        result['month'] = {
            'return': 0,
            'date': None
        }
    
    # Worst year - using 'YE' instead of 'Y' for year end frequency
    try:
        yearly_returns = returns.resample('YE').apply(lambda x: (1 + x).prod() - 1)
        worst_year_idx = yearly_returns.idxmin()
        result['year'] = {
            'return': yearly_returns.loc[worst_year_idx],
            'date': worst_year_idx
        }
    except Exception as e:
        logger.error(f"Error calculating worst year: {e}")
        result['year'] = {
            'return': 0,
            'date': None
        }
    
    return result


def analyze_best_periods(returns: pd.Series) -> Dict[str, Dict[str, Any]]:
    """
    Analyze the best performing periods of various lengths.
    
    Parameters:
    -----------
    returns : pandas.Series
        Series of returns
    
    Returns:
    --------
    Dict[str, Dict[str, Any]]
        Dictionary with best day, week, month, and year information
    """
    result = {}
    
    # Ensure we have a datetime index
    if not isinstance(returns.index, pd.DatetimeIndex):
        returns.index = pd.to_datetime(returns.index)
    
    # Best day (already in daily format)
    if len(returns) > 0:
        best_day_idx = returns.idxmax()
        result['day'] = {
            'return': returns.loc[best_day_idx],
            'date': best_day_idx
        }
    else:
        result['day'] = {
            'return': 0,
            'date': None
        }
    
    # Best week
    try:
        weekly_returns = returns.resample('W').apply(lambda x: (1 + x).prod() - 1)
        best_week_idx = weekly_returns.idxmax()
        result['week'] = {
            'return': weekly_returns.loc[best_week_idx],
            'date': best_week_idx
        }
    except Exception as e:
        logger.error(f"Error calculating best week: {e}")
        result['week'] = {
            'return': 0,
            'date': None
        }
    
    # Best month - using 'ME' instead of 'M' for month end frequency
    try:
        monthly_returns = returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)
        best_month_idx = monthly_returns.idxmax()
        result['month'] = {
            'return': monthly_returns.loc[best_month_idx],
            'date': best_month_idx
        }
    except Exception as e:
        logger.error(f"Error calculating best month: {e}")
        result['month'] = {
            'return': 0,
            'date': None
        }
    
    # Best year - using 'YE' instead of 'Y' for year end frequency
    try:
        yearly_returns = returns.resample('YE').apply(lambda x: (1 + x).prod() - 1)
        best_year_idx = yearly_returns.idxmax()
        result['year'] = {
            'return': yearly_returns.loc[best_year_idx],
            'date': best_year_idx
        }
    except Exception as e:
        logger.error(f"Error calculating best year: {e}")
        result['year'] = {
            'return': 0,
            'date': None
        }
    
    return result


def calculate_stress_test(returns: pd.Series, scenario_name: str) -> Dict[str, float]:
    """
    Perform a stress test by simulating specific historical market scenario.
    
    Parameters:
    -----------
    returns : pandas.Series
        Series of returns
    scenario_name : str
        Name of the historical scenario to simulate
    
    Returns:
    --------
    Dict[str, float]
        Dictionary with stress test results
    """
    # Define historical stress scenarios
    scenarios = {
        '2008_financial_crisis': {
            'start': '2008-09-01',
            'end': '2009-03-31',
            'description': '2008 Global Financial Crisis'
        },
        'covid_crash': {
            'start': '2020-02-19',
            'end': '2020-03-23',
            'description': 'COVID-19 Market Crash'
        },
        'dotcom_bubble': {
            'start': '2000-03-10',
            'end': '2002-10-09',
            'description': 'Dot-com Bubble Burst'
        },
        'black_monday': {
            'start': '1987-10-19',
            'end': '1987-10-20',
            'description': 'Black Monday Crash'
        }
    }
    
    scenario = scenarios.get(scenario_name, None)
    if scenario is None:
        logger.error(f"Stress scenario '{scenario_name}' not found")
        return {
            'scenario': scenario_name,
            'description': 'Unknown scenario',
            'max_drawdown': 0,
            'recovery_days': 0,
            'total_return': 0
        }
    
    try:
        # Get historical data for the scenario period
        scenario_returns = returns[scenario['start']:scenario['end']]
        
        if scenario_returns.empty:
            logger.warning(f"No data available for scenario {scenario_name}")
            return {
                'scenario': scenario_name,
                'description': scenario['description'],
                'max_drawdown': 0,
                'recovery_days': 0,
                'total_return': 0
            }
        
        # Calculate cumulative return for the period
        cum_return = (1 + scenario_returns).prod() - 1
        
        # Calculate max drawdown during the scenario
        cum_returns = (1 + scenario_returns).cumprod()
        running_max = cum_returns.cummax()
        drawdowns = (cum_returns / running_max) - 1
        max_drawdown = drawdowns.min()
        
        # Calculate recovery time (days)
        if cum_return < 0:
            # Find the first date after the scenario where we recover to pre-scenario level
            post_scenario_returns = returns[scenario['end']:]
            full_cum_returns = (1 + returns).cumprod()
            pre_scenario_level = full_cum_returns[scenario['start']]
            
            recovery_date = None
            for date, value in full_cum_returns[scenario['end']:].items():
                if value >= pre_scenario_level:
                    recovery_date = date
                    break
            
            if recovery_date:
                recovery_days = (recovery_date - pd.to_datetime(scenario['end'])).days
            else:
                recovery_days = None  # Never recovered
        else:
            recovery_days = 0  # No loss during scenario
        
        return {
            'scenario': scenario_name,
            'description': scenario['description'],
            'max_drawdown': max_drawdown,
            'recovery_days': recovery_days,
            'total_return': cum_return
        }
    
    except Exception as e:
        logger.error(f"Error performing stress test {scenario_name}: {e}")
        return {
            'scenario': scenario_name,
            'description': scenario.get('description', 'Unknown'),
            'max_drawdown': 0,
            'recovery_days': 0,
            'total_return': 0
        }


def calculate_tail_risk_metrics(returns: pd.Series) -> Dict[str, float]:
    """
    Calculate various tail risk metrics.
    
    Parameters:
    -----------
    returns : pandas.Series
        Series of returns
    
    Returns:
    --------
    Dict[str, float]
        Dictionary with tail risk metrics
    """
    result = {}
    
    try:
        # Skewness (negative values indicate increased likelihood of large negative returns)
        result['skewness'] = returns.skew()
        
        # Kurtosis (higher values indicate fatter tails)
        result['kurtosis'] = returns.kurt()
        
        # Jarque-Bera test for normality
        jb_stat, jb_pvalue = stats.jarque_bera(returns.dropna())
        result['jarque_bera'] = jb_stat
        result['jarque_bera_pvalue'] = jb_pvalue
        
        # Tail risk ratios (Sortino ratio - only considers downside risk)
        downside_returns = returns[returns < 0]
        downside_deviation = np.sqrt((downside_returns ** 2).mean())
        
        if downside_deviation > 0:
            sortino_ratio = returns.mean() / downside_deviation
            result['sortino_ratio'] = sortino_ratio * np.sqrt(252)  # Annualized
        else:
            result['sortino_ratio'] = float('inf')  # No downside volatility
        
        # Maximum consecutive loss days
        neg_streak = (returns < 0).astype(int)
        neg_streak_with_reset = neg_streak.mul(neg_streak.groupby((neg_streak != neg_streak.shift()).cumsum()).cumcount() + 1)
        result['max_consecutive_loss'] = neg_streak_with_reset.max()
        
        # Calmar ratio (annualized return / max drawdown)
        _, max_dd, _ = calculate_drawdowns(returns)
        annualized_return = returns.mean() * 252
        
        if max_dd != 0:
            result['calmar_ratio'] = abs(annualized_return / max_dd)
        else:
            result['calmar_ratio'] = float('inf')  # No drawdown
        
        return result
    
    except Exception as e:
        logger.error(f"Error calculating tail risk metrics: {e}")
        return {
            'skewness': 0,
            'kurtosis': 0,
            'jarque_bera': 0,
            'jarque_bera_pvalue': 1,
            'sortino_ratio': 0,
            'max_consecutive_loss': 0,
            'calmar_ratio': 0
        }