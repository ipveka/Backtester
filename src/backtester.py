import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import os

# Import local modules
from src.data_loader import load_etf_data, load_dummy_data, load_inflation_data
from src.portfolio import Portfolio, PortfolioSnapshot
from src.analysis import (
    calculate_cagr,
    calculate_sharpe_ratio,
    calculate_volatility,
    calculate_drawdowns,
    calculate_rolling_returns,
    analyze_monthly_returns,
    analyze_annual_returns
)
from src.risk import (
    calculate_var,
    calculate_expected_shortfall,
    analyze_worst_periods,
    analyze_best_periods  # Add this line to import the new function
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def normalize_datetime_index(df):
    """
    Normalize a DataFrame index to ensure consistent datetime format without timezone information.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with a datetime index that may contain timezone info
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with a normalized datetime index (no timezone)
    """
    if df is None or df.empty:
        return df
    
    if isinstance(df.index, pd.DatetimeIndex):
        # Create a new DataFrame with timezone-naive index
        normalized_df = pd.DataFrame(df.values, 
                                    index=df.index.tz_localize(None) if df.index.tz else df.index,
                                    columns=df.columns)
        return normalized_df
    
    return df


@dataclass
class BacktestResults:
    """
    Class to store the results of a backtest.
    """
    # Portfolio data
    etfs: List[str]
    weights: Dict[str, float]
    initial_investment: float
    monthly_contribution: float
    start_date: str
    end_date: str
    country: str
    benchmark: Optional[str]
    
    # Performance metrics
    final_value: float
    total_contributions: float
    total_gain: float
    total_return: float
    cagr: float
    sharpe_ratio: float
    
    # Risk metrics
    volatility: float
    max_drawdown: float
    longest_drawdown_days: int
    var_95: float
    var_99: float
    expected_shortfall: float
    
    # Time series data
    portfolio_value_history: pd.Series
    returns_df: pd.DataFrame
    drawdown_series: pd.Series
    
    # Statistical metrics
    mean_return: float
    median_return: float
    return_std: float
    skewness: float
    kurtosis: float
    positive_days_pct: float
    negative_days_pct: float
    
    # Worst periods
    worst_day_return: float
    worst_day_date: datetime
    worst_week_return: float
    worst_week_date: datetime
    worst_month_return: float
    worst_month_date: datetime
    worst_year_return: float
    worst_year_date: datetime
    
    # Add best periods
    best_day_return: float
    best_day_date: datetime
    best_week_return: float
    best_week_date: datetime
    best_month_return: float
    best_month_date: datetime
    best_year_return: float
    best_year_date: datetime
    
    # Rolling returns
    rolling_returns_1yr_min: float
    rolling_returns_1yr_max: float
    rolling_returns_1yr_avg: float
    
    # Attributes that may be None for short backtests
    rolling_returns_3yr_min: Optional[float] = None
    rolling_returns_3yr_max: Optional[float] = None
    rolling_returns_3yr_avg: Optional[float] = None
    rolling_returns_5yr_min: Optional[float] = None
    rolling_returns_5yr_max: Optional[float] = None
    rolling_returns_5yr_avg: Optional[float] = None
    rolling_returns_10yr_min: Optional[float] = None
    rolling_returns_10yr_max: Optional[float] = None
    rolling_returns_10yr_avg: Optional[float] = None
    
    # Monthly/annual returns analysis
    monthly_returns: Optional[pd.DataFrame] = None
    annual_returns: Optional[pd.Series] = None
    
    # Inflation data
    inflation_data: Optional[pd.DataFrame] = None
    inflation_adjusted_returns: Optional[pd.DataFrame] = None
    
    # Benchmark comparison
    benchmark_returns: Optional[pd.DataFrame] = None
    benchmark_cagr: Optional[float] = None
    benchmark_sharpe: Optional[float] = None
    benchmark_max_drawdown: Optional[float] = None
    
    # Multi-asset analysis
    correlation_matrix: Optional[pd.DataFrame] = None
    asset_statistics: Optional[pd.DataFrame] = None
    
    # Snapshot data
    initial_snapshot: Optional[PortfolioSnapshot] = None
    final_snapshot: Optional[PortfolioSnapshot] = None


def backtest(
    portfolio: Portfolio,
    initial_investment: float,
    start_date: str,
    end_date: str,
    country: str = "US",
    benchmark: Optional[str] = None,
    monthly_contribution: float = 0,
    rebalance_frequency: str = "quarterly",
    risk_free_rate: float = 0.02
) -> BacktestResults:
    """
    Run a backtest for a given portfolio configuration.
    
    Parameters:
    -----------
    portfolio : Portfolio
        Portfolio object containing ETFs and weights
    initial_investment : float
        Initial investment amount
    start_date : str
        Start date for the backtest (YYYY-MM-DD)
    end_date : str
        End date for the backtest (YYYY-MM-DD)
    country : str, optional
        Country for inflation data
    benchmark : str, optional
        Ticker for benchmark comparison
    monthly_contribution : float, optional
        Monthly contribution amount
    rebalance_frequency : str, optional
        Frequency for portfolio rebalancing ('never', 'monthly', 'quarterly', 'annually')
    risk_free_rate : float, optional
        Annual risk-free rate for Sharpe ratio calculation
    
    Returns:
    --------
    BacktestResults
        Object containing all backtest results
    """
    logger.info(f"Starting backtest from {start_date} to {end_date}")
    logger.info(f"Portfolio: {portfolio.etfs}")
    
    # Validate the portfolio
    if not portfolio.is_valid():
        logger.warning("Portfolio weights don't sum to 1. Normalizing weights.")
        # Normalize the weights
        total_weight = sum(portfolio.get_weights().values())
        for ticker in portfolio.etfs:
            portfolio.etfs[ticker] /= total_weight
    
    # Load ETF price data
    tickers = portfolio.get_etfs()
    etf_data = load_etf_data(tickers, start_date, end_date)
    
    if etf_data.empty:
        logger.warning("No ETF data available. Using dummy data.")
        etf_data = load_dummy_data(tickers, start_date, end_date)
    
    # Add benchmark if specified
    if benchmark:
        try:
            benchmark_data = load_etf_data([benchmark], start_date, end_date)
            if not benchmark_data.empty:
                benchmark_returns = benchmark_data.pct_change().dropna()
            else:
                logger.warning(f"No data available for benchmark {benchmark}. Proceeding without benchmark.")
                benchmark_returns = None
        except Exception as e:
            logger.error(f"Error loading benchmark data: {e}")
            benchmark_returns = None
    else:
        benchmark_returns = None
    
    # Initialize the backtest variables
    current_date = pd.to_datetime(start_date)
    end_datetime = pd.to_datetime(end_date)
    
    # We'll store the portfolio value at each date
    portfolio_values = {}
    
    # Initialize holdings with the initial investment
    holdings = {ticker: 0 for ticker in tickers}  # Shares of each ETF
    cash = initial_investment
    
    # Get the first available date in the data
    first_date = etf_data.index[0]
    
    # Track total contributions
    total_contributions = initial_investment
    
    # Create initial snapshot
    etf_holdings = {}
    for ticker in tickers:
        etf_holdings[ticker] = {
            'shares': 0,
            'price': etf_data.loc[first_date, ticker]
        }
    initial_snapshot = PortfolioSnapshot(first_date, etf_holdings, cash)
    
    # Rebalancing schedule
    if rebalance_frequency == 'monthly':
        rebalance_months = 1
    elif rebalance_frequency == 'quarterly':
        rebalance_months = 3
    elif rebalance_frequency == 'annually':
        rebalance_months = 12
    else:  # 'never'
        rebalance_months = None
    
    # Track last rebalance date
    last_rebalance_date = None
    
    # Run the backtest
    for date, prices in etf_data.iterrows():
        current_date = date
        
        # Check if it's time to add the monthly contribution
        if monthly_contribution > 0 and date.day == 1 and date > first_date:
            cash += monthly_contribution
            total_contributions += monthly_contribution
        
        # Check if it's time to rebalance
        should_rebalance = False
        if rebalance_months and (last_rebalance_date is None or 
                                (date.year - last_rebalance_date.year) * 12 + 
                                (date.month - last_rebalance_date.month) >= rebalance_months):
            should_rebalance = True
            last_rebalance_date = date
        
        # On the first day or rebalance days, invest according to target weights
        if date == first_date or should_rebalance:
            # Calculate the current portfolio value
            portfolio_value = cash
            for ticker in tickers:
                portfolio_value += holdings[ticker] * prices[ticker]
            
            # Calculate the target value for each ETF
            target_values = {}
            for ticker, weight in portfolio.etfs.items():
                target_values[ticker] = portfolio_value * weight
            
            # Rebalance the portfolio
            for ticker in tickers:
                # Calculate the current value and the difference to target
                current_value = holdings[ticker] * prices[ticker]
                target_value = target_values.get(ticker, 0)
                
                # Calculate how many shares to buy/sell
                shares_to_adjust = (target_value - current_value) / prices[ticker]
                
                # Update the holdings and cash
                if shares_to_adjust > 0:  # Buy
                    cash_required = shares_to_adjust * prices[ticker]
                    if cash_required <= cash:  # Check if we have enough cash
                        holdings[ticker] += shares_to_adjust
                        cash -= cash_required
                    else:  # Adjust to the available cash
                        affordable_shares = cash / prices[ticker]
                        holdings[ticker] += affordable_shares
                        cash = 0
                else:  # Sell
                    holdings[ticker] += shares_to_adjust  # Negative shares_to_adjust
                    cash -= shares_to_adjust * prices[ticker]  # Add to cash (negative * negative = positive)
        
        # Calculate the portfolio value for this date
        portfolio_value = cash
        for ticker in tickers:
            if ticker in prices:
                portfolio_value += holdings[ticker] * prices[ticker]
        
        # Store the portfolio value
        portfolio_values[date] = portfolio_value
    
    # Create the portfolio value series
    portfolio_value_series = pd.Series(portfolio_values)
    
    # Create final snapshot
    etf_holdings = {}
    for ticker in tickers:
        if ticker in etf_data.iloc[-1]:
            etf_holdings[ticker] = {
                'shares': holdings[ticker],
                'price': etf_data.iloc[-1][ticker]
            }
    final_snapshot = PortfolioSnapshot(etf_data.index[-1], etf_holdings, cash)
    
    # Calculate returns
    returns_df = pd.DataFrame(index=portfolio_value_series.index)
    returns_df['Value'] = portfolio_value_series
    returns_df['Daily Returns'] = returns_df['Value'].pct_change()
    returns_df['Cumulative Returns'] = (1 + returns_df['Daily Returns']).cumprod() - 1
    
    # Normalize returns dataframe to remove timezone info
    returns_df = normalize_datetime_index(returns_df)
    
    # Load inflation data if requested
    try:
        inflation_data = load_inflation_data(country, start_date, end_date)
        
        # Normalize inflation data datetime index to remove timezone info
        inflation_data = normalize_datetime_index(inflation_data)
        
        # Calculate inflation-adjusted returns
        inflation_adjusted_returns = pd.DataFrame(index=returns_df.index)
        
        # Resample inflation data to match portfolio value frequency
        monthly_inflation = inflation_data['inflation_rate'].resample('D').ffill()
        monthly_inflation = monthly_inflation.reindex(returns_df.index, method='ffill')
        
        # Calculate cumulative inflation factor
        cumulative_inflation = (1 + monthly_inflation).cumprod()
        
        # Calculate inflation-adjusted portfolio value
        inflation_adjusted_returns['Inflation-Adjusted Value'] = returns_df['Value'] / cumulative_inflation
        inflation_adjusted_returns['Inflation-Adjusted Returns'] = inflation_adjusted_returns['Inflation-Adjusted Value'].pct_change()
        inflation_adjusted_returns['Inflation-Adjusted Cumulative Returns'] = (1 + inflation_adjusted_returns['Inflation-Adjusted Returns']).cumprod() - 1
    
    except Exception as e:
        logger.error(f"Error loading inflation data: {e}")
        import traceback
        logger.error(traceback.format_exc())
        inflation_data = None
        inflation_adjusted_returns = None
    
    # Calculate performance metrics
    final_value = portfolio_value_series.iloc[-1]
    total_gain = final_value - total_contributions
    total_return = total_gain / initial_investment
    cagr = calculate_cagr(portfolio_value_series, ann_factor=252)
    volatility = calculate_volatility(returns_df['Daily Returns'])
    sharpe = calculate_sharpe_ratio(returns_df['Daily Returns'], risk_free_rate, ann_factor=252)
    
    # Calculate drawdowns
    drawdown_series, max_drawdown, longest_drawdown = calculate_drawdowns(returns_df['Daily Returns'])
    
    # Calculate rolling returns
    rolling_returns = calculate_rolling_returns(returns_df['Daily Returns'])
    
    # Calculate VaR and Expected Shortfall
    var_95 = calculate_var(returns_df['Daily Returns'], confidence=0.95)
    var_99 = calculate_var(returns_df['Daily Returns'], confidence=0.99)
    expected_shortfall = calculate_expected_shortfall(returns_df['Daily Returns'], confidence=0.95)
    
    # Analyze monthly and annual returns
    monthly_returns_analysis = analyze_monthly_returns(returns_df['Daily Returns'])
    annual_returns_analysis = analyze_annual_returns(returns_df['Daily Returns'])
    
    # Calculate statistical metrics
    mean_return = returns_df['Daily Returns'].mean()
    median_return = returns_df['Daily Returns'].median()
    return_std = returns_df['Daily Returns'].std()
    skewness = returns_df['Daily Returns'].skew()
    kurtosis = returns_df['Daily Returns'].kurtosis()
    positive_days = (returns_df['Daily Returns'] > 0).sum() / len(returns_df['Daily Returns'])
    negative_days = (returns_df['Daily Returns'] < 0).sum() / len(returns_df['Daily Returns'])
    
    # Analyze worst periods
    worst_periods = analyze_worst_periods(returns_df['Daily Returns'])
    
    # Analyze best periods
    best_periods = analyze_best_periods(returns_df['Daily Returns'])
    
    # Calculate correlation matrix and asset statistics if multiple assets
    if len(tickers) > 1:
        # Get ETF returns
        etf_returns = etf_data.pct_change().dropna()
        
        # Calculate correlation matrix
        correlation_matrix = etf_returns.corr()
        
        # Calculate asset statistics
        asset_statistics = pd.DataFrame({
            'Annualized Return': calculate_cagr(etf_data, ann_factor=252),
            'Volatility': etf_returns.std() * np.sqrt(252),
            'Sharpe Ratio': (etf_returns.mean() * 252 - risk_free_rate) / (etf_returns.std() * np.sqrt(252)),
            'Max Drawdown': [calculate_drawdowns(etf_returns[ticker])[1] for ticker in tickers]
        })
    else:
        correlation_matrix = None
        asset_statistics = None
    
    # Calculate benchmark metrics
    if benchmark_returns is not None and not benchmark_returns.empty:
        try:
            # Make sure we're working with a single column Series for the benchmark
            if isinstance(benchmark_returns, pd.DataFrame) and benchmark_returns.shape[1] > 0:
                benchmark_return_series = benchmark_returns.iloc[:, 0]
            else:
                benchmark_return_series = benchmark_returns
                
            # Get the benchmark data Series for CAGR calculation
            if isinstance(benchmark_data, pd.DataFrame) and benchmark_data.shape[1] > 0:
                benchmark_price_series = benchmark_data.iloc[:, 0]
            else:
                benchmark_price_series = benchmark_data
            
            # Calculate benchmark metrics
            benchmark_cagr = calculate_cagr(benchmark_price_series, ann_factor=252)
            benchmark_sharpe = calculate_sharpe_ratio(benchmark_return_series, risk_free_rate, ann_factor=252)
            benchmark_dd, benchmark_max_dd, _ = calculate_drawdowns(benchmark_return_series)
        except Exception as e:
            logger.error(f"Error calculating benchmark metrics: {e}")
            import traceback
            logger.error(traceback.format_exc())
            benchmark_cagr = None
            benchmark_sharpe = None
            benchmark_max_dd = None
    else:
        benchmark_cagr = None
        benchmark_sharpe = None
        benchmark_max_dd = None
    
    # Create and return the results object
    results = BacktestResults(
        # Portfolio data
        etfs=tickers,
        weights=portfolio.get_weights(),
        initial_investment=initial_investment,
        monthly_contribution=monthly_contribution,
        start_date=start_date,
        end_date=end_date,
        country=country,
        benchmark=benchmark,
        
        # Performance metrics
        final_value=final_value,
        total_contributions=total_contributions,
        total_gain=total_gain,
        total_return=total_return,
        cagr=cagr,
        sharpe_ratio=sharpe,
        
        # Risk metrics
        volatility=volatility,
        max_drawdown=max_drawdown,
        longest_drawdown_days=longest_drawdown,
        var_95=var_95,
        var_99=var_99,
        expected_shortfall=expected_shortfall,
        
        # Time series data
        portfolio_value_history=portfolio_value_series,
        returns_df=returns_df,
        drawdown_series=drawdown_series,
        
        # Statistical metrics
        mean_return=mean_return,
        median_return=median_return,
        return_std=return_std,
        skewness=skewness,
        kurtosis=kurtosis,
        positive_days_pct=positive_days,
        negative_days_pct=negative_days,
        
        # Worst periods
        worst_day_return=worst_periods['day']['return'],
        worst_day_date=worst_periods['day']['date'],
        worst_week_return=worst_periods['week']['return'],
        worst_week_date=worst_periods['week']['date'],
        worst_month_return=worst_periods['month']['return'],
        worst_month_date=worst_periods['month']['date'],
        worst_year_return=worst_periods['year']['return'],
        worst_year_date=worst_periods['year']['date'],
        
        # Best periods - New additions
        best_day_return=best_periods['day']['return'],
        best_day_date=best_periods['day']['date'],
        best_week_return=best_periods['week']['return'],
        best_week_date=best_periods['week']['date'],
        best_month_return=best_periods['month']['return'],
        best_month_date=best_periods['month']['date'],
        best_year_return=best_periods['year']['return'],
        best_year_date=best_periods['year']['date'],
        
        # Rolling returns
        rolling_returns_1yr_min=rolling_returns['1Y']['min'],
        rolling_returns_1yr_max=rolling_returns['1Y']['max'],
        rolling_returns_1yr_avg=rolling_returns['1Y']['avg'],
        rolling_returns_3yr_min=rolling_returns.get('3Y', {}).get('min'),
        rolling_returns_3yr_max=rolling_returns.get('3Y', {}).get('max'),
        rolling_returns_3yr_avg=rolling_returns.get('3Y', {}).get('avg'),
        rolling_returns_5yr_min=rolling_returns.get('5Y', {}).get('min'),
        rolling_returns_5yr_max=rolling_returns.get('5Y', {}).get('max'),
        rolling_returns_5yr_avg=rolling_returns.get('5Y', {}).get('avg'),
        rolling_returns_10yr_min=rolling_returns.get('10Y', {}).get('min'),
        rolling_returns_10yr_max=rolling_returns.get('10Y', {}).get('max'),
        rolling_returns_10yr_avg=rolling_returns.get('10Y', {}).get('avg'),
        
        # Monthly/annual returns analysis
        monthly_returns=monthly_returns_analysis,
        annual_returns=annual_returns_analysis,
        
        # Inflation data
        inflation_data=inflation_data,
        inflation_adjusted_returns=inflation_adjusted_returns,
        
        # Benchmark comparison
        benchmark_returns=benchmark_returns,
        benchmark_cagr=benchmark_cagr,
        benchmark_sharpe=benchmark_sharpe,
        benchmark_max_drawdown=benchmark_max_dd,
        
        # Multi-asset analysis
        correlation_matrix=correlation_matrix,
        asset_statistics=asset_statistics,
        
        # Snapshot data
        initial_snapshot=initial_snapshot,
        final_snapshot=final_snapshot
    )
    
    logger.info(f"Backtest completed. Final value: ${final_value:.2f}, CAGR: {cagr:.2%}")
    
    return results