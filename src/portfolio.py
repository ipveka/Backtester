import pandas as pd
import numpy as np
from datetime import datetime

class Portfolio:
    """
    Portfolio class for holding and managing a collection of ETFs with their weights.
    This class serves as the primary container for the portfolio configuration.
    """
    
    def __init__(self):
        """
        Initialize an empty portfolio.
        """
        self.etfs = {}  # Dictionary to store ETFs and their weights
        self.cash = 0   # Cash component (if any)
    
    def add_etf(self, ticker, weight):
        """
        Add an ETF to the portfolio with a specific weight.
        
        Parameters:
        -----------
        ticker : str
            The ticker symbol of the ETF to add
        weight : float
            The weight of the ETF in the portfolio (0-1)
        """
        if weight < 0 or weight > 1:
            raise ValueError("Weight must be between 0 and 1")
        
        # Normalize existing weights if needed
        if ticker not in self.etfs and sum(self.etfs.values()) + weight > 1:
            self._normalize_weights(reserved_weight=weight)
        
        self.etfs[ticker] = weight
    
    def remove_etf(self, ticker):
        """
        Remove an ETF from the portfolio.
        
        Parameters:
        -----------
        ticker : str
            The ticker symbol of the ETF to remove
        """
        if ticker in self.etfs:
            del self.etfs[ticker]
            self._normalize_weights()
    
    def set_cash(self, weight):
        """
        Set the cash component of the portfolio.
        
        Parameters:
        -----------
        weight : float
            The weight of cash in the portfolio (0-1)
        """
        if weight < 0 or weight > 1:
            raise ValueError("Weight must be between 0 and 1")
        
        # Adjust ETF weights
        etf_total_weight = 1 - weight
        old_etf_total = sum(self.etfs.values())
        
        if old_etf_total > 0:
            # Scale all ETF weights proportionally
            for ticker in self.etfs:
                self.etfs[ticker] = self.etfs[ticker] * (etf_total_weight / old_etf_total)
        
        self.cash = weight
    
    def get_weights(self):
        """
        Get the current weights of all assets in the portfolio.
        
        Returns:
        --------
        dict
            Dictionary with tickers as keys and weights as values
        """
        weights = self.etfs.copy()
        if self.cash > 0:
            weights['CASH'] = self.cash
        return weights
    
    def get_etfs(self):
        """
        Get the list of ETF tickers in the portfolio.
        
        Returns:
        --------
        list
            List of ETF ticker strings
        """
        return list(self.etfs.keys())
    
    def is_valid(self):
        """
        Check if the portfolio is valid (weights sum to 1).
        
        Returns:
        --------
        bool
            True if portfolio is valid, False otherwise
        """
        total_weight = sum(self.etfs.values()) + self.cash
        return abs(total_weight - 1.0) < 1e-6  # Allow for small floating-point errors
    
    def _normalize_weights(self, reserved_weight=0):
        """
        Normalize the weights of all ETFs to sum to (1 - reserved_weight - cash).
        
        Parameters:
        -----------
        reserved_weight : float
            Weight to reserve for a new asset
        """
        available_weight = 1 - reserved_weight - self.cash
        current_total = sum(self.etfs.values())
        
        if current_total > 0:
            # Scale existing weights
            for ticker in self.etfs:
                self.etfs[ticker] = self.etfs[ticker] * (available_weight / current_total)
    
    def __str__(self):
        """String representation of the portfolio."""
        portfolio_str = "Portfolio:\n"
        for ticker, weight in self.etfs.items():
            portfolio_str += f"  {ticker}: {weight:.2%}\n"
        
        if self.cash > 0:
            portfolio_str += f"  CASH: {self.cash:.2%}\n"
        
        return portfolio_str


class PortfolioSnapshot:
    """
    Represents a snapshot of a portfolio at a specific point in time,
    with details about holdings, values, and allocations.
    """
    
    def __init__(self, date, etf_holdings, cash=0):
        """
        Initialize a portfolio snapshot.
        
        Parameters:
        -----------
        date : datetime
            The date of the snapshot
        etf_holdings : dict
            Dictionary with ETF tickers as keys and a dict of data for each ETF
            (containing at least 'shares' and 'price')
        cash : float
            The cash amount in the portfolio
        """
        self.date = date
        self.holdings = etf_holdings
        self.cash = cash
        
        # Calculate total value
        self.total_value = cash
        for ticker, data in etf_holdings.items():
            self.total_value += data['shares'] * data['price']
        
        # Calculate allocations
        self.allocations = {}
        for ticker, data in etf_holdings.items():
            value = data['shares'] * data['price']
            self.allocations[ticker] = value / self.total_value if self.total_value > 0 else 0
        
        if cash > 0 and self.total_value > 0:
            self.allocations['CASH'] = cash / self.total_value
    
    def get_value(self):
        """Get the total portfolio value."""
        return self.total_value
    
    def get_holdings_df(self):
        """
        Get the holdings as a DataFrame.
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame with holdings information
        """
        data = []
        for ticker, holding_data in self.holdings.items():
            data.append({
                'Ticker': ticker,
                'Shares': holding_data['shares'],
                'Price': holding_data['price'],
                'Value': holding_data['shares'] * holding_data['price'],
                'Allocation': self.allocations.get(ticker, 0)
            })
        
        if self.cash > 0:
            data.append({
                'Ticker': 'CASH',
                'Shares': None,
                'Price': None,
                'Value': self.cash,
                'Allocation': self.allocations.get('CASH', 0)
            })
        
        return pd.DataFrame(data)
    
    def __str__(self):
        """String representation of the portfolio snapshot."""
        snapshot_str = f"Portfolio Snapshot ({self.date.strftime('%Y-%m-%d')}):\n"
        snapshot_str += f"  Total Value: ${self.total_value:.2f}\n"
        snapshot_str += "  Holdings:\n"
        
        for ticker, data in self.holdings.items():
            value = data['shares'] * data['price']
            snapshot_str += f"    {ticker}: {data['shares']} shares @ ${data['price']:.2f} = ${value:.2f} ({self.allocations.get(ticker, 0):.2%})\n"
        
        if self.cash > 0:
            snapshot_str += f"    CASH: ${self.cash:.2f} ({self.allocations.get('CASH', 0):.2%})\n"
        
        return snapshot_str