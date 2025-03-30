import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the modules to test
from src.analysis import (
    calculate_cagr,
    calculate_volatility,
    calculate_sharpe_ratio,
    calculate_drawdowns
)


class TestAnalysis(unittest.TestCase):
    """
    Test the financial analysis functions.
    """
    
    def setUp(self):
        """
        Set up test data.
        """
        # Create a test price series
        dates = pd.date_range(start='2020-01-01', end='2022-12-31', freq='B')
        
        # Create prices with some realistic volatility (starting at 100)
        np.random.seed(42)  # For reproducibility
        returns = np.random.normal(0.0004, 0.01, len(dates))  # Positive drift
        prices = 100 * (1 + returns).cumprod()
        
        self.prices = pd.Series(prices, index=dates)
        self.returns = self.prices.pct_change().dropna()
    
    def test_calculate_cagr(self):
        """
        Test the CAGR calculation.
        """
        cagr = calculate_cagr(self.prices)
        
        # The CAGR should be positive (given our positive drift in returns)
        self.assertGreater(cagr, 0)
        
        # Test with a known example
        fixed_prices = pd.Series(
            [100, 110, 121, 133.1],
            index=pd.date_range(start='2020-01-01', periods=4, freq='365D')
        )
        expected_cagr = 0.10  # 10% annual growth
        self.assertAlmostEqual(calculate_cagr(fixed_prices), expected_cagr, places=2)
    
    def test_calculate_volatility(self):
        """
        Test the volatility calculation.
        """
        volatility = calculate_volatility(self.returns)
        
        # Volatility should be positive
        self.assertGreater(volatility, 0)
        
        # Test with a known example - using the actual calculation from our code
        fixed_returns = np.array([0.01, -0.01, 0.02, -0.02, 0.01])
        expected_vol = np.std(fixed_returns, ddof=1) * np.sqrt(252)
        self.assertAlmostEqual(calculate_volatility(fixed_returns), expected_vol)
    
    def test_calculate_sharpe_ratio(self):
        """
        Test the Sharpe ratio calculation.
        """
        sharpe = calculate_sharpe_ratio(self.returns, risk_free_rate=0.02)
        
        # Should return a finite value
        self.assertTrue(np.isfinite(sharpe))
        
        # Test with a known example - using the actual calculation from our code
        fixed_returns = np.array([0.01, -0.01, 0.02, -0.02, 0.01])
        mean_return = np.mean(fixed_returns) * 252
        std_return = np.std(fixed_returns, ddof=1) * np.sqrt(252)
        expected_sharpe = (mean_return - 0.02) / std_return
        
        self.assertAlmostEqual(
            calculate_sharpe_ratio(fixed_returns, risk_free_rate=0.02),
            expected_sharpe
        )
    
    def test_calculate_drawdowns(self):
        """
        Test the drawdown calculation.
        """
        drawdowns, max_dd, longest_dd = calculate_drawdowns(self.returns)
        
        # Drawdowns should be non-positive
        self.assertLessEqual(max_dd, 0)
        
        # Max drawdown should be the minimum value in the drawdown series
        self.assertAlmostEqual(max_dd, drawdowns.min())
        
        # Test with a known drawdown scenario
        fixed_prices = pd.Series(
            [100, 110, 105, 95, 90, 95, 100, 110],
            index=pd.date_range(start='2020-01-01', periods=8, freq='D')
        )
        fixed_returns = fixed_prices.pct_change().dropna()
        
        # Calculated manually: The max drawdown should be (90-110)/110 = -0.1818...
        expected_max_dd = -0.1818
        _, max_dd, _ = calculate_drawdowns(fixed_returns)
        
        self.assertAlmostEqual(max_dd, expected_max_dd, places=4)


if __name__ == '__main__':
    unittest.main()