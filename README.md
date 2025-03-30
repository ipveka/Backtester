# Backtester: Advanced ETF Portfolio Backtesting Platform

Backtester is a powerful Streamlit-based application that allows investors to backtest ETF portfolios with sophisticated analytics, visualizations, and reporting capabilities.

## 🌟 Features

- **Multi-region ETF support**: Backtest portfolios with ETFs from US and European markets
- **Custom portfolio construction**: Create personalized ETF portfolios with custom weights
- **Flexible investment options**: Model one-time investments or regular contributions
- **Comprehensive analytics**: Calculate CAGR, Sharpe ratio, maximum drawdown, Value at Risk, and more
- **Inflation adjustment**: Compare returns against country-specific inflation rates
- **Interactive visualizations**: View portfolio performance with detailed charts and graphs
- **Benchmarking**: Compare your portfolio against other ETFs or indices
- **Exportable reports**: Generate and download detailed PDF reports
- **User-friendly interface**: Intuitive Streamlit UI for seamless user experience

## 📋 Prerequisites

- Python 3.9 or higher
- Git (for cloning the repository)

## 🚀 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/backtester.git
   cd backtester
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Open your web browser and navigate to the URL shown in your terminal (typically http://localhost:8501)

3. Follow the steps in the application:
   - Select your country (for inflation calculations)
   - Set your initial investment amount and monthly contribution (if any)
   - Choose ETFs from the available list
   - Assign weights to each ETF
   - Define the backtesting period
   - View and analyze the results
   - Download a detailed report (optional)

## 📊 Example

```python
# Example code demonstrating how to use the backtester module directly
from src.backtester import backtest
from src.portfolio import Portfolio

# Create a portfolio
portfolio = Portfolio()
portfolio.add_etf("SPY", weight=0.6)
portfolio.add_etf("VEUR", weight=0.4)

# Run backtest
results = backtest(
    portfolio=portfolio,
    initial_investment=10000,
    monthly_contribution=500,
    start_date="2015-01-01",
    end_date="2023-12-31",
    country="US"
)

# Print results
print(f"Final Portfolio Value: ${results.final_value:.2f}")
print(f"CAGR: {results.cagr:.2%}")
print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
```

## 📁 Project Structure

```
backtester/
│
├── app.py                       # Main Streamlit application entry point
├── README.md                    # Project documentation and setup instructions
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore file
│
├── assets/                      # Static assets
│   ├── css/                     # Custom CSS styles
│   └── images/                  # Images for the app
│
├── data/                        # Data storage
│   ├── etf_metadata.csv         # ETF database with tickers, names, regions
│   └── inflation/               # Inflation data by country
│
├── src/                         # Source code
│   ├── __init__.py              # Package initialization
│   ├── data_loader.py           # ETF and inflation data loading functions
│   ├── backtester.py            # Core backtesting logic
│   ├── portfolio.py             # Portfolio construction and analysis
│   ├── analysis.py              # Financial metrics calculation (CAGR, Sharpe, etc.)
│   ├── risk.py                  # Risk metrics (VaR, drawdowns, etc.)
│   ├── visualization.py         # Charts and visualization functions
│   └── report.py                # PDF report generation
│
└── tests/                       # Unit tests
```

## 🧪 Testing

Run tests with pytest:
```bash
pytest
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📧 Contact

Your Name - your.email@example.com

Project Link: [https://github.com/yourusername/backtester](https://github.com/yourusername/backtester)