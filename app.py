import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import sys

# Add the src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

# Import project modules
from src.data_loader import load_etf_data, load_inflation_data, get_available_etfs, save_etf_metadata, BENCHMARK_TICKERS
from src.backtester import backtest
from src.portfolio import Portfolio
from src.visualization import (
    plot_portfolio_performance,
    plot_annual_returns,
    plot_returns_histogram,
    plot_drawdown_periods,
    plot_monthly_returns_heatmap,
    plot_correlation_matrix
)
from src.report import generate_report

# Set page configuration
st.set_page_config(
    page_title="Backtester - ETF Portfolio Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #333;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f7f7f7;
        border-radius: 5px;
        padding: 1rem;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1E88E5;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
    }
    .positive {
        color: #4CAF50;
    }
    .negative {
        color: #F44336;
    }
    .center-container {
        display: flex;
        justify-content: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar - User inputs
with st.sidebar:
    st.markdown("<h1 style='text-align: center;'>Backtester</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Country selection for inflation data
    countries = ["US", "Germany", "France", "UK", "Spain", "Italy", "Netherlands", "Switzerland"]
    selected_country = st.selectbox("Select Country (for inflation data)", countries)
    
    # Investment parameters
    st.markdown("### Investment Parameters")
    initial_investment = st.number_input("Initial Investment Amount", min_value=100, value=10000, step=1000)
    monthly_contribution = st.number_input("Monthly Contribution (optional)", min_value=0, value=500, step=100)
    
    # Date range selection
    st.markdown("### Backtest Period")
    max_date = datetime.now().date()
    min_date = max_date - timedelta(days=365*15)  # 15 years ago
    
    start_date = st.date_input("Start Date", value=max_date - timedelta(days=365*10), min_value=min_date, max_value=max_date)
    end_date = st.date_input("End Date", value=max_date, min_value=start_date, max_value=max_date)
    
    # Benchmark selection
    st.markdown("### Benchmark")
    benchmark_options = ["S&P 500 (SPY)", "MSCI World (URTH)", "MSCI Europe (IEUR)", "None"]
    selected_benchmark = st.selectbox("Select Benchmark", benchmark_options)
    
    # Submit button
    run_backtest = st.button("Run Backtest", type="primary")

# Main area
st.markdown("<div class='main-header'>ETF Portfolio Backtester</div>", unsafe_allow_html=True)
st.markdown(
    """
    Build and test your ETF portfolio with historical data. Analyze performance metrics,
    visualize returns, and compare against benchmarks and inflation.
    """
)

# Portfolio configuration
st.markdown("<div class='sub-header'>Portfolio Configuration</div>", unsafe_allow_html=True)

# Load available ETFs
available_etfs = get_available_etfs()

# Tabs for US and European ETFs
tab1, tab2 = st.tabs(["US ETFs", "European ETFs"])

with tab1:
    us_etfs = available_etfs[available_etfs['Region'] == 'US']
    
    # Create columns for selection
    col1, col2, col3 = st.columns(3)
    
    selected_us_etfs = []
    etf_weights = {}
    
    with col1:
        st.markdown("### US Equities")
        us_equity_etfs = us_etfs[us_etfs['Category'] == 'Equity']
        for idx, row in us_equity_etfs.iterrows():
            if st.checkbox(f"{row['Ticker']} - {row['Name']}", key=f"us_eq_{row['Ticker']}"):
                selected_us_etfs.append(row['Ticker'])
                etf_weights[row['Ticker']] = 0  # Initialize weight
    
    with col2:
        st.markdown("### US Bonds")
        us_bond_etfs = us_etfs[us_etfs['Category'] == 'Bond']
        for idx, row in us_bond_etfs.iterrows():
            if st.checkbox(f"{row['Ticker']} - {row['Name']}", key=f"us_bond_{row['Ticker']}"):
                selected_us_etfs.append(row['Ticker'])
                etf_weights[row['Ticker']] = 0  # Initialize weight
    
    with col3:
        st.markdown("### US Other")
        us_other_etfs = us_etfs[~us_etfs['Category'].isin(['Equity', 'Bond'])]
        for idx, row in us_other_etfs.iterrows():
            if st.checkbox(f"{row['Ticker']} - {row['Name']}", key=f"us_other_{row['Ticker']}"):
                selected_us_etfs.append(row['Ticker'])
                etf_weights[row['Ticker']] = 0  # Initialize weight

with tab2:
    eu_etfs = available_etfs[available_etfs['Region'] == 'EU']
    
    # Create columns for selection
    col1, col2, col3 = st.columns(3)
    
    selected_eu_etfs = []
    
    with col1:
        st.markdown("### European Equities")
        eu_equity_etfs = eu_etfs[eu_etfs['Category'] == 'Equity']
        for idx, row in eu_equity_etfs.iterrows():
            if st.checkbox(f"{row['Ticker']} - {row['Name']}", key=f"eu_eq_{row['Ticker']}"):
                selected_eu_etfs.append(row['Ticker'])
                etf_weights[row['Ticker']] = 0  # Initialize weight
    
    with col2:
        st.markdown("### European Bonds")
        eu_bond_etfs = eu_etfs[eu_etfs['Category'] == 'Bond']
        for idx, row in eu_bond_etfs.iterrows():
            if st.checkbox(f"{row['Ticker']} - {row['Name']}", key=f"eu_bond_{row['Ticker']}"):
                selected_eu_etfs.append(row['Ticker'])
                etf_weights[row['Ticker']] = 0  # Initialize weight
    
    with col3:
        st.markdown("### European Other")
        eu_other_etfs = eu_etfs[~eu_etfs['Category'].isin(['Equity', 'Bond'])]
        for idx, row in eu_other_etfs.iterrows():
            if st.checkbox(f"{row['Ticker']} - {row['Name']}", key=f"eu_other_{row['Ticker']}"):
                selected_eu_etfs.append(row['Ticker'])
                etf_weights[row['Ticker']] = 0  # Initialize weight

# Combine all selected ETFs
selected_etfs = selected_us_etfs + selected_eu_etfs

if not selected_etfs:
    st.warning("Please select at least one ETF to continue.")
else:
    # Configure weights
    st.markdown("<div class='sub-header'>Portfolio Weights</div>", unsafe_allow_html=True)
    st.markdown("Assign percentage weights to your selected ETFs (total must sum to 100%).")
    
    # Setup columns for better layout
    col1, col2, col3 = st.columns(3)
    for i, ticker in enumerate(selected_etfs):
        with col1 if i % 3 == 0 else col2 if i % 3 == 1 else col3:
            etf_info = available_etfs[available_etfs['Ticker'] == ticker].iloc[0]
            etf_weights[ticker] = st.slider(
                f"{ticker} - {etf_info['Name']}",
                min_value=0,
                max_value=100,
                value=round(100 / len(selected_etfs)),  # Equal weight by default
                step=1,
                key=f"weight_{ticker}"
            )
    
    # Check if weights sum to 100%
    total_weight = sum(etf_weights.values())
    if total_weight != 100:
        st.warning(f"Total weight is {total_weight}%. Please adjust to sum to 100%.")
    
    # Display the portfolio composition in a pie chart
    if selected_etfs:
        st.markdown("<h3 style='text-align: center;'>Portfolio Composition</h3>", unsafe_allow_html=True)
        
        # Create a larger, centered pie chart
        fig = px.pie(
            values=[etf_weights[ticker] for ticker in selected_etfs],
            names=selected_etfs,
            color_discrete_sequence=px.colors.qualitative.Set3,
            height=500,  # Make it taller
            width=800,   # Make it wider
        )
        
        # Update layout for a cleaner look
        fig.update_layout(
            margin=dict(t=0, b=0, l=0, r=0),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.1,
                xanchor="center",
                x=0.5
            )
        )
        
        # Center the chart using columns
        col1, col2, col3 = st.columns([1, 10, 1])
        with col2:
            st.plotly_chart(fig, use_container_width=True)

# Main functionality - Run backtest when button is clicked
if run_backtest and selected_etfs and total_weight == 100:
    with st.spinner("Running backtest... This may take a moment."):
        try:
            # Create portfolio object
            portfolio = Portfolio()
            for ticker in selected_etfs:
                portfolio.add_etf(ticker, weight=etf_weights[ticker] / 100)
            
            # Get the actual ticker for the benchmark
            benchmark_ticker = BENCHMARK_TICKERS.get(selected_benchmark)
            
            # Run backtest
            results = backtest(
                portfolio=portfolio,
                initial_investment=initial_investment,
                monthly_contribution=monthly_contribution,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
                country=selected_country,
                benchmark=benchmark_ticker
            )
            
            # Store results in session state for report generation
            st.session_state['backtest_results'] = results
            st.session_state['portfolio_config'] = {
                'etfs': selected_etfs,
                'weights': etf_weights,
                'initial_investment': initial_investment,
                'monthly_contribution': monthly_contribution,
                'start_date': start_date,
                'end_date': end_date,
                'country': selected_country,
                'benchmark': benchmark_ticker
            }
            
            # Display results
            st.markdown("<div class='sub-header'>Backtest Results</div>", unsafe_allow_html=True)
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-value'>${results.final_value:,.2f}</div>", unsafe_allow_html=True)
                st.markdown("<div class='metric-label'>Final Value</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                cagr_class = "positive" if results.cagr > 0 else "negative"
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-value {cagr_class}'>{results.cagr:.2%}</div>", unsafe_allow_html=True)
                st.markdown("<div class='metric-label'>CAGR</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col3:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-value'>{results.sharpe_ratio:.2f}</div>", unsafe_allow_html=True)
                st.markdown("<div class='metric-label'>Sharpe Ratio</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col4:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-value negative'>{results.max_drawdown:.2%}</div>", unsafe_allow_html=True)
                st.markdown("<div class='metric-label'>Max Drawdown</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Performance chart
            st.markdown("### Portfolio Performance")
            performance_chart = plot_portfolio_performance(results)
            st.plotly_chart(performance_chart, use_container_width=True)
            
            # Additional metrics in tabs
            tab1, tab2, tab3, tab4 = st.tabs(["Annual Returns", "Risk Metrics", "Return Distribution", "Correlation"])
            
            with tab1:
                # Annual returns bar chart - vertical layout, no monthly returns
                annual_returns_chart = plot_annual_returns(results)
                st.plotly_chart(annual_returns_chart, use_container_width=True)
            
            with tab2:
                # Risk metrics table
                st.markdown("### Risk Metrics")
                risk_data = {
                    "Metric": [
                        "Value at Risk (95%)",
                        "Value at Risk (99%)",
                        "Expected Shortfall (95%)",
                        "Maximum Drawdown",
                        "Longest Drawdown Duration",
                        "Annual Volatility"
                    ],
                    "Value": [
                        f"{results.var_95:.2%}" if hasattr(results, 'var_95') else "N/A",
                        f"{results.var_99:.2%}" if hasattr(results, 'var_99') else "N/A",
                        f"{results.expected_shortfall:.2%}" if hasattr(results, 'expected_shortfall') else "N/A",
                        f"{results.max_drawdown:.2%}" if hasattr(results, 'max_drawdown') else "N/A",
                        f"{results.longest_drawdown_days} days" if hasattr(results, 'longest_drawdown_days') else "N/A",
                        f"{results.volatility:.2%}" if hasattr(results, 'volatility') else "N/A"
                    ]
                }
                st.table(pd.DataFrame(risk_data))
                
                # Worst periods table
                st.markdown("### Worst Periods")
                worst_periods = pd.DataFrame({
                    "Period": ["Worst Day", "Worst Week", "Worst Month", "Worst Year"],
                    "Return": [
                        f"{results.worst_day_return:.2%}" if hasattr(results, 'worst_day_return') else "N/A",
                        f"{results.worst_week_return:.2%}" if hasattr(results, 'worst_week_return') else "N/A",
                        f"{results.worst_month_return:.2%}" if hasattr(results, 'worst_month_return') else "N/A",
                        f"{results.worst_year_return:.2%}" if hasattr(results, 'worst_year_return') else "N/A"
                    ],
                    "Date": [
                        results.worst_day_date.strftime("%Y-%m-%d") if hasattr(results, 'worst_day_date') and results.worst_day_date else "N/A",
                        results.worst_week_date.strftime("%Y-%m-%d") if hasattr(results, 'worst_week_date') and results.worst_week_date else "N/A",
                        results.worst_month_date.strftime("%Y-%m-%d") if hasattr(results, 'worst_month_date') and results.worst_month_date else "N/A",
                        results.worst_year_date.strftime("%Y") if hasattr(results, 'worst_year_date') and results.worst_year_date else "N/A"
                    ]
                })
                st.table(worst_periods)
                
                # Best periods table
                st.markdown("### Best Periods")
                if hasattr(results, 'best_day_return'):
                    best_periods = pd.DataFrame({
                        "Period": ["Best Day", "Best Week", "Best Month", "Best Year"],
                        "Return": [
                            f"{results.best_day_return:.2%}" if hasattr(results, 'best_day_return') else "N/A",
                            f"{results.best_week_return:.2%}" if hasattr(results, 'best_week_return') else "N/A",
                            f"{results.best_month_return:.2%}" if hasattr(results, 'best_month_return') else "N/A",
                            f"{results.best_year_return:.2%}" if hasattr(results, 'best_year_return') else "N/A"
                        ],
                        "Date": [
                            results.best_day_date.strftime("%Y-%m-%d") if hasattr(results, 'best_day_date') and results.best_day_date else "N/A",
                            results.best_week_date.strftime("%Y-%m-%d") if hasattr(results, 'best_week_date') and results.best_week_date else "N/A",
                            results.best_month_date.strftime("%Y-%m-%d") if hasattr(results, 'best_month_date') and results.best_month_date else "N/A",
                            results.best_year_date.strftime("%Y") if hasattr(results, 'best_year_date') and results.best_year_date else "N/A"
                        ]
                    })
                    st.table(best_periods)
                else:
                    # Create a placeholder with sample data
                    best_periods = pd.DataFrame({
                        "Period": ["Best Day", "Best Week", "Best Month", "Best Year"],
                        "Return": ["N/A", "N/A", "N/A", "N/A"],
                        "Date": ["N/A", "N/A", "N/A", "N/A"]
                    })
                    st.table(best_periods)
                    st.info("Add 'analyze_best_periods' to the risk.py module to show actual best performance periods.")
            
            with tab3:
                # Simple Returns histogram
                st.markdown("### Return Distribution")
                
                if 'Daily Returns' in results.returns_df:
                    # Simple histogram with Plotly
                    returns = results.returns_df['Daily Returns'].dropna() * 100  # Convert to percentage
                    
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(
                        x=returns,
                        nbinsx=30,
                        marker_color='rgba(30, 136, 229, 0.6)',
                        marker_line_color='rgba(30, 136, 229, 1)',
                        marker_line_width=1
                    ))
                    
                    # Update layout for a cleaner look
                    fig.update_layout(
                        title='Daily Returns Distribution',
                        xaxis_title='Daily Return (%)',
                        yaxis_title='Frequency',
                        template='plotly_white',
                        height=400
                    )
                    
                    # Add a vertical line for the mean
                    mean_return = returns.mean()
                    fig.add_shape(
                        type="line",
                        x0=mean_return,
                        y0=0,
                        x1=mean_return,
                        y1=1,
                        yref="paper",
                        line=dict(color="rgba(0, 0, 0, 0.7)", width=2, dash="dash")
                    )
                    
                    # Add annotation for the mean
                    fig.add_annotation(
                        x=mean_return,
                        y=0.95,
                        yref="paper",
                        text=f"Mean: {mean_return:.2f}%",
                        showarrow=True,
                        arrowhead=2,
                        arrowcolor="rgba(0, 0, 0, 0.7)",
                        arrowsize=1,
                        arrowwidth=2,
                        bgcolor="white",
                        bordercolor="black",
                        borderwidth=1,
                        borderpad=4,
                        opacity=0.8
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Return data not available")
                
                # Return statistics table
                st.markdown("### Return Statistics")
                returns_stats = pd.DataFrame({
                    "Statistic": [
                        "Mean Daily Return",
                        "Median Daily Return",
                        "Daily Return Std. Dev.",
                        "Skewness",
                        "Kurtosis",
                        "Positive Days (%)",
                        "Negative Days (%)"
                    ],
                    "Value": [
                        f"{results.mean_return:.4%}" if hasattr(results, 'mean_return') else "N/A",
                        f"{results.median_return:.4%}" if hasattr(results, 'median_return') else "N/A",
                        f"{results.return_std:.4%}" if hasattr(results, 'return_std') else "N/A",
                        f"{results.skewness:.4f}" if hasattr(results, 'skewness') else "N/A",
                        f"{results.kurtosis:.4f}" if hasattr(results, 'kurtosis') else "N/A",
                        f"{results.positive_days_pct:.2%}" if hasattr(results, 'positive_days_pct') else "N/A",
                        f"{results.negative_days_pct:.2%}" if hasattr(results, 'negative_days_pct') else "N/A"
                    ]
                })
                st.table(returns_stats)
                
            with tab4:
                # Correlation matrix
                if hasattr(results, 'correlation_matrix') and results.correlation_matrix is not None:
                    st.markdown("### Correlation Matrix")
                    correlation_chart = plot_correlation_matrix(results)
                    st.plotly_chart(correlation_chart, use_container_width=True)
                    
                    # Asset statistics
                    st.markdown("### Asset Performance")
                    if hasattr(results, 'asset_statistics') and results.asset_statistics is not None:
                        st.table(results.asset_statistics)
                    else:
                        st.info("Asset statistics not available.")
                else:
                    st.info("Correlation data only available for portfolios with multiple assets.")
            
            # Download report button
            st.markdown("### Download Report")
            if st.button("Generate PDF Report"):
                with st.spinner("Generating report..."):
                    report_path = generate_report(results, st.session_state['portfolio_config'])
                    
                    # Create download button for the report
                    with open(report_path, "rb") as file:
                        btn = st.download_button(
                            label="Download PDF Report",
                            data=file,
                            file_name="backtester_report.pdf",
                            mime="application/pdf"
                        )
                    
                    # Remove the temporary file
                    os.remove(report_path)
        
        except Exception as e:
            st.error(f"An error occurred during the backtest: {str(e)}")
            import traceback
            st.error(traceback.format_exc())

if __name__ == "__main__":
    st.set_option('deprecation.showPyplotGlobalUse', False)
    save_etf_metadata()  # Ensure ETF metadata is created on first run