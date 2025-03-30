import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from typing import Optional, Dict, Any, List
import calendar

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def plot_portfolio_performance(results: Any) -> go.Figure:
    """
    Plot portfolio performance over time.
    
    Parameters:
    -----------
    results : BacktestResults
        Backtest results object
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    # Create the base figure
    fig = go.Figure()
    
    # Plot portfolio value
    fig.add_trace(go.Scatter(
        x=results.portfolio_value_history.index,
        y=results.portfolio_value_history.values,
        mode='lines',
        name='Portfolio Value',
        line=dict(color='#1E88E5', width=2)
    ))
    
    # Add total contributions line
    if results.monthly_contribution > 0:
        # Calculate cumulative contributions
        dates = results.portfolio_value_history.index
        months_passed = [(d.year - dates[0].year) * 12 + d.month - dates[0].month for d in dates]
        contributions = [results.initial_investment + results.monthly_contribution * max(0, m) for m in months_passed]
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=contributions,
            mode='lines',
            name='Total Contributions',
            line=dict(color='#4CAF50', width=2, dash='dash')
        ))
    
    # Add benchmark if available
    if hasattr(results, 'benchmark_returns') and results.benchmark_returns is not None:
        # Normalize benchmark to match the initial investment
        benchmark_values = results.initial_investment * (1 + results.benchmark_returns.iloc[:, 0].cumsum())
        
        fig.add_trace(go.Scatter(
            x=benchmark_values.index,
            y=benchmark_values.values,
            mode='lines',
            name=f'Benchmark ({results.benchmark})',
            line=dict(color='#FF9800', width=2, dash='dot')
        ))
    
    # Customize the layout
    fig.update_layout(
        title='Portfolio Performance',
        xaxis=dict(title='Date'),
        yaxis=dict(title='Value ($)', gridcolor='lightgray'),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=10, r=10, t=40, b=10),
        hovermode='x unified',
        template='plotly_white'
    )
    
    # Add annotations for key metrics
    fig.add_annotation(
        x=0.01,
        y=0.98,
        xref="paper",
        yref="paper",
        text=f"Final Value: ${results.final_value:.2f}<br>CAGR: {results.cagr:.2%}<br>Sharpe: {results.sharpe_ratio:.2f}",
        showarrow=False,
        font=dict(size=12),
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="gray",
        borderwidth=1,
        borderpad=4,
        align="left"
    )
    
    return fig


def plot_annual_returns(results: Any) -> go.Figure:
    """
    Plot annual returns as a bar chart.
    
    Parameters:
    -----------
    results : BacktestResults
        Backtest results object
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    if not hasattr(results, 'annual_returns') or results.annual_returns is None or len(results.annual_returns) == 0:
        # Create an empty figure with a message
        fig = go.Figure()
        fig.add_annotation(
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            text="Not enough data to calculate annual returns",
            showarrow=False,
            font=dict(size=14)
        )
        return fig
    
    # Format annual returns for plotting
    annual_returns = results.annual_returns.copy()
    annual_returns.index = annual_returns.index.strftime('%Y')
    
    # Create the figure
    fig = go.Figure()
    
    # Add the bars with colors based on positive/negative
    colors = ['#4CAF50' if x >= 0 else '#F44336' for x in annual_returns.values]
    
    fig.add_trace(go.Bar(
        x=annual_returns.index,
        y=annual_returns.values * 100,  # Convert to percentage
        marker_color=colors,
        name='Annual Returns'
    ))
    
    # Add a horizontal line at y=0
    fig.add_shape(
        type="line",
        x0=-0.5,
        y0=0,
        x1=len(annual_returns) - 0.5,
        y1=0,
        line=dict(color="black", width=1.5, dash="solid")
    )
    
    # Add average annual return
    avg_return = annual_returns.mean() * 100
    fig.add_trace(go.Scatter(
        x=annual_returns.index,
        y=[avg_return] * len(annual_returns),
        mode='lines',
        name=f'Average ({avg_return:.2f}%)',
        line=dict(color='rgba(0, 0, 0, 0.7)', width=2, dash='dash')
    ))
    
    # Customize the layout
    fig.update_layout(
        title='Annual Returns',
        xaxis=dict(title='Year', tickangle=45),
        yaxis=dict(title='Return (%)', gridcolor='lightgray'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=40, b=10),
        hovermode='x unified',
        template='plotly_white',
        height=500  # Make it taller for better visibility
    )
    
    return fig


def plot_returns_histogram(results: Any) -> go.Figure:
    """
    Plot a simpler histogram of daily returns.
    
    Parameters:
    -----------
    results : BacktestResults
        Backtest results object
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    returns = results.returns_df['Daily Returns'].dropna() * 100  # Convert to percentage
    
    # Create a more straightforward histogram
    fig = go.Figure()
    
    # Add the histogram with nicer styling
    fig.add_trace(go.Histogram(
        x=returns,
        nbinsx=30,
        marker=dict(
            color='rgba(30, 136, 229, 0.7)',
            line=dict(color='rgba(30, 136, 229, 1)', width=1)
        ),
        name='Daily Returns'
    ))
    
    # Customize the layout
    fig.update_layout(
        title='Return Distribution',
        xaxis=dict(
            title='Daily Return (%)',
            gridcolor='lightgray',
            zeroline=True,
            zerolinecolor='black',
            zerolinewidth=1.5
        ),
        yaxis=dict(
            title='Frequency',
            gridcolor='lightgray'
        ),
        margin=dict(l=10, r=10, t=50, b=10),
        bargap=0.05,
        template='plotly_white',
        height=500  # Make it taller for better visibility
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
    
    return fig


def plot_drawdown_periods(results: Any) -> go.Figure:
    """
    Plot drawdown periods.
    
    Parameters:
    -----------
    results : BacktestResults
        Backtest results object
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    drawdowns = results.drawdown_series.copy() * 100  # Convert to percentage
    
    # Create the figure
    fig = go.Figure()
    
    # Add the drawdowns trace
    fig.add_trace(go.Scatter(
        x=drawdowns.index,
        y=drawdowns.values,
        mode='lines',
        name='Drawdowns',
        line=dict(color='#F44336', width=2),
        fill='tozeroy',
        fillcolor='rgba(244, 67, 54, 0.3)'
    ))
    
    # Add a horizontal line at y=0
    fig.add_shape(
        type="line",
        x0=drawdowns.index[0],
        y0=0,
        x1=drawdowns.index[-1],
        y1=0,
        line=dict(color="black", width=1.5, dash="solid")
    )
    
    # Customize the layout
    fig.update_layout(
        title='Drawdown Periods',
        xaxis=dict(title='Date', gridcolor='lightgray'),
        yaxis=dict(title='Drawdown (%)', gridcolor='lightgray', autorange="reversed"),  # Reverse to show downside
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=40, b=10),
        hovermode='x unified',
        template='plotly_white'
    )
    
    # Add annotation for maximum drawdown
    fig.add_annotation(
        x=0.01,
        y=0.01,
        xref="paper",
        yref="paper",
        text=f"Max Drawdown: {results.max_drawdown:.2%}<br>Longest Drawdown: {results.longest_drawdown_days} days",
        showarrow=False,
        font=dict(size=12),
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="gray",
        borderwidth=1,
        borderpad=4,
        align="left"
    )
    
    return fig


def plot_monthly_returns_heatmap(results: Any) -> go.Figure:
    """
    Plot monthly returns as a heatmap.
    
    Parameters:
    -----------
    results : BacktestResults
        Backtest results object
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    if not hasattr(results, 'monthly_returns') or results.monthly_returns is None or results.monthly_returns.empty:
        # Create an empty figure with a message
        fig = go.Figure()
        fig.add_annotation(
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            text="Not enough data to calculate monthly returns",
            showarrow=False,
            font=dict(size=14)
        )
        return fig
    
    # Get the monthly returns data
    monthly_data = results.monthly_returns.copy() * 100  # Convert to percentage
    
    # Replace column numbers with month names
    month_names = {
        1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
        7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
    }
    monthly_data.columns = [month_names.get(col, col) for col in monthly_data.columns]
    
    # Create the heatmap
    fig = go.Figure(data=go.Heatmap(
        z=monthly_data.values,
        x=monthly_data.columns,
        y=monthly_data.index,
        colorscale=[
            [0, 'rgb(178, 24, 43)'],      # Dark red for strong negative
            [0.25, 'rgb(244, 67, 54)'],   # Red for negative
            [0.45, 'rgb(252, 180, 162)'],  # Light red for slight negative
            [0.5, 'rgb(247, 247, 247)'],   # White for zero
            [0.55, 'rgb(186, 228, 188)'],  # Light green for slight positive
            [0.75, 'rgb(76, 175, 80)'],    # Green for positive
            [1, 'rgb(0, 100, 0)']          # Dark green for strong positive
        ],
        colorbar=dict(
            title='Return (%)',
            title_side='right',  # Changed from titleside to title_side
            thickness=15
        ),
        hovertemplate='Year: %{y}<br>Month: %{x}<br>Return: %{z:.2f}%<extra></extra>'
    ))
    
    # Customize the layout
    fig.update_layout(
        title='Monthly Returns Heatmap',
        xaxis=dict(
            title='Month',
            side='top'  # Put month labels on top
        ),
        yaxis=dict(
            title='Year',
            autorange='reversed'  # Latest year at the top
        ),
        margin=dict(l=10, r=10, t=60, b=10),
        template='plotly_white'
    )
    
    return fig


def plot_correlation_matrix(results: Any) -> go.Figure:
    """
    Plot correlation matrix of ETFs in the portfolio.
    
    Parameters:
    -----------
    results : BacktestResults
        Backtest results object
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    if not hasattr(results, 'correlation_matrix') or results.correlation_matrix is None or results.correlation_matrix.empty:
        # Create an empty figure with a message
        fig = go.Figure()
        fig.add_annotation(
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            text="Correlation matrix not available",
            showarrow=False,
            font=dict(size=14)
        )
        return fig
    
    # Create the heatmap
    corr_matrix = results.correlation_matrix.copy()
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale=[
            [0, 'rgb(178, 24, 43)'],      # Dark red for strong negative correlation
            [0.25, 'rgb(244, 67, 54)'],   # Red for negative correlation
            [0.45, 'rgb(252, 180, 162)'],  # Light red for slight negative correlation
            [0.5, 'rgb(247, 247, 247)'],   # White for zero correlation
            [0.55, 'rgb(186, 228, 188)'],  # Light green for slight positive correlation
            [0.75, 'rgb(76, 175, 80)'],    # Green for positive correlation
            [1, 'rgb(0, 100, 0)']          # Dark green for strong positive correlation
        ],
        colorbar=dict(
            title='Correlation',
            title_side='right',  # Changed from titleside to title_side
            thickness=15
        ),
        hovertemplate='%{x} & %{y}<br>Correlation: %{z:.2f}<extra></extra>'
    ))
    
    # Add text annotations for correlation values
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.index)):
            fig.add_annotation(
                x=corr_matrix.columns[i],
                y=corr_matrix.index[j],
                text=str(round(corr_matrix.iloc[j, i], 2)),
                showarrow=False,
                font=dict(
                    color='black' if abs(corr_matrix.iloc[j, i]) < 0.7 else 'white'
                )
            )
    
    # Customize the layout
    fig.update_layout(
        title='Correlation Matrix',
        margin=dict(l=10, r=10, t=40, b=10),
        template='plotly_white',
        height=600,  # Make it taller for better visibility with multiple assets
        width=700    # Make it wider for better visibility with multiple assets
    )
    
    return fig