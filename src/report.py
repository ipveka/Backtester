import os
import pandas as pd
import numpy as np
import logging
import tempfile
from datetime import datetime
from typing import Dict, Any, List
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.linecharts import HorizontalLineChart
from reportlab.graphics.charts.piecharts import Pie
from reportlab.lib.units import inch
import io
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_report(results: Any, config: Dict[str, Any]) -> str:
    """
    Generate a PDF report of backtest results.
    
    Parameters:
    -----------
    results : BacktestResults
        Backtest results object
    config : Dict[str, Any]
        Configuration used for the backtest
    
    Returns:
    --------
    str
        Path to the generated PDF file
    """
    logger.info("Generating backtest report...")
    
    # Create a temporary file for the report
    fd, temp_path = tempfile.mkstemp(suffix='.pdf')
    os.close(fd)
    
    # Create document
    doc = SimpleDocTemplate(
        temp_path,
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )
    
    # Get the default styles
    styles = getSampleStyleSheet()
    
    # Create custom styles
    title_style = ParagraphStyle(
        'Title',
        parent=styles['Title'],
        fontSize=20,
        leading=24,
        textColor=colors.blue
    )
    
    heading1_style = ParagraphStyle(
        'Heading1',
        parent=styles['Heading1'],
        fontSize=16,
        leading=20,
        spaceBefore=12,
        spaceAfter=6
    )
    
    heading2_style = ParagraphStyle(
        'Heading2',
        parent=styles['Heading2'],
        fontSize=14,
        leading=18,
        spaceBefore=10,
        spaceAfter=6
    )
    
    normal_style = styles["Normal"]
    
    # Create the story (list of elements to add to the document)
    story = []
    
    # Add title
    title_text = "ETF Portfolio Backtest Report"
    story.append(Paragraph(title_text, title_style))
    story.append(Spacer(1, 12))
    
    # Add date
    date_text = f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    story.append(Paragraph(date_text, normal_style))
    story.append(Spacer(1, 24))
    
    # Add summary section
    story.append(Paragraph("Portfolio Summary", heading1_style))
    
    # Portfolio composition
    story.append(Paragraph("Portfolio Composition", heading2_style))
    
    # Create table of ETFs and weights
    etf_data = []
    etf_data.append(["ETF", "Weight"])
    for ticker, weight in zip(results.etfs, [results.weights[t] for t in results.etfs]):
        etf_data.append([ticker, f"{weight:.2%}"])
    
    # Create table
    table = Table(etf_data, colWidths=[2*inch, 1*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('ALIGN', (1, 1), (1, -1), 'RIGHT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('GRID', (0, 0), (-1, -1), 1, colors.lightgrey)
    ]))
    
    # Portfolio diversification section (if multiple ETFs)
    if len(results.etfs) > 1 and hasattr(results, 'correlation_matrix') and results.correlation_matrix is not None:
        story.append(Paragraph("Portfolio Diversification", heading1_style))
        
        # Correlation matrix
        story.append(Paragraph("Correlation Matrix", heading2_style))
        
        # Create matplotlib figure for correlation matrix
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111)
        
        # Plot correlation heatmap
        sns.heatmap(results.correlation_matrix, annot=True, cmap='RdYlGn', vmin=-1, vmax=1, ax=ax, 
                    annot_kws={"size": 8}, fmt='.2f', linewidths=0.5)
        
        ax.set_title('ETF Correlation Matrix')
        
        # Save figure to memory
        buffer = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buffer, format='png', dpi=100)
        buffer.seek(0)
        
        # Create ReportLab Image
        img = Image(buffer)
        img.drawHeight = 5 * inch
        img.drawWidth = 6 * inch
        
        story.append(img)
        story.append(Spacer(1, 12))
        
        # Asset statistics table
        if hasattr(results, 'asset_statistics') and results.asset_statistics is not None:
            story.append(Paragraph("Asset Performance Metrics", heading2_style))
            
            # Prepare data
            asset_stats = results.asset_statistics.copy()
            asset_stats = asset_stats.reset_index()
            asset_stats.columns = ['ETF'] + list(asset_stats.columns[1:])
            
            # Convert to list for ReportLab table
            stat_data = [list(asset_stats.columns)]
            for _, row in asset_stats.iterrows():
                formatted_row = [row['ETF']]
                for col in asset_stats.columns[1:]:
                    if 'Return' in col or 'Drawdown' in col:
                        formatted_row.append(f"{row[col]:.2%}")
                    else:
                        formatted_row.append(f"{row[col]:.2f}")
                stat_data.append(formatted_row)
            
            # Create table
            table = Table(stat_data, colWidths=[1.0*inch] + [1.2*inch] * (len(stat_data[0]) - 1))
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                ('ALIGN', (1, 1), (-1, -1), 'RIGHT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                ('GRID', (0, 0), (-1, -1), 1, colors.lightgrey)
            ]))
            
            story.append(table)
            story.append(Spacer(1, 24))
    
    # Inflation comparison (if available)
    if hasattr(results, 'inflation_adjusted_returns') and results.inflation_adjusted_returns is not None:
        story.append(Paragraph("Inflation Comparison", heading1_style))
        
        # Create matplotlib figure for inflation comparison
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111)
        
        # Plot nominal and inflation-adjusted returns
        ax.plot(results.returns_df.index, results.returns_df['Cumulative Returns'] * 100, 
                label='Nominal Returns', color='blue')
        ax.plot(results.inflation_adjusted_returns.index, 
                results.inflation_adjusted_returns['Inflation-Adjusted Returns'] * 100,
                label='Inflation-Adjusted Returns', color='green')
        
        ax.set_title('Nominal vs. Inflation-Adjusted Returns')
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Returns (%)')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        # Save figure to memory
        buffer = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buffer, format='png', dpi=100)
        buffer.seek(0)
        
        # Create ReportLab Image
        img = Image(buffer)
        img.drawHeight = 4 * inch
        img.drawWidth = 6 * inch
        
        story.append(img)
        story.append(Spacer(1, 12))
        
        # Inflation impact text
        final_nominal_return = results.returns_df['Cumulative Returns'].iloc[-1]
        final_real_return = results.inflation_adjusted_returns['Inflation-Adjusted Returns'].iloc[-1]
        inflation_impact = final_nominal_return - final_real_return
        
        impact_text = (
            f"Impact of Inflation: The portfolio's nominal return was {final_nominal_return:.2%}, "
            f"while the inflation-adjusted (real) return was {final_real_return:.2%}. "
            f"Inflation reduced returns by {inflation_impact:.2%} over the investment period."
        )
        
        story.append(Paragraph(impact_text, normal_style))
        story.append(Spacer(1, 24))
    
    # Rolling returns (if available)
    if hasattr(results, 'rolling_returns_1yr_min') and results.rolling_returns_1yr_min is not None:
        story.append(Paragraph("Rolling Returns", heading1_style))
        
        rolling_data = [
            ["Period", "Minimum", "Average", "Maximum"],
            ["1 Year", f"{results.rolling_returns_1yr_min:.2%}", 
             f"{results.rolling_returns_1yr_avg:.2%}", f"{results.rolling_returns_1yr_max:.2%}"]
        ]
        
        # Add other periods if available
        if hasattr(results, 'rolling_returns_3yr_min') and results.rolling_returns_3yr_min is not None:
            rolling_data.append(["3 Years", f"{results.rolling_returns_3yr_min:.2%}", 
                               f"{results.rolling_returns_3yr_avg:.2%}", f"{results.rolling_returns_3yr_max:.2%}"])
        
        if hasattr(results, 'rolling_returns_5yr_min') and results.rolling_returns_5yr_min is not None:
            rolling_data.append(["5 Years", f"{results.rolling_returns_5yr_min:.2%}", 
                               f"{results.rolling_returns_5yr_avg:.2%}", f"{results.rolling_returns_5yr_max:.2%}"])
        
        if hasattr(results, 'rolling_returns_10yr_min') and results.rolling_returns_10yr_min is not None:
            rolling_data.append(["10 Years", f"{results.rolling_returns_10yr_min:.2%}", 
                               f"{results.rolling_returns_10yr_avg:.2%}", f"{results.rolling_returns_10yr_max:.2%}"])
        
        # Create table
        table = Table(rolling_data, colWidths=[1.0*inch, 1.5*inch, 1.5*inch, 1.5*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('ALIGN', (1, 1), (-1, -1), 'RIGHT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.lightgrey)
        ]))
        
        story.append(table)
        story.append(Spacer(1, 24))
    
    # Conclusion
    story.append(Paragraph("Conclusion", heading1_style))
    
    # Generate conclusion text based on performance
    if results.cagr > 0.08:  # 8% threshold for good performance
        performance_level = "strong"
    elif results.cagr > 0.04:  # 4% threshold for decent performance
        performance_level = "solid"
    elif results.cagr > 0:  # Positive but not great
        performance_level = "modest"
    else:  # Negative
        performance_level = "negative"
    
    conclusion_text = (
        f"This portfolio has shown {performance_level} performance over the tested period, "
        f"with a Compound Annual Growth Rate (CAGR) of {results.cagr:.2%}. "
    )
    
    if results.sharpe_ratio > 1:
        conclusion_text += f"The Sharpe ratio of {results.sharpe_ratio:.2f} indicates a favorable risk-adjusted return. "
    elif results.sharpe_ratio > 0.5:
        conclusion_text += f"The Sharpe ratio of {results.sharpe_ratio:.2f} suggests an acceptable risk-adjusted return. "
    else:
        conclusion_text += f"The Sharpe ratio of {results.sharpe_ratio:.2f} indicates poor risk-adjusted return. "
    
    if results.max_drawdown > 0.3:
        conclusion_text += f"The maximum drawdown of {results.max_drawdown:.2%} represents significant volatility and risk. "
    elif results.max_drawdown > 0.15:
        conclusion_text += f"The maximum drawdown of {results.max_drawdown:.2%} represents moderate volatility and risk. "
    else:
        conclusion_text += f"The maximum drawdown of {results.max_drawdown:.2%} indicates relatively low volatility. "
    
    if len(results.etfs) > 1 and hasattr(results, 'correlation_matrix'):
        avg_correlation = results.correlation_matrix.values[np.triu_indices_from(results.correlation_matrix.values, 1)].mean()
        if avg_correlation < 0.4:
            conclusion_text += "The low average correlation between assets suggests good diversification. "
        elif avg_correlation < 0.7:
            conclusion_text += "There is moderate diversification between assets in the portfolio. "
        else:
            conclusion_text += "The high correlation between assets indicates limited diversification benefits. "
    
    conclusion_text += (
        f"Based on the backtest results, an initial investment of ${results.initial_investment:,.2f} "
        f"would have grown to ${results.final_value:,.2f} over the {results.start_date} to {results.end_date} period."
    )
    
    story.append(Paragraph(conclusion_text, normal_style))
    
    # Disclaimer
    story.append(Spacer(1, 24))
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.grey
    )
    
    disclaimer_text = (
        "DISCLAIMER: Past performance is not indicative of future results. This report is for informational purposes only and should not be considered investment advice. "
        "The backtest results are based on historical data and do not account for taxes, fees, or other expenses that would reduce returns. "
        "Investing involves risk, including the potential loss of principal."
    )
    
    story.append(Paragraph(disclaimer_text, disclaimer_style))
    
    # Build the PDF
    doc.build(story)
    
    # Portfolio configuration
    story.append(Paragraph("Backtest Configuration", heading2_style))
    
    # Create table for configuration
    config_data = [
        ["Parameter", "Value"],
        ["Initial Investment", f"${config['initial_investment']:,.2f}"],
        ["Monthly Contribution", f"${config['monthly_contribution']:,.2f}"],
        ["Start Date", config['start_date'].strftime("%Y-%m-%d")],
        ["End Date", config['end_date'].strftime("%Y-%m-%d")],
        ["Country", config['country']],
        ["Benchmark", config['benchmark'] if config['benchmark'] else "None"]
    ]
    
    # Create table
    table = Table(config_data, colWidths=[2*inch, 2*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('GRID', (0, 0), (-1, -1), 1, colors.lightgrey)
    ]))
    
    story.append(table)
    story.append(Spacer(1, 24))
    
    # Performance summary
    story.append(Paragraph("Performance Summary", heading1_style))
    
    # Key metrics table
    story.append(Paragraph("Key Metrics", heading2_style))
    
    metrics_data = [
        ["Metric", "Value"],
        ["Final Portfolio Value", f"${results.final_value:,.2f}"],
        ["Total Gain", f"${results.total_gain:,.2f}"],
        ["Total Return", f"{results.total_return:.2%}"],
        ["CAGR", f"{results.cagr:.2%}"],
        ["Sharpe Ratio", f"{results.sharpe_ratio:.2f}"],
        ["Volatility (Annualized)", f"{results.volatility:.2%}"],
        ["Maximum Drawdown", f"{results.max_drawdown:.2%}"],
        ["Value at Risk (95%)", f"{results.var_95:.2%}"]
    ]
    
    if hasattr(results, 'benchmark_cagr') and results.benchmark_cagr is not None:
        metrics_data.append(["Benchmark CAGR", f"{results.benchmark_cagr:.2%}"])
        metrics_data.append(["Benchmark Sharpe", f"{results.benchmark_sharpe:.2f}"])
    
    # Create table
    table = Table(metrics_data, colWidths=[2.5*inch, 1.5*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('ALIGN', (1, 1), (1, -1), 'RIGHT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('GRID', (0, 0), (-1, -1), 1, colors.lightgrey),
        # Color positive/negative returns
        ('TEXTCOLOR', (1, 3), (1, 3), colors.green if results.total_return > 0 else colors.red),
        ('TEXTCOLOR', (1, 4), (1, 4), colors.green if results.cagr > 0 else colors.red)
    ]))
    
    story.append(table)
    story.append(Spacer(1, 12))
    
    # Portfolio growth chart
    story.append(Paragraph("Portfolio Growth", heading2_style))
    
    # Create matplotlib figure for portfolio growth
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    
    # Plot portfolio value
    ax.plot(results.portfolio_value_history.index, results.portfolio_value_history.values, label='Portfolio Value', color='blue')
    
    # Add total contributions line if monthly contribution > 0
    if results.monthly_contribution > 0:
        # Calculate cumulative contributions
        dates = results.portfolio_value_history.index
        months_passed = [(d.year - dates[0].year) * 12 + d.month - dates[0].month for d in dates]
        contributions = [results.initial_investment + results.monthly_contribution * max(0, m) for m in months_passed]
        
        ax.plot(dates, contributions, label='Total Contributions', color='green', linestyle='--')
    
    # Add benchmark if available
    if hasattr(results, 'benchmark_returns') and results.benchmark_returns is not None:
        # Normalize benchmark to match the initial investment
        benchmark_values = results.initial_investment * (1 + results.benchmark_returns.iloc[:, 0].cumsum())
        ax.plot(benchmark_values.index, benchmark_values.values, label=f'Benchmark ({results.benchmark})', color='orange', linestyle=':')
    
    ax.set_title('Portfolio Performance')
    ax.set_xlabel('Date')
    ax.set_ylabel('Value ($)')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    # Save figure to memory
    buffer = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buffer, format='png', dpi=100)
    buffer.seek(0)
    
    # Create ReportLab Image
    img = Image(buffer)
    img.drawHeight = 4 * inch
    img.drawWidth = 6 * inch
    
    story.append(img)
    story.append(Spacer(1, 12))
    
    # Annual returns chart
    if hasattr(results, 'annual_returns') and results.annual_returns is not None and len(results.annual_returns) > 0:
        story.append(Paragraph("Annual Returns", heading2_style))
        
        # Create matplotlib figure for annual returns
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111)
        
        # Format annual returns for plotting
        annual_returns = results.annual_returns.copy() * 100  # Convert to percentage
        annual_returns.index = annual_returns.index.strftime('%Y')
        
        # Plot bars
        bars = ax.bar(annual_returns.index, annual_returns.values, color=['green' if x >= 0 else 'red' for x in annual_returns.values])
        
        ax.set_title('Annual Returns')
        ax.set_xlabel('Year')
        ax.set_ylabel('Return (%)')
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Add horizontal line at y=0
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add average annual return
        avg_return = annual_returns.mean()
        ax.axhline(y=avg_return, color='black', linestyle='--', alpha=0.7, label=f'Average ({avg_return:.2f}%)')
        ax.legend()
        
        # Rotate x-axis labels
        plt.xticks(rotation=45)
        
        # Save figure to memory
        buffer = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buffer, format='png', dpi=100)
        buffer.seek(0)
        
        # Create ReportLab Image
        img = Image(buffer)
        img.drawHeight = 4 * inch
        img.drawWidth = 6 * inch
        
        story.append(img)
        story.append(Spacer(1, 12))
    
    # Drawdown chart
    story.append(Paragraph("Drawdowns", heading2_style))
    
    # Create matplotlib figure for drawdowns
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    
    # Plot drawdowns
    drawdowns = results.drawdown_series.copy() * 100  # Convert to percentage
    ax.fill_between(drawdowns.index, drawdowns.values, 0, color='red', alpha=0.3)
    ax.plot(drawdowns.index, drawdowns.values, color='red')
    
    ax.set_title('Portfolio Drawdowns')
    ax.set_xlabel('Date')
    ax.set_ylabel('Drawdown (%)')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Invert y-axis for better visualization
    ax.invert_yaxis()
    
    # Add annotation for maximum drawdown
    ax.annotate(f'Max Drawdown: {results.max_drawdown:.2%}',
                xy=(0.02, 0.02), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # Save figure to memory
    buffer = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buffer, format='png', dpi=100)
    buffer.seek(0)
    
    # Create ReportLab Image
    img = Image(buffer)
    img.drawHeight = 4 * inch
    img.drawWidth = 6 * inch
    
    story.append(img)
    story.append(Spacer(1, 24))
    
    # Risk analysis section
    story.append(Paragraph("Risk Analysis", heading1_style))
    
    # Return distribution
    story.append(Paragraph("Return Distribution", heading2_style))
    
    # Create matplotlib figure for return distribution
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    
    # Get daily returns
    returns = results.returns_df['Daily Returns'].dropna() * 100  # Convert to percentage
    
    # Plot histogram
    sns.histplot(returns, bins=30, kde=True, ax=ax, color='blue', alpha=0.6)
    
    # Add vertical line for mean
    ax.axvline(returns.mean(), color='black', linestyle='--', label=f'Mean ({returns.mean():.2f}%)')
    
    # Add vertical line for VaR (95%)
    var_95 = np.percentile(returns, 5)
    ax.axvline(var_95, color='red', linestyle='--', label=f'VaR 95% ({var_95:.2f}%)')
    
    ax.set_title('Daily Returns Distribution')
    ax.set_xlabel('Daily Return (%)')
    ax.set_ylabel('Frequency')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    # Save figure to memory
    buffer = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buffer, format='png', dpi=100)
    buffer.seek(0)
    
    # Create ReportLab Image
    img = Image(buffer)
    img.drawHeight = 4 * inch
    img.drawWidth = 6 * inch
    
    story.append(img)
    story.append(Spacer(1, 12))
    
    # Risk metrics table
    story.append(Paragraph("Risk Metrics", heading2_style))
    
    risk_data = [
        ["Metric", "Value"],
        ["Annual Volatility", f"{results.volatility:.2%}"],
        ["Maximum Drawdown", f"{results.max_drawdown:.2%}"],
        ["Longest Drawdown Duration", f"{results.longest_drawdown_days} days"],
        ["Value at Risk (95%)", f"{results.var_95:.2%}"],
        ["Value at Risk (99%)", f"{results.var_99:.2%}"],
        ["Expected Shortfall (95%)", f"{results.expected_shortfall:.2%}"],
        ["Positive Days", f"{results.positive_days_pct:.2%}"],
        ["Negative Days", f"{results.negative_days_pct:.2%}"]
    ]
    
    # Create table
    table = Table(risk_data, colWidths=[2.5*inch, 1.5*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('ALIGN', (1, 1), (1, -1), 'RIGHT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('GRID', (0, 0), (-1, -1), 1, colors.lightgrey)
    ]))
    
    story.append(table)
    story.append(Spacer(1, 12))

    logger.info(f"Report generated at {temp_path}")
    
    return temp_path