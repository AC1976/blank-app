import streamlit as st

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from matplotlib.backends.backend_pdf import PdfPages
import sys

# Import the simulator classes (assumes the main code is in a file called portfolio_simulator.py)
# If running as standalone, include all the classes from the previous artifact here
# For this example, I'll include the essential classes inline

from dataclasses import dataclass
from typing import Literal, Optional, List, Tuple
from enum import Enum


class ContributionFrequency(Enum):
    """Enum for contribution frequencies"""
    MONTHLY = 12
    QUARTERLY = 4
    ANNUAL = 1


class TaxModel(Enum):
    """Enum for taxation models"""
    NONE = "none"
    REALIZATION = "realization"
    MARK_TO_MARKET = "mark_to_market"


@dataclass
class IndexParameters:
    """Parameters for different market indices"""
    name: str
    annual_return: float
    annual_volatility: float
    dividend_yield: float
    
    @property
    def monthly_return(self) -> float:
        return (1 + self.annual_return) ** (1/12) - 1
    
    @property
    def monthly_volatility(self) -> float:
        return self.annual_volatility / np.sqrt(12)


@dataclass
class TaxSettings:
    """Tax configuration settings"""
    capital_gains_rate: float
    dividend_tax_rate: float
    tax_payment_month: int = 4
    cost_basis_method: str = "FIFO"


class IndexLibrary:
    """Library of historical index parameters"""
    
    @staticmethod
    def get_sp500() -> IndexParameters:
        return IndexParameters(
            name="S&P 500",
            annual_return=0.1068,
            annual_volatility=0.18,
            dividend_yield=0.0175
        )
    
    @staticmethod
    def get_nasdaq100() -> IndexParameters:
        return IndexParameters(
            name="NASDAQ 100",
            annual_return=0.1250,
            annual_volatility=0.22,
            dividend_yield=0.0075
        )
    
    @staticmethod
    def get_russell2000() -> IndexParameters:
        return IndexParameters(
            name="Russell 2000",
            annual_return=0.0950,
            annual_volatility=0.23,
            dividend_yield=0.0125
        )
    
    @staticmethod
    def get_msci_world() -> IndexParameters:
        return IndexParameters(
            name="MSCI World",
            annual_return=0.0875,
            annual_volatility=0.17,
            dividend_yield=0.0200
        )
    
    @staticmethod
    def get_all_indices():
        return {
            'S&P 500': IndexLibrary.get_sp500(),
            'NASDAQ 100': IndexLibrary.get_nasdaq100(),
            'Russell 2000': IndexLibrary.get_russell2000(),
            'MSCI World': IndexLibrary.get_msci_world()
        }


class TaxLot:
    """Represents a tax lot for cost basis tracking"""
    def __init__(self, shares: float, cost_basis_per_share: float, purchase_month: int):
        self.shares = shares
        self.cost_basis_per_share = cost_basis_per_share
        self.purchase_month = purchase_month
        self.total_cost_basis = shares * cost_basis_per_share


# Note: Include the full PortfolioMonteCarloSimulator class here from the previous artifact
# For brevity in this response, I'm showing the structure - you'll need to copy the full class

class PortfolioMonteCarloSimulator:
    """Monte Carlo simulator - COPY FULL CLASS FROM PREVIOUS ARTIFACT"""
    # [Copy entire class implementation here]
    pass


# Streamlit App Configuration
st.set_page_config(
    page_title="Portfolio Tax Impact Simulator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-size: 1.1rem;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.75rem;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">📊 Portfolio Tax Impact Simulator</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Monte Carlo Analysis for Policy Evaluation</p>', unsafe_allow_html=True)

# Sidebar - Input Parameters
st.sidebar.header("⚙️ Simulation Parameters")

# Investment Parameters
st.sidebar.subheader("💰 Investment Details")
initial_investment = st.sidebar.number_input(
    "Initial Investment ($)",
    min_value=0,
    value=10000,
    step=1000,
    help="Starting portfolio value"
)

regular_contribution = st.sidebar.number_input(
    "Regular Contribution Amount ($)",
    min_value=0,
    value=500,
    step=100,
    help="Amount contributed each period"
)

contribution_freq = st.sidebar.selectbox(
    "Contribution Frequency",
    options=["Monthly", "Quarterly", "Annual"],
    index=0,
    help="How often contributions are made"
)

investment_years = st.sidebar.slider(
    "Investment Period (Years)",
    min_value=5,
    max_value=50,
    value=30,
    step=1,
    help="Total investment time horizon"
)

# Index Selection
st.sidebar.subheader("📈 Market Index")
index_options = list(IndexLibrary.get_all_indices().keys())
selected_index_name = st.sidebar.selectbox(
    "Select Index",
    options=index_options,
    index=0,
    help="Choose the market index to simulate"
)

# Tax Model Selection
st.sidebar.subheader("🏛️ Taxation Model")
tax_model_option = st.sidebar.radio(
    "Select Tax Model",
    options=["No Taxes (Baseline)", "Realization-Based", "Mark-to-Market"],
    index=1,
    help="Choose how gains are taxed"
)

# Tax Rates (only show if not "No Taxes")
if tax_model_option != "No Taxes (Baseline)":
    st.sidebar.subheader("💵 Tax Rates")
    
    capital_gains_rate = st.sidebar.slider(
        "Capital Gains Tax Rate (%)",
        min_value=0.0,
        max_value=50.0,
        value=20.0,
        step=0.5,
        help="Tax rate on capital gains"
    ) / 100
    
    dividend_tax_rate = st.sidebar.slider(
        "Dividend Tax Rate (%)",
        min_value=0.0,
        max_value=50.0,
        value=15.0,
        step=0.5,
        help="Tax rate on dividend income"
    ) / 100
else:
    capital_gains_rate = 0.0
    dividend_tax_rate = 0.0

# Simulation Settings
st.sidebar.subheader("🎲 Simulation Settings")
num_simulations = st.sidebar.select_slider(
    "Number of Simulations",
    options=[100, 500, 1000, 2000, 5000],
    value=1000,
    help="More simulations = more accurate but slower"
)

random_seed = st.sidebar.number_input(
    "Random Seed (for reproducibility)",
    min_value=0,
    value=42,
    step=1,
    help="Use same seed for consistent results"
)

# Run Simulation Button
st.sidebar.markdown("---")
run_button = st.sidebar.button("🚀 Run Simulation", type="primary")

# Main Content Area
if run_button:
    # Map selections to enums
    freq_map = {
        "Monthly": ContributionFrequency.MONTHLY,
        "Quarterly": ContributionFrequency.QUARTERLY,
        "Annual": ContributionFrequency.ANNUAL
    }
    
    tax_map = {
        "No Taxes (Baseline)": TaxModel.NONE,
        "Realization-Based": TaxModel.REALIZATION,
        "Mark-to-Market": TaxModel.MARK_TO_MARKET
    }
    
    contribution_frequency = freq_map[contribution_freq]
    tax_model = tax_map[tax_model_option]
    
    # Get index parameters
    all_indices = IndexLibrary.get_all_indices()
    index_params = all_indices[selected_index_name]
    
    # Create tax settings if needed
    tax_settings = None
    if tax_model != TaxModel.NONE:
        tax_settings = TaxSettings(
            capital_gains_rate=capital_gains_rate,
            dividend_tax_rate=dividend_tax_rate,
            tax_payment_month=4,
            cost_basis_method="FIFO"
        )
    
    # Show progress
    with st.spinner(f'Running {num_simulations:,} simulations... This may take a minute...'):
        # NOTE: You need to copy the full PortfolioMonteCarloSimulator class here
        # or import it from the main module
        
        # Create and run simulator
        simulator = PortfolioMonteCarloSimulator(
            index_params=index_params,
            initial_investment=initial_investment,
            regular_contribution=regular_contribution,
            contribution_frequency=contribution_frequency,
            years=investment_years,
            num_simulations=num_simulations,
            tax_model=tax_model,
            tax_settings=tax_settings,
            random_seed=random_seed
        )
        
        simulator.run_simulation()
        stats = simulator.get_statistics()
    
    st.success("✅ Simulation Complete!")
    
    # Display Results in Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Summary", "📈 Visualizations", "📄 Detailed Report"])
    
    with tab1:
        st.header("Summary Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Contributed",
                f"${stats['total_contributed']:,.0f}"
            )
        
        with col2:
            st.metric(
                "Median Final Value",
                f"${stats['median_final_value']:,.0f}",
                delta=f"+${stats['median_final_value'] - stats['total_contributed']:,.0f}"
            )
        
        with col3:
            st.metric(
                "Mean Final Value",
                f"${stats['mean_final_value']:,.0f}",
                f"${stats['percentile_75']:,.0f}",
                f"${stats['percentile_90']:,.0f}",
                f"${stats['max_value']:,.0f}",
                f"${stats['std_dev']:,.0f}"
            )
        }
        
        st.table(stats_data)
        
        if tax_model != TaxModel.NONE:
            st.subheader("Total Taxes Paid Statistics")
            
            tax_stats_data = {
                "Statistic": [
                    "10th Percentile",
                    "Median (50th)",
                    "Mean",
                    "90th Percentile"
                ],
                "Value ($)": [
                    f"${stats['total_taxes_10th']:,.0f}",
                    f"${stats['median_total_taxes']:,.0f}",
                    f"${stats['mean_total_taxes']:,.0f}",
                    f"${stats['total_taxes_90th']:,.0f}"
                ]
            }
            
            st.table(tax_stats_data)
            
            st.markdown("---")
            st.subheader("Tax Impact Summary")
            
            # Calculate impact metrics
            median_final = stats['median_final_value']
            median_taxes = stats['median_total_taxes']
            total_contributed = stats['total_contributed']
            
            median_growth = median_final + median_taxes - total_contributed
            tax_percentage = (median_taxes / (median_final + median_taxes)) * 100
            
            impact_col1, impact_col2, impact_col3 = st.columns(3)
            
            with impact_col1:
                st.metric("Market Growth (Median)", f"${median_growth:,.0f}")
            
            with impact_col2:
                st.metric("Taxes Paid (Median)", f"${median_taxes:,.0f}")
            
            with impact_col3:
                st.metric("Tax Burden %", f"{tax_percentage:.1f}%")
            
            st.info(f"""
            **Key Insight:** In the median scenario, an investor contributes ${total_contributed:,.0f} 
            and experiences ${median_growth:,.0f} in market growth. However, ${median_taxes:,.0f} 
            ({tax_percentage:.1f}% of pre-tax portfolio value) goes to taxes under the {tax_model_option} model, 
            leaving a final after-tax value of ${median_final:,.0f}.
            """)
        
        # Download report button
        st.markdown("---")
        st.subheader("📥 Export Results")
        
        # Create downloadable HTML report
        html_report = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Portfolio Tax Impact Simulation Report</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 40px;
                    line-height: 1.6;
                }}
                h1 {{
                    color: #1f77b4;
                    text-align: center;
                    border-bottom: 3px solid #1f77b4;
                    padding-bottom: 10px;
                }}
                h2 {{
                    color: #333;
                    margin-top: 30px;
                    border-bottom: 2px solid #ddd;
                    padding-bottom: 5px;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 12px;
                    text-align: left;
                }}
                th {{
                    background-color: #1f77b4;
                    color: white;
                }}
                tr:nth-child(even) {{
                    background-color: #f2f2f2;
                }}
                .summary-box {{
                    background-color: #e8f4f8;
                    border-left: 5px solid #1f77b4;
                    padding: 15px;
                    margin: 20px 0;
                }}
                .metric {{
                    display: inline-block;
                    margin: 10px 20px;
                    padding: 15px;
                    background-color: #f9f9f9;
                    border-radius: 5px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .metric-value {{
                    font-size: 24px;
                    font-weight: bold;
                    color: #1f77b4;
                }}
                .metric-label {{
                    font-size: 14px;
                    color: #666;
                }}
                @media print {{
                    body {{
                        margin: 20px;
                    }}
                }}
            </style>
        </head>
        <body>
            <h1>📊 Portfolio Tax Impact Simulation Report</h1>
            
            <div class="summary-box">
                <h3>Executive Summary</h3>
                <p>This report presents a Monte Carlo simulation analysis of portfolio growth under the 
                <strong>{tax_model_option}</strong> taxation model.</p>
            </div>
            
            <h2>Investment Configuration</h2>
            <table>
                <tr><th>Parameter</th><th>Value</th></tr>
                <tr><td>Market Index</td><td>{index_params.name}</td></tr>
                <tr><td>Initial Investment</td><td>${initial_investment:,.0f}</td></tr>
                <tr><td>Regular Contribution</td><td>${regular_contribution:,.0f} ({contribution_freq})</td></tr>
                <tr><td>Investment Period</td><td>{investment_years} years</td></tr>
                <tr><td>Tax Model</td><td>{tax_model_option}</td></tr>
                {f'<tr><td>Capital Gains Tax Rate</td><td>{capital_gains_rate*100:.1f}%</td></tr>' if tax_model != TaxModel.NONE else ''}
                {f'<tr><td>Dividend Tax Rate</td><td>{dividend_tax_rate*100:.1f}%</td></tr>' if tax_model != TaxModel.NONE else ''}
                <tr><td>Number of Simulations</td><td>{num_simulations:,}</td></tr>
            </table>
            
            <h2>Key Results</h2>
            <div style="text-align: center;">
                <div class="metric">
                    <div class="metric-label">Total Contributed</div>
                    <div class="metric-value">${stats['total_contributed']:,.0f}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Median Final Value</div>
                    <div class="metric-value">${stats['median_final_value']:,.0f}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Mean Final Value</div>
                    <div class="metric-value">${stats['mean_final_value']:,.0f}</div>
                </div>
                {f'''<div class="metric">
                    <div class="metric-label">Median Taxes Paid</div>
                    <div class="metric-value">${stats['median_total_taxes']:,.0f}</div>
                </div>''' if tax_model != TaxModel.NONE else ''}
            </div>
            
            <h2>Final Portfolio Value Statistics</h2>
            <table>
                <tr><th>Statistic</th><th>Value</th></tr>
                <tr><td>Minimum</td><td>${stats['min_value']:,.0f}</td></tr>
                <tr><td>10th Percentile</td><td>${stats['percentile_10']:,.0f}</td></tr>
                <tr><td>25th Percentile</td><td>${stats['percentile_25']:,.0f}</td></tr>
                <tr><td>Median (50th Percentile)</td><td>${stats['median_final_value']:,.0f}</td></tr>
                <tr><td>Mean</td><td>${stats['mean_final_value']:,.0f}</td></tr>
                <tr><td>75th Percentile</td><td>${stats['percentile_75']:,.0f}</td></tr>
                <tr><td>90th Percentile</td><td>${stats['percentile_90']:,.0f}</td></tr>
                <tr><td>Maximum</td><td>${stats['max_value']:,.0f}</td></tr>
                <tr><td>Standard Deviation</td><td>${stats['std_dev']:,.0f}</td></tr>
            </table>
        """
        
        if tax_model != TaxModel.NONE:
            median_final = stats['median_final_value']
            median_taxes = stats['median_total_taxes']
            median_growth = median_final + median_taxes - stats['total_contributed']
            tax_percentage = (median_taxes / (median_final + median_taxes)) * 100
            
            html_report += f"""
            <h2>Tax Impact Analysis</h2>
            <table>
                <tr><th>Statistic</th><th>Value</th></tr>
                <tr><td>10th Percentile Taxes</td><td>${stats['total_taxes_10th']:,.0f}</td></tr>
                <tr><td>Median Taxes Paid</td><td>${stats['median_total_taxes']:,.0f}</td></tr>
                <tr><td>Mean Taxes Paid</td><td>${stats['mean_total_taxes']:,.0f}</td></tr>
                <tr><td>90th Percentile Taxes</td><td>${stats['total_taxes_90th']:,.0f}</td></tr>
            </table>
            
            <div class="summary-box">
                <h3>Tax Burden Summary (Median Scenario)</h3>
                <p><strong>Market Growth:</strong> ${median_growth:,.0f}</p>
                <p><strong>Taxes Paid:</strong> ${median_taxes:,.0f}</p>
                <p><strong>Tax Burden:</strong> {tax_percentage:.1f}% of pre-tax portfolio value</p>
                <p><strong>Final After-Tax Value:</strong> ${median_final:,.0f}</p>
                <p><strong>Net Gain After Taxes:</strong> ${median_final - stats['total_contributed']:,.0f}</p>
            </div>
            """
        
        html_report += """
            <h2>Methodology</h2>
            <p>This analysis uses Monte Carlo simulation to model portfolio growth under uncertainty. 
            The simulation:</p>
            <ul>
                <li>Models monthly returns using historical mean returns and volatility</li>
                <li>Includes dividend reinvestment</li>
                <li>Tracks cost basis using FIFO (First-In-First-Out) method</li>
                <li>Applies taxation according to the selected model</li>
                <li>Generates thousands of possible future paths</li>
            </ul>
            
            <h2>Disclaimer</h2>
            <p><em>This simulation is for educational and policy analysis purposes only. 
            Past performance does not guarantee future results. Actual investment outcomes may vary 
            significantly. Consult with financial and tax professionals before making investment decisions.</em></p>
            
            <hr>
            <p style="text-align: center; color: #666; font-size: 12px;">
                Generated by Portfolio Tax Impact Simulator | Monte Carlo Analysis Tool
            </p>
        </body>
        </html>
        """
        
        # Create download button
        st.download_button(
            label="📄 Download HTML Report",
            data=html_report,
            file_name=f"portfolio_simulation_report_{tax_model_option.replace(' ', '_').lower()}.html",
            mime="text/html"
        )

else:
    # Show instructions when simulation hasn't been run yet
    st.info("👈 Configure your simulation parameters in the sidebar and click 'Run Simulation' to begin.")
    
    st.markdown("""
    ## About This Tool
    
    This Portfolio Tax Impact Simulator helps policymakers and investors understand how different 
    taxation models affect long-term investment outcomes.
    
    ### Features:
    - **Monte Carlo Simulation**: Models thousands of possible market scenarios
    - **Multiple Tax Models**: Compare no taxes, realization-based, and mark-to-market taxation
    - **Multiple Indices**: Choose from S&P 500, NASDAQ 100, Russell 2000, or MSCI World
    - **Flexible Parameters**: Customize investment amounts, periods, and contribution frequencies
    - **Comprehensive Analysis**: View distributions, percentiles, and tax burden metrics
    - **Exportable Reports**: Download results as HTML for presentations
    
    ### Tax Models Explained:
    
    **Realization-Based Taxation**
    - Capital gains are taxed only when shares are sold
    - Dividends are taxed annually
    - Allows for tax-deferred growth
    - Final liquidation event taxes all remaining gains
    
    **Mark-to-Market Taxation**
    - Unrealized gains are taxed annually (as if sold and repurchased)
    - Dividends are taxed annually
    - Tax basis steps up each year after taxation
    - No deferral benefit
    
    ### How to Use:
    1. Set your investment parameters (amount, frequency, period)
    2. Choose a market index
    3. Select a taxation model and set tax rates
    4. Adjust simulation settings (number of paths, random seed)
    5. Click "Run Simulation"
    6. Review results in the three tabs
    7. Download HTML report for sharing
    """)
    
    st.markdown("---")
    st.markdown("**Note:** Before running the simulation, make sure you have copied the full "
                "`PortfolioMonteCarloSimulator` class into this file where indicated in the code.")


# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="text-align: center; font-size: 0.8rem; color: #666;">
    <p><strong>Portfolio Tax Impact Simulator</strong></p>
    <p>Monte Carlo Analysis Tool</p>
    <p>For policy evaluation purposes</p>
</div>
""", unsafe_allow_html=True)f}"
            )
        
        with col4:
            if tax_model != TaxModel.NONE:
                st.metric(
                    "Median Taxes Paid",
                    f"${stats['median_total_taxes']:,.0f}"
                )
            else:
                st.metric(
                    "Tax Model",
                    "None"
                )
        
        # Percentile ranges
        st.subheader("Final Portfolio Value Ranges")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info(f"**10th Percentile (Pessimistic)**\n\n${stats['percentile_10']:,.0f}")
        
        with col2:
            st.success(f"**50th Percentile (Median)**\n\n${stats['median_final_value']:,.0f}")
        
        with col3:
            st.warning(f"**90th Percentile (Optimistic)**\n\n${stats['percentile_90']:,.0f}")
    
    with tab2:
        st.header("Visual Analysis")
        
        # Generate plots
        st.subheader("Portfolio Growth Simulation")
        
        # Create figure for portfolio paths
        fig1, ax = plt.subplots(figsize=(12, 6))
        
        time_years = np.linspace(0, investment_years, simulator.months + 1)
        
        # Show sample paths
        num_paths_to_show = min(100, num_simulations)
        indices_to_plot = np.random.choice(num_simulations, size=num_paths_to_show, replace=False)
        
        for idx in indices_to_plot:
            ax.plot(time_years, simulator.portfolio_paths[idx], 
                   alpha=0.1, color='blue', linewidth=0.5)
        
        # Percentile lines
        p10 = np.percentile(simulator.portfolio_paths, 10, axis=0)
        p50 = np.percentile(simulator.portfolio_paths, 50, axis=0)
        p90 = np.percentile(simulator.portfolio_paths, 90, axis=0)
        
        ax.plot(time_years, p10, 'r--', linewidth=2, label='10th Percentile')
        ax.plot(time_years, p50, 'g-', linewidth=2.5, label='Median')
        ax.plot(time_years, p90, 'r--', linewidth=2, label='90th Percentile')
        
        ax.set_xlabel('Years', fontsize=12)
        ax.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax.set_title(f'{index_params.name} - {tax_model_option}', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig1)
        
        # Distribution plots
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Final Value Distribution")
            fig2, ax2 = plt.subplots(figsize=(6, 5))
            ax2.hist(simulator.final_values, bins=50, alpha=0.7, color='green', edgecolor='black')
            ax2.axvline(np.median(simulator.final_values), color='red', linestyle='--',
                       linewidth=2, label=f'Median: ${np.median(simulator.final_values):,.0f}')
            ax2.set_xlabel('Final Portfolio Value ($)')
            ax2.set_ylabel('Frequency')
            ax2.legend()
            ax2.grid(True, alpha=0.3, axis='y')
            st.pyplot(fig2)
        
        with col2:
            if tax_model != TaxModel.NONE:
                st.subheader("Tax Distribution")
                fig3, ax3 = plt.subplots(figsize=(6, 5))
                ax3.hist(simulator.total_taxes_paid, bins=50, alpha=0.7, color='red', edgecolor='black')
                ax3.axvline(np.median(simulator.total_taxes_paid), color='darkred', linestyle='--',
                           linewidth=2, label=f'Median: ${np.median(simulator.total_taxes_paid):,.0f}')
                ax3.set_xlabel('Total Taxes Paid ($)')
                ax3.set_ylabel('Frequency')
                ax3.legend()
                ax3.grid(True, alpha=0.3, axis='y')
                st.pyplot(fig3)
        
        # Tax Impact Summary (if applicable)
        if tax_model != TaxModel.NONE:
            st.subheader("Tax Impact Analysis")
            
            # Prepare data for impact chart
            total_contributed = stats['total_contributed']
            
            median_final = np.median(simulator.final_values)
            median_taxes = np.median(simulator.total_taxes_paid)
            median_growth = median_final + median_taxes - total_contributed
            
            p10_final = np.percentile(simulator.final_values, 10)
            p10_taxes = np.percentile(simulator.total_taxes_paid, 10)
            p10_growth = p10_final + p10_taxes - total_contributed
            
            p90_final = np.percentile(simulator.final_values, 90)
            p90_taxes = np.percentile(simulator.total_taxes_paid, 90)
            p90_growth = p90_final + p90_taxes - total_contributed
            
            # Create impact visualization
            fig4, (ax4, ax5) = plt.subplots(1, 2, figsize=(14, 5))
            
            # Stacked bar chart
            scenarios = ['10th\nPercentile', 'Median', '90th\nPercentile']
            contributions = [total_contributed] * 3
            growth = [p10_growth, median_growth, p90_growth]
            taxes = [p10_taxes, median_taxes, p90_taxes]
            
            x = np.arange(len(scenarios))
            width = 0.6
            
            ax4.bar(x, contributions, width, label='Contributions', color='#2E86AB')
            ax4.bar(x, growth, width, bottom=contributions, label='Growth', color='#06A77D')
            ax4.bar(x, taxes, width, bottom=np.array(contributions) + np.array(growth),
                   label='Taxes', color='#D62828')
            
            ax4.set_ylabel('Amount ($)', fontsize=12)
            ax4.set_title('Where Does Your Money Go?', fontsize=13, fontweight='bold')
            ax4.set_xticks(x)
            ax4.set_xticklabels(scenarios)
            ax4.legend()
            ax4.grid(True, alpha=0.3, axis='y')
            
            # Tax percentage chart
            percentages = [
                (p10_taxes / (p10_final + p10_taxes)) * 100,
                (median_taxes / (median_final + median_taxes)) * 100,
                (p90_taxes / (p90_final + p90_taxes)) * 100
            ]
            
            bars = ax5.bar(scenarios, percentages, width=0.6, 
                          color=['#FF6B6B', '#FF8C42', '#DC2F02'])
            
            for bar, pct in zip(bars, percentages):
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width()/2., height/2,
                        f'{pct:.1f}%', ha='center', va='center',
                        fontweight='bold', fontsize=12, color='white')
            
            ax5.set_ylabel('Tax Burden (%)', fontsize=12)
            ax5.set_title('Taxes as % of Pre-Tax Value', fontsize=13, fontweight='bold')
            ax5.grid(True, alpha=0.3, axis='y')
            
            st.pyplot(fig4)
    
    with tab3:
        st.header("Detailed Statistical Report")
        
        st.subheader("Investment Configuration")
        config_col1, config_col2 = st.columns(2)
        
        with config_col1:
            st.write(f"**Index:** {index_params.name}")
            st.write(f"**Initial Investment:** ${initial_investment:,.0f}")
            st.write(f"**Regular Contribution:** ${regular_contribution:,.0f}")
            st.write(f"**Contribution Frequency:** {contribution_freq}")
            st.write(f"**Investment Period:** {investment_years} years")
        
        with config_col2:
            st.write(f"**Tax Model:** {tax_model_option}")
            if tax_model != TaxModel.NONE:
                st.write(f"**Capital Gains Rate:** {capital_gains_rate*100:.1f}%")
                st.write(f"**Dividend Tax Rate:** {dividend_tax_rate*100:.1f}%")
            st.write(f"**Number of Simulations:** {num_simulations:,}")
            st.write(f"**Random Seed:** {random_seed}")
        
        st.markdown("---")
        
        st.subheader("Final Portfolio Value Statistics")
        
        stats_data = {
            "Statistic": [
                "Total Contributed",
                "Minimum",
                "10th Percentile",
                "25th Percentile",
                "Median (50th)",
                "Mean",
                "75th Percentile",
                "90th Percentile",
                "Maximum",
                "Standard Deviation"
            ],
            "Value ($)": [
                f"${stats['total_contributed']:,.0f}",
                f"${stats['min_value']:,.0f}",
                f"${stats['percentile_10']:,.0f}",
                f"${stats['percentile_25']:,.0f}",
                f"${stats['median_final_value']:,.0f}",
                f"${stats['mean_final_value']:,.0
