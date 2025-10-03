import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, List, Tuple
from enum import Enum


# ============================================================================
# CORE SIMULATION CLASSES
# ============================================================================

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


class PortfolioMonteCarloSimulator:
    """
    Monte Carlo simulator for portfolio planning with regular contributions and taxation.
    """
    
    def __init__(
        self,
        index_params: IndexParameters,
        initial_investment: float,
        regular_contribution: float,
        contribution_frequency: ContributionFrequency,
        years: int,
        num_simulations: int = 1000,
        tax_model: TaxModel = TaxModel.NONE,
        tax_settings: Optional[TaxSettings] = None,
        random_seed: Optional[int] = None
    ):
        self.index_params = index_params
        self.initial_investment = initial_investment
        self.regular_contribution = regular_contribution
        self.contribution_frequency = contribution_frequency
        self.years = years
        self.num_simulations = num_simulations
        self.tax_model = tax_model
        self.tax_settings = tax_settings
        self.random_seed = random_seed
        
        if tax_model != TaxModel.NONE and tax_settings is None:
            raise ValueError("tax_settings required when using a tax model")
        
        self.months = years * 12
        self.contribution_months = self._get_contribution_months()
        
        self.portfolio_paths = None
        self.final_values = None
        self.total_taxes_paid = None
        
    def _get_contribution_months(self) -> np.ndarray:
        """Determine which months have contributions"""
        freq = self.contribution_frequency.value
        return np.arange(0, self.months, 12 // freq)
    
    def _run_simulation_no_tax(self) -> np.ndarray:
        """Run simulation without taxes"""
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        
        portfolio_paths = np.zeros((self.num_simulations, self.months + 1))
        portfolio_paths[:, 0] = self.initial_investment
        
        monthly_returns = np.random.normal(
            self.index_params.monthly_return,
            self.index_params.monthly_volatility,
            (self.num_simulations, self.months)
        )
        
        monthly_dividend_rate = self.index_params.dividend_yield / 12
        
        for month in range(self.months):
            portfolio_paths[:, month + 1] = portfolio_paths[:, month] * (1 + monthly_returns[:, month])
            portfolio_paths[:, month + 1] += portfolio_paths[:, month] * monthly_dividend_rate
            
            if month in self.contribution_months:
                portfolio_paths[:, month + 1] += self.regular_contribution
        
        return portfolio_paths
    
    def _run_single_path_with_tax(self, monthly_returns: np.ndarray, path_idx: int) -> Tuple[np.ndarray, float]:
        """Run a single simulation path with tax treatment"""
        portfolio_values = np.zeros(self.months + 1)
        portfolio_values[0] = self.initial_investment
        
        shares = self.initial_investment
        price_per_share = 1.0
        
        tax_lots: List[TaxLot] = [TaxLot(shares, price_per_share, 0)]
        loss_carryforward = 0.0
        total_tax_paid = 0.0
        dividends_received_ytd = 0.0
        
        mtm_tax_basis = self.initial_investment
        
        monthly_dividend_rate = self.index_params.dividend_yield / 12
        
        for month in range(self.months):
            price_per_share *= (1 + monthly_returns[month])
            
            dividend_income = shares * price_per_share * monthly_dividend_rate
            dividends_received_ytd += dividend_income
            
            new_shares_from_dividends = dividend_income / price_per_share
            shares += new_shares_from_dividends
            tax_lots.append(TaxLot(new_shares_from_dividends, price_per_share, month))
            
            if month in self.contribution_months:
                new_shares_from_contribution = self.regular_contribution / price_per_share
                shares += new_shares_from_contribution
                tax_lots.append(TaxLot(new_shares_from_contribution, price_per_share, month))
            
            portfolio_values[month + 1] = shares * price_per_share
            
            current_year = month // 12
            month_in_year = month % 12
            
            if month_in_year == self.tax_settings.tax_payment_month and month > 0:
                tax_due = 0.0
                
                if dividends_received_ytd > 0:
                    dividend_tax = dividends_received_ytd * self.tax_settings.dividend_tax_rate
                    tax_due += dividend_tax
                    dividends_received_ytd = 0.0
                
                if self.tax_model == TaxModel.MARK_TO_MARKET:
                    current_value = shares * price_per_share
                    unrealized_gain = current_value - mtm_tax_basis
                    
                    if unrealized_gain > 0:
                        taxable_gain = max(0, unrealized_gain - loss_carryforward)
                        capital_gains_tax = taxable_gain * self.tax_settings.capital_gains_rate
                        tax_due += capital_gains_tax
                        
                        loss_carryforward = max(0, loss_carryforward - unrealized_gain)
                        mtm_tax_basis = current_value
                    else:
                        loss_carryforward += abs(unrealized_gain)
                        mtm_tax_basis = current_value
                
                if tax_due > 0:
                    shares_to_sell = tax_due / price_per_share
                    
                    if shares_to_sell >= shares:
                        shares_to_sell = shares
                        tax_due = shares * price_per_share
                    
                    if self.tax_model == TaxModel.REALIZATION:
                        realized_gain = self._sell_shares_fifo(
                            tax_lots, shares_to_sell, price_per_share
                        )
                        
                        if realized_gain > 0:
                            taxable_gain = max(0, realized_gain - loss_carryforward)
                            additional_tax = taxable_gain * self.tax_settings.capital_gains_rate
                            
                            if additional_tax > 0:
                                additional_shares_to_sell = additional_tax / price_per_share
                                if additional_shares_to_sell < shares:
                                    shares_to_sell += additional_shares_to_sell
                                    self._sell_shares_fifo(tax_lots, additional_shares_to_sell, price_per_share)
                                    tax_due += additional_tax
                            
                            loss_carryforward = max(0, loss_carryforward - realized_gain)
                        else:
                            loss_carryforward += abs(realized_gain)
                    else:
                        self._sell_shares_fifo(tax_lots, shares_to_sell, price_per_share)
                    
                    shares -= shares_to_sell
                    total_tax_paid += tax_due
                    portfolio_values[month + 1] = shares * price_per_share
        
        if self.tax_model == TaxModel.REALIZATION:
            final_dividend_tax = 0.0
            if dividends_received_ytd > 0:
                final_dividend_tax = dividends_received_ytd * self.tax_settings.dividend_tax_rate
                total_tax_paid += final_dividend_tax
            
            final_price = price_per_share
            final_portfolio_value = shares * final_price
            
            realized_gain = self._calculate_total_realized_gain(tax_lots, shares, final_price)
            
            final_capital_gains_tax = 0.0
            if realized_gain > 0:
                taxable_gain = max(0, realized_gain - loss_carryforward)
                final_capital_gains_tax = taxable_gain * self.tax_settings.capital_gains_rate
                total_tax_paid += final_capital_gains_tax
            
            portfolio_values[-1] = final_portfolio_value - final_capital_gains_tax - final_dividend_tax
        
        return portfolio_values, total_tax_paid
    
    def _sell_shares_fifo(self, tax_lots: List[TaxLot], shares_to_sell: float, 
                          current_price: float) -> float:
        """Sell shares using FIFO method and return realized gain/loss"""
        realized_gain = 0.0
        remaining_to_sell = shares_to_sell
        
        lots_to_remove = []
        
        for i, lot in enumerate(tax_lots):
            if remaining_to_sell <= 0:
                break
            
            if lot.shares <= remaining_to_sell:
                sale_proceeds = lot.shares * current_price
                cost_basis = lot.shares * lot.cost_basis_per_share
                realized_gain += (sale_proceeds - cost_basis)
                remaining_to_sell -= lot.shares
                lots_to_remove.append(i)
            else:
                sale_proceeds = remaining_to_sell * current_price
                cost_basis = remaining_to_sell * lot.cost_basis_per_share
                realized_gain += (sale_proceeds - cost_basis)
                lot.shares -= remaining_to_sell
                lot.total_cost_basis = lot.shares * lot.cost_basis_per_share
                remaining_to_sell = 0
        
        for i in reversed(lots_to_remove):
            tax_lots.pop(i)
        
        return realized_gain
    
    def _calculate_total_realized_gain(self, tax_lots: List[TaxLot], shares_to_sell: float,
                                       current_price: float) -> float:
        """Calculate realized gain without modifying tax lots"""
        realized_gain = 0.0
        remaining_to_sell = shares_to_sell
        
        for lot in tax_lots:
            if remaining_to_sell <= 0:
                break
            
            shares_from_lot = min(lot.shares, remaining_to_sell)
            sale_proceeds = shares_from_lot * current_price
            cost_basis = shares_from_lot * lot.cost_basis_per_share
            realized_gain += (sale_proceeds - cost_basis)
            remaining_to_sell -= shares_from_lot
        
        return realized_gain
    
    def run_simulation(self) -> np.ndarray:
        """Run the Monte Carlo simulation"""
        if self.tax_model == TaxModel.NONE:
            self.portfolio_paths = self._run_simulation_no_tax()
            self.total_taxes_paid = np.zeros(self.num_simulations)
        else:
            if self.random_seed is not None:
                np.random.seed(self.random_seed)
            
            all_returns = np.random.normal(
                self.index_params.monthly_return,
                self.index_params.monthly_volatility,
                (self.num_simulations, self.months)
            )
            
            portfolio_paths = np.zeros((self.num_simulations, self.months + 1))
            total_taxes = np.zeros(self.num_simulations)
            
            for i in range(self.num_simulations):
                portfolio_paths[i], total_taxes[i] = self._run_single_path_with_tax(
                    all_returns[i], i
                )
            
            self.portfolio_paths = portfolio_paths
            self.total_taxes_paid = total_taxes
        
        self.final_values = self.portfolio_paths[:, -1]
        return self.portfolio_paths
    
    def get_statistics(self) -> dict:
        """Calculate summary statistics from the simulation"""
        if self.final_values is None:
            raise ValueError("Must run simulation first")
        
        total_contributed = (
            self.initial_investment + 
            self.regular_contribution * len(self.contribution_months)
        )
        
        stats = {
            'total_contributed': total_contributed,
            'mean_final_value': np.mean(self.final_values),
            'median_final_value': np.median(self.final_values),
            'percentile_10': np.percentile(self.final_values, 10),
            'percentile_25': np.percentile(self.final_values, 25),
            'percentile_75': np.percentile(self.final_values, 75),
            'percentile_90': np.percentile(self.final_values, 90),
            'min_value': np.min(self.final_values),
            'max_value': np.max(self.final_values),
            'std_dev': np.std(self.final_values)
        }
        
        if self.tax_model != TaxModel.NONE:
            stats['mean_total_taxes'] = np.mean(self.total_taxes_paid)
            stats['median_total_taxes'] = np.median(self.total_taxes_paid)
            stats['total_taxes_10th'] = np.percentile(self.total_taxes_paid, 10)
            stats['total_taxes_90th'] = np.percentile(self.total_taxes_paid, 90)
        
        return stats


# ============================================================================
# STREAMLIT APP
# ============================================================================

st.set_page_config(
    page_title="Portfolio Tax Impact Simulator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">📊 Portfolio Tax Impact Simulator</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Monte Carlo Analysis for Policy Evaluation</p>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("⚙️ Simulation Parameters")

st.sidebar.subheader("💰 Investment Details")
initial_investment = st.sidebar.number_input(
    "Initial Investment ($)",
    min_value=0,
    value=10000,
    step=1000
)

regular_contribution = st.sidebar.number_input(
    "Regular Contribution Amount ($)",
    min_value=0,
    value=500,
    step=100
)

contribution_freq = st.sidebar.selectbox(
    "Contribution Frequency",
    options=["Monthly", "Quarterly", "Annual"],
    index=0
)

investment_years = st.sidebar.slider(
    "Investment Period (Years)",
    min_value=5,
    max_value=50,
    value=30,
    step=1
)

st.sidebar.subheader("📈 Market Index")
index_options = list(IndexLibrary.get_all_indices().keys())
selected_index_name = st.sidebar.selectbox(
    "Select Index",
    options=index_options,
    index=0
)

st.sidebar.subheader("🏛️ Taxation Model")
tax_model_option = st.sidebar.radio(
    "Select Tax Model",
    options=["No Taxes (Baseline)", "Realization-Based", "Mark-to-Market"],
    index=1
)

if tax_model_option != "No Taxes (Baseline)":
    st.sidebar.subheader("💵 Tax Rates")
    
    capital_gains_rate = st.sidebar.slider(
        "Capital Gains Tax Rate (%)",
        min_value=0.0,
        max_value=50.0,
        value=20.0,
        step=0.5
    ) / 100
    
    dividend_tax_rate = st.sidebar.slider(
        "Dividend Tax Rate (%)",
        min_value=0.0,
        max_value=50.0,
        value=15.0,
        step=0.5
    ) / 100
else:
    capital_gains_rate = 0.0
    dividend_tax_rate = 0.0

st.sidebar.subheader("🎲 Simulation Settings")
num_simulations = st.sidebar.select_slider(
    "Number of Simulations",
    options=[100, 500, 1000, 2000, 5000],
    value=1000
)

random_seed = st.sidebar.number_input(
    "Random Seed",
    min_value=0,
    value=42,
    step=1
)

st.sidebar.markdown("---")
run_button = st.sidebar.button("🚀 Run Simulation", type="primary")

# Main content
if run_button:
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
    
    all_indices = IndexLibrary.get_all_indices()
    index_params = all_indices[selected_index_name]
    
    tax_settings = None
    if tax_model != TaxModel.NONE:
        tax_settings = TaxSettings(
            capital_gains_rate=capital_gains_rate,
            dividend_tax_rate=dividend_tax_rate,
            tax_payment_month=4,
            cost_basis_method="FIFO"
        )
    
    with st.spinner(f'Running {num_simulations:,} simulations...'):
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
    
    tab1, tab2, tab3 = st.tabs(["📊 Summary", "📈 Visualizations", "📄 Report"])
    
    with tab1:
        st.header("Summary Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Contributed", f"${stats['total_contributed']:,.0f}")
        
        with col2:
            st.metric(
                "Median Final Value",
                f"${stats['median_final_value']:,.0f}",
                delta=f"+${stats['median_final_value'] - stats['total_contributed']:,.0f}"
            )
        
        with col3:
            st.metric("Mean Final Value", f"${stats['mean_final_value']:,.0f}")
        
        with col4:
            if tax_model != TaxModel.NONE:
                st.metric("Median Taxes", f"${stats['median_total_taxes']:,.0f}")
            else:
                st.metric("Tax Model", "None")
        
        st.subheader("Final Portfolio Value Ranges")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info(f"**10th Percentile**\n\n${stats['percentile_10']:,.0f}")
        
        with col2:
            st.success(f"**Median**\n\n${stats['median_final_value']:,.0f}")
        
        with col3:
            st.warning(f"**90th Percentile**\n\n${stats['percentile_90']:,.0f}")
    
    with tab2:
        st.header("Visual Analysis")
        
        fig1, ax = plt.subplots(figsize=(12, 6))
        
        time_years = np.linspace(0, investment_years, simulator.months + 1)
        
        num_paths_to_show = min(100, num_simulations)
        indices_to_plot = np.random.choice(num_simulations, size=num_paths_to_show, replace=False)
        
        for idx in indices_to_plot:
            ax.plot(time_years, simulator.portfolio_paths[idx], 
                   alpha=0.1, color='blue', linewidth=0.5)
        
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
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Final Value Distribution")
            fig2, ax2 = plt.subplots(figsize=(6, 5))
            ax2.hist(simulator.final_values, bins=50, alpha=0.7, color='green', edgecolor='black')
            ax2.axvline(np.median(simulator.final_values), color='red', linestyle='--',
                       linewidth=2, label=f'Median: ${np.median(simulator.final_values):,.0f}')
            ax2.set_xlabel('Final Value ($)')
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
                ax3.set_xlabel('Total Taxes ($)')
                ax3.set_ylabel('Frequency')
                ax3.legend()
                ax3.grid(True, alpha=0.3, axis='y')
                st.pyplot(fig3)
        
        if tax_model != TaxModel.NONE:
            st.subheader("Tax Impact Analysis")
            
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
            
            fig4, (ax4, ax5) = plt.subplots(1, 2, figsize=(14, 5))
            
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
        st.header("Detailed Report")
        
        st.subheader("Configuration")
        config_col1, config_col2 = st.columns(2)
        
        with config_col1:
            st.write(f"**Index:** {index_params.name}")
            st.write(f"**Initial Investment:** ${initial_investment:,.0f}")
            st.write(f"**Regular Contribution:** ${regular_contribution:,.0f}")
            st.write(f"**Frequency:** {contribution_freq}")
            st.write(f"**Period:** {investment_years} years")
        
        with config_col2:
            st.write(f"**Tax Model:** {tax_model_option}")
            if tax_model != TaxModel.NONE:
                st.write(f"**Capital Gains Rate:** {capital_gains_rate*100:.1f}%")
                st.write(f"**Dividend Rate:** {dividend_tax_rate*100:.1f}%")
