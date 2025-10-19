#
# Description:
#   This script expands on the parametric Monte Carlo concept to model the long-term
#   financial outcomes of multiple sports betting strategies, including both simple
#   rules-based strategies and machine learning models.
#
#   The simulation is based on statistical "rules" derived from prior empirical
#   backtesting. The core parameters for each strategy—the true win probability and
#   the effective payout odds—are reverse-engineered from backtested Win Rate and ROI.
#
#   The simulation models a large population for each strategy, generating a
#   statistical distribution of final bankrolls and calculating advanced risk/return metrics.
#
# Output:
#   - report_parametric_monte_carlo_summary.txt: A comprehensive statistical report
#     with advanced metrics for all simulated strategies.
#   - monte_carlo_home_team.png: A histogram for the "Bet on Home Team" strategy.
#   - monte_carlo_random_forest.png: A histogram for the Random Forest model.
#
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats
import time

# --- Configuration ---
OUTPUT_REPORT_FILE = 'report_parametric_monte_carlo_summary.txt'

# --- Simulation Parameters ---
NUM_SIMULATIONS = 20000
INITIAL_BANKROLL = 1000
BETS_PER_SIMULATION = 500
STAKE_PER_BET = 10

# --- RULES OF THE GAME (Derived from the two data snippets you provided) ---
STRATEGIES_DATA = {
    "Bet on Home Team": {"win_prob": 0.457,  "roi": -0.0382},
    "Bet on Away Team": {"win_prob": 0.278,  "roi": -0.1632},
    "Bet on Favorite":  {"win_prob": 0.539,  "roi": -0.0415},
    "Bet on Underdog":  {"win_prob": 0.204,  "roi": -0.1081},
    "Random Bet":       {"win_prob": 0.323,  "roi": -0.1168},
    "XGBoost":          {"win_prob": 0.5151, "roi": -0.0142},
    "LightGBM":         {"win_prob": 0.5228, "roi": -0.0146},
    "Random Forest":    {"win_prob": 0.4881, "roi": -0.0154}
}
ML_STRATEGIES = ["XGBoost", "LightGBM", "Random Forest"]

# --- Helper Functions ---

def calculate_odds_from_roi(win_prob, roi):
    """
    Reverse-engineers the payout odds from Win Rate and ROI.
    """
    if win_prob == 0:
        return 1
    return (roi + 1) / win_prob

def run_simulation_for_strategy(win_prob, odds):
    final_bankrolls = np.full(NUM_SIMULATIONS, float(INITIAL_BANKROLL))
    active_mask = np.ones(NUM_SIMULATIONS, dtype=bool)

    for _ in range(BETS_PER_SIMULATION):
        if not np.any(active_mask):
            break
        
        outcomes = np.random.rand(np.sum(active_mask)) < win_prob
        profit_loss = np.where(outcomes, STAKE_PER_BET * (odds - 1), -STAKE_PER_BET)
        final_bankrolls[active_mask] += profit_loss
        active_mask[active_mask] = final_bankrolls[active_mask] >= STAKE_PER_BET

    final_bankrolls[final_bankrolls < STAKE_PER_BET] = 0
    return final_bankrolls


def generate_plot(bankrolls, strategy_name, filename):
    print(f"  -> Generating plot for '{strategy_name}' -> '{filename}'")
    population_term = "Machines" if strategy_name in ML_STRATEGIES else "Players"
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(20, 11), facecolor='#f8f9fa')
    ax.set_facecolor('#f8f9fa')
    sns.histplot(bankrolls, bins=1000, color='#001019', alpha=0.75, kde=False, fill=True, ax=ax)

    mean_val = bankrolls.mean()
    percent_won = np.sum(bankrolls > INITIAL_BANKROLL) / NUM_SIMULATIONS * 100
    percent_bankrupt = np.sum(bankrolls <= 0) / NUM_SIMULATIONS * 100

    ax.axvline(INITIAL_BANKROLL, color='#d62828', linestyle='--', linewidth=2.5, label='Starting Bankroll')
    ax.axvline(mean_val, color='#f77f00', linestyle='-', linewidth=2.5, label='Average Final Bankroll')
    
    params_text = (f"$\\bf{{Simulation\ Parameters}}$\n"
                   f"Initial Bankroll: ${INITIAL_BANKROLL:,}\n"
                   f"Bet Stake: ${STAKE_PER_BET}\n"
                   f"───────────────\n"
                   f"$\\bf{{Key\ Values}}$\n"
                   f"Starting Bankroll: ${INITIAL_BANKROLL:,.0f}\n"
                   f"Average Final Bankroll: ${mean_val:,.0f}\n")
    ax.text(0.99, 0.99, params_text, transform=ax.transAxes, fontsize=12, va='top', ha='right', bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7))

    segmentation_text = (f"$\\bf{{Population\ Segmentation}}$\n"
                         f"• Profitable: {percent_won:.1f}%\n"
                         f"• Unprofitable Majority: {(100 - percent_won - percent_bankrupt):.1f}%\n"
                         f"• Bankrupt ('Risk of Ruin'): {percent_bankrupt:.1f}%")
    ax.text(0.05, 0.99, segmentation_text, transform=ax.transAxes, fontsize=12, va='top', bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7))

    fig.suptitle(f'Monte Carlo Simulation: "{strategy_name}" Strategy', fontsize=20, fontweight='bold')
    ax.set_title(f'Final bankroll distribution for {NUM_SIMULATIONS:,} {population_term.lower()}, each making {BETS_PER_SIMULATION} random bets', fontsize=14, pad=10)
    ax.set_xlabel('Final Bankroll ($)', fontsize=12)
    ax.set_ylabel(f'Number of {population_term} (Frequency)', fontsize=12)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'${int(x):,}'))
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_ylim(bottom=0)
    sns.despine(left=True, bottom=True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def get_report_section(bankrolls, strategy_name, win_prob, odds):
    ev_per_bet = (win_prob * (STAKE_PER_BET * (odds - 1))) - ((1 - win_prob) * STAKE_PER_BET)
    theoretical_mean = INITIAL_BANKROLL + (ev_per_bet * BETS_PER_SIMULATION)
    
    mean_val = bankrolls.mean()
    median_val = np.median(bankrolls)
    std_dev = bankrolls.std()
    
    returns = (bankrolls - INITIAL_BANKROLL) / INITIAL_BANKROLL
    downside_returns = returns[returns < 0]
    expected_return = returns.mean()
    downside_std = downside_returns.std() if len(downside_returns) > 1 else 0
    sortino_ratio = expected_return / downside_std if downside_std > 0 else 0
    value_at_risk_95 = np.percentile(returns, 5) * INITIAL_BANKROLL

    profitable_count = np.sum(bankrolls > INITIAL_BANKROLL)
    bankrupt_count = np.sum(bankrolls <= 0)
    unprofitable_count = NUM_SIMULATIONS - profitable_count - bankrupt_count

    report = [
        f"\n{'='*85}",
        f"## STRATEGY ANALYSIS: {strategy_name.upper()}",
        f"{'='*85}\n",
        "## Mathematical Expectation",
        f"  - True Win Probability: {win_prob:.2%}",
        f"  - Calculated Payout Odds: {odds:.4f}",
        f"  - Expected Value (EV) per ${STAKE_PER_BET} Bet: ${ev_per_bet:.4f}\n",
        "## Central Tendency",
        f"  - Theoretical Mean Final Bankroll: ${theoretical_mean:,.2f}",
        f"  - Actual Mean Final Bankroll (Simulated): ${mean_val:,.2f}",
        f"  - Median Final Bankroll (Typical Outcome): ${median_val:,.2f}\n",
        "## Risk & Return Profile",
        f"  - Standard Deviation: ${std_dev:,.2f}",
        f"  - Sortino Ratio (Downside Risk-Adjusted Return): {sortino_ratio:.4f}",
        f"  - 95% Value at Risk (VaR): You can expect to lose at least ${-value_at_risk_95:,.2f} in 5% of cases.\n",
        "## Final Population Segmentation",
        f"  - Profitable: {profitable_count:,} ({profitable_count / NUM_SIMULATIONS:.1%})",
        f"  - Unprofitable Majority: {unprofitable_count:,} ({unprofitable_count / NUM_SIMULATIONS:.1%})",
        f"  - Bankrupt ('Risk of Ruin'): {bankrupt_count:,} ({bankrupt_count / NUM_SIMULATIONS:.1%})"
    ]
    return "\n".join(report)

# --- Main Simulation Script ---
def main():
    print("Running Parametric Monte Carlo Simulation for all strategies...")
    start_time = time.time()
    full_report_content = [
        f"{'='*85}",
        "        PARAMETRIC MONTE CARLO: COMPREHENSIVE STATISTICAL REPORT",
        f"{'='*85}\n",
        "## Global Simulation Parameters",
        f"  - Simulation Population (Players/Machines): {NUM_SIMULATIONS:,}",
        f"  - Initial Bankroll: ${INITIAL_BANKROLL:,.0f}",
        f"  - Bets per Simulation: {BETS_PER_SIMULATION}",
        f"  - Stake per Bet: ${STAKE_PER_BET:,.0f}"
    ]

    for name, data in STRATEGIES_DATA.items():
        print(f"\nSimulating strategy: '{name}'...")
        win_prob = data['win_prob']
        roi = data['roi']
        odds = calculate_odds_from_roi(win_prob, roi)
        
        bankrolls = run_simulation_for_strategy(win_prob, odds)
        
        if name == "Bet on Home Team":
            generate_plot(bankrolls, name, 'monte_carlo_home_team.png')
        elif name == "Random Forest":
            generate_plot(bankrolls, name, 'monte_carlo_random_forest.png')
            
        report_section = get_report_section(bankrolls, name, win_prob, odds)
        full_report_content.append(report_section)
    
    print(f"\nAll simulations complete. Total time: {time.time() - start_time:.2f} seconds.")
    print(f"Generating final comprehensive report -> '{OUTPUT_REPORT_FILE}'...")
    with open(OUTPUT_REPORT_FILE, 'w') as f:
        f.write("\n".join(full_report_content))

    print("\nFinal analysis complete.")
    print(f"- Final Figures: 'fmonte_carlo_home_team.png', 'monte_carlo_random_forest.png'")
    print(f"- Final Statistical Report: '{OUTPUT_REPORT_FILE}'")

if __name__ == '__main__':
    main()
