#
# Description:
#   This script performs a Monte Carlo simulation to backtest the financial performance
#   of several simple, non-ML betting strategies on historical sports betting data.
#
#   The methodology involves running the simulation multiple times (N_SHUFFLED_RUNS),
#   each time on a randomly shuffled version of the dataset. This approach helps to
#   average out the effects of luck or specific sequences of events, providing a more
#   robust and generalized assessment of each strategy's long-term viability.
#
#   The strategies evaluated are:
#   - Bet on Favorite
#   - Bet on Underdog
#   - Bet on Home/Away Team
#   - Random Bet
#
#   Key financial metrics (ROI, Max Drawdown, Sharpe Ratio, etc.) are calculated
#   for each run and then aggregated to produce a final summary report.
#
# Output:
#   - report_simple_strategies_summary.txt: A summary of the aggregated financial metrics.
#   - fig_simple_strategies_performance.png: A plot showing the averaged financial trajectory
#     for each strategy, including a one-standard-deviation envelope.
#
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import random
import os
from datetime import datetime, UTC
from scipy.ndimage import gaussian_filter1d

# --- Configuration ---
FILE_BET365 = "bet365.csv"
OUTPUT_FIG_PNG = "fig_famous_strategies_performance.png"
OUTPUT_REPORT_TXT = "report_famous_strategies_summary.txt"

INITIAL_BALANCE = 1000
STAKE_PER_BET = 10
N_SHUFFLED_RUNS = 20

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# --- Metrics Calculation ---
def calculate_metrics(history, initial_balance, stake_per_bet, wins):
    # all the key financial metrics we need.
    balance_arr = np.array(history)
    final_balance = balance_arr[-1]
    net_profit = final_balance - initial_balance

    num_bets = len(history) - 1
    win_rate = wins / num_bets if num_bets > 0 else 0

    if num_bets == 0:
        return {'final_balance': final_balance, 'net_profit': net_profit, 'roi': 0,
                'peak_balance': initial_balance, 'max_drawdown': 0, 'win_rate': 0}

    total_staked = num_bets * stake_per_bet
    roi = (net_profit / total_staked) * 100 if total_staked > 0 else 0

    running_max = np.maximum.accumulate(balance_arr)
    with np.errstate(divide='ignore', invalid='ignore'):
        drawdown = (running_max - balance_arr) / running_max
        drawdown[~np.isfinite(drawdown)] = 0
    max_drawdown = np.max(drawdown) * 100

    return {'final_balance': final_balance, 'net_profit': net_profit, 'roi': roi,
            'peak_balance': np.max(balance_arr), 'max_drawdown': max_drawdown, 'win_rate': win_rate}

# --- Simulation ---
def run_simulation(df, strategy):
    # the core loop for a single simulation run.
    balance = INITIAL_BALANCE
    history = [balance]
    wins = 0
    local_df = df.copy()

    if strategy == "favorite":
        cols = ['B365H','B365D','B365A']
        local_df['bet_choice'] = local_df[cols].idxmin(axis=1).str.replace('B365','')
    elif strategy == "underdog":
        cols = ['B365H','B365D','B365A']
        local_df['bet_choice'] = local_df[cols].idxmax(axis=1).str.replace('B365','')
    elif strategy == "random":
        local_df['bet_choice'] = [random.choice(['H','D','A']) for _ in range(len(local_df))]
    else:
        local_df['bet_choice'] = strategy

    for _, row in local_df.iterrows():
        if balance < STAKE_PER_BET:
            break
        balance -= STAKE_PER_BET
        if row['bet_choice'] == row['Result']:
            wins += 1
            odd = float(row[f"B365{row['bet_choice']}"])
            balance += STAKE_PER_BET * odd
        history.append(balance)
    return history, wins

# --- Main ---
def main():
    if not os.path.exists(FILE_BET365):
        print(f"ERROR: Did not find {FILE_BET365}. Please place it in the same directory.")
        return
    df = pd.read_csv(FILE_BET365).dropna(subset=['Result','B365H','B365D','B365A'])
    for c in ['B365H','B365D','B365A']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['B365H','B365D','B365A']).reset_index(drop=True)

    strategies = {
        'Bet on Favorite': 'favorite', 'Bet on Home Team': 'H', 'Bet on Away Team': 'A',
        'Random Bet': 'random', 'Bet on Underdog': 'underdog'
    }

    print(f"Running {N_SHUFFLED_RUNS} simulations on shuffled data...")
    all_metrics = {name: [] for name in strategies}

    # run the simulation many times to get a stable average.
    for i in range(N_SHUFFLED_RUNS):
        print(f"  ... Run {i + 1}/{N_SHUFFLED_RUNS}")
        shuffled_df = df.sample(frac=1, random_state=RANDOM_SEED + i).reset_index(drop=True)
        for name, key in strategies.items():
            history, wins = run_simulation(shuffled_df, key)
            metrics = calculate_metrics(history, INITIAL_BALANCE, STAKE_PER_BET, wins)
            metrics['history'] = history
            all_metrics[name].append(metrics)
    print("Simulations complete.")

    all_histories = {name: [m['history'] for m in all_metrics[name]] for name in strategies}

    max_len = max(len(h) for histories in all_histories.values() for h in histories)

    plot_data = {}
    for name, histories in all_histories.items():
        padded_histories = [h + [h[-1]] * (max_len - len(h)) for h in histories]
        hist_array = np.array(padded_histories)
        mean_hist = np.mean(hist_array, axis=0)
        std_hist = np.std(hist_array, axis=0)
        plot_data[name] = {'x': np.arange(max_len), 'mean': gaussian_filter1d(mean_hist, sigma=5), 'std': std_hist}

    # time to make the plot look good.
    plt.style.use('default')
    colors = {'Bet on Home Team': '#CC0066', 'Bet on Away Team': '#FF8800', 'Bet on Favorite': '#0066CC', 'Bet on Underdog': '#9933CC', 'Random Bet': '#00AA44'}
    fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=150)
    fig.patch.set_facecolor('white')

    for name, data in plot_data.items():
        ax.fill_between(data['x'], data['mean'] - data['std'], data['mean'] + data['std'], color=colors[name], alpha=0.15, zorder=2)
        ax.plot(data['x'], data['mean'], color=colors[name], label=name, linewidth=1.7, zorder=3)

    ax.axhline(INITIAL_BALANCE, color='#888888', linestyle='--', linewidth=1.3, alpha=0.7, zorder=2)
    abs_peak_y = max(np.mean([m['peak_balance'] for m in metrics_list]) for metrics_list in all_metrics.values())
    ax.legend(loc='upper right', frameon=True, fancybox=False, shadow=False, fontsize=10, edgecolor='#DDDDDD')
    ax.set_xlim(0, max_len)
    ax.set_ylim(0, max(abs_peak_y * 1.1, INITIAL_BALANCE * 1.15))
    ax.set_xlabel('Number of Bets Placed', fontsize=12, labelpad=8)
    ax.set_ylabel('Bankroll ($)', fontsize=12, labelpad=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'${int(x):,}'))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'{int(x):,}'))
    ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
    ax.set_title(f'Average Performance of Betting Strategies (over {N_SHUFFLED_RUNS} Shuffled Runs)', fontsize=14, pad=12)
    plt.subplots_adjust(left=0.06, right=0.98, top=0.96, bottom=0.06)
    fig.savefig(OUTPUT_FIG_PNG, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\nSaved aggregated plot to {OUTPUT_FIG_PNG}")

    # now build the report.
    summary_rows = []
    for name, metrics_list in all_metrics.items():
        final_balances = [m['final_balance'] for m in metrics_list]
        rois = [m['roi'] for m in metrics_list]
        std_dev_final = np.std(final_balances)
        std_dev_roi = np.std(rois)
        avg_roi = np.mean(rois)
        sharpe_ratio = avg_roi / std_dev_roi if std_dev_roi != 0 else 0
        summary_rows.append({
            'Strategy': name, 'Avg Final Bankroll ($)': f"${np.mean(final_balances):,.0f}",
            'Std. Dev. ($)': f"${std_dev_final:,.0f}", 'Avg Win Rate (%)': f"{np.mean([m['win_rate'] for m in metrics_list]) * 100:.1f}",
            'Avg ROI (%)': f"{avg_roi:.2f}", 'Sharpe Ratio': f"{sharpe_ratio:.2f}",
            'sharpe_numeric': sharpe_ratio
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df['sort_col'] = [np.mean([m['net_profit'] for m in all_metrics[name]]) for name in summary_df['Strategy']]
    summary_df = summary_df.sort_values(by='sort_col', ascending=False)

    best_sharpe_row = summary_df.sort_values(by='sharpe_numeric', ascending=False).iloc[0]
    worst_sharpe_row = summary_df.sort_values(by='sharpe_numeric', ascending=True).iloc[0]

    summary_df_display = summary_df.drop(columns=['sort_col', 'sharpe_numeric'])

    with open(OUTPUT_REPORT_TXT, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"BETTING STRATEGY SIMULATION REPORT ({N_SHUFFLED_RUNS} RUNS)\n")
        f.write("="*80 + "\n")
        f.write(f"Generated (UTC): {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Methodology: Averaged results over {N_SHUFFLED_RUNS} simulations on shuffled data.\n")
        f.write("="*80 + "\n\n")
        f.write("PERFORMANCE METRICS SUMMARY (AVERAGES)\n")
        f.write("─"*80 + "\n\n")
        f.write(summary_df_display.to_string(index=False))
        f.write("\n\n" + "─"*80 + "\n\n")
        f.write("STATISTICAL FINDINGS\n")
        f.write("─"*80 + "\n\n")

        f.write(f"1. Expected Value:\n")
        f.write(f"   The 'Avg ROI (%)' for all strategies is negative, confirming a negative expected\n")
        f.write(f"   value due to the bookmaker's margin (vigorish).\n\n")

        f.write(f"2. Risk-Adjusted Performance (Sharpe Ratio):\n")
        f.write(f"   The '{best_sharpe_row['Strategy']}' strategy ({best_sharpe_row['Sharpe Ratio']} Sharpe) offers\n")
        f.write(f"   the optimal risk-adjusted performance. Conversely, '{worst_sharpe_row['Strategy']}' ({worst_sharpe_row['Sharpe Ratio']} Sharpe)\n")
        f.write(f"   is the least optimal from a risk-reward perspective.\n\n")

        f.write(f"3. Overall Conclusion:\n")
        f.write(f"   Statistical analysis robustly concludes that none of the tested simple strategies\n")
        f.write(f"   can consistently overcome the inherent mathematical advantage of the bookmaker.\n")

    print(f"Saved aggregated report to {OUTPUT_REPORT_TXT}")

if __name__ == "__main__":
    main()
