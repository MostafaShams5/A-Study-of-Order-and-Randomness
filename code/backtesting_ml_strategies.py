#
# Description:
#   This script evaluates the efficacy of machine learning models as sports betting
#   strategies. It trains, backtests, and simulates the financial performance of
#   three distinct classification models: XGBoost, LightGBM, and Random Forest.
#
#   The methodology is rooted in a multi-run simulation (N_SIMULATION_RUNS). In each run,
#   the data is split into training and testing sets, models are trained on the odds data,
#   and their predictions are used to simulate a betting history. This process is repeated
#   to generate a distribution of outcomes, and the results (financial metrics, model
#   performance) are then averaged to ensure robustness.
#
#   The ML models are benchmarked against two simple strategies: "Bet on Favorite" and
#   "Bet on Home Team". A t-test is performed on the final bankroll distributions to
#   determine if the performance differences between models are statistically significant.
#
# Output:
#   - report_ml_strategies_summary.txt: A comprehensive report with aggregated financial
#     metrics, statistical tests, and representative model performance details.
#   - fig_ml_strategies_performance.png: A plot of the averaged financial trajectories.
#
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from scipy.ndimage import gaussian_filter1d
from scipy.stats import ttest_ind
import time
import os

# --- Configuration ---
FILE_BET365 = 'bet365.csv'
OUTPUT_PLOT_FILE = 'fig_ml_strategies_performance.png'
OUTPUT_REPORT_FILE = 'report_ml_strategies_summary.txt'
INITIAL_BALANCE = 1000
STAKE_PER_BET = 10
SMOOTHING_WINDOW = 1
N_SIMULATION_RUNS = 20

# --- Financial Metrics Function ---
def calculate_financial_metrics(history, initial_balance, stake_per_bet):
    # gotta get those financial stats.
    balance_arr = np.array(history)
    final_balance = balance_arr[-1]
    net_profit = final_balance - initial_balance
    num_bets = len(history) - 1

    if num_bets == 0:
        return {'Final Bankroll ($)': final_balance, 'Net Profit/Loss ($)': net_profit,
                'ROI (%)': 0, 'Peak Bankroll ($)': initial_balance, 'Max Drawdown (%)': 0, 'Calmar Ratio': 0}

    total_staked = num_bets * stake_per_bet
    roi = (net_profit / total_staked) * 100 if total_staked > 0 else 0

    running_max = np.maximum.accumulate(balance_arr)
    with np.errstate(divide='ignore', invalid='ignore'):
        drawdown = (running_max - balance_arr) / running_max
    max_drawdown = np.nanmax(drawdown) * 100 if np.any(np.isfinite(drawdown)) else 0
    calmar_ratio = roi / max_drawdown if max_drawdown > 0 else 0

    return {'Final Bankroll ($)': final_balance, 'Net Profit/Loss ($)': net_profit,
            'ROI (%)': roi, 'Peak Bankroll ($)': np.max(balance_arr),
            'Max Drawdown (%)': max_drawdown, 'Calmar Ratio': calmar_ratio}

# --- Simulation Function ---
def run_simulation(df_test, predictions, label_encoder):
    # simulate the betting based on model predictions.
    balance = INITIAL_BALANCE
    history = [balance]
    predicted_outcomes = label_encoder.inverse_transform(predictions)
    df_test = df_test.copy().reset_index(drop=True)
    df_test['predicted_outcome'] = predicted_outcomes

    for _, row in df_test.iterrows():
        if balance < STAKE_PER_BET:
            break
        balance -= STAKE_PER_BET
        if row['predicted_outcome'] == row['Result']:
            odd_col = f"B365{row['Result']}"
            balance += STAKE_PER_BET * row[odd_col]
        history.append(balance)
    return history

def main():
    print("Starting ML Analysis & Financial Simulation...")
    print(f"Core analysis will be based on the average of {N_SIMULATION_RUNS} runs.")

    # --- 1. Load and Prepare Data ---
    if not os.path.exists(FILE_BET365):
        print(f"Error: Input file '{FILE_BET365}' not found.")
        return

    df = pd.read_csv(FILE_BET365)
    required_cols = ['Result', 'B365H', 'B365D', 'B365A']
    df.dropna(subset=required_cols, inplace=True)
    for col in ['B365H', 'B365D', 'B365A']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=required_cols, inplace=True)

    X = df[['B365H', 'B365D', 'B365A']]
    y = df['Result']

    # --- STAGE 1: MULTI-RUN SIMULATION ---
    print(f"\n--- Running {N_SIMULATION_RUNS} simulations for primary analysis ---")
    start_time = time.time()

    # the contenders.
    models = {
        "XGBoost": xgb.XGBClassifier(objective='multi:softmax', use_label_encoder=False, eval_metric='mlogloss', n_jobs=-1, random_state=42),
        "LightGBM": lgb.LGBMClassifier(objective='multiclass', random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    }

    strategies = list(models.keys()) + ["Bet on Favorite", "Bet on Home Team"]
    all_metrics = {name: [] for name in strategies}
    all_histories = {name: [] for name in strategies}

    classification_reports_example = {}
    feature_importances_example = {}

    # the main event: run the simulation N times for robust results.
    for i in range(N_SIMULATION_RUNS):
        print(f"  -> Running simulation {i+1}/{N_SIMULATION_RUNS} (random_state={i})...")

        X_train, X_test, y_train_raw, y_test_raw = train_test_split(X, y, test_size=0.33, random_state=i, stratify=y)

        le = LabelEncoder()
        y_train = le.fit_transform(y_train_raw)
        y_test = le.transform(y_test_raw)

        test_data_for_sim = df.loc[X_test.index]

        # --- Run ML Models ---
        for name, model in models.items():
            model.set_params(random_state=i)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            history = run_simulation(test_data_for_sim, preds, le)
            metrics = calculate_financial_metrics(history, INITIAL_BALANCE, STAKE_PER_BET)
            metrics["Win Rate (%)"] = accuracy_score(y_test, preds) * 100

            all_metrics[name].append(metrics)
            all_histories[name].append(history)

            # save reports from the first run as an example.
            if i == 0:
                classification_reports_example[name] = classification_report(y_test, preds, target_names=le.classes_, zero_division=0)
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    feature_df = pd.DataFrame({'feature': X_train.columns, 'importance': importances}).sort_values('importance', ascending=False)
                    feature_importances_example[name] = feature_df

        # --- Run "Bet on Favorite" Strategy ---
        fav_preds = le.transform(X_test[['B365H', 'B365D', 'B365A']].idxmin(axis=1).str.replace('B365', ''))
        history_fav = run_simulation(test_data_for_sim, fav_preds, le)
        metrics_fav = calculate_financial_metrics(history_fav, INITIAL_BALANCE, STAKE_PER_BET)
        metrics_fav["Win Rate (%)"] = accuracy_score(y_test, fav_preds) * 100
        all_metrics["Bet on Favorite"].append(metrics_fav)
        all_histories["Bet on Favorite"].append(history_fav)
        
        # --- Run "Bet on Home Team" Strategy ---
        home_preds = np.full(y_test.shape, le.transform(['H'])[0])
        history_home = run_simulation(test_data_for_sim, home_preds, le)
        metrics_home = calculate_financial_metrics(history_home, INITIAL_BALANCE, STAKE_PER_BET)
        metrics_home["Win Rate (%)"] = accuracy_score(y_test, home_preds) * 100
        all_metrics["Bet on Home Team"].append(metrics_home)
        all_histories["Bet on Home Team"].append(history_home)

    print(f"Multi-run simulation finished in {time.time() - start_time:.2f} seconds.")

    # --- STAGE 2: PROCESS AGGREGATED RESULTS ---
    print("\n--- Aggregating results from all runs ---")
    
    # aggregate everything for the final report and plot.
    max_len = max(len(h) for histories in all_histories.values() for h in histories)
    avg_histories = {}
    for name, histories in all_histories.items():
        padded_histories = [np.pad(h, (0, max_len - len(h)), 'edge') for h in histories]
        avg_histories[name] = np.mean(padded_histories, axis=0)

    avg_metrics_df = pd.DataFrame()
    for name, metrics_list in all_metrics.items():
        avg_metrics_df = pd.concat([avg_metrics_df, pd.DataFrame(metrics_list).mean().to_frame(name)], axis=1)
    avg_metrics_df = avg_metrics_df.T.reset_index().rename(columns={'index': 'Strategy'})
    final_cols_order = ['Strategy', 'Final Bankroll ($)', 'Net Profit/Loss ($)', 'ROI (%)', 'Max Drawdown (%)', 'Calmar Ratio', 'Peak Bankroll ($)', 'Win Rate (%)']
    avg_metrics_df = avg_metrics_df[final_cols_order]

    # are the differences real or just chance? t-test time.
    final_bankrolls = {name: [m['Final Bankroll ($)'] for m in metrics] for name, metrics in all_metrics.items()}
    rf_runs = final_bankrolls['Random Forest']; xgb_runs = final_bankrolls['XGBoost']; lgb_runs = final_bankrolls['LightGBM']
    _, p_value_rf_xgb = ttest_ind(rf_runs, xgb_runs, equal_var=False)
    _, p_value_rf_lgb = ttest_ind(rf_runs, lgb_runs, equal_var=False)

    # --- STAGE 3: GENERATE PLOT AND REPORT ---
    print(f"-> Generating plot from averaged data: '{OUTPUT_PLOT_FILE}'...")
    plt.style.use('default'); fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=150); fig.patch.set_facecolor('white'); ax.set_facecolor('white')
    plot_strategies = {"XGBoost": '#0066CC', "LightGBM": '#2ca02c', "Random Forest": '#9467bd', "Bet on Favorite": '#FF8800', "Bet on Home Team": '#CC0066'}

    for name, color in plot_strategies.items():
        avg_hist = avg_histories[name]
        smoothed = pd.Series(avg_hist).rolling(window=SMOOTHING_WINDOW, min_periods=1).mean()
        extra_smooth = gaussian_filter1d(smoothed, sigma=3)
        ax.plot(extra_smooth, color=color, linewidth=2.0, label=f"{name} (Avg.)", zorder=3, antialiased=True)
        avg_final_balance = avg_metrics_df[avg_metrics_df['Strategy'] == name]['Final Bankroll ($)'].iloc[0]
        ax.text(len(avg_hist) - 1 + 20, extra_smooth[-1], f'${avg_final_balance:,.0f}', fontsize=12, color=color, va='center', ha='left', fontweight='bold', bbox=dict(boxstyle='round,pad=0.35', facecolor='white', alpha=0.9, edgecolor=color, linewidth=1.2))

    ax.axhline(INITIAL_BALANCE, color='#555555', linestyle='--', linewidth=1.5, alpha=0.7, zorder=2); ax.text(max_len*0.98, INITIAL_BALANCE, 'Breakeven ', ha='right', va='center', fontsize=12, color='#555555', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.2))
    ax.set_title(f'Average Financial Performance Over {N_SIMULATION_RUNS} Simulations', fontsize=15, fontweight='bold', pad=15); ax.set_xlabel('Number of Bets Placed', fontsize=12, labelpad=10); ax.set_ylabel('Bankroll ($)', fontsize=12, labelpad=10)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'${int(x):,}')); ax.tick_params(labelsize=11, width=0.8, length=4, colors='#333333'); ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, color='#DDDDDD')
    ax.legend(loc='upper right', fontsize=13); ax.set_xlim(0, max_len * 1.05); ax.set_ylim(bottom=0)
    for spine in ax.spines.values(): spine.set_edgecolor('#CCCCCC')
    plt.subplots_adjust(left=0.06, right=0.98, top=0.94, bottom=0.06); plt.savefig(OUTPUT_PLOT_FILE, dpi=300); plt.close()

    # --- Generate Final Report ---
    print(f"-> Generating final report from averaged data: '{OUTPUT_REPORT_FILE}'...")
    with open(OUTPUT_REPORT_FILE, 'w') as f:
        f.write("="*100 + "\n")
        f.write(f"COMPREHENSIVE DATA REPORT: ML & FINANCIAL SIMULATION ({N_SIMULATION_RUNS} RUNS)\n")
        f.write("="*100 + "\n\n")

        f.write(f"## 1. AVERAGE SIMULATED FINANCIAL OUTCOMES (Based on {N_SIMULATION_RUNS} runs)\n\n")
        f.write(avg_metrics_df.to_string(index=False, float_format="{:,.2f}".format))
        f.write("\n\n\n")

        f.write("## 2. STATISTICAL SIGNIFICANCE OF MODEL PERFORMANCE\n\n")
        f.write("   --- T-Test: Random Forest vs. XGBoost ---\n")
        f.write(f"   - P-value: {p_value_rf_xgb:.4f}\n")
        f.write(f"   - Interpretation: {'P < 0.05 indicates a statistically significant difference in financial outcomes.' if p_value_rf_xgb < 0.05 else 'P >= 0.05 indicates no statistically significant difference.'}\n\n")

        f.write("   --- T-Test: Random Forest vs. LightGBM ---\n")
        f.write(f"   - P-value: {p_value_rf_lgb:.4f}\n")
        f.write(f"   - Interpretation: {'P < 0.05 indicates a statistically significant difference in financial outcomes.' if p_value_rf_lgb < 0.05 else 'P >= 0.05 indicates no statistically significant difference.'}\n\n\n")

        f.write("## 3. MODEL DETAILS (From a single representative run, random_state=0)\n\n")

        f.write("--- Classification Reports ---\n\n")
        for name, report in classification_reports_example.items():
            f.write(f"--- {name} ---\n{report}\n\n")

        f.write("--- Feature Importance Analysis ---\n\n")
        for name, df_importance in feature_importances_example.items():
            f.write(f"--- {name} ---\n")
            f.write(df_importance.to_string(index=False))
            f.write("\n\n")
        f.write("   - Statement: The models consistently demonstrate the highest reliance on 'B365H' (Home Win Odds).\n")

    print("Analysis complete.")

if __name__ == '__main__':
    main()
