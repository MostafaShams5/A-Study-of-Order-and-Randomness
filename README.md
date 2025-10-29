# A Study of Order and Randomness

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17475734.svg)](https://doi.org/10.5281/zenodo.17475734)

This repository contains the complete dataset, analysis scripts, and simulation code for the research paper: **"Does the House Really Always Win? What AI Found in the Chaos: A Study of Order and Randomness."**

This is the official repository for the paper, as cited in the publication.

This project is a comprehensive, data-driven investigation into the enduring maxim that "the house always wins." It deploys artificial intelligence not as a gambler, but as a detective to hunt for a predictive "edge" in the chaos of modern betting markets.

---

## The Core Investigation

To test this claim, the analysis was split into two distinct environments to first calibrate the AI and then test it against a complex, real-world system.

### Part I: The Algorithmic Void (The 'Crash' Game)

Before the AI could be trusted to find a *real* pattern, it had to be proven that it wouldn't be fooled by *fake* ones.

* **Objective:** To test the AI's ability to discern true, mathematical randomness.
* **Method:** A live web scraper (`crash_game_scraper.py`) collected thousands of rounds (2,596 total) from a provably random online 'crash' game. This data was then subjected to a rigorous statistical analysis (`analysis_crash_game.py`), including the Augmented Dickey-Fuller test, ACF/PACF plots, and testing sequence-based neural networks (LSTM, Autoencoders).
* **Conclusion:** The AI and statistical tests correctly confirmed the game was pure chaos. The models could find no exploitable patterns, proving they were well-calibrated and would not "see ghosts" in random noise.

### Part II: The Human Colosseum (The English Premier League)

With the tools calibrated, the investigation turned to a complex, human-driven system: 20 years of the English Premier League.

* **Objective:** To determine if a sophisticated AI could find a profitable, exploitable pattern in a real-world, information-rich market.
* **Method:** Three machine learning models (XGBoost, LightGBM, and Random Forest) were trained on 7,404 matches using *only* the bookmaker's odds as features. Their financial performance was then rigorously backtested (`backtesting_ml_strategies.py`) and benchmarked against simple human strategies (e.g., "Bet on Favorite," "Bet on Home Team").
* **Conclusion:** The AI models **failed to generate a net profit**. However, they **dramatically outperformed** all human-heuristic strategies by mastering the art of **capital preservation**.

---

## Key Findings

The central discovery of this research was not a secret to winning, but a confirmation of *why* the house wins.

1.  **The AI Played the Book, Not the Game:** A feature-importance analysis revealed the AI didn't learn how to predict football. It learned how to **interpret the market**. The bookmaker's odds (specifically the Home Win odd, B365H) were consistently the most important predictive feature.

2.  **The "Favorite-Longshot Bias" Paradox:** The research provides a classic demonstration of this bias. The human strategy "Bet on Favorite" had the highest win rate (53.9%), making the player *feel* successful. However, it was one of the worst financial performers (ROI -4.15%), proving that **winning bets is not the same as making money**.

3.  **Markets are Efficient:** The AI's success at mitigating loss proves the bookmaker's odds are a profoundly efficient signal. They already reflect all available information so accurately that no simple edge can be found, a core tenet of the Efficient Market Hypothesis.

4.  **The AI's "Edge" is Catastrophe Aversion:** While human strategies lost money rapidly, the AI models achieved a near-zero Risk of Ruin (0.0%). Their advantage was not in profit generation, but in superior risk management and the avoidance of catastrophic losses.

5.  **A Mirror to Human Psychology:** The findings move from economics to psychology. The human desire to bet, even when faced with a structural disadvantage, is explained by powerful cognitive biases like the **"Illusion of Control"** (believing our research gives us an edge) and **"Prospect Theory"** (feeling the pain of a loss twice as much as the pleasure of a win).

---

## About This Repository

This project is organized into its core components:

* **`Does_the_House_Really_Always_Win?_What_AI_Found_in_the_Chaos:_Study_of_Order_and_Randomness.pdf`**: The complete research paper. This is the best place to start.

* **`/code`**: All Python scripts used for the analysis.
    * `crash_game_scraper.py`: The Playwright-based scraper used to acquire data from the live 'crash' game.
    * `analysis_crash_game.py`: Performs the full statistical analysis, randomness testing (ADF), and neural network calibration on the crash game data.
    * `backtesting_famous_strategies.py`: Simulates the financial performance of simple human strategies (Bet on Favorite, Home Team, etc.) on the EPL dataset.
    * `backtesting_ml_strategies.py`: Trains, backtests, and compares the financial performance of the three AI models (XGBoost, LightGBM, Random Forest).
    * `simulation_parametric_monte_carlo.py`: Runs the large-scale (20,000-player) Monte Carlo simulations to generate the final risk/return profiles (e.g., Figures 8 and 10).

* **`/data`**: The raw datasets used in the study.
    * `1xbet_crash_data.csv`: The scraped dataset of 2,596 crash game rounds.
    * `bet365.csv`: The historical dataset of 7,404 EPL matches and their odds (Seasons 2000-01 to 2020-21).

* **`/figures`**: All plots and charts generated by the analysis scripts, as seen in the paper.

* **`/reports`**: The raw `.txt` output files containing detailed statistical summaries from the analysis scripts.

---

## How to Run the Analysis

To reproduce the findings in this paper:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/MostafaShams5/A-Study-of-Order-and-Randomness.git](https://github.com/MostafaShams5/A-Study-of-Order-and-Randomness.git)
    cd A-Study-of-Order-and-Randomness
    ```

2.  **Create a virtual environment (Recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Install Playwright browsers (for the scraper):**
    ```bash
    playwright install
    ```

5.  **Run the analysis scripts:**
    You can run the Python scripts located in the `/code` directory to regenerate all reports and figures.

---

## License

This project (including the paper, code, and data) is licensed under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license.

This means you are free to share and adapt this work for any purpose, as long as you give appropriate credit by citing the paper (see the "How to Cite" section).

---

## How to Cite

This repository is the official code and data for the research paper. If you use this work, please cite the paper using the following:

**APA Style**
Shams, M. (2025). *Does the House Really Always Win? What AI Found in the Chaos: A Study of Order and Randomness* (Version 1.1) [Working Paper]. Zenodo. https://doi.org/10.5281/zenodo.17475734

**BibTeX**
```bibtex
@software{shams_2025_17475734,
  author       = {Shams, Mostafa},
  title        = {{Does the House Really Always Win? What AI Found in 
                   the Chaos: A Study of Order and Randomness}},
  version      = {1.1},
  publisher    = {Zenodo},
  year         = {2025},
  doi          = {10.5281/zenodo.17475734},
  url          = {[https://doi.org/10.5281/zenodo.17475734](https://doi.org/10.5281/zenodo.17475734)}
}
