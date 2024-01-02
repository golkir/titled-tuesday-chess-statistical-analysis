# Chess.com "Titled Tuesday" Statistical Analysis

This repository contains tools for statistical analysis of chess games played in "Titled Tuesday" events on chess.com between July 2022 and December 2023.

## Dataset

- `scrape_and_save.py`: This script retrieves game data from "Titled Tuesday" via chess.com API and web pages scraping, preprocesses it, and saves to MongoDB. The data includes games of all players who ranked 1-25 in any of the "Titled Tuesday" event.

- `generate_dataset.py`: This script turns MongoDB players collection into Pandas DataFrame and then saves it as a `.csv` dataset.

## Statistical Analysis

The `analysis.py` script contains code to perform various types of statistical analysis with chess data:

- Descriptive statistics (mean, number of wins, losses, rank distribution).
- Game accuracy distribution analysis between tournaments rounds to find differences in accuracy between initial and final rounds
- K-means clustering and k-nearest neighbors analysis.
- Linear regression to predict game results based on player accuracy.
- Statistical tests (Kolmogorov-Smirnov test, Shapiro-Wilk normality test).
- Kernel Density Estimation (KDE) and visualization
- Maximum Likelihood Estimation of mean and standard deviation assuming chess data normality
- Correlation analysis of score and accuracy (Spearman's rank corellation coefficient)
