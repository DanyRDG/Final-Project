# **Predicting International Football Success from Pre-Tournament Data**
**Statistical Analysis & Sports Data Visualization**

---

## Problem Statement / Motivation

Football has always been my first passion, and before every major international tournament, I often find myself debating with friends about which team is most likely to win based on their recent performances. These discussions usually rely on intuition or subjective impressions of “form”, but there is rarely a quantitative basis for such claims.

The goal of this project is to examine whether pre-tournament information can be used to predict how national teams perform in major international tournaments and which model of machine-learning works the best. More specifically, the project investigates whether data available before a tournament starts can help anticipate tournament outcomes, including identifying teams likely to make deep runs or win the competition. Using historical international football data from the Kaggle dataset “International Football Results from 1872 to 2025”, the analysis focuses on tournaments from the modern football era (2000 onward).

---

## Planned Approach & Technologies

The project is implemented entirely in Python, using libraries such as pandas and NumPy for data processing, matplotlib for visualization, and scikit-learn and XGBoost for predictive modeling.

### Data Preparation

- Filter international matches keeping only the games from 2000 and onward.
- Identify major tournaments (World Cup, Euro, Copa América, African Cup of Nations, etc...) to then create a dataset with the results of those tournaments since 2010.
- Create an elo system for every national team.
- Compute pre-tournament “form metrics” based on each team’s N (Most likely 12) games before the competition like:
  - Team's pre-tournament elo
  - Win rate
  - Average goal scored/game
  - Average goal conceded/game
  - Goal difference/game
  - Average elo of opponents on those N games

I will also use a dataset that I will create about the group composition of each tournaments based on the processed dataset imported to compute those metrics:
 - Average Elo of teams within the group
 - Maximum Elo in the group
 - Team’s Elo relative to the group average
 - Team’s Elo rank within the group

---

## Modeling & Analysis

-Define the target variable as the tournament stage reached by each team: Champion, Runner-up, Semi-Finalist, Quarter-Finalist, Round of 16, or Group Stage.
-Introduce an additional aggregated evaluation with three macro categories: Deep Run, Knockouts, and Group Stage.
-Train and compare three classification models:
 -Multinomial Logistic Regression
 -Random Forest
 -Gradient Boosting (XGBoost)
-Apply a Leave-One-Tournament-Out (LOTO) validation strategy, where each tournament from 2016 onward is predicted using only earlier tournaments as training data.
- Visualize results using confusion matrices, feature importance plots, correlation heatmaps, and tournament-level performance comparisons.

---

## Expected Challenges & Mitigation

- Missing contextual factors: External influences like injuries, travel, or team chemistry aren’t captured in the dataset.
- Ambiguity in defining “form”: Compare multiple definitions (last 5 vs. 10 games, weighted by opponent quality).
- Small number of tournaments
- Tournaments have small number of games per team, anything can happen

---

## Success Criteria

The project will be successful if:
-It demonstrates measurable relationships between pre-tournament features and tournament outcomes.
-The models show consistent predictive performance across different competitions.
-The approach provides interpretable insights into which factors are most strongly associated with tournament success.

---

## Stretch Goals (If Time Permits)

- Extend the analysis to tournaments after 2025 and try to predict the winner of the next world cup in 2026.
