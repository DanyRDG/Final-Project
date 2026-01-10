# **Predicting International Football Tournament Outcomes**

This project investigates whether pre-tournament data can be used to predict how national teams perform in major international football tournaments. Using historical match results, Elo ratings, recent form indicators, and group-stage context, the project builds and evaluates several predictive models and applies the best-performing approach to generate probabilistic forecasts for the 2026 FIFA World Cup.

---

## **Project pipeline**
The pipeline is implemented in main.py as follows:

 1. Download raw datasets (results + shootouts)
 2. Clean and save processed data
 3. Build tournaments_info.csv
 4. Build tournaments_results.csv (with exclusions + shootout logic)
 5. Build group_stage_composition.csv
 6. Compute Elo ratings (since 2000)
 7. Build final features (form + group strength)
 8. Train and evaluate models (LogReg + RandomForest + XGBoost)
 9. Build 2026 World Cup group composition
 10. Predict 2026 World Cup results based on best model (LogReg)
 11. Generate all visualizations

--- 

## **Setup**

### **1. Clone the repository**
```bash
git clone <repository-url>
cd Final-Project
```

### **2. Create and activate the Conda environment**
```bash
conda env create -f environment.yml
conda activate football-project
```
### **3. Run the full pipeline**
```bash
python main.py
```

---

## **Requirements**
 - Python
 - pandas
 - numpy
 - matplotlib
 - seaborn
 - scikit-learn>=1.2.2
 - xgboost
 - joblib
 - requests
 - statsmodels

---

## **Output results**
After running the full pipeline the following result outputs are geenrated:

### **/results/models/**
 - logreg_model.joblib
 - logreg_scaler.joblib
 - randomforest_model.joblib
 - randomforest_scaler.joblib
 - xgboost_model.joblib
 - xgboost_scaler.joblib

### **/results/predictions/**
 - predictions_worldcup2026.csv

### **/results/visualizations/**
 - confusion_matrix_aggregated_logreg.png
 - confusion_matrix_detailed_logreg.png
 - correlation_matrix.png
 - feature_importance_logreg.png
 - top10_champion_probability_2026.png
 - top10_tournament_success_2026.png
 - tournament_performance_logreg.png

---

## **Final end structure**
```bash
Final-Project/
├── main.py                                 # Main entry point (runs full pipeline)
├── environment.yml                         # Conda environment specification
├── requirements.txt                        # pip dependencies
├── README.md                               # Project documentation
├── PROPOSAL.md                             # Initial project proposal
│
├── src/                                    # Source code
│   ├── __init__.py                         # Package initializer
│   ├── data_loading_cleaning.py            # Raw data loading and cleaning
│   ├── elo_calculator.py                   # Elo rating computation
│   ├── features.py                         # Feature engineering
│   ├── group_stage_composition_builder.py  # Group-stage reconstruction
│   ├── models.py                           # Model training and evaluation
│   ├── tournament_builder.py               # Tournament structure creation
│   ├── tournament_results_builder.py       # Tournament results processing
│   ├── Visualization.py                    # Data visualization utilities
│   └── World_Cup2026.py                    # 2026 World Cup predictions
│
├── data/                                   # Data directory
│   ├── raw/                                # Raw, unprocessed data
│   │   ├── results.csv
│   │   └── shootouts.csv
│   │
│   └── processed/                          # Processed datasets (generated)
│
├── results/                                # Model outputs
│   ├── models/                             # Trained models and scalers
│   ├── predictions/                        # Prediction outputs
│   └── visualizations/                     # Figures and plots
```