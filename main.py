"""
Main entry point for the Final Project:
Predicting International Football Success from Pre-Tournament Form
------------------------------------------------------------------
Pipeline overview:
1. Download raw datasets (results + shootouts)
2. Clean and save processed data
3. Build tournaments_info.csv
4. Build tournaments_results.csv (with exclusions + shootout logic)
5. Build group_stage_composition.csv
6. Compute Elo ratings (since 2000)
7. Build final features (form + group strength)
8. Train and evaluate models (LogReg + RandomForest + XGBoost)
9. Build 2026 World Cup group composition
10. Predict 2026 World Cup results
11. Generate all visualizations
"""

from pathlib import Path
import pandas as pd

# Data and feature pipeline imports
from src.data_loading_cleaning import download_dataset, load_dataframes, clean_and_save_all
from src.tournament_builder import build_tournaments_info
from src.tournament_results_builder import build_tournament_results
from src.group_stage_composition_builder import build_group_stage_composition
from src.elo_calculator import build_elo_files
from src.features import (
    build_pre_tournament_features,
    add_group_difficulty_features,
    save_final_features,
)

# Model imports - from single models.py file
from src.models import loto_evaluation, load_data

# Prediction module - NOW USING THE COMBINED World_Cup2026.py
from src.World_Cup2026 import (
    build_group_stage_composition_2026,
    load_pre_tournament_features,
    predict_worldcup_2026
)

# Visualization module - Import the complete visualization suite
from src.Visualization import generate_all_visualizations

def main():
    print("\n========== FOOTBALL PROJECT DATA PIPELINE ==========\n")

    # Step 1: Download raw data
    print("[1/11] Downloading raw datasets...")
    raw_dir = download_dataset()

    # Step 2: Load raw datasets
    print("[2/11] Loading raw CSV files...")
    dfs = load_dataframes(raw_dir)

    # Step 3: Clean and save processed data
    print("[3/11] Cleaning and saving processed datasets...")
    clean_and_save_all(dfs)
    processed_dir = Path(__file__).resolve().parent / "data" / "processed"

    # Step 4: Build tournaments_info.csv
    print("[4/11] Building tournaments_info.csv...")
    results_path = processed_dir / "results_cleaned.csv"
    tournaments_info_path = processed_dir / "tournaments_info.csv"
    build_tournaments_info(results_path, tournaments_info_path)

    # Step 5: Build tournaments_results.csv (with exclusions)
    print("[5/11] Generating tournaments_results.csv...")
    results_df = pd.read_csv(results_path, parse_dates=["date"])
    shootouts_df = pd.read_csv(processed_dir / "shootouts_cleaned.csv")
    tournaments_info_df = pd.read_csv(tournaments_info_path, parse_dates=["start_date", "end_date"])

    tournaments_results_path = processed_dir / "tournaments_results.csv"
    build_tournament_results(results_df, shootouts_df, tournaments_info_df, tournaments_results_path)

    # Step 6: Build group_stage_composition.csv
    print("[6/11] Building group_stage_composition.csv...")
    build_group_stage_composition(
        results_path, 
        tournaments_info_path, 
        processed_dir / "group_stage_composition.csv"
    )

    # Step 7: Compute Elo ratings (since 2000)
    print("[7/11] Computing Elo ratings (since 2000)...")
    build_elo_files()

    # Step 8: Build final feature dataset
    print("[8/11] Generating final features dataset...")
    
    # Load all processed data
    tournaments_info = pd.read_csv(tournaments_info_path, parse_dates=["start_date", "end_date"])
    tournaments_results = pd.read_csv(tournaments_results_path)
    group_composition = pd.read_csv(processed_dir / "group_stage_composition.csv")
    matches_elo = pd.read_csv(processed_dir / "elo_matches_since_2000.csv", parse_dates=["date"])

    # Build features
    form_features = build_pre_tournament_features(
        tournaments_info, tournaments_results, matches_elo, lookback_matches=12
    )
    complete_features = add_group_difficulty_features(form_features, group_composition)
    save_final_features(complete_features)

    print("\n✅ Feature dataset created successfully.")
    print("Sample preview:")
    print(complete_features.head(5).to_string(index=False))

    # Step 9: Train and evaluate all three models
    print("\n[9/11] Training and evaluating models...")
    
    # Load features for modeling
    features_path = processed_dir / "final_features.csv"
    df = load_data(features_path)
    
    # Train Logistic Regression, Random Forest, and XGBoost
    for model_type in ["logreg", "randomforest", "xgboost"]:
        print(f"\n{'='*70}")
        print(f"  Training {model_type.upper()} model...")
        print(f"{'='*70}")
        
        results_df, predictions_df = loto_evaluation(df, model_type=model_type)
        
        # Save results
        results_path = processed_dir / f"loto_results_{model_type}.csv"
        predictions_path = processed_dir / f"loto_predictions_{model_type}.csv"
        results_df.to_csv(results_path, index=False)
        predictions_df.to_csv(predictions_path, index=False)
        
        print(f"\n✅ {model_type.upper()} model evaluation complete.")
        print(f"   → Results: {results_path}")
        print(f"   → Predictions: {predictions_path}")
        
        # Summary statistics
        print(f"\n{'='*70}")
        print(f"  OVERALL SUMMARY - {model_type.upper()}")
        print(f"{'='*70}")
        print(results_df.describe()[["accuracy_stage", "mae_stage", "accuracy_macro3"]].T)

    # Step 10: Build 2026 World Cup groups and generate predictions
    print(f"\n{'='*70}")
    print("  [10/11] Building 2026 World Cup groups and generating predictions...")
    print(f"{'='*70}")

    results_dir = Path(__file__).resolve().parent / "results" / "predictions"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Create 2026 World Cup group composition
    groups_2026_path = processed_dir / "group_stage_composition_2026.csv"
    build_group_stage_composition_2026(groups_2026_path)

    # Build features for 2026 teams
    features_2026 = load_pre_tournament_features(processed_dir)
    
    # Generate predictions
    predictions_2026 = predict_worldcup_2026(features_2026, results_dir)

    # Step 11: Generate all visualizations
    print(f"\n{'='*70}")
    print("  [11/11] Generating all visualizations...")
    print(f"{'='*70}")
    
    # Generate all 7 visualizations (historical evaluation + tournament performance + 2026 predictions)
    generate_all_visualizations()

    print("\n" + "="*70)
    print("  PIPELINE COMPLETED SUCCESSFULLY")
    print("="*70)
    print(f"✅ All results stored in: {results_dir}")
    print(f"✅ Model comparison available in: {processed_dir}")
    print(f"✅ Files: loto_results_[logreg|randomforest|xgboost].csv")
    print(f"✅ 2026 World Cup predictions: {results_dir / 'predictions_worldcup2026.csv'}")
    print(f"✅ All visualizations generated in: results/visualizations/")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()