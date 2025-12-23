"""
Main entry point for the Final Project:
Predicting International Football Success from Pre-Tournament Form
------------------------------------------------------------------
Pipeline overview:
1. Download raw datasets (results + shootouts)
2. Clean and save processed data
3. Build tournaments_info.csv
4. Build tournaments_results.csv (with exclusions + shootout logic)
5. Filter tournaments from 2000 onward
6. Build group_stage_composition.csv
7. Compute Elo ratings (since 2000)
8. Build final features (form + group strength)
9. Train and evaluate Logistic Regression model
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
from src.models import loto_evaluation


def main():
    print("\n========== FOOTBALL PROJECT DATA PIPELINE ==========\n")

    # --- Step 1: Download raw data ---
    print("[1/9] Downloading raw datasets...")
    raw_dir = download_dataset()

    # --- Step 2: Load raw datasets ---
    print("[2/9] Loading raw CSV files...")
    dfs = load_dataframes(raw_dir)

    # --- Step 3: Clean and save processed data ---
    print("[3/9] Cleaning and saving processed datasets...")
    clean_and_save_all(dfs)
    processed_dir = Path(__file__).resolve().parent / "data" / "processed"

    # --- Step 4: Build tournaments_info.csv ---
    print("[4/9] Building tournaments_info.csv...")
    results_path = processed_dir / "results_cleaned.csv"
    tournaments_info_path = processed_dir / "tournaments_info.csv"
    tournaments_info = build_tournaments_info(results_path, tournaments_info_path)

    # --- Step 5: Build tournaments_results.csv (with exclusions) ---
    print("[5/9] Generating tournaments_results.csv with exclusions...")
    results_df = pd.read_csv(processed_dir / "results_cleaned.csv")
    shootouts_df = pd.read_csv(processed_dir / "shootouts_cleaned.csv")
    tournaments_info_df = pd.read_csv(tournaments_info_path)

    # Ensure datetime formats
    results_df["date"] = pd.to_datetime(results_df["date"], errors="coerce")
    tournaments_info_df["start_date"] = pd.to_datetime(tournaments_info_df["start_date"])
    tournaments_info_df["end_date"] = pd.to_datetime(tournaments_info_df["end_date"])

    tournaments_results_path = processed_dir / "tournaments_results.csv"
    tournament_results_summary = build_tournament_results(
        results_df, shootouts_df, tournaments_info_df, tournaments_results_path
    )

    # Filter tournaments (>=2000)
    tournament_results_summary = tournament_results_summary[
        tournament_results_summary["year"] >= 2000
    ].reset_index(drop=True)
    tournament_results_summary.to_csv(tournaments_results_path, index=False)
    print(f"  → tournaments_results.csv saved ({len(tournament_results_summary)} rows)")

    # --- Step 6: Build group_stage_composition.csv ---
    print("[6/9] Building group_stage_composition.csv...")
    group_stage_output = processed_dir / "group_stage_composition.csv"
    build_group_stage_composition(results_path, tournaments_info_path, group_stage_output)

    # --- Step 7: Compute Elo ratings (since 2000) ---
    print("[7/9] Computing Elo ratings (since 2000)...")
    build_elo_files()

    # --- Step 8: Build final feature dataset ---
    print("[8/9] Generating final features dataset...")

    tournaments_info_path = processed_dir / "tournaments_info.csv"
    tournaments_results_path = processed_dir / "tournaments_results.csv"
    group_stage_path = processed_dir / "group_stage_composition.csv"
    elo_matches_path = processed_dir / "elo_matches_since_2000.csv"

    # Load processed data
    tournaments_info = pd.read_csv(tournaments_info_path, parse_dates=["start_date", "end_date"])
    tournaments_results = pd.read_csv(tournaments_results_path)
    group_composition = pd.read_csv(group_stage_path)
    matches_elo = pd.read_csv(elo_matches_path, parse_dates=["date"])

    # Build features
    form_features = build_pre_tournament_features(
        tournaments_info=tournaments_info,
        tournaments_results=tournaments_results,
        matches_with_elo=matches_elo,
        lookback_matches=12,
    )
    complete_features = add_group_difficulty_features(form_features, group_composition)
    save_final_features(complete_features)

    print("\n✅ Feature dataset created successfully.")
    print("Sample preview:")
    print(complete_features.head(5).to_string(index=False))

    # --- Step 9: Train and evaluate Logistic Regression model ---
    print("\n[9/9] Training and evaluating Logistic Regression model...")

    results_log, preds_log = loto_evaluation(complete_features)
    results_log_path = processed_dir / "loto_results_logreg.csv"
    preds_log_path = processed_dir / "loto_predictions_logreg.csv"
    results_log.to_csv(results_log_path, index=False)
    preds_log.to_csv(preds_log_path, index=False)
    print("✅ Logistic Regression model evaluation complete.")

    # --- Summary table ---
    print("\n========== OVERALL SUMMARY (Logistic Regression) ==========")
    print(results_log.describe()[["accuracy_stage", "mae_stage", "accuracy_macro3"]].T)
    print("===========================================================")

    print("\nPipeline completed successfully.")
    print(f"✅ Results saved to: {processed_dir}")
    print(f"   - Tournament results: {results_log_path}")
    print(f"   - Team predictions: {preds_log_path}")


if __name__ == "__main__":
    main()