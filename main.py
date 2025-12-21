"""
Main entry point for the Final Project:
Predicting International Football Success from Pre-Tournament Form
Pipeline overview:
1. Download raw datasets (results + shootouts)
2. Clean and save processed data
3. Build tournaments_info.csv
4. Build tournaments_results.csv (with exclusions + shootout logic)
5. Build group_stage_composition.csv
6. Compute Elo ratings (since 2000)
7. Build final features (form + group strength)
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


def main():
    print("\n========== FOOTBALL PROJECT DATA PIPELINE ==========\n")

    # Step 1: Download raw data
    print("[1/7] Downloading raw datasets...")
    raw_dir = download_dataset()

    # Step 2: Load raw datasets
    print("[2/7] Loading raw CSV files...")
    dfs = load_dataframes(raw_dir)

    # Step 3: Clean and save processed data
    print("[3/7] Cleaning and saving processed datasets...")
    clean_and_save_all(dfs)
    processed_dir = Path(__file__).resolve().parent / "data" / "processed"

    # Step 4: Build tournaments_info.csv
    print("[4/7] Building tournaments_info.csv...")
    results_path = processed_dir / "results_cleaned.csv"
    tournaments_info_path = processed_dir / "tournaments_info.csv"
    build_tournaments_info(results_path, tournaments_info_path)

    # Step 5: Build tournaments_results.csv (with exclusions)
    print("[5/7] Generating tournaments_results.csv...")
    results_df = pd.read_csv(results_path, parse_dates=["date"])
    shootouts_df = pd.read_csv(processed_dir / "shootouts_cleaned.csv")
    tournaments_info_df = pd.read_csv(tournaments_info_path, parse_dates=["start_date", "end_date"])

    tournaments_results_path = processed_dir / "tournaments_results.csv"
    build_tournament_results(results_df, shootouts_df, tournaments_info_df, tournaments_results_path)

    # Step 6: Build group_stage_composition.csv
    print("[6/7] Building group_stage_composition.csv...")
    build_group_stage_composition(
        results_path, 
        tournaments_info_path, 
        processed_dir / "group_stage_composition.csv"
    )

    # Step 7: Compute Elo ratings (since 2000)
    print("[7/7] Computing Elo ratings (since 2000)...")
    build_elo_files()

    # Step 8: Build final feature dataset
    print("[8/8] Generating final features dataset...")
    
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

    # Show preview
    print("\n✅ Feature dataset created successfully.")
    print("Sample preview:")
    print(complete_features.head(5).to_string(index=False))
    
    print("\nPipeline completed successfully.")
    print(f"✅ Final features saved to: {processed_dir / 'final_features.csv'}")


if __name__ == "__main__":
    main()