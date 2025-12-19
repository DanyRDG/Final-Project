"""
Main entry point for the Final Project:
Predicting International Football Success from Pre-Tournament Form
------------------------------------------------------------------
Pipeline overview (Steps 1-3):
1. Download raw datasets (results + shootouts)
2. Clean and save processed data
3. Build tournaments_info.csv
"""

from pathlib import Path

# Data pipeline imports
from src.data_loading_cleaning import download_dataset, load_dataframes, clean_and_save_all
from src.tournament_builder import build_tournaments_info


def main():
    print("\n========== FOOTBALL PROJECT DATA PIPELINE ==========\n")

    # --- Step 1: Download raw data ---
    print("[1/3] Downloading raw datasets...")
    raw_dir = download_dataset()

    # --- Step 2: Load and clean datasets ---
    print("[2/3] Loading and cleaning datasets...")
    dfs = load_dataframes(raw_dir)
    clean_and_save_all(dfs)

    # --- Step 3: Build tournaments_info.csv ---
    print("[3/3] Building tournaments_info.csv...")
    processed_dir = Path(__file__).resolve().parent / "data" / "processed"
    results_path = processed_dir / "results_cleaned.csv"
    tournaments_info_path = processed_dir / "tournaments_info.csv"
    tournaments_info = build_tournaments_info(results_path, tournaments_info_path)

    print(f"\n✅ Pipeline completed successfully.")
    print(f"✅ All processed files saved in: {processed_dir}")
    print(f"\n📊 Sample of tournaments_info:")
    print(tournaments_info.head(10))


if __name__ == "__main__":
    main()