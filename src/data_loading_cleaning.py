"""
Data preparation module for the football project. Combines data loading and cleaning operations.

"""

from pathlib import Path
import pandas as pd
import requests

# GLOBAL PROJECT PATH

# Determines the project root path by going up one level from the current file
project_root = Path(__file__).resolve().parents[1]
RAW_DIR = project_root / "data" / "raw"
PROCESSED_DIR = project_root / "data" / "processed"


# 1. DATA DOWNLOADING

def download_dataset(destination: str = "data/raw/") -> Path:
    """
    Download the 'results' and 'shootouts' datasets from open GitHub mirrors
    and store them in data/raw/.
    """
    destination_path = project_root / destination
    destination_path.mkdir(parents=True, exist_ok=True)

    # Dictionary mapping local filenames to download URLs
    datasets = {
        "results.csv": "https://raw.githubusercontent.com/martj42/international_results/master/results.csv",
        "shootouts.csv": "https://raw.githubusercontent.com/martj42/international_results/master/shootouts.csv",
    }

    # Download each file
    for filename, url in datasets.items():
        print(f" Downloading {filename}...")
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            (destination_path / filename).write_bytes(response.content)
            print(f"✅  {filename} downloaded successfully")
        except Exception as e:
            print(f"❌  Failed to download {filename}: {e}")

    print(f"\n All datasets downloaded in: {destination_path}")
    return destination_path


# 2. DATA LOADING

def load_dataframes(data_path: Path) -> dict:
    """Load all CSVs in data/raw/ into pandas DataFrames."""
    dfs = {}
    for csv_file in data_path.glob("*.csv"):
        try:
            df = pd.read_csv(csv_file)
            dfs[csv_file.stem] = df
            print(f" Loaded {csv_file.name} ({len(df)} rows)")
        except Exception as e:
            print(f" Could not read {csv_file.name}: {e}")
    return dfs


# 3. CLEANING UTILITIES

def clean_dataframe(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Clean a single DataFrame by:
    - Converting date columns to datetime format
    - Removing empty/duplicate rows
    - Dropping rows with missing team names
    - Removing specific unnecessary columns that are "city", "country", "neutral"
    """
    print(f"\n Cleaning {name} dataset...")

    # Convert date columns to datetime 
    date_cols = [col for col in df.columns if "date" in col.lower()]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    # Remove empty rows and any duplicates
    initial_rows = len(df)
    df = df.dropna(how="all").drop_duplicates()
    
    # Remove rows with missing team names
    if "home_team" in df.columns and "away_team" in df.columns:
        df = df.dropna(subset=["home_team", "away_team"])

    # Removing unuseful columns
    columns_to_drop = []
    if name.lower() == "shootouts" and "first_shooter" in df.columns:
        columns_to_drop.append("first_shooter")
    if name.lower() == "results":
        columns_to_drop.extend([c for c in ["city", "country", "neutral"] if c in df.columns])
    
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)
        print(f" Removed columns: {', '.join(columns_to_drop)}")

    print(f" Cleaned: {initial_rows} → {len(df)} rows (removed {initial_rows - len(df)})")
    return df


def dataset_summary(df: pd.DataFrame, name: str):
    """Print a summary of the dataset."""
    print(f"\n Summary for {name}:")
    print(f"Rows: {df.shape[0]:,} | Columns: {df.shape[1]}")

    # Show missing values if any
    missing = df.isna().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        print(" Missing values:", dict(missing))
    else:
        print("✅ No missing values")


# 4. FULL CLEANING PIPELINE

def clean_and_save_all(dfs: dict):
    """
    Clean all DataFrames and save both cleaned and filtered (since 2000) versions.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    cleaned_dfs, filtered_dfs = {}, {}

    for name, df in dfs.items():
        # Clean the DataFrame
        cleaned_df = clean_dataframe(df, name)
        dataset_summary(cleaned_df, name)

        # Save cleaned version under /data/processed
        cleaned_path = PROCESSED_DIR / f"{name}_cleaned.csv"
        cleaned_df.to_csv(cleaned_path, index=False)
        print(f" Saved: {cleaned_path}")

        # Create and save filtered version with the games after 2000 under /data/proccesed
        date_cols = [col for col in cleaned_df.columns if "date" in col.lower()]
        if date_cols:
            year = 2000
            filtered_df = cleaned_df[cleaned_df[date_cols[0]] >= pd.Timestamp(year=year, month=1, day=1)].copy()
            filtered_path = PROCESSED_DIR / f"{name}_since_{year}.csv"
            filtered_df.to_csv(filtered_path, index=False)
            print(f" Saved filtered (since {year}): {filtered_path} ({len(filtered_df)} rows)")
            filtered_dfs[f"{name}_since_{year}"] = filtered_df

        cleaned_dfs[name] = cleaned_df

    print(f"\n✅ All datasets saved in: {PROCESSED_DIR}")
    return cleaned_dfs, filtered_dfs


# 5. RUNNING CODE COMMAND

if __name__ == "__main__":
    print("\n========== DATA PREPARATION PIPELINE ==========\n")

    # Step 1: Download raw data from GitHub
    raw_data_path = download_dataset()

    # Step 2: Load all CSV files into memory
    dfs = load_dataframes(raw_data_path)

    # Step 3: Clean and save processed data
    clean_and_save_all(dfs)

    print("\n✅ Data preparation complete — all files ready in /data/processed/\n")