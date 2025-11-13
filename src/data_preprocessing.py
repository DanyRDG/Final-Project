from pathlib import Path
import shutil
import pandas as pd
import kagglehub


def download_dataset(destination: str = "Final-Project/data/raw"):
    """
    Download the dataset 'All International Football Results'
    (https://www.kaggle.com/datasets/patateriedata/all-international-football-results)
    from Kaggle using KaggleHub

    Args:
        destination (str): Directory to save the dataset.
    Returns:
        Path: Path to the local data/raw directory.
    """
    # Define local target path
    dest = Path(destination)
    dest.mkdir(parents=True, exist_ok=True)

    # Target file paths
    matches_file = dest / "all_matches.csv"
    countries_file = dest / "countries_names.csv"

    # If both files already exist → skip download
    if matches_file.exists() and countries_file.exists():
        print(f" Files already exist in {dest}, skipping download.")
        return dest

    # Download from KaggleHub
    dataset_id = "patateriedata/all-international-football-results"
    print(f" Downloading dataset from KaggleHub: {dataset_id}")
    dl_dir = Path(kagglehub.dataset_download(dataset_id))
    print(f"✅ Downloaded to: {dl_dir}")

    # Find CSV files
    csvs = list(dl_dir.rglob("*.csv"))
    if not csvs:
        raise FileNotFoundError("❌ No CSV files found in downloaded dataset.")

    # Copy files with same original names
    for csv in csvs:
        target_path = dest / csv.name
        shutil.copy(csv, target_path)
        print(f"✅ Copied {csv.name} → {target_path}")

    print("\n Current contents of data/raw:")
    print([p.name for p in dest.glob('*')])

    return dest


def load_full_data(raw_dir: str = "Final-Project/data/raw") -> dict:
    """
    Load the original CSV files (all_matches.csv and countries_names.csv)
    into pandas DataFrames. Downloads them automatically if missing.

    Returns:
        dict: {'matches': DataFrame, 'countries': DataFrame or None}
    """
    raw_path = Path(raw_dir)
    matches_file = raw_path / "all_matches.csv"
    countries_file = raw_path / "countries_names.csv"

    # If missing it downloads again
    if not matches_file.exists() or not countries_file.exists():
        print("⚠️ Data missing — downloading now...")
        download_dataset(str(raw_path))

    # Load matches
    if not matches_file.exists():
        raise FileNotFoundError("❌ all_matches.csv not found in data/raw.")
    matches = pd.read_csv(matches_file)
    print(f"✅ Loaded all_matches.csv with {len(matches):,} rows.")

    #Ensure 'date' column is in datetime format
    if "date" in matches.columns:
        matches["date"] = pd.to_datetime(matches["date"], errors="coerce")
        print(" Converted 'date' column to datetime format.")

    # Load countries
    if countries_file.exists():
        countries = pd.read_csv(countries_file)
        print(f"✅ Loaded countries_names.csv with {len(countries):,} rows.")
    else:
        countries = None
        print("⚠️ countries_names.csv not found — continuing without it.")

    return {"matches": matches, "countries": countries}


if __name__ == "__main__":
    data = load_full_data()
    print("\n Matches preview:")
    print(data["matches"].head())

    if data["countries"] is not None:
        print("\n📋 Countries preview:")
        print(data["countries"].head())
