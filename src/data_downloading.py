from pathlib import Path
import shutil
import pandas as pd
import kagglehub


def download_dataset(destination: str = "Final-Project/data/raw"):
    """
    Download the 'All International Football Results' dataset from KaggleHub.
    Saves those files:
        - all_matches.csv
        - countries_names.csv
    """
    dest = Path(destination)
    dest.mkdir(parents=True, exist_ok=True)

    match_file = dest / "all_matches.csv"
    countries_file = dest / "countries_names.csv"

    # Skip if both files already exist
    if match_file.exists() and countries_file.exists():
        print(f"📦 Main dataset already exists in {dest}")
        return dest

    dataset_id = "patateriedata/all-international-football-results"
    print(f"⬇️ Downloading MATCHES dataset: {dataset_id}")

    dl_dir = Path(kagglehub.dataset_download(dataset_id))
    print(f"   → Downloaded to: {dl_dir}")

    csv_files = list(dl_dir.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError("❌ No CSV files found in the matches dataset.")

    for csv in csv_files:
        if csv.name == "all_matches.csv":
            shutil.copy(csv, match_file)
            print(f"   → Saved all_matches.csv")
        elif csv.name == "countries_names.csv":
            shutil.copy(csv, countries_file)
            print(f"   → Saved countries_names.csv")

    print(f"✅ Matches dataset ready in {dest}")
    return dest


def load_full_data(raw_dir: str = "Final-Project/data/raw") -> dict:
    """
    Load the MATCHES + COUNTRIES datasets.
    Returns:
        {
            "matches": DataFrame,
            "countries": DataFrame or None
        }
    """
    raw_path = Path(raw_dir)

    match_file = raw_path / "all_matches.csv"
    countries_file = raw_path / "countries_names.csv"

    # If missing: download automatically
    if not match_file.exists() or not countries_file.exists():
        print(" Main data missing → downloading...")
        download_dataset(raw_dir)

    # Load matches file
    matches = pd.read_csv(match_file, parse_dates=["date"])
    print(f"✅ Loaded all_matches.csv → {len(matches):,} rows")

    # Load countries file
    if countries_file.exists():
        countries = pd.read_csv(countries_file)
        print(f"✅ Loaded countries_names.csv → {len(countries):,} rows")
    else:
        print("No countries file found.")
        countries = None

    return {"matches": matches, "countries": countries}


def download_shootouts_dataset(destination: str = "Final-Project/data/raw"):
    """
    Download the penalty shootouts dataset and save Penalty_Shootouts.csv
    as penalty_shootouts.csv
    """
    dest = Path(destination)
    dest.mkdir(parents=True, exist_ok=True)

    target = dest / "penalty_shootouts.csv"

    if target.exists():
        print("Penalty shootouts dataset already exists.")
        return

    dataset_id = "muhammadehsan02/global-football-results-18722024"
    print(f"⬇️ Downloading SHOOTOUT dataset: {dataset_id}")

    dl_dir = Path(kagglehub.dataset_download(dataset_id))
    print(f"   → Downloaded to: {dl_dir}")

    csv_files = list(dl_dir.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError("❌ No CSVs found in shootout dataset.")

    for csv in csv_files:
        if csv.name.lower() == "penalty_shootouts.csv":
            shutil.copy(csv, target)
            print(f"   → Saved penalty_shootouts.csv")
            return

    raise FileNotFoundError("penalty_shootouts.csv not found in dataset!")


def download_all_data(destination: str = "Final-Project/data/raw"):
    """
    Download ALL required datasets:
    - All Matches Dataset
    - Penalty Shootouts Dataset
    """
    print("\n===== DOWNLOADING ALL DATASETS =====\n")

    download_dataset(destination)
    download_shootouts_dataset(destination)

    print("\n All datasets downloaded successfully!\n")


def load_all_data(raw_dir: str = "Final-Project/data/raw") -> dict:
    """
    Load all datasets:
    - matches
    - countries
    - penalty shootouts

    Returns a dictionary:
        {
            "matches": df,
            "countries": df or None,
            "shootouts": df or None
        }
    """
    data = load_full_data(raw_dir)

    raw_path = Path(raw_dir)
    shootout_file = raw_path / "penalty_shootouts.csv"

    if shootout_file.exists():
        shootouts = pd.read_csv(shootout_file, parse_dates=["date"])
        print(f"✅ Loaded penalty shootouts → {len(shootouts):,} rows")
    else:
        print("No penalty_shootouts.csv found.")
        shootouts = None

    data["shootouts"] = shootouts
    return data


