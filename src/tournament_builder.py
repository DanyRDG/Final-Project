"""
Tournament Builder
Generates tournaments_info.csv containing:
- tournament name (lowercase)
- year
- start_date / end_date
- confederation
- continent
Includes only main tournaments:
fifa world cup, uefa euro, copa américa, african cup of nations, afc asian cup, and gold cup.
"""
from pathlib import Path
import pandas as pd


# Dictionary mapping tournament names to their confederation and continent
TOURNAMENT_INFO = {
    "fifa world cup": ("world", "global"),
    "uefa euro": ("uefa", "europe"),
    "copa américa": ("conmebol", "south america"),
    "african cup of nations": ("caf", "africa"),
    "afc asian cup": ("afc", "asia"),
    "gold cup": ("concacaf", "north & central america")
}


# 1.FILE BUILDER

def build_tournaments_info(results_path: Path, output_path: Path):
    """
    Build tournaments_info.csv from cleaned match results.
    Keeps only the main tournaments defined in TOURNAMENT_INFO.
    """
    print(f"\n🏗️ Building tournaments_info.csv from {results_path.name}...")
    
    # Load data and normalize tournament names to lowercases
    df = pd.read_csv(results_path, parse_dates=["date"])
    df["tournament"] = df["tournament"].str.lower()
    
    # Filter to keep only the tournaments I want
    df = df[df["tournament"].isin(TOURNAMENT_INFO.keys())]
    
    # Extract year and group by tournament + year to get start - end dates of each tournaments
    df["year"] = df["date"].dt.year
    summary = (
        df.groupby(["tournament", "year"])
        .agg(start_date=("date", "min"), end_date=("date", "max"))
        .reset_index()
    )
    
    # Create empty columns for confederation and continent
    summary["confederation"] = ""
    summary["continent"] = ""
    
    # Use manual loop to fill confederation and continent for each row
    for i in range(len(summary)):
        # Get the tournament name for the row
        tournament_name = summary.loc[i, "tournament"]
        
        # Look up the tournament info in the dictionary
        confederation, continent = TOURNAMENT_INFO[tournament_name]
        
        # Assign confederation and continent inputs to the current row
        summary.loc[i, "confederation"] = confederation
        summary.loc[i, "continent"] = continent
    
    # Sort and save
    summary = summary.sort_values(["tournament", "year"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    
    print(f"✅ tournaments_info.csv created successfully at: {output_path}")
    print(f"Total tournaments found: {len(summary)}")
    return summary


# 2.RUNNING CODE COMMAND

if __name__ == "__main__":
    print("Running tournament_builder.py...")
    
    # Set up paths
    project_root = Path(__file__).resolve().parents[1]
    results_file = project_root / "data" / "processed" / "results_since_1980.csv"
    output_file = project_root / "data" / "processed" / "tournaments_info.csv"
    
    # Build tournaments info
    tournaments_info = build_tournaments_info(results_file, output_file)
    
    # Display sample
    print("\nSample of tournaments_info:")
    print(tournaments_info.head(10))