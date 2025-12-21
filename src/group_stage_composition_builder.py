"""
Group Stage Composition Builder
------------------------------------------------------
Creates group_stage_composition.csv by identifying which teams played together
in the group stage of each tournament (2000 onward).

Logic:
- For each tournament, find each team's first 3 opponents
- Teams with the same 3 opponents are in the same group
- Assign group letters: A, B, C, D, etc.

Output: tournament | year | team | group
"""

from pathlib import Path
import pandas as pd


# 1. FINDING TEAM'S FIRST 3 OPPONENTS BY TOURNAMENT

def get_team_opponents(team: str, matches: pd.DataFrame) -> tuple:
    """Get a team's first 3 opponents in chronological order."""
    team_matches = matches[
        (matches["home_team"] == team) | (matches["away_team"] == team)
    ].sort_values("date").head(3)
    
    opponents = [
        row["away_team"] if row["home_team"] == team else row["home_team"]
        for _, row in team_matches.iterrows()
    ]
    return tuple(sorted(opponents))
    

# 2. ASSIGNING TEAMS TO A GROUP 

def assign_group_letters(team_opponents: dict) -> dict:
    """Assign group letters (A, B, C, ...) to teams based on shared opponents."""
    # Group teams by their opponent set
    unique_groups = {}
    for team, opponents in team_opponents.items():
        group_key = tuple(sorted([team] + list(opponents)))
        unique_groups.setdefault(group_key, []).append(team)
    
    # Assign group letters A, B, C, ...
    group_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']
    team_to_group = {}
    
    for idx, teams in enumerate(unique_groups.values()):
        letter = group_letters[idx] if idx < len(group_letters) else f"Group_{idx + 1}"
        print(f"   Group {letter}: {', '.join(sorted(teams))}")
        for team in teams:
            team_to_group[team] = letter
    
    return team_to_group


# 3. BUILDING THE GROUP STAGE DATAFRAME

def build_group_stage_composition(
    results_path: Path, 
    tournaments_info_path: Path, 
    output_path: Path
) -> pd.DataFrame:
    """Build the group stage composition file for all tournaments from 2000 onward"""
    print("\nBuilding group_stage_composition.csv (2000 onward)...")

    # Load and standardize data
    results_df = pd.read_csv(results_path, parse_dates=["date"])
    tournaments_info = pd.read_csv(tournaments_info_path, parse_dates=["start_date", "end_date"])
    results_df.columns = results_df.columns.str.lower()
    tournaments_info.columns = tournaments_info.columns.str.lower()

    # Excluding tournaments with weird formats for feasibility
    excluded = {
        ("african cup of nations", 2010),
        ("gold cup", 2000), ("gold cup", 2002), ("gold cup", 2003),
    }

    all_records = []

    for _, t in tournaments_info.iterrows():
        t_name, t_year = t["tournament"], int(t["year"])
        
        # Skip pre-2000 tournaments
        if t_year < 2000:
            continue

        # Check exclusions
        base_name = next((name for name in ["fifa world cup", "uefa euro", "copa américa", 
                         "gold cup", "african cup of nations", "afc asian cup"]
                         if name in t_name.lower()), None)
        
        if (base_name, t_year) in excluded:
            print(f"Skipping {t_name} ({t_year})")
            continue

        # Filter tournament matches
        matches = results_df[
            (results_df["tournament"].str.lower() == t_name.lower()) &
            (results_df["date"] >= t["start_date"]) &
            (results_df["date"] <= t["end_date"])
        ].sort_values("date")

        if matches.empty:
            continue

        print(f"\nProcessing {t_name} {t_year}...")

        # Get all teams and their opponents
        teams = pd.unique(matches[["home_team", "away_team"]].values.ravel())
        team_opponents = {team: get_team_opponents(team, matches) for team in teams}

        # Assign groups
        team_groups = assign_group_letters(team_opponents)

        # Store results
        for team, group in team_groups.items():
            all_records.append({
                "tournament": t_name,
                "year": t_year,
                "team": team,
                "group": group,
            })

    # Create and save DataFrame
    group_df = pd.DataFrame(all_records).sort_values(["year", "tournament", "group", "team"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    group_df.to_csv(output_path, index=False)

    print(f"\n✅ Saved to: {output_path}")
    print(f"Total entries: {len(group_df)} | Tournaments: {group_df['tournament'].nunique()}")

    return group_df


# 4. RUNNING CODE COMMAND

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    processed = project_root / "data" / "processed"
    
    build_group_stage_composition(
        processed / "results_cleaned.csv",
        processed / "tournaments_info.csv",
        processed / "group_stage_composition.csv"
    )