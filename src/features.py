"""
Feature Engineering for Tournament Performance Prediction
Builds a comprehensive feature dataset for each team's tournament participation,
including pre-tournament form, Elo ratings, and group strength metrics

Inputs:
- tournaments_info.csv, tournaments_results.csv, group_stage_composition.csv, elo_matches_since_2000.csv

Output:
- final_features.csv: Complete feature set for modeling
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd

# Project directory structure
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# 1. TEAM ALO RATING LOOKUP

def get_team_elo_before_tournament(matches_with_elo: pd.DataFrame, team: str, tournament_start: pd.Timestamp) -> float:
    """
    Get a team's Elo rating just before a tournament starts
    
    Finds the team's most recent match before the tournament and returns
    their post-match Elo rating from that game
    
    Returns:
        float: Team's Elo rating, or NaN if no prior matches exist
    """
    # Filter for all matches involving this team before tournament start
    team_matches = matches_with_elo[
        ((matches_with_elo["home_team"] == team) | (matches_with_elo["away_team"] == team))
        & (matches_with_elo["date"] < tournament_start)
    ].sort_values("date", ascending=False)

    # No match history available
    if team_matches.empty:
        return np.nan

    # Get the most recent match
    latest = team_matches.iloc[0]
    
    # Return appropriate Elo based on whether team was home or away
    return latest["home_elo_post"] if latest["home_team"] == team else latest["away_elo_post"]


# 2. PRE-TOURNMAENT FORM METRICS CALCULATION

def calculate_team_form_metrics(
    team: str, tournament_start: pd.Timestamp, matches_with_elo: pd.DataFrame, lookback_matches: int = 12
) -> Dict[str, float]:
    """
    Calculate team's recent form before a tournament
    
    Analyzes the team's last N matches to compute:
    - Win rate (1=win, 0.5=draw, 0=loss)
    - Average goal difference per match
    - Goals scored and conceded per match
    - Average opponent strength (Elo rating)
    """
    # Get all matches before the tournament involving this team
    prior_matches = matches_with_elo[
        ((matches_with_elo["home_team"] == team) | (matches_with_elo["away_team"] == team))
        & (matches_with_elo["date"] < tournament_start)
    ].copy()

    # No match history - return empty metrics
    if prior_matches.empty:
        return {
            "n_matches_form": 0, "win_rate": np.nan, "avg_goal_diff": np.nan,
            "goals_for_per_match": np.nan, "goals_against_per_match": np.nan, "avg_opponent_elo": np.nan
        }

    # Take only the most recent N matches
    recent = prior_matches.sort_values("date", ascending=False).head(lookback_matches)

    def extract_stats(match):
        """
        Extract team-specific statistics from a match.
        
        Handles both home and away perspectives to get:
        - Goals for/against from team's perspective
        - Opponent's pre-match Elo rating
        - Match result (win/draw/loss)
        """
        is_home = match["home_team"] == team
        
        # Get goals from team's perspective
        gf = match["home_score"] if is_home else match["away_score"]  # Goals for
        ga = match["away_score"] if is_home else match["home_score"]  # Goals against
        
        # Get opponent's Elo rating
        opp_elo = match["away_elo_pre"] if is_home else match["home_elo_pre"]
        
        # Calculate match result: 1=win, 0.5=draw, 0=loss
        result = 1.0 if gf > ga else (0.5 if gf == ga else 0.0)
        
        return pd.Series({"gf": gf, "ga": ga, "result": result, "opp_elo": opp_elo})

    # Apply extraction to all recent matches
    stats = recent.apply(extract_stats, axis=1)

    # Calculate aggregate metrics across all matches
    return {
        "n_matches_form": len(stats), 
        "win_rate": stats["result"].mean(), 
        "avg_goal_diff": (stats["gf"] - stats["ga"]).mean(), 
        "goals_for_per_match": stats["gf"].mean(), 
        "goals_against_per_match": stats["ga"].mean(), 
        "avg_opponent_elo": stats["opp_elo"].mean(), 
    }


# 3. Build Complete Pre-Tournament Feature Dataset

def build_pre_tournament_features(
    tournaments_info: pd.DataFrame,
    tournaments_results: pd.DataFrame,
    matches_with_elo: pd.DataFrame,
    lookback_matches: int = 12,
) -> pd.DataFrame:
    """
    Build pre-tournament features for each team-tournament combination
    
    For every team in every tournament, calculates:
    - Pre-tournament Elo rating
    - Recent form metrics (last N matches)
    - Tournament outcome (stage reached)
    """
    print("Building pre-tournament form dataset...")

    # Standardize column names to lowercase for consistency
    tournaments_info.columns = tournaments_info.columns.str.lower()
    tournaments_results.columns = tournaments_results.columns.str.lower()
    
    # Merge tournament dates with team results
    team_data = tournaments_results.merge(
        tournaments_info[["tournament", "start_date", "year"]], 
        on=["tournament", "year"], 
        how="inner"
    )
    
    # Handle potential duplicate year columns from merge operation
    if "year" not in team_data.columns:
        year_col = "year_x" if "year_x" in team_data.columns else "year_y"
        team_data.rename(columns={year_col: "year"}, inplace=True)

    # Filter to only include tournaments from 2000 onwards
    team_data = team_data[team_data["year"] >= 2000].reset_index(drop=True)

    # Build feature records for each team-tournament
    records = []
    for _, row in team_data.iterrows():
        tournament_start = pd.to_datetime(row["start_date"])
        
        # Calculate recent form metrics
        form_stats = calculate_team_form_metrics(
            row["team"], tournament_start, matches_with_elo, lookback_matches
        )
        
        # Get pre-tournament Elo rating
        elo_rating = get_team_elo_before_tournament(
            matches_with_elo, row["team"], tournament_start
        )
        
        # Combine all features into a single record
        records.append({
            "tournament": row["tournament"],
            "year": row["year"],
            "team": row["team"],
            "stage": row["stage"],  # Tournament outcome
            "pre_tournament_elo": elo_rating,
            **form_stats  # Unpack form metrics
        })

    # Convert to DataFrame and remove duplicates
    features_df = pd.DataFrame(records).drop_duplicates(
        subset=["tournament", "year", "team"], 
        keep="first"
    )

    # Create numeric encoding for tournament stages (for machine learning)
    stage_hierarchy = [
        "champion",          # 0 - Winner
        "runner-up",         # 1 - Lost final
        "semi-finalist",     # 2 - Lost semi-final
        "quarter-finalist",  # 3 - Lost quarter-final
        "round of 16",       # 4 - Lost round of 16
        "group stage"        # 5 - Eliminated in groups
    ]
    features_df["stage_order"] = features_df["stage"].str.lower().map(
        {s: i for i, s in enumerate(stage_hierarchy)}
    )
    
    print(f"✅ Built form dataset with {len(features_df)} unique rows (2000+ only).")
    return features_df.round(3)


# 4. GROUP STAGE DIFFICULTY FEATURES

def add_group_difficulty_features(features_df: pd.DataFrame, group_composition: pd.DataFrame) -> pd.DataFrame:
    """
    Add group difficulty metrics to the feature dataset
    
    For teams that played in a group stage, calculates:
    - Average Elo of all teams in their group
    - Maximum Elo in their group (strongest opponent)
    - Team's Elo relative to group average
    - Team's Elo rank within their group (1 = strongest)
    """
    print("\nAdding group strength features...")

    # Standardize column names
    features_df.columns = features_df.columns.str.lower()
    group_composition.columns = group_composition.columns.str.lower()

    # Merge group assignments with features
    merged = features_df.merge(
        group_composition, 
        on=["tournament", "year", "team"], 
        how="left"
    )
    
    # Keep only teams that had a group stage (some tournaments skip groups)
    merged = merged.dropna(subset=["group"]).reset_index(drop=True)

    # Calculate aggregate statistics for each group
    group_stats = merged.groupby(["tournament", "year", "group"]).agg(
        avg_group_elo=("pre_tournament_elo", "mean"),  
        max_group_elo=("pre_tournament_elo", "max")   
    ).reset_index()

    # Add group statistics back to each team
    merged = merged.merge(group_stats, on=["tournament", "year", "group"], how="left")
    
    # Calculate team's Elo relative to their group average
    # Positive value = stronger than group average
    # Negative value = weaker than group average
    merged["elo_minus_avg_group"] = merged["pre_tournament_elo"] - merged["avg_group_elo"]
    
    # Rank teams within their group by Elo (1 = strongest in group)
    merged["group_elo_rank"] = merged.groupby(["tournament", "year", "group"])["pre_tournament_elo"].rank(
        ascending=False,  
        method="dense"  
    )

    print(f"✅ Added group-level Elo features for {merged['tournament'].nunique()} tournaments.")
    return merged.round(3)


# 5. SAVE FEATURES DATASET

def save_final_features(df: pd.DataFrame, output_path: Path = None) -> Path:
    """
    Save the complete feature dataset to CSV
    """
    # Use default path if none provided
    output_path = output_path or PROCESSED_DATA_DIR / "final_features.csv"
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Saved final features to: {output_path}")
    return output_path


# 6. RUNNING CODE COMMAND

if __name__ == "__main__":
    print("\n========== BUILDING FULL FEATURE DATASET ==========\n")

    # Load all required input data
    print("Loading input data...")
    tournaments_info = pd.read_csv(
        PROCESSED_DATA_DIR / "tournaments_info.csv", 
        parse_dates=["start_date", "end_date"]
    )
    tournaments_results = pd.read_csv(PROCESSED_DATA_DIR / "tournaments_results.csv")
    group_composition = pd.read_csv(PROCESSED_DATA_DIR / "group_stage_composition.csv")
    matches_with_elo = pd.read_csv(
        PROCESSED_DATA_DIR / "elo_matches_since_2000.csv", 
        parse_dates=["date"]
    )

    # Step 1: Build pre-tournament features
    form_features = build_pre_tournament_features(
        tournaments_info, 
        tournaments_results, 
        matches_with_elo, 
        lookback_matches=12
    )

    # Step 2: Add group stage difficulty features
    complete_features = add_group_difficulty_features(form_features, group_composition)

    # Step 3: Save final dataset for modeling
    save_final_features(complete_features)

    print("\n✅ Done — final_features.csv created successfully.\n")