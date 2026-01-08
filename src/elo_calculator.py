"""
Elo Rating Calculator Module (Since 2000 Only)
Computes Elo ratings for international football matches using the Elo rating system

Inputs:
- /data/processed/results_since_2000.csv

Outputs:
- /data/processed/elo_matches_since_2000.csv (match-by-match ratings)
- /data/processed/elo_timeseries_since_2000.csv (team rating history)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import pandas as pd
from pathlib import Path


# 1. ELO CONFIGURARTION

@dataclass
class EloConfig:
    """
    Configuration parameters for Elo rating calculations
    
    Attributes:
        initial_rating: Starting rating for new teams (1500 is standard)
        k_factor: How much ratings change per match (higher = more volatile)
        home_advantage: Elo points bonus for playing at home
    """
    initial_rating: float = 1500.0
    k_factor: float = 30.0
    home_advantage: float = 100.0


# 2. CORE ELO FUNCTION

def calculate_expected_score(rating_a: float, rating_b: float) -> float:
    """
    Calculate the expected probability that team A wins against team B
    
    Formula: 1 / (1 + 10^((rating_b - rating_a) / 400))
    
    Returns a value between 0 and 1:
    - 0.5 means evenly matched teams
    - Higher values mean team A is favored
    """
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


# 3. ELO RESULTS CAULCULATION AFTER THE MATCH
    
def update_ratings(
    rating_a: float,
    rating_b: float,
    actual_score_a: float,
    actual_score_b: float,
    config: EloConfig,
) -> Tuple[float, float]:
    """
    Update both teams' ratings based on match result
    
    The rating change is proportional to:
    - The difference between actual and expected results
    - The k_factor (sensitivity of rating changes)
    
    Args:
        rating_a: Team A's rating before the match
        rating_b: Team B's rating before the match
        actual_score_a: Actual result for A (1=win, 0.5=draw, 0=loss)
        actual_score_b: Actual result for B (1=win, 0.5=draw, 0=loss)
        config: Elo configuration parameters
        
    Returns:
        Tuple of (new_rating_a, new_rating_b)
    """
    # Calculate what we expected to happen
    expected_a = calculate_expected_score(rating_a, rating_b)
    expected_b = calculate_expected_score(rating_b, rating_a)
    
    # Update ratings based on expectation vs reality
    new_rating_a = rating_a + config.k_factor * (actual_score_a - expected_a)
    new_rating_b = rating_b + config.k_factor * (actual_score_b - expected_b)
    
    return new_rating_a, new_rating_b


# 4. ELO COMPUTATION FOR MATCHES

def compute_elo_ratings(matches: pd.DataFrame, config: EloConfig = None) -> pd.DataFrame:
    """
    Calculate Elo ratings for all matches chronologically
    
    Process:
    1. Sort matches by date (chronological order is crucial)
    2. For each match:
       - Get current ratings for both teams
       - Apply home advantage
       - Determine match result (win/draw/loss)
       - Update ratings based on result
       - Store pre- and post-match ratings
    """
    if config is None:
        config = EloConfig()

    df = matches.copy().sort_values("date").reset_index(drop=True)
    
    # Dictionary to track each team's current rating
    current_ratings: Dict[str, float] = {}
    
    # Lists to store pre/post ratings for each match
    pre_home, pre_away, post_home, post_away = [], [], [], []

    for _, match in df.iterrows():
        home = match["home_team"]
        away = match["away_team"]
        home_goals = match["home_score"]
        away_goals = match["away_score"]

        # Get current ratings (or use initial rating for new teams)
        home_rating = current_ratings.get(home, config.initial_rating)
        away_rating = current_ratings.get(away, config.initial_rating)

        # Apply home advantage to home team's effective rating
        home_rating_effective = home_rating + config.home_advantage
        away_rating_effective = away_rating

        # Convert match result to scores (1=win, 0.5=draw, 0=loss)
        if home_goals > away_goals:
            home_result, away_result = 1.0, 0.0
        elif home_goals < away_goals:
            home_result, away_result = 0.0, 1.0
        else:
            home_result, away_result = 0.5, 0.5

        # Calculate new ratings with home advantage applied
        home_new_effective, away_new_effective = update_ratings(
            home_rating_effective, away_rating_effective, 
            home_result, away_result, config
        )

        # Remove home advantage from home team's final rating
        home_new = home_new_effective - config.home_advantage
        away_new = away_new_effective

        # Update the rating tracker
        current_ratings[home] = home_new
        current_ratings[away] = away_new

        # Store ratings for the match (rounded to 2 decimal places)
        pre_home.append(round(home_rating, 2))
        pre_away.append(round(away_rating, 2))
        post_home.append(round(home_new, 2))
        post_away.append(round(away_new, 2))

    # Add rating columns to dataframe
    df["home_elo_pre"] = pre_home
    df["away_elo_pre"] = pre_away
    df["home_elo_post"] = post_home
    df["away_elo_post"] = post_away

    return df


# 4. Elo Time Series Construction

def build_elo_timeseries(matches_with_elo: pd.DataFrame) -> pd.DataFrame:
    """
    Convert match-level Elo data into a team-level time series
    
    Creates one row per team per date showing their rating after all matches that day
    Useful for tracking a team's rating evolution over time (specially last date before a tournament)
    """
    # Extract home team ratings after each match
    home_series = matches_with_elo[["date", "home_team", "home_elo_post"]].rename(
        columns={"home_team": "team", "home_elo_post": "elo"}
    )
    
    # Extract away team ratings after each match
    away_series = matches_with_elo[["date", "away_team", "away_elo_post"]].rename(
        columns={"away_team": "team", "away_elo_post": "elo"}
    )

    # Combine both series
    timeseries = pd.concat([home_series, away_series], ignore_index=True)
    
    # If a team played multiple matches on the same day, keep only the final rating
    timeseries = (
        timeseries.groupby(["team", "date"], as_index=False)
        .agg(elo=("elo", "last"))  # Keep the last rating from that day
        .sort_values(["team", "date"])
        .reset_index(drop=True)
    )
    
    # Round Elo ratings to 2 decimal places
    timeseries["elo"] = timeseries["elo"].round(2)

    return timeseries
    
# 5. TEAMS' ELO FOR SPECIFIC DATE 

def get_team_rating_before_date(
    elo_timeseries: pd.DataFrame,
    team: str,
    target_date: pd.Timestamp,
    default_rating: float = 1500.0,
) -> float:
    """
    Get a team's Elo rating just before a specific date
    
    Useful for predictions: "What was Brazil's rating before the 2022 World Cup?"
    """
    # Get all ratings for this team before the target date
    team_history = elo_timeseries[
        (elo_timeseries["team"] == team) & 
        (elo_timeseries["date"] < target_date)
    ]
    
    # If no history exists, return default rating
    if team_history.empty:
        return default_rating
    
    # Return the most recent rating (rounded to 2 decimal places)
    return round(float(team_history.sort_values("date")["elo"].iloc[-1]), 2)


# 6. BUILDING CODE FOR THE FILES

def build_elo_files():
    """
    Main function that runs the complete Elo calculation pipeline.
    
    Steps:
    Load match results from 2000 onwards
    Calculate Elo ratings for each match chronologically
    Build team-level time series
    Save both outputs to CSV files
    """
    # Set up file paths
    project_root = Path(__file__).resolve().parents[1]
    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    input_file = processed_dir / "results_since_2000.csv"
    
    # Check if input file exists
    if not input_file.exists():
        print(f"Missing {input_file.name}, cannot compute Elo ratings.")
        return

    print(f"\n Computing Elo ratings from 2000 onward...")
    
    # Load and sort matches by date
    matches = pd.read_csv(input_file, parse_dates=["date"])
    matches = matches.sort_values("date")

    # Run Elo calculations
    matches_with_elo = compute_elo_ratings(matches, EloConfig())
    elo_timeseries = build_elo_timeseries(matches_with_elo)

    # Save outputs
    matches_output = processed_dir / "elo_matches_since_2000.csv"
    timeseries_output = processed_dir / "elo_timeseries_since_2000.csv"

    matches_with_elo.to_csv(matches_output, index=False)
    elo_timeseries.to_csv(timeseries_output, index=False)

    # Print summary
    print(f"✅ Saved match-level Elo to {matches_output}")
    print(f"✅ Saved team-level Elo time series to {timeseries_output}")
    print(f"Date range: {matches['date'].min().date()} → {matches['date'].max().date()}")
    print(f"Teams covered: {elo_timeseries['team'].nunique()}")


# 7. RUNNING CODE COMMAND

if __name__ == "__main__":
    build_elo_files()