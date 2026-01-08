"""
World_Cup2026.py
------------------------------------
Predicts 2026 FIFA World Cup results using the pre-trained Logistic Regression model

Pipeline Steps:
1. Define official 2026 World Cup groups (48 teams, 12 groups)
2. Extract team features: Elo ratings, recent form (last 12 matches), group strength
3. Generate predictions: probability of reaching each tournament stage
4. Rank teams by predicted tournament success

Requirements:
- Trained model files: logreg_model.joblib and logreg_scaler.joblib
- Historical data: elo_timeseries_since_2000.csv and results_cleaned.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np
import joblib


# 1. DEFINE 2026 WORLD CUP GROUPS

def build_group_stage_composition_2026(output_path: Path) -> pd.DataFrame:
    """
    Creates the official group composition for 2026 FIFA World Cup
    
    Format: 48 teams divided into 12 groups (A-L) with 4 teams each
    
    Returns:
        DataFrame with columns [tournament, year, team, group]
    """
    print("\n Creating 2026 World Cup groups...")
    
    # Official group assignments for the 2026 World Cup
    groups = {
        "A": ["Mexico", "South Africa", "South Korea", "Denmark"],
        "B": ["Canada", "Italy", "Qatar", "Switzerland"],
        "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
        "D": ["United States", "Paraguay", "Australia", "Turkey"],
        "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
        "F": ["Netherlands", "Japan", "Albania", "Tunisia"],
        "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
        "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
        "I": ["France", "Senegal", "Bolivia", "Norway"],
        "J": ["Argentina", "Algeria", "Austria", "Jordan"],
        "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
        "L": ["England", "Croatia", "Ghana", "Panama"],
    }
    
    # Convert groups dictionary to DataFrame
    rows = []
    for group_letter, teams in groups.items():
        for team in teams:
            rows.append({
                "tournament": "fifa world cup",
                "year": 2026,
                "team": team,
                "group": group_letter
            })
    
    df = pd.DataFrame(rows)
    
    # Save to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"✅ Saved {len(df)} teams across {df['group'].nunique()} groups")
    return df


# 2. BUILD THE FEATURES FOR EACH TEAM

def load_pre_tournament_features(processed_dir: Path) -> pd.DataFrame:
    """
    Constructs prediction features for each 2026 World Cup team
    
    Same features as the training model:
    1. Pre-tournament Elo (team strength rating)
    2. Recent form from last 12 matches (win rate, goals, etc.)
    3. Group difficulty (average/max Elo in group, relative position)
    
    Returns:
        DataFrame with all features for each team
    """
    print("\n Building team features...")
    
    # Load required data files
    groups_df = pd.read_csv(processed_dir / "group_stage_composition_2026.csv")
    elo_df = pd.read_csv(processed_dir / "elo_timeseries_since_2000.csv", parse_dates=["date"])
    results_df = pd.read_csv(processed_dir / "results_cleaned.csv", parse_dates=["date"])
    
    # Set cutoff date (January 1st 2026)
    cutoff_date = pd.Timestamp("2026-01-01")

    # Feature 1: Latest Elo Rating
    # Get most recent Elo for each team before the tournament
    pre_tournament_elo = (
        elo_df[elo_df["date"] < cutoff_date]
        .sort_values("date")
        .groupby("team")
        .last()[["elo"]]
        .rename(columns={"elo": "pre_tournament_elo"})
        .reset_index()
    )
    
    # Features 2-7: Recent Form (Last 12 Matches)
    form_data = []
    
    for team in groups_df["team"].unique():
        # Get team's last 12 matches before the cutoff date
        team_matches = results_df[
            ((results_df["home_team"] == team) | (results_df["away_team"] == team))
            & (results_df["date"] < cutoff_date)
        ].sort_values("date", ascending=False).head(12)
        
        if team_matches.empty:
            continue
        
        n_matches = len(team_matches)
        
        # Count wins (home wins or away wins)
        wins = (
            ((team_matches["home_team"] == team) & (team_matches["home_score"] > team_matches["away_score"])) |
            ((team_matches["away_team"] == team) & (team_matches["away_score"] > team_matches["home_score"]))
        ).sum()
        
        # Calculate goals scored and conceded
        goals_for = team_matches.apply(
            lambda row: row["home_score"] if row["home_team"] == team else row["away_score"], 
            axis=1
        ).sum()
        
        goals_against = team_matches.apply(
            lambda row: row["away_score"] if row["home_team"] == team else row["home_score"], 
            axis=1
        ).sum()
        
        # Get average opponent strength (Elo rating)
        opponent_names = team_matches.apply(
            lambda row: row["away_team"] if row["home_team"] == team else row["home_team"], 
            axis=1
        )
        
        opponent_elos = elo_df[
            (elo_df["team"].isin(opponent_names)) & (elo_df["date"] < cutoff_date)
        ].sort_values("date").groupby("team").last()["elo"]
        
        avg_opponent_elo = opponent_elos.mean() if not opponent_elos.empty else np.nan
        
        # Store form metrics
        form_data.append({
            "team": team,
            "n_matches_form": n_matches,                              
            "win_rate": wins / n_matches,                              
            "avg_goal_diff": (goals_for - goals_against) / n_matches,  
            "goals_for_per_match": goals_for / n_matches,              
            "goals_against_per_match": goals_against / n_matches,      
            "avg_opponent_elo": avg_opponent_elo,                      
        })
    
    form_df = pd.DataFrame(form_data)
    print(f"✅ Computed form for {len(form_df)} teams (last 12 matches)")
    
    # Combine group info, Elo, and form data
    features = (
        groups_df
        .merge(pre_tournament_elo, on="team", how="left")
        .merge(form_df, on="team", how="left")
    )
    
    # Features 8-11: Group Context Features
    # Calculate average and maximum Elo within each group
    group_stats = (
        features.groupby("group")["pre_tournament_elo"]
        .agg(["mean", "max"])
        .rename(columns={"mean": "avg_group_elo", "max": "max_group_elo"})
        .reset_index()
    )
    
    features = features.merge(group_stats, on="group", how="left")
    
    # Calculate team's Elo relative to their group average
    features["elo_minus_avg_group"] = features["pre_tournament_elo"] - features["avg_group_elo"]
    
    # Rank teams within their group (1 = strongest, 4 = weakest)
    features["group_elo_rank"] = features.groupby("group")["pre_tournament_elo"].rank(ascending=False)
    
    # Save feature dataset
    output_path = processed_dir / "final_features_2026.csv"
    features.to_csv(output_path, index=False)
    print(f"✅ Saved features to {output_path}")
    
    return features


# 3. GENERATE PREDICTIONS 

def predict_worldcup_2026(features_df: pd.DataFrame, results_dir: Path) -> pd.DataFrame:
    """
    Generates World Cup predictions using the pre-trained Logistic Regression model
    
    Process:
    1. Load trained model and scaler
    2. Standardize features (same normalization as training)
    3. Predict probability of reaching each stage (Champion, Runner-up, etc.)
    4. Calculate "tournament success score" for ranking teams
    5. Save predictions sorted by success probability
    
    Returns:
        DataFrame with predictions and stage probabilities for all teams
    """
    print("\n Generating 2026 World Cup predictions...")
    
    # Define feature columns (must match training data)
    feature_cols = [
        "pre_tournament_elo",
        "n_matches_form",
        "win_rate",
        "avg_goal_diff",
        "goals_for_per_match",
        "goals_against_per_match",
        "avg_opponent_elo",
        "avg_group_elo",
        "max_group_elo",
        "elo_minus_avg_group",
        "group_elo_rank",
    ]
    
    # Prepare feature matrix (fill any missing values with 0)
    X = features_df[feature_cols].fillna(0)
    
    # Load pre-trained model and scaler
    model_path = results_dir.parent / "models" / "logreg_model.joblib"
    scaler_path = results_dir.parent / "models" / "logreg_scaler.joblib"
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    # Standardize features (mean=0, std=1)
    X_scaled = scaler.transform(X)
    
    # Generate predictions
    # Get probability of reaching each tournament stage
    probs = model.predict_proba(X_scaled)
    
    # Stage labels
    stage_names = {
        0: "Champion",
        1: "Runner-up",
        2: "Semi-Finalist",
        3: "Quarter-Finalist",
        4: "Round of 16",
        5: "Group Stage",
    }
    
    # Convert probabilities to DataFrame with stage names as columns
    prob_df = pd.DataFrame(probs, columns=[stage_names[i] for i in model.classes_])
    
    # Combine original data with predictions
    predictions = pd.concat([features_df.reset_index(drop=True), prob_df], axis=1)
    
    # Calculate tournament success score
    # Weighted sum: Champion (6 pts) > Runner-up (5 pts) > ... > Group Stage (1 pt)
    predictions["tournament_success_score"] = (
        predictions["Champion"] * 6 +
        predictions["Runner-up"] * 5 +
        predictions["Semi-Finalist"] * 4 +
        predictions["Quarter-Finalist"] * 3 +
        predictions["Round of 16"] * 2 +
        predictions["Group Stage"] * 1
    )
    
    # Rank teams by success score (highest = most likely to succeed)
    predictions = predictions.sort_values("tournament_success_score", ascending=False).reset_index(drop=True)
    
    # Round probabilities for readability
    predictions = predictions.round(3)
    
    # Save and display results
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "predictions_worldcup2026.csv"
    predictions.to_csv(output_path, index=False)
    
    print("\n Top 10 Predicted Teams:")
    print(predictions[["team", "group", "tournament_success_score"]].head(10).to_string(index=False))
    print(f"\n✅ Full predictions saved to {output_path}")
    
    return predictions


# 4. RUNNING CODE COMMAND

if __name__ == "__main__":
    print("\n" + "="*70)
    print("  2026 FIFA WORLD CUP PREDICTION PIPELINE")
    print("="*70)
    
    # Define directory structure
    project_root = Path(__file__).resolve().parents[1]
    processed_dir = project_root / "data" / "processed"
    results_dir = project_root / "results" / "predictions"
    
    # Step 1: Create group stage composition
    groups_output = processed_dir / "group_stage_composition_2026.csv"
    build_group_stage_composition_2026(groups_output)
    
    # Step 2: Build feature set for all teams
    features_2026 = load_pre_tournament_features(processed_dir)
    
    # Step 3: Generate predictions using trained model
    predictions_2026 = predict_worldcup_2026(features_2026, results_dir)
    
    print("\n" + "="*70)
    print("  ✅ PREDICTION PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70 + "\n")