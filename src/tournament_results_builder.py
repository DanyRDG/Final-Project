"""
Tournament Results Builder
Generates tournaments_results.csv including accurate Stage classification,
and excludes tournaments with irregular formats or disqualifications

Version: restricted to tournaments from 2000 onward
"""

from pathlib import Path
import pandas as pd

# 1. FINDING THE CHAMPION OF EACH TOURNAMENT SINCE 2000

def get_final_winner(final_match, shootouts_df):
    """Determine winner of final match with pennalty shootouts if needed"""
    final_teams = [final_match["home_team"], final_match["away_team"]]
    
    # Check if there is a winner in regular time
    if final_match["home_score"] != final_match["away_score"]:
        return (final_match["home_team"] if final_match["home_score"] > final_match["away_score"] 
                else final_match["away_team"])
    
    # Draw in regular time -> check for shootout
    shootout = shootouts_df[
        (shootouts_df["date"] == final_match["date"].strftime("%Y-%m-%d")) &
        (shootouts_df["home_team"].isin(final_teams)) &
        (shootouts_df["away_team"].isin(final_teams))
    ]
    
    if not shootout.empty:
        return shootout.iloc[0]["winner"]


# 2. CLASSIFICATION STAGE "FORMULA"

def classify_team_stage(team, games, max_games, champion, runner_up, third_place_teams):
    """Classify what stage a team reached in the tournament
        Takes the maximum of games played a team inside a tournament = max_games
        Semi-finalists played max_games - 1 etc...
    """
    if team == champion:
        return "Champion"
    elif team == runner_up:
        return "Runner-up"
    elif team in third_place_teams or games == max_games - 1:
        return "Semi-Finalist"
    elif games == max_games - 2:
        return "Quarter-Finalist"
    elif games == max_games - 3:
        return "Round of 16"
    else:
        return "Group Stage"


# 3. CLASSIFYING THE TEAMS FOR EACH TOURNAMENTS SINCE 2000

def classify_stages(df, results_df, shootouts_df):
    """Add stage classification to tournament results."""
    results = []

    for (tournament, year), group in df.groupby(["tournament", "year"]):
        # Get matches for this tournament
        matches = results_df[
            (results_df["tournament"].str.lower() == tournament.lower()) &
            (results_df["date"].dt.year == year)
        ].sort_values("date")

        if matches.empty:
            continue

        # Identify the runner-up from final match
        final_match = matches.iloc[-1]
        third_place_match = matches.iloc[-2]
        
        champion = get_final_winner(final_match, shootouts_df)
        final_teams = [final_match["home_team"], final_match["away_team"]]
        runner_up = [t for t in final_teams if t != champion][0]
        third_place_teams = [third_place_match["home_team"], third_place_match["away_team"]]
        
        max_games = group["matches_played"].max()

        # Classify each team
        for _, row in group.iterrows():
            stage = classify_team_stage(
                row["team"], row["matches_played"], max_games, 
                champion, runner_up, third_place_teams
            )
            record = row.to_dict()
            record["stage"] = stage
            results.append(record)

    return pd.DataFrame(results)


#4. STATISTICS BY TEAM FOR EACH TOURNAMENTS SINCE 2000

def calculate_team_stats(team, matches):
    """Calculate statistics for a team in their tournament matches."""
    team_matches = matches[(matches["home_team"] == team) | (matches["away_team"] == team)]
    
    # Count wins
    home_wins = ((team_matches["home_team"] == team) & 
                 (team_matches["home_score"] > team_matches["away_score"])).sum()
    away_wins = ((team_matches["away_team"] == team) & 
                 (team_matches["away_score"] > team_matches["home_score"])).sum()
    
    # Calculate goals
    goals_for = team_matches.apply(
        lambda r: r["home_score"] if r["home_team"] == team else r["away_score"], axis=1
    ).sum()
    goals_against = team_matches.apply(
        lambda r: r["away_score"] if r["home_team"] == team else r["home_score"], axis=1
    ).sum()
    
    return {
        "matches_played": len(team_matches),
        "wins": home_wins + away_wins,
        "goals_for": goals_for,
        "goals_conceded": goals_against
    }


#5. SORTING BY INSIDE EACH TOURNAMENT BY STAGE FOR LISIBILITY REASONS

def sort_by_stage(df):
    """Sort DataFrame so teams are ordered by stage within each tournament."""
    # Define stage order (from best to worst)
    stage_order = {
        "Champion": 1,
        "Runner-up": 2,
        "Semi-Finalist": 3,
        "Quarter-Finalist": 4,
        "Round of 16": 5,
        "Group Stage": 6
    }
    
    # Add sorting column
    df["stage_order"] = df["stage"].map(stage_order)
    
    # Sort by year, tournament, then stage order
    df = df.sort_values(["year", "tournament", "stage_order", "team"])
    
    # Remove the temporary sorting column
    df = df.drop(columns=["stage_order"])
    
    return df

#6. BUILDING THE DATAFRAME

def build_tournament_results(results_df, shootouts_df, tournaments_info, output_path: Path):
    """Build tournaments_results.csv excluding irregular tournaments (2000+ only)."""
    print("\n🏗️ Building tournaments_results.csv (2000 onward)...")

    tournaments_info.columns = tournaments_info.columns.str.lower()

    # Tournaments with irregular formats to exclude for feasibility reasons
    excluded_tournaments = {
        ("african cup of nations", 2010), ("gold cup", 2000),
        ("gold cup", 2002), ("gold cup", 2003),
    }

    records = []

    for _, tournament in tournaments_info.iterrows():
        # Skip pre-2000 tournaments
        if tournament["year"] < 2000:
            continue

        # Check if tournament should be excluded
        base_name = next(
            (name for name in ["fifa world cup", "uefa euro", "copa américa", "gold cup",
                               "african cup of nations", "afc asian cup"] 
             if name in tournament["tournament"].lower()), None
        )
        
        if (base_name, tournament["year"]) in excluded_tournaments:
            print(f"🚫 Skipping {tournament['tournament']} ({tournament['year']}) — excluded due to irregular format.")
            continue

        # Get matches within tournament dates
        matches = results_df[
            (results_df["tournament"].str.lower() == tournament["tournament"].lower()) &
            (results_df["date"] >= tournament["start_date"]) &
            (results_df["date"] <= tournament["end_date"])
        ]
        
        if matches.empty:
            continue

        # Get all participating teams
        teams = pd.unique(matches[["home_team", "away_team"]].values.ravel())

        # Calculate stats for each team
        for team in teams:
            stats = calculate_team_stats(team, matches)
            records.append({
                "tournament": tournament["tournament"],
                "year": tournament["year"],
                "team": team,
                **stats
            })

    # Create DataFrame and add stage classifications
    summary = pd.DataFrame(records)
    summary = classify_stages(summary, results_df, shootouts_df)
    
    # Sort by stage within each tournament
    summary = sort_by_stage(summary)

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)

    print(f"✅ tournaments_results.csv saved to {output_path}")
    print(f"{len(summary)} rows across {summary['tournament'].nunique()} tournaments.")
    print(f"Tournaments from {summary['year'].min()} to {summary['year'].max()}")
    
    return summary


#7. RUNNING CODE COMMAND

if __name__ == "__main__":
    # Set up paths
    project_root = Path(__file__).resolve().parents[1]
    processed_dir = project_root / "data" / "processed"

    # Load data
    print("Loading processed data...")
    results_df = pd.read_csv(processed_dir / "results_cleaned.csv")
    shootouts_df = pd.read_csv(processed_dir / "shootouts_cleaned.csv")
    tournaments_info = pd.read_csv(processed_dir / "tournaments_info.csv")

    # Convert dates
    results_df["date"] = pd.to_datetime(results_df["date"], errors="coerce")
    tournaments_info["start_date"] = pd.to_datetime(tournaments_info["start_date"])
    tournaments_info["end_date"] = pd.to_datetime(tournaments_info["end_date"])

    # Build results
    build_tournament_results(
        results_df, shootouts_df, tournaments_info,
        processed_dir / "tournaments_results.csv"
    )