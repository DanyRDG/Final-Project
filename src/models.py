"""
Model Training: Leave-One-Tournament-Out (LOTO)
------------------------------------------------
Trains and evaluates three models to predict tournament success:
1. Logistic Regression (baseline linear model)
2. XGBoost (gradient boosting)
3. Random Forest (ensemble method)

Evaluation:
- Trains on tournaments < 2016, predicts tournaments >= 2016
- Uses standardized features
- Computes per-tournament accuracy, MAE, and macro-3 accuracy
- Includes 3-category evaluation (Deep Run / Knockouts / Group Stage)
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from joblib import dump
import warnings

# 1.LOADING THE DATA AND FEATURES

def load_data(path: Path) -> pd.DataFrame:
    """
    Load and clean the features dataset
    Removes rows with missing target values (stage_order) or missing Elo ratings (should be none)
    """
    df = pd.read_csv(path)
    # Drop rows where we don't know the outcome or pre-tournament strength
    df = df.dropna(subset=["stage_order", "pre_tournament_elo"])
    return df


def get_features_targets(df: pd.DataFrame):
    """
    Split dataset into features (X) and target (y).
    
    Features include:
    - Pre-tournament Elo rating (team strength)
    - Recent form metrics (last 12 matches)
    - Group stage difficulty
    
    Target is stage_order:
    0=Champion, 1=Runner-up, 2=Semi-Finalist, 3=Quarter-Finalist, 4=Round of 16, 5=Group Stage
    """
    feature_cols = [
        "pre_tournament_elo",         # Team's rating before tournament
        "n_matches_form",             # Number of matches in form window (should be 12)
        "win_rate",                   # Percentage of wins on last 12 games
        "avg_goal_diff",              # Average goal difference per match last 12 games
        "goals_for_per_match",        # Offensive strength last 12 games
        "goals_against_per_match",    # Defensive strength last 12 games
        "avg_opponent_elo",           # Average strength of opponents last 12 games
        "avg_group_elo",              # Average Elo in team's group
        "max_group_elo",              # Strongest opponent in group
        "elo_minus_avg_group",        # Team's Elo relative to group average
        "group_elo_rank",             # Team's ranking within their group
    ]
    X = df[feature_cols]
    y = df["stage_order"]
    return X, y, feature_cols


def stage_decoder():
    """
    Mapping between numeric stage codes and labels
    Used to interpret model predictions and display results.
    """
    return {
        0: "Champion",           # Won the tournament
        1: "Runner-up",          # Lost in the final
        2: "Semi-Finalist",      # Lost in semi-final
        3: "Quarter-Finalist",   # Lost in quarter-final
        4: "Round of 16",        # Lost in round of 16
        5: "Group Stage",        # Eliminated in group stage
    }


def collapse_stage(stage_label: str) -> str:
    """
    Reduce detailed stages into 3 macro categories for simplified evaluation
    
    Categories:
    - Deep Run: Champion, Runner-up, Semi-Finalist (top 4 teams)
    - Knockouts: Quarter-Finalist, Round of 16 (made it past groups)
    - Group Stage: Eliminated in groups
    """
    if stage_label in ["Champion", "Runner-up", "Semi-Finalist"]:
        return "Deep Run"
    elif stage_label in ["Quarter-Finalist", "Round of 16"]:
        return "Knockouts"
    else:
        return "Group Stage"

        
# 2. MODEL PARAMETERS

def get_model(model_type: str):
    """
    Create and return the specified models
    
    Models:
    - logreg: Logistic Regression (linear baseline, interpretable)
    - randomforest: Random Forest (ensemble of decision trees, handles non-linearity)
    - xgboost: XGBoost (gradient boosting, often best performance on tabular data)
    """
    if model_type == "logreg":
        return LogisticRegression(
            multi_class="multinomial",  
            solver="lbfgs",             
            C=1.0,                      
            max_iter=1000,              
            random_state=42,            
        )
    elif model_type == "randomforest":
        return RandomForestClassifier(
            n_estimators=300,           
            max_depth=10,               
            min_samples_split=5,        
            min_samples_leaf=2,         
            max_features="sqrt",        
            random_state=42,
            n_jobs=-1,                  
            class_weight="balanced"     
        )
    elif model_type == "xgboost":
        return XGBClassifier(
            n_estimators=400,           
            max_depth=6,                
            learning_rate=0.05,         
            subsample=0.8,              
            colsample_bytree=0.8,       
            objective="multi:softprob", 
            eval_metric="mlogloss",     
            random_state=42,
            verbosity=0                 
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# 3. lEAVE ONE TOURNAMENT OUT MODEL CREATION

def loto_evaluation(df: pd.DataFrame, model_type: str = "logreg"):
    """
    Perform Leave-One-Tournament-Out Cross-Validation   
    How does the model work:
    - For each tournament from 2016 onwards (test set):
      - Train on all tournaments between 2000 to 2016 (training set)
      - Predict the stage each team will reach
      - Compare predictions to actual outcomes
    
    This mimics real-world prediction: using historical data to predict future tournaments.
    """
    # Get list of all tournaments sorted by year
    tournaments = df[["tournament", "year"]].drop_duplicates().sort_values(["year"])
    results = []
    all_predictions = []
    stage_map = stage_decoder()

    print(f"\n{'='*70}")
    print(f"  MODEL: {model_type.upper()}")
    print(f"{'='*70}")

    # Evaluate on each tournament
    for _, row in tournaments.iterrows():
        t_name, t_year = row["tournament"], row["year"]
        
        # Only predict tournaments from 2016 onward (test set)
        if t_year < 2016:
            continue

        # Split data: this tournament is test, pre-2016 is training
        test_mask = (df["tournament"] == t_name) & (df["year"] == t_year)
        train_mask = (df["year"] < 2016)
        train_df, test_df = df[train_mask], df[test_mask]

        # Extract features and targets
        X_train, y_train, feature_cols = get_features_targets(train_df)
        X_test, y_test, _ = get_features_targets(test_df)

        # Handle missing values by filling with 0 (neutral value after scaling)
        X_train = X_train.fillna(0)
        X_test = X_test.fillna(0)

        # Standardize features: mean=0, std=1
        # This ensures all features contribute equally (Elo is ~1500, win_rate is 0-1)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Get and train the specified model
        model = get_model(model_type)
        model.fit(X_train_scaled, y_train)

        # Save model and scaler for future use (2026 World Cup predictions)
        models_dir = Path(__file__).resolve().parents[1] / "results" / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        dump(model, models_dir / f"{model_type}_model.joblib")
        dump(scaler, models_dir / f"{model_type}_scaler.joblib")

        # Get probability predictions for each stage
        # Shape: (n_teams, n_classes) where each row sums to 1.0
        probs = model.predict_proba(X_test_scaled)
        prob_df = pd.DataFrame(probs, columns=[stage_map[i] for i in model.classes_])
        
        # Round probabilities to 5 decimal places for easier CSV readability
        prob_df = prob_df.round(5)
        
        temp = pd.concat([test_df.reset_index(drop=True), prob_df], axis=1)

        # Compute overall tournament success score
        # Weight stages by importance: Champion=6 points, Runner-up=5, ..., Group Stage=1
        temp["tournament_success_score"] = (
            temp.get("Champion", 0) * 6 +
            temp.get("Runner-up", 0) * 5 +
            temp.get("Semi-Finalist", 0) * 4 +
            temp.get("Quarter-Finalist", 0) * 3 +
            temp.get("Round of 16", 0) * 2 +
            temp.get("Group Stage", 0) * 1
        )
        
        # Sort teams by predicted success (best teams first)
        temp = temp.sort_values("tournament_success_score", ascending=False).reset_index(drop=True)
        total_teams = len(temp)

        # Determine actual tournament structure
        # Count how many teams actually reached each stage
        real_stage_counts = test_df["stage_order"].map(stage_map).value_counts().to_dict()
        real_stage_counts["Champion"] = 1    # Always exactly 1 champion
        real_stage_counts["Runner-up"] = 1   # Always exactly 1 runner-up
        valid_stages = list(real_stage_counts.keys())

        # Assign predicted stages based on ranking
        # Top team = predicted champion, next team = predicted runner-up, etc.
        temp["predicted_stage_label"] = "Group Stage"
        assigned = 0
        for stage in ["Champion", "Runner-up", "Semi-Finalist", "Quarter-Finalist", "Round of 16"]:
            if stage not in valid_stages:
                continue
            # Assign this stage to the next N highest-ranked teams
            n = real_stage_counts[stage]
            end = min(assigned + n, total_teams)
            temp.loc[assigned:end - 1, "predicted_stage_label"] = stage
            assigned = end

        # Add true and predicted labels
        temp["true_stage_label"] = temp["stage_order"].map(stage_map)
        temp["predicted_stage_order"] = temp["predicted_stage_label"].map({v: k for k, v in stage_map.items()})

        # Create macro categories (3 classes instead of 6)
        temp["true_macro_label"] = temp["true_stage_label"].apply(collapse_stage)
        temp["pred_macro_label"] = temp["predicted_stage_label"].apply(collapse_stage)

        # Calculate evaluation metrics
        # 1. Stage accuracy: exact match of predicted vs actual stage
        acc_total = (temp["predicted_stage_label"] == temp["true_stage_label"]).mean()
        
        # 2. Mean Absolute Error: average difference in stage numbers
        #    Example: predicting Semi-Finalist (2) when actual is Champion (0) = error of 2
        mae_stage = np.mean(np.abs(temp["predicted_stage_order"] - temp["stage_order"]))
        
        # 3. Macro-3 accuracy: accuracy on 3 broader categories
        acc_macro3 = accuracy_score(temp["true_macro_label"], temp["pred_macro_label"])
        
        # Check if we predicted the champion correctly
        actual_champion = temp.loc[temp["true_stage_label"] == "Champion", "team"].values
        predicted_champion = temp.loc[temp["predicted_stage_label"] == "Champion", "team"].values
        actual_champion = actual_champion[0] if len(actual_champion) else "N/A"
        predicted_champion = predicted_champion[0] if len(predicted_champion) else "N/A"
        champion_correct = actual_champion == predicted_champion
        
        # Print tournament results
        print(f"\n{t_name} {t_year} ({total_teams} teams)")
        print(f"   → Predicted Champion: {predicted_champion}")
        print(f"   → Actual Champion:    {actual_champion} {'✅' if champion_correct else '❌'}")
        print(f"   → Stage Accuracy:     {acc_total:.3f}")
        print(f"   → Stage MAE:          {mae_stage:.3f}")
        print(f"   → Macro-3 Accuracy:   {acc_macro3:.3f}")
        
        # Show top 8 predicted teams
        print("\n  Top 8 predicted teams:")
        leaderboard = temp[["team", "predicted_stage_label", "pred_macro_label", "tournament_success_score"]].head(8)
        print(leaderboard.to_string(index=False))

        # Store results for this tournament
        results.append({
            "tournament": t_name,
            "year": t_year,
            "n_teams": total_teams,
            "accuracy_stage": acc_total,
            "mae_stage": mae_stage,
            "accuracy_macro3": acc_macro3,
            "predicted_champion": predicted_champion,
            "actual_champion": actual_champion,
            "champion_correct": champion_correct,
        })
        all_predictions.append(temp)

    # Combine all tournament predictions into single DataFrames
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    results_df = pd.DataFrame(results)
    return results_df, predictions_df


# 4. RUNNING CODE COMMAND

if __name__ == "__main__":
    print("\n" + "="*70)
    print("  MODEL TRAINING: LEAVE-ONE-TOURNAMENT-OUT (LOTO)")
    print("="*70)
    
    # Load feature dataset
    processed_dir = Path(__file__).resolve().parents[1] / "data" / "processed"
    features_path = processed_dir / "final_features.csv"
    
    try:
        df = load_data(features_path)
    except FileNotFoundError:
        print(f"Error: Features file not found at {features_path}")
        print("Please ensure the data processing step has been run.")
        exit(1)
    
    # Train and evaluate all three models for comparison
    models = ["logreg", "randomforest", "xgboost"]
    
    for model_type in models:
        # Run Leave-One-Tournament-Out evaluation
        results_df, predictions_df = loto_evaluation(df, model_type=model_type)

        # Save tournament-level results (one row per tournament)
        results_path = processed_dir / f"loto_results_{model_type}.csv"
        results_df.to_csv(results_path, index=False)
        print(f"\n✅ Tournament-level results saved to {results_path}")
        
        # Save team-level predictions (one row per team per tournament)
        predictions_path = processed_dir / f"loto_predictions_{model_type}.csv"
        predictions_df.to_csv(predictions_path, index=False)
        print(f"✅ Full team-level predictions saved to {predictions_path}")
        
        # Display summary statistics across all tournaments
        print(f"\n{'='*70}")
        print(f"  OVERALL SUMMARY - {model_type.upper()}")
        print(f"{'='*70}")
        print(results_df.describe()[["accuracy_stage", "mae_stage", "accuracy_macro3"]].T)
    
    print("\n" + "="*70)
    print("  ✅ All three models trained and evaluated successfully!")
    print("="*70 + "\n")