"""
Model Training: Leave-One-Tournament-Out (LOTO)
------------------------------------------------
Predicts tournament success (stage_order) using pre-tournament features.
Version: Logistic Regression baseline
-------------------------------------
- Trains on tournaments <= 2016, predicts tournaments >= 2016
- Uses standardized features and L2 regularization
- Computes per-tournament accuracy, MAE, and macro-3 accuracy
- Adds reduced 3-category evaluation (Deep Run / Knockouts / Group Stage)
"""
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")


def load_data(path: Path) -> pd.DataFrame:
    """Load and clean the features dataset."""
    df = pd.read_csv(path)
    df = df.dropna(subset=["stage_order", "pre_tournament_elo"])
    return df

def get_features_targets(df: pd.DataFrame):
    """Split into features (X) and target (y)."""
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
    X = df[feature_cols]
    y = df["stage_order"]
    return X, y, feature_cols

def stage_decoder():
    """Mapping between encoded stages and readable labels."""
    return {
        0: "Champion",
        1: "Runner-up",
        2: "Semi-Finalist",
        3: "Quarter-Finalist",
        4: "Round of 16",
        5: "Group Stage",
    }

def collapse_stage(stage_label: str) -> str:
    """Reduce detailed stages into 3 macro categories."""
    if stage_label in ["Champion", "Runner-up", "Semi-Finalist"]:
        return "Deep Run"
    elif stage_label in ["Quarter-Finalist", "Round of 16"]:
        return "Knockouts"
    else:
        return "Group Stage"

def loto_evaluation(df: pd.DataFrame):
    """Perform Leave-One-Tournament-Out CV (restricted to post-2016 tournaments)."""
    tournaments = df[["tournament", "year"]].drop_duplicates().sort_values(["year"])
    results = []
    all_predictions = []
    stage_map = stage_decoder()

    for _, row in tournaments.iterrows():
        t_name, t_year = row["tournament"], row["year"]
        
        # ✅ Predict tournaments from 2016 onward
        if t_year < 2016:
            continue

        test_mask = (df["tournament"] == t_name) & (df["year"] == t_year)
        train_mask = (df["year"] < 2016)
        train_df, test_df = df[train_mask], df[test_mask]
        
        if len(test_df) < 4:
            continue

        X_train, y_train, feature_cols = get_features_targets(train_df)
        X_test, y_test, _ = get_features_targets(test_df)

        # Handle missing values (impute NaNs)
        X_train = X_train.fillna(0)
        X_test = X_test.fillna(0)

        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Logistic Regression Model
        model = LogisticRegression(
            multi_class="multinomial",
            solver="lbfgs",
            C=1.0,
            max_iter=1000,
            random_state=42,
        )
        model.fit(X_train_scaled, y_train)

        # Save trained model and scaler (for World Cup 2026 predictions)
        from joblib import dump
        models_dir = Path(__file__).resolve().parents[1] / "results" / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        dump(model, models_dir / "logreg_model.joblib")
        dump(scaler, models_dir / "scaler.joblib")

        # Get probabilities and predictions
        probs = model.predict_proba(X_test_scaled)
        prob_df = pd.DataFrame(probs, columns=[stage_map[i] for i in model.classes_])
        temp = pd.concat([test_df.reset_index(drop=True), prob_df], axis=1)

        # Compute tournament-wide ranking 
        temp["tournament_success_score"] = (
            temp.get("Champion", 0) * 6
            + temp.get("Runner-up", 0) * 5
            + temp.get("Semi-Finalist", 0) * 4
            + temp.get("Quarter-Finalist", 0) * 3
            + temp.get("Round of 16", 0) * 2
            + temp.get("Group Stage", 0) * 1
        )
        temp = temp.sort_values("tournament_success_score", ascending=False).reset_index(drop=True)
        total_teams = len(temp)

        # Determine actual tournament structure
        real_stage_counts = test_df["stage_order"].map(stage_map).value_counts().to_dict()
        real_stage_counts["Champion"] = 1
        real_stage_counts["Runner-up"] = 1
        valid_stages = list(real_stage_counts.keys())

        # --- Assign predicted stages according to real structure ---
        temp["predicted_stage_label"] = "Group Stage"
        assigned = 0
        for stage in ["Champion", "Runner-up", "Semi-Finalist", "Quarter-Finalist", "Round of 16"]:
            if stage not in valid_stages:
                continue
            n = real_stage_counts[stage]
            end = min(assigned + n, total_teams)
            temp.loc[assigned:end - 1, "predicted_stage_label"] = stage
            assigned = end

        temp["true_stage_label"] = temp["stage_order"].map(stage_map)
        temp["predicted_stage_order"] = temp["predicted_stage_label"].map({v: k for k, v in stage_map.items()})

        # Collapse both into 3 macro categories
        temp["true_macro_label"] = temp["true_stage_label"].apply(collapse_stage)
        temp["pred_macro_label"] = temp["predicted_stage_label"].apply(collapse_stage)

        # Metrics 
        acc_total = (temp["predicted_stage_label"] == temp["true_stage_label"]).mean()
        mae_stage = np.mean(np.abs(temp["predicted_stage_order"] - temp["stage_order"]))
        acc_macro3 = accuracy_score(temp["true_macro_label"], temp["pred_macro_label"])
        
        actual_champion = temp.loc[temp["true_stage_label"] == "Champion", "team"].values
        predicted_champion = temp.loc[temp["predicted_stage_label"] == "Champion", "team"].values
        actual_champion = actual_champion[0] if len(actual_champion) else "N/A"
        predicted_champion = predicted_champion[0] if len(predicted_champion) else "N/A"
        champion_correct = actual_champion == predicted_champion
        
        print(f"\n{t_name} {t_year} ({total_teams} teams)")
        print(f" → Predicted Champion: {predicted_champion}")
        print(f" → Actual Champion: {actual_champion} {'✅' if champion_correct else '❌'}")
        print(f" → Stage Accuracy: {acc_total:.3f}")
        print(f" → Stage MAE: {mae_stage:.3f}")
        print(f" → Macro-3 Accuracy: {acc_macro3:.3f}")
        
        print("\n Top 8 predicted teams:")
        leaderboard = temp[["team", "predicted_stage_label", "pred_macro_label", "tournament_success_score"]].head(8)
        print(leaderboard.to_string(index=False))

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

    predictions_df = pd.concat(all_predictions, ignore_index=True)
    results_df = pd.DataFrame(results)
    return results_df, predictions_df


# RUNNING COD COMMAND

if __name__ == "__main__":
    print("\n========== MODEL TRAINING: LOGISTIC REGRESSION (LOTO) ==========\n")
    
    processed_dir = Path(__file__).resolve().parents[1] / "data" / "processed"
    features_path = processed_dir / "final_features.csv"
    
    # Load data
    try:
        df = load_data(features_path)
    except FileNotFoundError:
        print(f"Error: Features file not found at {features_path}")
        print("Please ensure the data processing step has been run.")
        exit(1)
        
    # Run LOTO evaluation
    results_df, predictions_df = loto_evaluation(df)

    # Save results
    results_path = processed_dir / "loto_results_logreg.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n✅ Tournament-level results saved to {results_path}")
    
    predictions_path = processed_dir / "loto_predictions_logreg.csv"
    predictions_df.to_csv(predictions_path, index=False)
    print(f"✅ Full team-level predictions saved to {predictions_path}")
    
    print("\n========== OVERALL SUMMARY ==========")
    print(results_df.describe()[["accuracy_stage", "mae_stage", "accuracy_macro3"]].T)
    print("\n✅ Logistic Regression LOTO completed successfully.\n")