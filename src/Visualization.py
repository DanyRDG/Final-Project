"""
Visualization.py

Generates 7 visualizations:
HISTORICAL MODEL EVALUATION (4 plots):
1. Detailed Confusion Matrix - All tournament stages
2. Aggregated Confusion Matrix - Deep Run/Knockouts/Group Stage categories
3. Feature Importance - Most predictive features
4. Correlation Matrix - Feature relationships

TOURNAMENT PERFORMANCE (1 plot):
5. Performance by Tournament Type - Grouped bar chart

2026 PREDICTIONS (2 plots):
6. Top 10 Champion Probabilities
7. Top 10 Tournament Success Scores

All saved in results/visualizations/
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix


# 1. CONFIGURATION 

def get_feature_columns():
    """Returns the 11 features used by the logistic regression model"""
    return [
        "pre_tournament_elo", "n_matches_form", "win_rate", "avg_goal_diff",
        "goals_for_per_match", "goals_against_per_match", "avg_opponent_elo",
        "avg_group_elo", "max_group_elo", "elo_minus_avg_group", "group_elo_rank"
    ]


def map_to_aggregated_category(stage):
    """
    Groups tournament stages into 3 broader categories:
    - Deep Run: Champion, Runner-up, Semi-Finalist
    - Knockouts: Quarter-Finalist, Round of 16
    - Group Stage: Eliminated in groups
    """
    stage_lower = str(stage).lower()
    if any(x in stage_lower for x in ['champion', 'runner-up', 'semi']):
        return "Deep Run"
    elif any(x in stage_lower for x in ['quarter', 'round of 16']):
        return "Knockouts"
    return "Group Stage"


# 1. DETAILED CONFUSION MATRIX 

def plot_confusion_matrix_detailed(results_dir, processed_dir):
    """Shows prediction accuracy across all 6 tournament stages"""
    print("\n Generating Detailed Confusion Matrix...")
    
    # Load predictions
    df = pd.read_csv(processed_dir / "loto_predictions_logreg.csv")
    
    # Find actual and predicted columns
    actual_col = 'true_stage_label' if 'true_stage_label' in df.columns else [c for c in df.columns if 'true' in c.lower()][0]
    pred_col = 'predicted_stage_label' if 'predicted_stage_label' in df.columns else [c for c in df.columns if 'pred' in c.lower()][0]
    
    # Define stage order (best to worst performance)
    stage_order = ["Champion", "Runner-up", "Semi-Finalist", "Quarter-Finalist", "Round of 16", "Group Stage"]
    labels = [s for s in stage_order if s in df[actual_col].values]
    
    # Create confusion matrix
    cm = confusion_matrix(df[actual_col], df[pred_col], labels=labels)
    
    # Configuration
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels,
                cbar_kws={"label": "Number of Teams"}, linewidths=0.5, linecolor='gray')
    plt.title("Confusion Matrix (Detailed Stages) - Logistic Regression\n", fontsize=16, fontweight='bold')
    plt.xlabel("Predicted Stage", fontsize=12, fontweight='bold')
    plt.ylabel("Actual Stage", fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(results_dir / "confusion_matrix_detailed_logreg.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: confusion_matrix_detailed_logreg.png")


# 3. AGGREGATED CONFUSION MATRIX (Macro-3)

def plot_confusion_matrix_aggregated(results_dir, processed_dir):
    """Shows prediction accuracy for Deep Run/Knockouts/Group Stage"""
    print("\n Generating Aggregated Confusion Matrix...")
    
    # Load and map stages to categories
    df = pd.read_csv(processed_dir / "loto_predictions_logreg.csv")
    actual_col = 'true_stage_label' if 'true_stage_label' in df.columns else [c for c in df.columns if 'true' in c.lower()][0]
    pred_col = 'predicted_stage_label' if 'predicted_stage_label' in df.columns else [c for c in df.columns if 'pred' in c.lower()][0]
    
    y_true = df[actual_col].apply(map_to_aggregated_category)
    y_pred = df[pred_col].apply(map_to_aggregated_category)
    
    # Create confusion matrix
    categories = ["Deep Run", "Knockouts", "Group Stage"]
    cm = confusion_matrix(y_true, y_pred, labels=categories)
    
    # Configiration
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=categories, yticklabels=categories,
                cbar_kws={"label": "Number of Teams"}, linewidths=0.5, linecolor='gray')
    plt.title("Confusion Matrix (Aggregated) - Logistic Regression\n", fontsize=16, fontweight='bold')
    plt.xlabel("Predicted Category", fontsize=12, fontweight='bold')
    plt.ylabel("Actual Category", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "confusion_matrix_aggregated_logreg.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: confusion_matrix_aggregated_logreg.png")


# 4. FEATURE IMPORTANCE

def plot_feature_importance(results_dir, processed_dir):
    """Shows which features most influence predictions (based on coefficient magnitude)"""
    print("\n Generating Feature Importance Plot...")
    
    # Find and load model
    project_root = processed_dir.parent.parent
    model_paths = [
        project_root / "results" / "models" / "logreg_model.joblib",
        project_root / "models" / "logreg_model.joblib"
    ]
    model_path = next((p for p in model_paths if p.exists()), None)
    model = joblib.load(model_path)
    
    # Calculate feature importance 
    feature_names = get_feature_columns()
    importance = np.abs(model.coef_).mean(axis=0)
    
    # Sort and plot
    df_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importance}).sort_values('Importance', ascending=True)
    
    plt.figure(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(df_importance)))
    plt.barh(df_importance['Feature'], df_importance['Importance'], color=colors)
    plt.xlabel('Average Absolute Coefficient', fontsize=12, fontweight='bold')
    plt.ylabel('Feature', fontsize=12, fontweight='bold')
    plt.title('Feature Importance - Logistic Regression\n', fontsize=16, fontweight='bold')
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(results_dir / "feature_importance_logreg.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: feature_importance_logreg.png")


# 5. CORRELATION MATRIX

def plot_correlation_matrix(results_dir, processed_dir):
    """Shows relationships between features (helps identify multicollinearity)"""
    print("\n Generating Correlation Matrix...")
    
    # Load features and calculate correlation
    df = pd.read_csv(processed_dir / "final_features.csv")
    X = df[get_feature_columns()].fillna(0)
    corr = X.corr()
    
    # Plot 
    plt.figure(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True,
                linewidths=0.5, cbar_kws={"shrink": 0.8, "label": "Correlation"},
                vmin=-1, vmax=1, mask=mask)
    plt.title("Feature Correlation Matrix\n", fontsize=16, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(results_dir / "correlation_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: correlation_matrix.png")


# 6. TOURNAMENT PERFORMANCE COMPARISON

def plot_tournament_performance(results_dir, processed_dir):
    """Compares model performance across different tournament types (World Cup, Euro, Copa, etc.)"""
    print("\n Generating Tournament Performance Chart...")
    
    # Load LOTO results
    df = pd.read_csv(processed_dir / "loto_results_logreg.csv")
    
    # Group by tournament and calculate mean metrics
    summary = df.groupby("tournament").agg({
        "accuracy_stage": "mean",
        "accuracy_macro3": "mean",
        "mae_stage": "mean"
    }).round(3).reset_index()
    
    summary = summary.sort_values("accuracy_stage", ascending=False)
    
    # Prepare data for grouped bar chart
    tournaments = summary["tournament"].str.title().tolist()
    x = np.arange(len(tournaments))
    width = 0.25
    
    # Plot three metrics side-by-side
    fig, ax = plt.subplots(figsize=(14, 7))
    bars1 = ax.bar(x - width, summary["accuracy_stage"], width, label='Stage Accuracy', color='#3498db', edgecolor='black')
    bars2 = ax.bar(x, summary["accuracy_macro3"], width, label='Macro-3 Accuracy', color='#2ecc71', edgecolor='black')
    bars3 = ax.bar(x + width, summary["mae_stage"], width, label='MAE (Lower is Better)', color='#e74c3c', edgecolor='black')
    
    # Add value labels on top of bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel("Tournament Type", fontsize=13, fontweight='bold')
    ax.set_ylabel("Metric Value", fontsize=13, fontweight='bold')
    ax.set_title("Logistic Regression Performance by Tournament Type", fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(tournaments, rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, max(summary["accuracy_stage"].max(), summary["accuracy_macro3"].max(), summary["mae_stage"].max()) * 1.12)
    plt.tight_layout()
    plt.savefig(results_dir / "tournament_performance_logreg.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: tournament_performance_logreg.png")


# 7. TOP 10 CHAMPION PROBABILITIES World Cup 2026

def plot_top_champions_2026(results_dir, predictions_path):
    """Shows the 10 teams most likely to win the 2026 World Cup"""
    print("\n Generating Top 10 Champion Probabilities (2026)...")
    
    # Load predictions
    df = pd.read_csv(predictions_path)
    top10 = df.nlargest(10, 'Champion')[['team', 'Champion', 'group']].sort_values('Champion', ascending=True)
    
    # Plot horizontal bars
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(top10)))
    ax.barh(range(len(top10)), top10['Champion'], color=colors, edgecolor='black', linewidth=0.5)
    
    # Add team labels and percentages
    team_labels = [f"{row['team']} (Group {row['group']})" for _, row in top10.iterrows()]
    ax.set_yticks(range(len(top10)))
    ax.set_yticklabels(team_labels, fontsize=11)
    
    for i, (_, row) in enumerate(top10.iterrows()):
        ax.text(row['Champion'] + 0.005, i, f"{row['Champion']*100:.1f}%", 
                va='center', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Champion Probability', fontsize=13, fontweight='bold')
    ax.set_ylabel('Team', fontsize=13, fontweight='bold')
    ax.set_title('Top 10 Teams by Champion Probability - 2026 World Cup\n', fontsize=16, fontweight='bold')
    ax.set_xlim(0, max(top10['Champion']) * 1.15)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(results_dir / "top10_champion_probability_2026.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: top10_champion_probability_2026.png")


# 8. TOP 10 TOURNAMENT SUCCESS SCORES World Cup 2026

def plot_top_success_2026(results_dir, predictions_path):
    """Shows the 10 teams with highest overall tournament success prediction"""
    print("\n Generating Top 10 Tournament Success Scores (2026)...")
    
    # Load predictions
    df = pd.read_csv(predictions_path)
    top10 = df.nlargest(10, 'tournament_success_score')[['team', 'tournament_success_score', 'group']].sort_values('tournament_success_score', ascending=True)
    
    # Plot horizontal bars
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.Greens(np.linspace(0.4, 0.9, len(top10)))
    ax.barh(range(len(top10)), top10['tournament_success_score'], color=colors, edgecolor='black', linewidth=0.5)
    
    # Add team labels and scores
    team_labels = [f"{row['team']} (Group {row['group']})" for _, row in top10.iterrows()]
    ax.set_yticks(range(len(top10)))
    ax.set_yticklabels(team_labels, fontsize=11)
    
    for i, (_, row) in enumerate(top10.iterrows()):
        ax.text(row['tournament_success_score'] + 0.03, i, f"{row['tournament_success_score']:.2f}",
                va='center', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Tournament Success Score', fontsize=13, fontweight='bold')
    ax.set_ylabel('Team', fontsize=13, fontweight='bold')
    ax.set_title('Top 10 Teams by Success Score - 2026 World Cup\n', fontsize=16, fontweight='bold')
    ax.set_xlim(0, max(top10['tournament_success_score']) * 1.15)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add explanation note
    ax.text(0.98, 0.02, 'Weights: Champion(6) > Runner-up(5) > Semi(4) > Quarter(3) > R16(2) > Group(1)',
            transform=ax.transAxes, fontsize=8, ha='right', style='italic',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(results_dir / "top10_tournament_success_2026.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: top10_tournament_success_2026.png")


# 9. RUNNING CODE COMMAND 

def generate_all_visualizations():
    """
    Generates all 7 visualizations for the Logistic Regression model:
    - 4 historical evaluation plots
    - 1 tournament comparison plot
    - 2 2026 prediction plots
    """
    print("\n" + "="*70)
    print("  GENERATING ALL VISUALIZATIONS - LOGISTIC REGRESSION")
    print("="*70)
    
    # Setup paths
    project_root = Path(__file__).resolve().parents[1]
    results_dir = project_root / "results" / "visualizations"
    processed_dir = project_root / "data" / "processed"
    predictions_2026 = project_root / "results" / "predictions" / "predictions_worldcup2026.csv"
    
    results_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Historical evaluation plots
        plot_confusion_matrix_detailed(results_dir, processed_dir)
        plot_confusion_matrix_aggregated(results_dir, processed_dir)
        plot_feature_importance(results_dir, processed_dir)
        plot_correlation_matrix(results_dir, processed_dir)
        
        # Tournament performance comparison
        plot_tournament_performance(results_dir, processed_dir)
        
        # 2026 predictions (if available)
        if predictions_2026.exists():
            plot_top_champions_2026(results_dir, predictions_2026)
            plot_top_success_2026(results_dir, predictions_2026)
        else:
            print(f"\n Skipping 2026 predictions (file not found)")
        
        print("\n" + "="*70)
        print("  ✅ ALL VISUALIZATIONS GENERATED SUCCESSFULLY")
        print("="*70)
        print(f" Location: {results_dir}")
        print("\n Generated files:")
        print("   1. confusion_matrix_detailed_logreg.png")
        print("   2. confusion_matrix_aggregated_logreg.png")
        print("   3. feature_importance_logreg.png")
        print("   4. correlation_matrix.png")
        print("   5. tournament_performance_logreg.png")
        if predictions_2026.exists():
            print("   6. top10_champion_probability_2026.png")
            print("   7. top10_tournament_success_2026.png")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        raise


if __name__ == "__main__":
    generate_all_visualizations()