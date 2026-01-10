# 0.0 Generic Modules
import os
import json
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2
# 0.1 My Modules
from tree import DecisionTreePruning

# 1. Hardcoded Continuous Features
CONTINUOUS_FEATURES = [
    "age", "physical_activity_minutes_per_week", "bmi", 
    "systolic_bp", "diastolic_bp", "cholesterol_total", 
    "ldl_cholesterol", "glucose_fasting", "glucose_postprandial", 
    "insulin_level", "hba1c"
]

def find_analytical_bins(df, feature_name, max_bins=15):
    temp_df = df[[feature_name, "diabetes_stage"]].dropna().copy()
    
    # 1. Divide by Quantiles
    from sklearn.preprocessing import KBinsDiscretizer
    kbd = KBinsDiscretizer(n_bins=60, encode='ordinal', strategy='quantile')
    temp_df[feature_name] = kbd.fit_transform(temp_df[[feature_name]])
    
    # 2. Use PRUNING tree to find the bins
    tree_finder = DecisionTreePruning(temp_df, [feature_name])
    
    # Set a threshold to delete bad bins
    tree_finder.min_samples = 25 
    
    # 3. Train
    root = tree_finder._training(None, [0], None) 
    
    if root.childs:
        # Collect only the thresholds the PRUNED tree kept
        bin_indices = sorted([child.value for child in root.childs])
        edges = kbd.bin_edges_[0]
        actual_thresholds = [edges[int(i)] for i in bin_indices if i < len(edges)]
        
        return [-float('inf')] + sorted(list(set(actual_thresholds))) + [float('inf')]
    return []

def binding():
    print("--- Starting Analytical Binding ---")
    dirname = os.path.dirname(__file__)
    df = pd.read_csv(os.path.join(dirname, '../data/diabetes.csv'))
    
    # Dictionary to store boundaries for the future
    bin_map = {}

    for col in CONTINUOUS_FEATURES:
        print(f"Finding entropy bins for: {col}")
        bins = find_analytical_bins(df, col)
        if bins:
            bin_map[col] = bins
            # Transform column to category codes 
            df[col] = pd.cut(df[col], bins=bins, labels=False, include_lowest=True)
    
    # Save the Categorized Data
    temp_path = os.path.join(dirname, '../temp/cat_diabetes.csv')
    os.makedirs(os.path.dirname(temp_path), exist_ok=True)
    df.to_csv(temp_path, index=False)
    
    # Save the Map (Just for Traces)
    with open(os.path.join(dirname, '../results/bin_mapping.json'), 'w') as f:
        json.dump(bin_map, f, indent=4)
    
    print(f"Successfully saved categorized data to {temp_path}")

def analyze_and_save(k=5):
    print(f"--- Running Feature Selection (Top {k}) ---")
    dirname = os.path.dirname(__file__)
    # Read the updated data
    df = pd.read_csv(os.path.join(dirname, '../temp/cat_diabetes.csv'))

    # Encode categorical strings (Gender, Ethnicity)
    for col in df.columns:
        if df[col].dtype == 'object' and col != "diabetes_stage":
            df[col] = df[col].astype('category').cat.codes

    # Ensure no NaNs 
    df = df.dropna()

    # Prep for Chi-Square
    # We use all columns except the label
    df = df.drop(columns=[col for col in ["diagnosed_diabetes", "diabetes_risk_score"] if col in df.columns])

    X_cols = [c for c in df.columns if c != "diabetes_stage"]
    X = df[X_cols]
    y = df["diabetes_stage"]

    selector = SelectKBest(score_func=chi2, k=k)
    selector.fit(X, y)

    feature_scores = pd.DataFrame({
        'Feature': X_cols,
        'Score': selector.scores_
    }).sort_values(by='Score', ascending=False)

    print("\n--- FEATURE RANKING BY CHI-SQUARE ---")
    print(feature_scores.head(28)) 
    
    best_features = X.columns[selector.get_support()].tolist()

    # Save best ones
    with open(os.path.join(dirname, '../results/selected_features.json'), 'w') as f:
        json.dump(best_features, f, indent=4)
    
    print(f"Done! Winners: {best_features}")

if __name__ == "__main__":
    binding()
    analyze_and_save(k=5)
