import os
import json
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.feature_selection import SelectKBest, chi2

# The full list from your CSV
FEATURE_COLS = ["age","gender","ethnicity","education_level","income_level","employment_status","smoking_status","alcohol_consumption_per_week","physical_activity_minutes_per_week","diet_score","sleep_hours_per_day","screen_time_hours_per_day","family_history_diabetes","hypertension_history","cardiovascular_history","bmi","waist_to_hip_ratio","systolic_bp","diastolic_bp","heart_rate","cholesterol_total","hdl_cholesterol","ldl_cholesterol","triglycerides","glucose_fasting","glucose_postprandial","insulin_level","hba1c"]
LABEL_COL = "diabetes_stage"

def analyze_and_save(k=10):
    dirname = os.path.dirname(__file__)
    file_path = os.path.join(dirname, '../data/diabetes.csv')
    df = pd.read_csv(file_path)

    # Encode strings to numbers for math
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('category').cat.codes

    # Binning logic (matching your tree setup)
    continuous = [c for c in FEATURE_COLS if df[c].nunique() > 2]
    kbd = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='kmeans')
    df[continuous] = kbd.fit_transform(df[continuous])

    # Chi-Square
    selector = SelectKBest(score_func=chi2, k=k)
    selector.fit(df[FEATURE_COLS], df[LABEL_COL])
    
    # Get names
    cols_idx = selector.get_support(indices=True)
    best_features = [FEATURE_COLS[i] for i in cols_idx]

    # SAVE TO JSON
    output_path = os.path.join(dirname, '../results/selected_features.json')
    with open(output_path, 'w') as f:
        json.dump(best_features, f, indent=4)
    
    print(f"Success! {k} features saved to /results/selected_features.json")
    print(f"Top 3: {best_features[:3]}")

if __name__ == "__main__":
    analyze_and_save()