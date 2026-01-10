# 1.0 Generic Modules
import pandas as pd
import os
import json
from math import trunc
# 1.1 My Modules
from node import Node

# 2.0 Constants
FEATURE_COLS= ["age","gender","ethnicity","education_level","income_level","employment_status","smoking_status","alcohol_consumption_per_week","physical_activity_minutes_per_week","diet_score","sleep_hours_per_day","screen_time_hours_per_day","family_history_diabetes","hypertension_history","cardiovascular_history","bmi","waist_to_hip_ratio","systolic_bp","diastolic_bp","heart_rate","cholesterol_total","hdl_cholesterol","ldl_cholesterol","triglycerides","glucose_fasting","glucose_postprandial","insulin_level","hba1c"]
CONTINUOUS_COLS = ["age", "alcohol_consumption_per_week", "physical_activity_minutes_per_week", "diet_score", "sleep_hours_per_day", "screen_time_hours_per_day", "bmi", "waist_to_hip_ratio", "systolic_bp", "diastolic_bp", "heart_rate", "cholesterol_total", "hdl_cholesterol", "ldl_cholesterol", "triglycerides", "glucose_fasting", "glucose_postprandial", "insulin_level", "hba1c"]
SIZE= 100000


# 3.2 Loading Function
def load(n_rows: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Read DB from Relative PATH
    dirname = os.path.dirname(__file__)

    with open(os.path.join(dirname, '../results/selected_features.json'), 'r') as f:
        selected_features= json.load(f)
    with open(os.path.join(dirname, '../results/bin_mapping.json'), 'r') as f:
        bin_mapping= json.load(f)

    df= pd.read_csv(os.path.join(dirname, '../data/diabetes.csv'))

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # We categorize continous values
    for col, boundaries in bin_mapping.items():
        if col in selected_features:
            df[col]= pd.cut(df[col], bins=boundaries, labels=False, include_lowest= True)

    # Handle categoricals
    for col in selected_features:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('category').cat.codes 
    
    df_final = df[selected_features + ["diabetes_stage"]].dropna()

    # We separate between training and validation
    training= df_final.iloc[:n_rows]
    validation= df_final.iloc[n_rows:]

    return (training, validation, selected_features)

# 4.1 Conversion Function to SAVE
def node_to_dict(node: Node) -> dict:
    if node is None:
        return None

    # If only value -> Leaf Node
    if node.childs is None and node.next is None:
        return {
            "type": "leaf",
            "value": node.value
        }
    # If no next -> Decision
    if node.next is None:
        return {
            "type": "decision",
            "column": node.value,
            "children": [
                node_to_dict(child)
                for child in node.childs
            ]
        }
    # If no children -> Branch
    if node.childs is None:
        return {
            "type": "branch",
            "value": node.value,
            "next": node_to_dict(node.next)
        }

# 4.2 Conversion Function to LOAD
def dict_to_node(d: dict) -> Node:
    # Instance a Node
    n= Node()

    if d["type"]== "leaf":
        n.value= d["value"]
        return n 

    if d["type"]== "decision":
        n.value= d["column"]
        n.childs= [dict_to_node(child) for child in d["children"]]
        return n 

    if d["type"]== "branch":
        n.value= d["value"]
        n.next= dict_to_node(d["next"])
        return n 

