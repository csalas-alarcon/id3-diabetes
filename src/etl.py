# 1.0 Generic Modules
import pandas as pd
import os
from math import trunc
# 1.1 My Modules
from node import Node

from sklearn.preprocessing import KBinsDiscretizer

# 2.0 Constants
FEATURE_COLS= ["age","gender","ethnicity","education_level","income_level","employment_status","smoking_status","alcohol_consumption_per_week","physical_activity_minutes_per_week","diet_score","sleep_hours_per_day","screen_time_hours_per_day","family_history_diabetes","hypertension_history","cardiovascular_history","bmi","waist_to_hip_ratio","systolic_bp","diastolic_bp","heart_rate","cholesterol_total","hdl_cholesterol","ldl_cholesterol","triglycerides","glucose_fasting","glucose_postprandial","insulin_level","hba1c"]
CONTINUOUS_COLS = ["age", "alcohol_consumption_per_week", "physical_activity_minutes_per_week", "diet_score", "sleep_hours_per_day", "screen_time_hours_per_day", "bmi", "waist_to_hip_ratio", "systolic_bp", "diastolic_bp", "heart_rate", "cholesterol_total", "hdl_cholesterol", "ldl_cholesterol", "triglycerides", "glucose_fasting", "glucose_postprandial", "insulin_level", "hba1c"]
SIZE= 100000

'''
# 3.0 Normalization Auxiliar Function
def normalize(value, min_val, max_val) -> float:
    return (value - min_val) / (max_val - min_val)

# 3.1 Categorization of Continious Features Auxiliar Function
def categorize(value, min_val, max_val):
    # We normalize each value
    norm = normalize(value, min_val, max_val)
    # Avoid 0 and 1
    norm = min(max(norm, 0), 0.9999)
    # We assign it a category
    cat_index = trunc(norm * 10)
    return cat_index 

'''

# 3.2 Loading Function
def load(n_rows: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Read DB from Relative PATH
    dirname = os.path.dirname(__file__)
    file = os.path.join(dirname, '../data/diabetes.csv')
    df= pd.read_csv(file)

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # We categorize continous values
    discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='kmeans')

    
    # 2. Fit and Transform the continuous columns
    # We do this before splitting to ensure training and validation use the same scales
    df[CONTINUOUS_COLS] = discretizer.fit_transform(df[CONTINUOUS_COLS])

    # We separate between training and validation
    training= df.iloc[:n_rows]
    validation= df.iloc[n_rows:]

    return (training, validation)

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

