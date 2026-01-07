import pandas as pd
import os

from node import Node
 
FEATURE_COLS= ["age","gender","ethnicity","education_level","income_level","employment_status","smoking_status","alcohol_consumption_per_week","physical_activity_minutes_per_week","diet_score","sleep_hours_per_day","screen_time_hours_per_day","family_history_diabetes","hypertension_history","cardiovascular_history","bmi","waist_to_hip_ratio","systolic_bp","diastolic_bp","heart_rate","cholesterol_total","hdl_cholesterol","ldl_cholesterol","triglycerides","glucose_fasting","glucose_postprandial","insulin_level","hba1c"]


CONTINUOUS_COLS = [
    "age",
    "alcohol_consumption_per_week",
    "physical_activity_minutes_per_week",
    "diet_score",
    "sleep_hours_per_day",
    "screen_time_hours_per_day",
    "bmi",
    "waist_to_hip_ratio",
    "systolic_bp",
    "diastolic_bp",
    "heart_rate",
    "cholesterol_total",
    "hdl_cholesterol",
    "ldl_cholesterol",
    "triglycerides",
    "glucose_fasting",
    "glucose_postprandial",
    "insulin_level",
    "hba1c"
]

SIZE= 100000

# Load the model
def load(n_rows: int) -> tuple[pd.DataFrame, pd.DataFrame]:

    dirname = os.path.dirname(__file__)
    file = os.path.join(dirname, '../data/diabetes.csv')

    df= pd.read_csv(file)

    for col in CONTINUOUS_COLS:
        min_val = df[col].min()
        max_val = df[col].max()
        df[col] = df[col].apply(lambda x: categorize(x, min_val, max_val))

    training= df.iloc[:n_rows]
    validation= df.iloc[n_rows:]

    return (training, validation)

def normalize(value, min_val, max_val) -> float:
    return (value - min_val) / (max_val - min_val)

def categorize(value, min_val, max_val):
    norm = normalize(value, min_val, max_val)
    # Clip to avoid rounding issues
    norm = min(max(norm, 0), 0.9999)
    bin_index = int(norm * 10)
    return bin_index  # returns 0..9

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
    
    else:
        print("GRAVÍSIMO ERROR; NO ENTRO EN NINGUNA CATEGORIA NODE TO DICT")

def dict_to_node(d: dict) -> Node:
    # instance a Node
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

    else:
        print("GRAVÍSIMO ERROR; NO ENTRO EN NINGUNA CATEGORIA DICTO_TO_NODE")


