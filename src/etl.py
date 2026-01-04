import pandas as pd
import typing
 
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

def discretize_dataframe(df: pd.DataFrame, columns: list[str], n_bins: int=5):
    df = df.copy()
    for col in columns:
        df[col] = pd.qcut(
            df[col],
            q=n_bins,
            duplicates="drop"
        ).astype(str)
    return df

def node_to_dict(node):
    if node is None:
        return None

    # Leaf node
    if node.childs is None and node.next is None:
        return {
            "type": "leaf",
            "value": node.value
        }

    # Branch node
    if node.childs is None and node.next is not None:
        return {
            "type": "branch",
            "value": node.value,
            "next": node_to_dict(node.next)
        }
    
    # Decision node (feature -> Children)
    return {
        "type": "decision",
        "feature": node.value,
        "children": [
            {
                "value": child.value,
                "next": node_to_dict(child.next)
            }
            for child in node.childs
        ]
    }

def dict_to_node(d: dict):
    if d["type"]== "leaf":
        n= Node()
        n.value= d["value"]
        return n 

    if d["type"]== "branch":
        n= Node()
        n.value= d["value"]
        n.next= dict_to_node(d["next"])
        return n 

    n= Node()
    n.value= d["feature"]
    n.childs= []
    for child in d["children"]:
        c= Node()
        c.value= child["value"]
        c.next= dict_to_node(child["next"])
        n.childs.append(c)
    
    return n



