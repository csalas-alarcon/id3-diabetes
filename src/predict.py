import json
import os
import pandas as pd

def predict_patient():
    dirname = os.path.dirname(__file__)
    
    # 1. Load your hard-earned logic
    with open(os.path.join(dirname, '../results/selected_features.json'), 'r') as f:
        features = json.load(f)
    with open(os.path.join(dirname, '../results/bin_mapping.json'), 'r') as f:
        bin_mapping = json.load(f)
    with open(os.path.join(dirname, '../results/decision_tree.json'), 'r') as f:
        tree = json.load(f)

    print("\n--- Diabetes Prediction Tool (89% Accuracy) ---")
    print(f"Please enter the following {len(features)} values:")
    
    # 2. Collect Inputs
    patient_data = {}
    for feat in features:
        val = float(input(f"Value for {feat.replace('_', ' ').title()}: "))
        
        # Apply the exact same Analytical Bins from your training
        bins = bin_mapping[feat]
        # Find which bin the value falls into (0, 1, 2...)
        patient_data[feat] = int(pd.cut([val], bins=bins, labels=False, include_lowest=True)[0])

    # 3. Traverse the Tree
    # This logic follows the branches of your JSON
    def walk_tree(node, data):
        # If the node has childs, we keep going down
        if "childs" in node and node["childs"]:
            feat_index = features.index(features[tree.index(node)]) # Current level feature
            val = data[features[feat_index]]
            
            # Find the branch that matches our patient's category
            for child in node["childs"]:
                if child["value"] == val:
                    return walk_tree(child, data)
            
            # If no branch matches, use the "Next" (Last Resort)
            if "next" in node and node["next"]:
                return walk_tree(node["next"], data)
        
        # If it's a leaf, the "value" is the Diagnosis
        return node["value"]

    # 4. Result
    # Classes map to your labels: 0: No, 1: Pre, 2: Type 1, 3: Type 2, 4: Gestational
    labels = ["No Diabetes", "Pre-Diabetes", "Type 1", "Type 2", "Gestational"]
    result_idx = walk_tree(tree, patient_data)
    
    print("\n" + "="*30)
    print(f"DIAGNOSIS: {labels[int(result_idx)]}")
    print("="*30)

if __name__ == "__main__":
    predict_patient()