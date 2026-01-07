# 1.0 Generic Modules
import pandas as pd 
import numpy as np 
import math
import os 
import json
# 1.1 My Modules
from etl import node_to_dict
from node import Node

# 2. Constants
FEATURE_COLS= ["age","gender","ethnicity","education_level","income_level","employment_status","smoking_status","alcohol_consumption_per_week","physical_activity_minutes_per_week","diet_score","sleep_hours_per_day","screen_time_hours_per_day","family_history_diabetes","hypertension_history","cardiovascular_history","bmi","waist_to_hip_ratio","systolic_bp","diastolic_bp","heart_rate","cholesterol_total","hdl_cholesterol","ldl_cholesterol","triglycerides","glucose_fasting","glucose_postprandial","insulin_level","hba1c"]
FEATURE_DICT= dict(enumerate(FEATURE_COLS))
LABEL_COLS= ["diabetes_stage"]

# 3.0 Pure Decision Tree Class
class DecisionTree():

    # 3.1 Constructor
    def __init__(self, database: pd.DataFrame):
        # Information
        self.data= np.array(database[FEATURE_COLS].copy()) # Where the features are
        self.results= np.array(database[LABEL_COLS].copy()) # Where the labels are
        self.length= np.shape(self.results)[0] # Initial Length
        self.min_samples= 10 # Minimum Instances per Child in Pre-Pruning Version
        
        self.node= None # Father Node Variable

    # 4.0 Entropy Auxiliar Function
    def _calculate_entropy(self, indices: list= None, value: str= None) -> float:
        if indices: # Filter by Indices
            newdata= self.data[indices]
            newresults= self.results[indices]
        else:
            # Basecase (if no indices)
            newdata= self.data
            newresults= self.results

        if value: # Filter by Value
            indices_value= np.where(np.any(newdata== value, axis=1))[0] # Get the (relative) indices that pass the condition
            
            newdata= newdata[indices_value]
            newresults= newresults[indices_value]

        newlength= np.shape(newresults)[0]

        # We get the patterns
        uniques, counts= np.unique(newresults, return_counts= True)
        newfrequency= dict(zip(uniques, counts))

        return sum([ # Apply the Entropy Formula
            -count/ newlength* math.log(count/ newlength, 2)
            if count else 0
            for count in newfrequency.values()
        ])

    # 4.1 Information Gain Auxiliar Function
    def _info_gain(self, indices: list[int], feature: int) -> float:
        initial_entropy= self._calculate_entropy(indices) # Calculate initial entropy with the given indices (total)
        data= self.data[indices] # We filter by the indices

        # Get Patterns
        uniques, counts= np.unique(data[:, feature], return_counts=True)
        feature_frequencies= dict(zip(uniques, counts))

        return initial_entropy - sum([  # Apply the Info Gain Formula
            value_freq/ self.length* self._calculate_entropy(indices, value)
            for value, value_freq in feature_frequencies.items()
        ])

    # 4.2 Best Information Gain Auxiliar Function
    def _get_max_info(self, indices: list[int], features: list[int]) -> tuple[str, int]:
        # We calculate the gain-per-feature
        gain_per_feature= [self._info_gain(indices, feature_col) for feature_col in features]

        # We get which is the maximum
        max_feature = gain_per_feature.index(max(gain_per_feature))
        best_col= features[max_feature]

        return (FEATURE_DICT[best_col], best_col)

    # 5.0 Training of the Decision Tree
    def _training(self, indices: list[int], features: list[int], node: Node) -> Node:
        # Create indices if not passed as an argument
        indices= [i for i in range(len(self.results))] if not indices else indices
        # Initialize node if not instanced
        node= Node() if node==None else ...

        # Filter our data scope
        newresults= self.results[indices]

        # If all results are the same -> Pure Node
        values, counts= np.unique(newresults, return_counts= True)
        if len(values) == 1:
            node.value = values[0]
            return node
        
        # If no more features to evaluate -> Return most likely
        if len(features) == 0:
            values, counts = np.unique(newresults, return_counts=True)
            node.value = values[np.argmax(counts)]
            return node

        # We choose the best feature
        best_name, best_col= self._get_max_info(indices, features)
        node.value= best_name

        # We get the unique values of that feature
        feature_values = set(
            self.data[i][best_col]
            for i in indices
            )

        node.childs= []
        # Create a Child per different value
        for value in feature_values:
            child = Node()
            child.value = value  # add a branch from the node to each feature value in our feature
            node.childs.append(child)  # append new child node to current node
            
            # We give the child the indices only with its value
            child_indices = [
                i 
                for i in indices 
                if self.data[i][best_col] == value
                ]

            # We create a "Normal Decision Node"
            # Remove the chosen feature
            myfeatures= features.copy()
            to_remove = myfeatures.index(best_col)
            myfeatures.pop(to_remove)
            # recursively call the algorithm to train it
            child.next = self._training(child_indices, myfeatures, child.next)
        return node
    
    # 6.0 The "Main" function
    def run(self):
        # We train it
        self.node = self._training(None, list(FEATURE_DICT), self.node)
        # We make it a Dictionary
        tree_dict= node_to_dict(self.node)
        # Convert it to JSON
        dirname = os.path.dirname(__file__)
        json_path = os.path.join(dirname, "../results/decision_tree.json")

        with open(json_path, "w") as f:
            json.dump(tree_dict, f, indent=4)

# 7.0 Pre-Pruning Decision Tree Class
class DecisionTreePruning(DecisionTree):
    
    #8.0 Adapted Version of the Decision Tree Training
    def _training(self, indices: list[int], features: list[int], node: Node) -> Node:
        # Create indices if not passed as an argument
        indices= [i for i in range(len(self.results))] if not indices else indices
        # Initialize node if not instanced
        node= Node() if node==None else ...

        # Filter our data scope
        newresults= self.results[indices]
        
        # If all results are the same -> Pure Node
        values, counts= np.unique(newresults, return_counts= True)
        if len(values) == 1:
            node.value = values[0]
            return node

        # If no more features to evaluate -> Return most likely
        if len(features) == 0:
            values, counts = np.unique(newresults, return_counts=True)
            node.value = values[np.argmax(counts)]
            return node

        # We choose the best feature
        best_name, best_col= self._get_max_info(indices, features)
        node.value= best_name

        # We get the unique values of that feature
        feature_values = set(
            self.data[i][best_col]
            for i in indices
            )

        node.childs= []
        # Create a Child per different value
        for value in feature_values:
            child = Node()
            child.value = value  # add a branch from the node to each feature value in our feature
            node.childs.append(child)  # append new child node to current node
            
            # We give the child the indices only with its value
            child_indices = [
                i 
                for i in indices 
                if self.data[i][best_col] == value
                ]

            # If there are too few examples, Convert into leaf Node
            if len(child_indices) < self.min_samples:
                values, counts = np.unique(self.results[child_indices], return_counts=True)
                child.next = Node()
                child.next.value = values[np.argmax(counts)]

                return node
            
            else:
                # We create a "Normal Decision Node"
                # Remove the chosen feature
                myfeatures= features.copy()
                to_remove = myfeatures.index(best_col)
                myfeatures.pop(to_remove)
                # recursively call the algorithm to train it
                child.next = self._training(child_indices, myfeatures, child.next)

                return node
            
       




                 