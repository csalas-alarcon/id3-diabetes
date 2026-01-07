# 1.0 Generic Modules
import pandas as pd
import random
import os
import json 
from sklearn.metrics import classification_report
# 1.1 My Modules
from etl import dict_to_node
from node import Node

# 2.0 Constants
LABEL_MAP = {
    "No Diabetes": 0,
    "Pre-Diabetes": 1,
    "Type 1": 2,
    "Type 2": 3,
    "Gestational": 4
}

# 3.0 Inference Engine Class
class Engine():
    # 3.1 Constructor
    def __init__(self, data: pd.DataFrame):
        self.data= data 
        self.pnode= self._get_model()

    # 3.2 Auxiliar Function to the Constructor
    def _get_model(self,) -> Node:
        # We get the Relative PATH and Open It
        dirname = os.path.dirname(__file__)
        model_path = os.path.join(dirname, '../results/decision_tree.json')

        with open(model_path) as f:
            tree_dict= json.load(f)
        # We convert the json to Objects
        pnode= dict_to_node(tree_dict)

        return pnode

    # 4.0 Tree Traversal Auxiliar Function
    def _traverse(self, data: pd.Series, tree: Node) -> float:
        # If it only has value -> Leaf
        if tree.childs is None and tree.next is None:
            return tree.value
        
        # If it does not have next -> Decision
        if tree.childs is not None:
            try:
                for child in tree.childs:
                    if str(child.value)==str(data[tree.value]):
                        return self._traverse(data, child)
                    
            # If the tree hasn't been exposed to that value, we choose a random value
            except Exception: 
                return self._traverse(data, random.choice(tree.childs))

        # If it doesn't have childs -> Branch
        if tree.next is not None:
            return self._traverse(data, tree.next)

        # As a Last Resort
        else:
            return "No Diabetes"

    # 4.1 Result generator Auxiliar Function
    def _get_results(self) -> list[float]:
        results: list[float]= []
        for _, row in self.data.iterrows():
            res= self._traverse(row, self.pnode)
            results.append(res)
        
        return results

    # 5.0 Metric Calculator Function
    def _validation(self, results: list[str]) -> str:
        # Get the Results
        labels = self.data["diabetes_stage"].tolist()

        # Parse them into ints
        y_true= [LABEL_MAP[label] for label in labels]
        y_pred= [LABEL_MAP[result] for result in results]

        # Generate the Report
        return classification_report(
        y_true,
        y_pred,
        target_names=LABEL_MAP.keys(), 
        output_dict=False 
    )

    # 6.0 The Main Function
    def run(self) -> str:
        prediction= self._get_results()
        report= self._validation(prediction)

        return report

