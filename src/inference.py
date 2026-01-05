import pandas as pd
import itertools
import os
import json 
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from etl import dict_to_node
from node import Node

class Engine():

    def __init__(self, data: pd.DataFrame):
        self.data= data 
        self.pnode= self._get_model()

    def _get_model(self,) -> Node:

        dirname = os.path.dirname(__file__)
        model_path = os.path.join(dirname, '../results/decision_tree.json')

        with open(model_path) as f:
            tree_dict= json.load(f)

        pnode= dict_to_node(tree_dict)
        return pnode

    def _traverse(self, data: pd.Series, tree: Node) -> float:
        # Leaf node
        if tree.childs is None and tree.next is None:
            return tree.value

        # Decision Node
        col= tree.value 
        value= data[col]

        for child in tree.childs: # Branch Node
            if child.value== value:
                return self._traverse(data, child.next) # Next decision node

        return tree.value

    def _get_results(self) -> list[float]:
        results: list[float]= []
        for _, row in self.data.iterrows():
            res= self._traverse(row, self.pnode)
            results.append(res)
        
        return results

    def _validation(self, results: list[float]) -> tuple[float, float, int]:
        labels= self.data["diabetes_risk_score"].tolist()
        mse= mean_squared_error(labels, results)
        rmse= root_mean_squared_error(labels, results)

        length= len(results)

        return (mse, rmse, length)

    def run(self) -> tuple[float, float, int]:
        prediction= self._get_results()
        mse, rmse, length= self._validation(prediction)

        return (mse, rmse, length)



    
