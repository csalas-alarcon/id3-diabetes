import os
import pandas as pd
import numpy as np

def get_unique_values():
    # Construct the relative file path
    dirname = os.path.dirname(__file__)
    file = os.path.join(dirname, '../data/diabetes.csv')
    
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file)
    
    # Get unique values in the 'diabetes_stage' column
    uniques, counts= np.unique(df["diabetes_stage"], return_counts= True)
    newfrequency= dict(zip(uniques, counts))
    
    # Print the unique values
    print("Unique values in 'diabetes_stage':")
    print(newfrequency)

if __name__ == "__main__":
    get_unique_values()
