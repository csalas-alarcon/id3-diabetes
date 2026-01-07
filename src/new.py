import os
import pandas as pd

def get_unique_values():
    # Construct the relative file path
    dirname = os.path.dirname(__file__)
    file = os.path.join(dirname, '../data/diabetes.csv')
    
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file)
    
    # Get unique values in the 'diabetes_stage' column
    unique_values = df['diabetes_stage'].unique()
    
    # Print the unique values
    print("Unique values in 'diabetes_stage':")
    print(unique_values)

if __name__ == "__main__":
    get_unique_values()
