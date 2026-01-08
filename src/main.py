# main.py
from etl import load
from tree import DecisionTreePruning
from inference import Engine

def main():
    # 1. Load the "Smart" Data
    # n_rows = training size
    training_data, validation_data, features = load(n_rows=85000)
    
    print(f"Training with {len(training_data)} rows and {len(features)} features.")
    
    # 2. Train the Analytical Tree
    # We pass the features list so the Tree knows exactly what to look at
    model = DecisionTreePruning(training_data, features)
    model.min_samples = 100 # Your winner threshold
    model.run() # This saves results/decision_tree.json
    
    print("Training complete. JSON saved.")

    # 3. Run Validation
    # Use your existing Inference Engine to see if we break 85-90%
    tester = Engine(validation_data)
    report = tester.run()
    
    print(report)

if __name__ == "__main__":
    main()