# main.py
from etl import load
from tree import DecisionTreePruning
from inference import Engine

def main():
    # 1. Load data
    # n_rows = training size
    training_data, validation_data, features = load(n_rows=85000)
    
    print(f"Training with {len(training_data)} rows and {len(features)} features.")
    
    # 2. Train the Analytical Tree

    model = DecisionTreePruning(training_data, features)
    model.min_samples = 100 
    model.run() # This saves results/decision_tree.json
    
    print("Training complete. JSON saved.")

    # 3. Run Validation
    tester = Engine(validation_data)
    report = tester.run()
    report_text, errors, resorts = report 

    print("\n" + "="*50)
    print("ANALYTICAL ID3 CLASSIFICATION REPORT")
    print("="*50)
    print(report_text)  
    print("-" * 50)
    print(f"Total Errors: {errors}")
    print(f"Last Resort Activations: {resorts}")
    print(f"Model Coverage: {((15000 - resorts) / 15000) * 100:.2f}%")
    print("="*50)

if __name__ == "__main__":
    main()