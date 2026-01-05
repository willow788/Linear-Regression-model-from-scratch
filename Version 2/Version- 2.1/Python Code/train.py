"""
Main training script for Version 2.1
This version demonstrates the problem with Version 1 (negative R² score)
"""

import numpy as np
from data_preprocessing import load_and_prepare_data, add_bias_term
from linear_regression import LinearRegression
from metrics import evaluate_model


def main():
    """Train and evaluate the linear regression model."""
    
    # Load and prepare data
    print("Loading and preparing data...")
    X, y = load_and_prepare_data('Advertising.csv')
    
    # Add bias term
    X_b = add_bias_term(X)
    print(f"Feature shape with bias: {X_b.shape}")
    
    # Initialize and train model
    # NOTE: These hyperparameters lead to poor performance (negative R²)
    # This is intentional to demonstrate the problem with Version 1
    print("\nTraining model...")
    model = LinearRegression(learn_rate=0.0001, iter=1000)
    model.fit(X_b, y)
    
    # Make predictions
    predictions = model.predict(X_b)
    
    # Evaluate model
    print("\nModel Evaluation:")
    metrics = evaluate_model(y, predictions)
    
    # Analysis of results
    print("\n" + "="*50)
    print("ANALYSIS:")
    print("="*50)
    if metrics['r2'] < 0:
        print("⚠️  WARNING:  Negative R² score detected!")
        print("This means the model performs worse than a horizontal line at the mean.")
        print("\nPossible reasons:")
        print("1. Learning rate is too high (0.0001)")
        print("2. Number of iterations is too low (1000)")
        print("3. Model is underfitted")
        print("\nSolution: Decrease learning rate and increase iterations")
        print("See Version 2.2 and 2.3 for improvements.")
    
    return model, metrics


if __name__ == "__main__":
    model, metrics = main()
