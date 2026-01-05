"""
Main training script for Version 2.2

Improvements over Version 2.1:
- Increased iterations from 1000 to 80000
- Changed learning rate from 0.0001 to 0.01
- Uses float64 instead of float32
- Corrected loss function and gradient calculation

Result: Still has negative R² (-0.73), but better than Version 2.1 (-4.55)
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
    print()
    
    # Initialize and train model
    print("Training model...")
    print("Hyperparameters:  learning_rate=0.01, iterations=80000")
    print()
    model = LinearRegression(learn_rate=0.01, iter=80000)
    model.fit(X_b, y)
    
    # Make predictions
    print("\nMaking predictions...")
    predictions = model.predict(X_b)
    
    # Evaluate model
    print("\nModel Evaluation:")
    print("="*50)
    metrics = evaluate_model(y, predictions)
    
    # Analysis of results
    print("\n" + "="*50)
    print("ANALYSIS:")
    print("="*50)
    print(f"Final Loss: {model.loss_history[-1]:. 4f}")
    print(f"Initial Loss: {model.loss_history[0]:.4f}")
    print(f"Loss Reduction: {model.loss_history[0] - model.loss_history[-1]:.4f}")
    
    if metrics['r2'] < 0:
        print("\n⚠️  WARNING: Still negative R² score!")
        print("Improvement over Version 2.1: -0.73 vs -4.55")
        print("\nRemaining issues:")
        print("1. Learning rate still too high (0.01)")
        print("2. Need even more iterations or better approach")
        print("3. Still using bias concatenated to features")
        print("\nSolution: See Version 2.3 - normalizes target variable")
    else:
        print("\n✓ Positive R² score achieved!")
    
    return model, metrics


if __name__ == "__main__":
    model, metrics = main()
