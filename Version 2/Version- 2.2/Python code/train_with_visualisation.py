"""
Training script with visualizations for Version 2.2
"""

import numpy as np
from data_preprocessing import load_and_prepare_data, add_bias_term
from linear_regression import LinearRegression
from metrics import evaluate_model
from visualization import plot_loss_history, plot_predictions_vs_actual, plot_residuals


def main():
    """Train, evaluate, and visualize the linear regression model."""
    
    # Load and prepare data
    print("Loading and preparing data...")
    X, y = load_and_prepare_data('Advertising.csv')
    
    # Add bias term
    X_b = add_bias_term(X)
    print(f"Feature shape with bias:  {X_b.shape}\n")
    
    # Initialize and train model
    print("Training model...")
    model = LinearRegression(learn_rate=0.01, iter=80000)
    model.fit(X_b, y)
    
    # Make predictions
    print("\nMaking predictions...")
    predictions = model.predict(X_b)
    
    # Evaluate model
    print("\nModel Evaluation:")
    print("="*50)
    metrics = evaluate_model(y, predictions)
    
    # Visualizations
    print("\nGenerating visualizations...")
    plot_loss_history(model.loss_history, "Version 2.2 - Loss Convergence")
    plot_predictions_vs_actual(y, predictions)
    plot_residuals(y, predictions)
    
    return model, metrics


if __name__ == "__main__":
    model, metrics = main()
