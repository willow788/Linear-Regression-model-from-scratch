"""
Main script for training and evaluating the Linear Regression model.

This script demonstrates the usage of all modules and runs the complete
pipeline including data preprocessing, model training, evaluation, and
visualization.
"""

import numpy as np
import pandas as pd
from data_preprocessing import load_and_preprocess_data, create_polynomial_features
from linear_regression import LinearRegression
from model_evaluation import calculate_metrics, print_metrics, cross_validation_score
from visualization import (plot_loss_convergence, plot_residuals, 
                          plot_correlation_matrix, plot_actual_vs_predicted,
                          plot_feature_importance)


def main():
    """
    Main function to run the complete pipeline. 
    """
    print("=" * 60)
    print("LINEAR REGRESSION MODEL - ADVERTISING SALES PREDICTION")
    print("=" * 60)
    
    # Load and preprocess data
    print("\n1. Loading and preprocessing data...")
    X_train, X_test, y_train, y_test = load_and_preprocess_data('Advertising.csv')
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")
    print(f"Number of features: {X_train. shape[1]}")
    
    # Define feature names
    feature_names = [
        "TV",
        "Radio",
        "Newspaper",
        "TV²",
        "Radio²",
        "Newspaper²",
        "TV × Radio",
        "TV × Newspaper",
        "Radio × Newspaper"
    ]
    
    # Perform cross-validation
    print("\n2. Performing 5-fold cross-validation...")
    cv_score = cross_validation_score(
        create_polynomial_features(pd.read_csv('Advertising.csv')[['TV', 'Radio', 'Newspaper']]. values),
        pd.read_csv('Advertising.csv')['Sales'].values. reshape(-1, 1),
        k=5
    )
    print(f"\nCross-validated R² Score: {cv_score:.4f}")
    
    # Train Batch Gradient Descent model
    print("\n3. Training model with Batch Gradient Descent...")
    model_batch = LinearRegression(
        learn_rate=0.02, 
        iter=50000, 
        method='batch', 
        l1_reg=0.1
    )
    model_batch.fit(X_train, y_train)
    
    # Print learned weights
    print("\n4. Model Coefficients:")
    for name, weight in zip(feature_names, model_batch.weights. flatten()):
        print(f"  {name: 20s}: {weight: 8.4f}")
    print(f"  {'Bias':20s}: {model_batch.bias:8.4f}")
    
    # Evaluate on test set
    print("\n5. Batch GD - Test Set Performance:")
    y_pred_test = model_batch.predict(X_test)
    test_metrics = calculate_metrics(y_test, y_pred_test)
    print_metrics(test_metrics)
    
    # Evaluate on training set
    print("\n6. Batch GD - Training Set Performance:")
    y_pred_train = model_batch.predict(X_train)
    train_metrics = calculate_metrics(y_train, y_pred_train)
    print_metrics(train_metrics)
    
    # Train Stochastic Gradient Descent model
    print("\n7. Training model with Stochastic Gradient Descent...")
    model_sgd = LinearRegression(
        learn_rate=0.01, 
        iter=50, 
        method='stochastic', 
        l1_reg=0.2
    )
    model_sgd.fit(X_train, y_train)
    
    print("\n8. SGD - Test Set Performance:")
    y_pred_sgd = model_sgd.predict(X_test)
    sgd_metrics = calculate_metrics(y_test, y_pred_sgd)
    print_metrics(sgd_metrics)
    
    # Train Mini-batch Gradient Descent model
    print("\n9. Training model with Mini-batch Gradient Descent...")
    model_mini = LinearRegression(
        learn_rate=0.01, 
        iter=1000, 
        method='mini-batch', 
        batch_size=16, 
        l1_reg=0.15
    )
    model_mini.fit(X_train, y_train)
    
    print("\n10. Mini-batch GD - Test Set Performance:")
    y_pred_mini = model_mini.predict(X_test)
    mini_metrics = calculate_metrics(y_test, y_pred_mini)
    print_metrics(mini_metrics)
    
    # Visualizations
    print("\n11. Generating visualizations...")
    
    # Plot loss convergence for batch GD
    plot_loss_convergence(model_batch.loss_history)
    
    # Plot residuals
    plot_residuals(y_test, y_pred_test)
    
    # Plot actual vs predicted
    plot_actual_vs_predicted(y_test, y_pred_test)
    
    # Plot feature importance
    plot_feature_importance(feature_names, model_batch.weights)
    
    # Plot correlation matrix
    data = pd.read_csv('Advertising.csv')
    plot_correlation_matrix(data)
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
