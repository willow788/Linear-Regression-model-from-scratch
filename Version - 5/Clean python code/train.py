"""
Main training script for linear regression model.
"""

import numpy as np
from data_preprocessing import load_and_prepare_data
from linear_regression import LinearRegression
from metrics import evaluate_model
from visualization import plot_loss_convergence, plot_predictions_vs_actual


def train_batch_gd():
    """Train model using batch gradient descent."""
    print("\n" + "="*50)
    print("Training with Batch Gradient Descent")
    print("="*50)
    
    # Load and prepare data
    X_train, X_test, y_train, y_test, *_ = load_and_prepare_data(
        'Advertising. csv', 
        normalize_y=True
    )
    
    # Initialize and train model
    model = LinearRegression(
        learn_rate=0.02, 
        iter=50000, 
        method='batch', 
        l2_reg=0.1
    )
    model.fit(X_train, y_train)
    
    # Make predictions
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    
    # Evaluate
    print("\nTraining Set Metrics:")
    evaluate_model(y_train, train_predictions, "Train")
    
    print("\nTest Set Metrics:")
    evaluate_model(y_test, test_predictions, "Test")
    
    # Visualize
    plot_loss_convergence(model.loss_history, "Batch GD Loss Convergence")
    plot_predictions_vs_actual(y_test, test_predictions, "Batch GD - Test Set")
    
    return model


def train_sgd():
    """Train model using stochastic gradient descent."""
    print("\n" + "="*50)
    print("Training with Stochastic Gradient Descent")
    print("="*50)
    
    # Load and prepare data
    X_train, X_test, y_train, y_test, *_ = load_and_prepare_data(
        'Advertising.csv', 
        normalize_y=True
    )
    
    # Initialize and train model
    model = LinearRegression(
        learn_rate=0.01, 
        iter=50, 
        method='stochastic', 
        l2_reg=0.2
    )
    model.fit(X_train, y_train)
    
    # Make predictions
    train_predictions = model. predict(X_train)
    test_predictions = model.predict(X_test)
    
    # Evaluate
    print("\nTraining Set Metrics:")
    evaluate_model(y_train, train_predictions, "Train")
    
    print("\nTest Set Metrics:")
    evaluate_model(y_test, test_predictions, "Test")
    
    return model


def train_mini_batch_gd():
    """Train model using mini-batch gradient descent."""
    print("\n" + "="*50)
    print("Training with Mini-Batch Gradient Descent")
    print("="*50)
    
    # Load and prepare data
    X_train, X_test, y_train, y_test, *_ = load_and_prepare_data(
        'Advertising.csv', 
        normalize_y=True
    )
    
    # Initialize and train model
    model = LinearRegression(
        learn_rate=0.01, 
        iter=1000, 
        method='mini-batch', 
        batch_size=16, 
        l2_reg=0.15
    )
    model.fit(X_train, y_train)
    
    # Make predictions
    train_predictions = model. predict(X_train)
    test_predictions = model.predict(X_test)
    
    # Evaluate
    print("\nTraining Set Metrics:")
    evaluate_model(y_train, train_predictions, "Train")
    
    print("\nTest Set Metrics:")
    evaluate_model(y_test, test_predictions, "Test")
    
    # Visualize
    plot_loss_convergence(model.loss_history, "Mini-Batch GD Loss Convergence")
    plot_predictions_vs_actual(y_test, test_predictions, "Mini-Batch GD - Test Set")
    
    return model


if __name__ == "__main__": 
    # Train with different methods
    batch_model = train_batch_gd()
    sgd_model = train_sgd()
    mini_batch_model = train_mini_batch_gd()
