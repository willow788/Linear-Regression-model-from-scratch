"""
Main training script for Linear Regression models
Trains three different models using batch, stochastic, and mini-batch gradient descent
"""

import numpy as np
import matplotlib.pyplot as plt
from data_preprocessing import load_and_preprocess_data
from linear_regression import LinearRegression
from evaluation import calculate_metrics


def plot_loss_convergence(model, title="Loss Convergence"):
    """
    Plot the loss convergence over iterations
    
    Args:
        model:  Trained LinearRegression model
        title: Plot title
    """
    plt. figure(figsize=(10, 6))
    plt.plot(model.loss_history)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.show()


def train_batch_gradient_descent(X_train, y_train, X_test, y_test):
    """Train model using Batch Gradient Descent"""
    print("\n" + "="*70)
    print("TRAINING WITH BATCH GRADIENT DESCENT")
    print("="*70)
    
    model = LinearRegression(
        learn_rate=0.02,
        iter=50000,
        method='batch',
        l1_reg=0.1
    )
    model.fit(X_train, y_train)
    
    # Make predictions
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    
    # Evaluate
    calculate_metrics(y_test, test_predictions, "Test Set")
    calculate_metrics(y_train, train_predictions, "Training Set")
    
    # Plot loss convergence
    plot_loss_convergence(model, "Loss Convergence - Batch GD")
    
    return model


def train_stochastic_gradient_descent(X_train, y_train, X_test, y_test):
    """Train model using Stochastic Gradient Descent"""
    print("\n" + "="*70)
    print("TRAINING WITH STOCHASTIC GRADIENT DESCENT")
    print("="*70)
    
    model_sgd = LinearRegression(
        learn_rate=0.01,
        iter=50,
        method='stochastic',
        l1_reg=0.2
    )
    model_sgd.fit(X_train, y_train)
    
    # Make predictions
    train_predictions = model_sgd.predict(X_train)
    test_predictions = model_sgd.predict(X_test)
    
    # Evaluate
    calculate_metrics(y_test, test_predictions, "Test Set (SGD)")
    calculate_metrics(y_train, train_predictions, "Training Set (SGD)")
    
    return model_sgd


def train_mini_batch_gradient_descent(X_train, y_train, X_test, y_test):
    """Train model using Mini-Batch Gradient Descent"""
    print("\n" + "="*70)
    print("TRAINING WITH MINI-BATCH GRADIENT DESCENT")
    print("="*70)
    
    model_mini = LinearRegression(
        learn_rate=0.01,
        iter=1000,
        method='mini-batch',
        batch_size=16,
        l1_reg=0.15
    )
    model_mini.fit(X_train, y_train)
    
    # Make predictions
    train_predictions = model_mini.predict(X_train)
    test_predictions = model_mini.predict(X_test)
    
    # Evaluate
    calculate_metrics(y_test, test_predictions, "Test Set (Mini-Batch GD)")
    calculate_metrics(y_train, train_predictions, "Training Set (Mini-Batch GD)")
    
    return model_mini


def main():
    """Main function to run all experiments"""
    print("="*70)
    print("LINEAR REGRESSION MODEL - VERSION 7")
    print("Polynomial Features with L1 Regularization")
    print("="*70)
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test, X_mean, X_std, y_mean, y_std = \
        load_and_preprocess_data('Advertising.csv')
    
    print(f"\nTraining set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Number of polynomial features: {X_train.shape[1]}")
    
    # Train models with different gradient descent methods
    model_batch = train_batch_gradient_descent(X_train, y_train, X_test, y_test)
    model_sgd = train_stochastic_gradient_descent(X_train, y_train, X_test, y_test)
    model_mini = train_mini_batch_gradient_descent(X_train, y_train, X_test, y_test)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
