"""
Model evaluation utilities for regression models.

This module provides functions for calculating various metrics and
performing cross-validation. 
"""

import numpy as np
from sklearn.model_selection import KFold
from linear_regression import LinearRegression


def calculate_metrics(y_true, y_pred):
    """
    Calculate regression metrics.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True target values
    y_pred :  np.ndarray
        Predicted target values
        
    Returns: 
    --------
    dict :  Dictionary containing MSE, RMSE, MAE, and R2 score
    """
    mse = np.mean((y_pred - y_true) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_pred - y_true))
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2_score = 1 - (ss_res / ss_tot)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2_score': r2_score
    }


def print_metrics(metrics, dataset_name=''):
    """
    Print evaluation metrics in a formatted way.
    
    Parameters:
    -----------
    metrics : dict
        Dictionary of metrics from calculate_metrics()
    dataset_name : str
        Name of the dataset (e.g., 'Test', 'Train')
    """
    prefix = f"{dataset_name} " if dataset_name else ""
    print(f"{prefix}Mean Squared Error:  {metrics['mse']:.6f}")
    print(f"{prefix}Root Mean Squared Error:  {metrics['rmse']:.6f}")
    print(f"{prefix}Mean Absolute Error: {metrics['mae']:.6f}")
    print(f"{prefix}R² Score: {metrics['r2_score']:.6f}")


def cross_validation_score(X, y, k=5, learn_rate=0.01, iter=50000, 
                          method='batch', l1_reg=0.0, early_stopping=True, 
                          patience=1000):
    """
    Perform k-fold cross-validation for the linear regression model.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target values
    k : int, default=5
        Number of folds
    learn_rate : float, default=0.01
        Learning rate for the model
    iter : int, default=50000
        Number of iterations
    method : str, default='batch'
        Gradient descent method
    l1_reg : float, default=0.0
        L1 regularization parameter
    early_stopping : bool, default=True
        Whether to use early stopping
    patience : int, default=1000
        Patience for early stopping
        
    Returns:
    --------
    float :  Mean R² score across all folds
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    r2_scores = []

    fold = 1
    for train_idx, val_idx in kf. split(X):
        X_training, X_validation = X[train_idx], X[val_idx]
        y_training, y_validation = y[train_idx], y[val_idx]

        # Normalize the data
        X_mean = X_training.mean(axis=0)
        X_std = X_training.std(axis=0)

        X_training_normalized = (X_training - X_mean) / X_std
        X_validation_normalized = (X_validation - X_mean) / X_std

        y_training_mean = y_training.mean()
        y_training_std = y_training.std()

        y_training_normalized = (y_training - y_training_mean) / y_training_std
        y_validation_normalized = (y_validation - y_training_mean) / y_training_std

        # Train model
        model = LinearRegression(
            learn_rate=learn_rate, 
            iter=iter, 
            method=method, 
            l1_reg=l1_reg, 
            early_stopping=early_stopping, 
            patience=patience
        )
        model.fit(X_training_normalized, y_training_normalized)
        y_val_pred = model.predict(X_validation_normalized)

        # Calculate R² score
        ss_total = np.sum((y_validation_normalized - 0) ** 2)
        ss_residual = np.sum((y_validation_normalized - y_val_pred) ** 2)
        r2_score = 1 - (ss_residual / ss_total)

        r2_scores.append(r2_score)
        print(f"Fold {fold} R² Score: {r2_score:.4f}")
        fold += 1

    mean_r2 = np.mean(r2_scores)
    print(f"Average R² Score across {k} folds: {mean_r2:.4f}")
    
    return mean_r2
