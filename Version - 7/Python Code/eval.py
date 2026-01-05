"""
Evaluation metrics for regression models
"""

import numpy as np


def calculate_metrics(y_true, y_pred, dataset_name=""):
    """
    Calculate and print regression metrics
    
    Args: 
        y_true: True target values
        y_pred:  Predicted values
        dataset_name:  Name to display (e.g., "test set", "training set")
    
    Returns:
        Dictionary containing all metrics
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n{'='*50}")
    if dataset_name:
        print(f"Metrics for {dataset_name}:")
    print(f"{'='*50}")
    print(f"Mean Squared Error:  {mse}")
    print(f"Root Mean Squared Error: {rmse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"R² Score:  {r2}")
    
    return {
        'mse':  mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }


def mean_squared_error(y_true, y_pred):
    """Calculate Mean Squared Error"""
    return np.mean((y_pred - y_true) ** 2)


def root_mean_squared_error(y_true, y_pred):
    """Calculate Root Mean Squared Error"""
    return np. sqrt(mean_squared_error(y_true, y_pred))


def mean_absolute_error(y_true, y_pred):
    """Calculate Mean Absolute Error"""
    return np.mean(np.abs(y_pred - y_true))


def r2_score(y_true, y_pred):
    """Calculate R² (coefficient of determination) score"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)
