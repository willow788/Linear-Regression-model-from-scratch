"""
Evaluation metrics for regression models - Version 4
Same as Version 3.1
"""

import numpy as np


def mean_squared_error(y_true, y_pred):
    """Calculate Mean Squared Error."""
    return np.mean((y_pred - y_true) ** 2)


def root_mean_squared_error(y_true, y_pred):
    """Calculate Root Mean Squared Error."""
    mse = mean_squared_error(y_true, y_pred)
    return np.sqrt(mse)


def mean_absolute_error(y_true, y_pred):
    """Calculate Mean Absolute Error."""
    return np. mean(np.abs(y_pred - y_true))


def r2_score(y_true, y_pred):
    """Calculate R-squared score."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


def evaluate_model(y_true, y_pred, dataset_name="", method_name=""):
    """
    Evaluate model and print all metrics.
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        True values
    y_pred : numpy.ndarray
        Predicted values
    dataset_name : str, default=""
        Name of the dataset (e.g., 'Train', 'Test')
    method_name : str, default=""
        Name of the method used (e.g., 'Batch GD', 'SGD')
    
    Returns:
    --------
    dict : Dictionary containing all metrics
    """
    prefix = ""
    if method_name:
        prefix = f"{method_name} - "
    if dataset_name:
        prefix += f"{dataset_name} "
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"{prefix}MSE: {mse:.6f}")
    print(f"{prefix}RMSE: {rmse:.6f}")
    print(f"{prefix}MAE:  {mae:.6f}")
    print(f"{prefix}R² Score: {r2:.6f}")
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }


def compare_methods(results_dict):
    """
    Compare performance of different gradient descent methods.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary with method names as keys and metrics as values
    """
    print("\n" + "="*70)
    print("COMPARISON OF GRADIENT DESCENT METHODS")
    print("="*70)
    
    print(f"\n{'Method':<20} {'Train R²':<12} {'Test R²':<12} {'Test MSE':<12}")
    print("-"*70)
    
    for method_name, metrics in results_dict. items():
        train_r2 = metrics['train']['r2']
        test_r2 = metrics['test']['r2']
        test_mse = metrics['test']['mse']
        print(f"{method_name:<20} {train_r2:<12.6f} {test_r2:<12.6f} {test_mse:<12.6f}")
    
    # Find best method
    best_method = max(results_dict. items(), 
                     key=lambda x:  x[1]['test']['r2'])
    
    print("\n" + "="*70)
    print(f"🏆 Best Method (by Test R²): {best_method[0]}")
    print(f"   Test R²: {best_method[1]['test']['r2']:.6f}")
    print("="*70)
