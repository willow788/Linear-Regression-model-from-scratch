"""
Evaluation metrics for regression models - Version 3
Now includes separate evaluation for train and test sets
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


def evaluate_model(y_true, y_pred, dataset_name=""):
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
    
    Returns:
    --------
    dict : Dictionary containing all metrics
    """
    prefix = f"{dataset_name} " if dataset_name else ""
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"{prefix}Mean Squared Error: {mse}")
    print(f"{prefix}Root Mean Squared Error: {rmse}")
    print(f"{prefix}Mean Absolute Error: {mae}")
    print(f"{prefix}R² Score: {r2}")
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }


def compare_train_test_performance(train_metrics, test_metrics):
    """
    Compare training and test set performance.
    
    Parameters:
    -----------
    train_metrics : dict
        Metrics from training set
    test_metrics : dict
        Metrics from test set
    """
    print("\n" + "="*60)
    print("TRAIN vs TEST COMPARISON")
    print("="*60)
    
    print(f"\nR² Score:")
    print(f"  Train: {train_metrics['r2']:.4f}")
    print(f"  Test:   {test_metrics['r2']:.4f}")
    print(f"  Difference: {abs(train_metrics['r2'] - test_metrics['r2']):.4f}")
    
    print(f"\nMSE:")
    print(f"  Train: {train_metrics['mse']:.4f}")
    print(f"  Test:  {test_metrics['mse']:.4f}")
    
    print(f"\nMAE:")
    print(f"  Train: {train_metrics['mae']:.4f}")
    print(f"  Test:  {test_metrics['mae']:.4f}")
    
    # Check for overfitting/underfitting
    if train_metrics['r2'] > 0 and test_metrics['r2'] > 0:
        r2_diff = train_metrics['r2'] - test_metrics['r2']
        if r2_diff > 0.1:
            print("\n⚠️  Possible overfitting (train R² much higher than test)")
        elif r2_diff < -0.1:
            print("\n⚠️  Unusual:  test R² higher than train")
        else:
            print("\n✓ Good generalization (similar train and test performance)")
    else:
        print("\n❌ Model has negative R² scores - severe underfitting!")
