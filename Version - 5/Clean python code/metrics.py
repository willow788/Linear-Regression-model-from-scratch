"""
Evaluation metrics for regression models.
"""

import numpy as np


def mean_squared_error(y_true, y_pred):
    """
    Calculate Mean Squared Error.
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        True values
    y_pred : numpy.ndarray
        Predicted values
    
    Returns:
    --------
    float : MSE value
    """
    return np. mean((y_pred - y_true) ** 2)


def root_mean_squared_error(y_true, y_pred):
    """
    Calculate Root Mean Squared Error.
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        True values
    y_pred :  numpy.ndarray
        Predicted values
    
    Returns: 
    --------
    float :  RMSE value
    """
    mse = mean_squared_error(y_true, y_pred)
    return np.sqrt(mse)


def mean_absolute_error(y_true, y_pred):
    """
    Calculate Mean Absolute Error. 
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        True values
    y_pred : numpy.ndarray
        Predicted values
    
    Returns:
    --------
    float : MAE value
    """
    return np.mean(np.abs(y_pred - y_true))


def r2_score(y_true, y_pred):
    """
    Calculate R-squared score.
    
    Parameters:
    -----------
    y_true :  numpy.ndarray
        True values
    y_pred : numpy. ndarray
        Predicted values
    
    Returns:
    --------
    float : R² score
    """
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
        Name of the dataset (e.g., 'train', 'test')
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
