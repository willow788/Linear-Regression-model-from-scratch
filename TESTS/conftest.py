"""
Pytest configuration and shared fixtures for testing. 
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path


@pytest.fixture
def sample_data():
    """
    Create sample data for testing. 
    
    Returns:
    --------
    dict :  Dictionary containing X, y, and expected shapes
    """
    np.random.seed(42)
    n_samples = 100
    n_features = 3
    
    X = np. random.randn(n_samples, n_features)
    true_weights = np.array([[2.5], [1.3], [-0.8]])
    bias = 1.5
    noise = np.random.randn(n_samples, 1) * 0.1
    
    y = X @ true_weights + bias + noise
    
    return {
        'X': X,
        'y': y,
        'n_samples': n_samples,
        'n_features': n_features,
        'true_weights': true_weights,
        'true_bias': bias
    }


@pytest.fixture
def normalized_data():
    """
    Create normalized sample data for testing.
    
    Returns:
    --------
    dict : Dictionary containing normalized X and y
    """
    np. random.seed(42)
    n_samples = 100
    n_features = 3
    
    X = np.random. randn(n_samples, n_features)
    y = np.random.randn(n_samples, 1)
    
    # Normalize
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_normalized = (X - X_mean) / X_std
    
    y_mean = y.mean()
    y_std = y.std()
    y_normalized = (y - y_mean) / y_std
    
    return {
        'X': X,
        'y': y,
        'X_normalized': X_normalized,
        'y_normalized': y_normalized,
        'X_mean': X_mean,
        'X_std': X_std,
        'y_mean': y_mean,
        'y_std':  y_std
    }


@pytest.fixture
def sample_csv(tmp_path):
    """
    Create a temporary CSV file for testing data loading.
    
    Parameters:
    -----------
    tmp_path : Path
        Pytest fixture providing temporary directory
        
    Returns:
    --------
    Path : Path to the created CSV file
    """
    data = pd.DataFrame({
        'Unnamed: 0': range(50),
        'TV': np.random.uniform(0, 300, 50),
        'Radio': np.random.uniform(0, 50, 50),
        'Newspaper': np.random.uniform(0, 120, 50),
        'Sales': np.random.uniform(5, 25, 50)
    })
    
    csv_path = tmp_path / "test_advertising.csv"
    data.to_csv(csv_path, index=False)
    
    return csv_path


@pytest.fixture
def sample_predictions():
    """
    Create sample predictions for evaluation metrics testing.
    
    Returns:
    --------
    dict : Dictionary containing y_true and y_pred
    """
    np.random.seed(42)
    y_true = np.array([[10], [20], [30], [40], [50]])
    y_pred = np.array([[12], [19], [31], [38], [51]])
    
    return {
        'y_true': y_true,
        'y_pred': y_pred
    }
