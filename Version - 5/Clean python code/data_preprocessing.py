"""
Data preprocessing utilities for linear regression model.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_and_prepare_data(filepath, test_size=0.2, random_state=42, normalize_y=True):
    """
    Load data from CSV and prepare it for training.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file containing the data
    test_size :  float, default=0.2
        Proportion of dataset to include in the test split
    random_state :  int, default=42
        Random state for reproducibility
    normalize_y : bool, default=True
        Whether to normalize the target variable
    
    Returns:
    --------
    tuple : (X_train, X_test, y_train, y_test, X_mean, X_std, y_mean, y_std)
    """
    # Load data
    data = pd.read_csv(filepath)
    print("Missing values check:")
    print(data.isna().sum())
    
    # Extract features and target
    X = data[['TV', 'Radio', 'Newspaper']].values
    y = data['Sales']. values. reshape(-1, 1)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Calculate normalization parameters from training data
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)
    
    # Normalize features
    X_train = (X_train - X_mean) / X_std
    X_test = (X_test - X_mean) / X_std
    
    # Normalize target if specified
    if normalize_y: 
        y_train_mean = y_train.mean()
        y_train_std = y_train.std()
        y_train = (y_train - y_train_mean) / y_train_std
        y_test = (y_test - y_train_mean) / y_train_std
    else:
        y_train_mean = 0
        y_train_std = 1
    
    return X_train, X_test, y_train, y_test, X_mean, X_std, y_train_mean, y_train_std


def normalize_features(X, mean, std):
    """
    Normalize features using provided mean and standard deviation.
    
    Parameters:
    -----------
    X : numpy. ndarray
        Features to normalize
    mean : numpy.ndarray
        Mean values for normalization
    std : numpy.ndarray
        Standard deviation values for normalization
    
    Returns:
    --------
    numpy.ndarray :  Normalized features
    """
    return (X - mean) / std
