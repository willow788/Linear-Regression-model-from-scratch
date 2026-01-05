"""
Data loading and preprocessing for Version 3

KEY ADDITION: Train/Test split for proper model evaluation
ISSUE: Forgot to normalize target variable (y) - causing negative R² scores again! 
"""

import pandas as pd
import numpy as np
from sklearn. model_selection import train_test_split


def load_and_split_data(filepath='Advertising. csv', test_size=0.2, random_state=42):
    """
    Load data from CSV, split into train/test, and normalize features. 
    
    KEY IMPROVEMENT: Uses train/test split for proper evaluation
    ISSUE: Does NOT normalize target variable (regression from Version 2.3)
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file containing the data
    test_size : float, default=0.2
        Proportion of dataset to include in test split
    random_state : int, default=42
        Random state for reproducibility
    
    Returns:
    --------
    tuple :  (X_train, X_test, y_train, y_test, X_mean, X_std)
    """
    # Load data
    data = pd.read_csv(filepath)
    print("Missing values check:")
    print(data.isna().sum())
    
    # Extract features and target
    X = data[['TV', 'Radio', 'Newspaper']].values
    y = data['Sales']. values. reshape(-1, 1)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Calculate mean and std for normalization from TRAINING data only
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)
    
    # Normalize the data using training statistics
    X_train = (X_train - X_mean) / X_std
    X_test = (X_test - X_mean) / X_std
    
    # NOTE: Target variable (y) is NOT normalized - this is the problem!
    # Should have: 
    # y_mean = y_train.mean()
    # y_std = y_train. std()
    # y_train = (y_train - y_mean) / y_std
    # y_test = (y_test - y_mean) / y_std
    
    return X_train, X_test, y_train, y_test, X_mean, X_std


def add_bias_term(X):
    """
    Add bias term to the feature matrix. 
    
    Parameters:
    -----------
    X : numpy.ndarray
        Feature matrix
    
    Returns:
    --------
    numpy.ndarray :  Feature matrix with bias term
    """
    expanded_X = np.ones((X.shape[0], 1))
    X_b = np.c_[expanded_X, X]
    # X_b will have the bias term added to the features
    return X_b
