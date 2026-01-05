"""
Data loading and preprocessing for Version 2.3
KEY IMPROVEMENT: Normalizes BOTH features AND target variable
This fixes the numerical instability and achieves positive R² scores!
"""

import pandas as pd
import numpy as np


def load_and_prepare_data(filepath='Advertising. csv'):
    """
    Load data from CSV and prepare it for training. 
    
    KEY CHANGE: Normalizes both X and y for better numerical stability. 
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file containing the data
    
    Returns:
    --------
    tuple :  (X, y) - normalized features and target
    """
    # Load data
    data = pd.read_csv(filepath)
    print("Missing values check:")
    print(data.isna().sum())
    
    # Extract features
    X = data[['TV', 'Radio', 'Newspaper']].values
    # X will have the values of the features
    
    # Normalizing the features
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    X = X.astype('float64')
    
    # Extract target
    y = data['Sales']. values. reshape(-1, 1)
    
    # KEY IMPROVEMENT: Normalize the target variable! 
    y = (y - y.mean()) / y.std()
    y = y.astype('float64')
    
    return X, y


def add_bias_term(X):
    """
    Add bias term to the feature matrix.
    
    Parameters:
    -----------
    X : numpy. ndarray
        Feature matrix
    
    Returns:
    --------
    numpy.ndarray :  Feature matrix with bias term
    """
    expanded_X = np.ones((X.shape[0], 1))
    X_b = np.c_[expanded_X, X]
    # X_b will have the bias term added to the features
    return X_b
