"""
Data preprocessing module for the Linear Regression model. 

This module handles loading the advertising dataset, creating polynomial features,
and normalizing the data for training.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_and_preprocess_data(filepath='Advertising.csv', test_size=0.2, random_state=42):
    """
    Load the advertising dataset and preprocess it with polynomial features.
    
    Parameters:
    -----------
    filepath :  str
        Path to the CSV file containing the advertising data
    test_size : float, default=0.2
        Proportion of the dataset to include in the test split
    random_state : int, default=42
        Random state for reproducibility
        
    Returns:
    --------
    tuple :  (X_train, X_test, y_train, y_test)
        Normalized training and test sets
    """
    # Load the dataset
    data = pd.read_csv(filepath)
    print(data.isna().sum())
    
    # Extract features and target
    X = data[['TV', 'Radio', 'Newspaper']].values
    y = data['Sales']. values. reshape(-1, 1)
    
    # Create polynomial features
    X_poly = create_polynomial_features(X)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_poly, y, test_size=test_size, random_state=random_state
    )
    
    # Normalize the data
    X_train_norm, X_test_norm, X_mean, X_std = normalize_features(X_train, X_test)
    y_train_norm, y_test_norm, y_mean, y_std = normalize_target(y_train, y_test)
    
    return X_train_norm, X_test_norm, y_train_norm, y_test_norm


def create_polynomial_features(X):
    """
    Create polynomial features from the input features.
    
    Includes: 
    - Original features (TV, Radio, Newspaper)
    - Squared features (TV^2, Radio^2, Newspaper^2)
    - Interaction features (TV*Radio, TV*Newspaper, Radio*Newspaper)
    
    Parameters:
    -----------
    X : np.ndarray
        Input feature matrix of shape (n_samples, 3)
        
    Returns: 
    --------
    X_poly : np.ndarray
        Polynomial feature matrix of shape (n_samples, 9)
    """
    TV = X[:, 0]. reshape(-1, 1)
    Radio = X[:, 1].reshape(-1, 1)
    Newspaper = X[:, 2].reshape(-1, 1)
    
    X_poly = np.hstack([
        TV,
        Radio,
        Newspaper,
        TV**2,
        Radio**2,
        Newspaper**2,
        TV * Radio,
        TV * Newspaper,
        Radio * Newspaper
    ])
    
    return X_poly


def normalize_features(X_train, X_test):
    """
    Normalize features using z-score normalization.
    
    Parameters:
    -----------
    X_train : np.ndarray
        Training feature matrix
    X_test :  np.ndarray
        Test feature matrix
        
    Returns: 
    --------
    tuple : (X_train_normalized, X_test_normalized, mean, std)
        Normalized training and test sets with normalization parameters
    """
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)
    
    X_train_normalized = (X_train - X_mean) / X_std
    X_test_normalized = (X_test - X_mean) / X_std
    
    return X_train_normalized, X_test_normalized, X_mean, X_std


def normalize_target(y_train, y_test):
    """
    Normalize target variable using z-score normalization. 
    
    Parameters:
    -----------
    y_train : np. ndarray
        Training target values
    y_test : np.ndarray
        Test target values
        
    Returns:
    --------
    tuple : (y_train_normalized, y_test_normalized, mean, std)
        Normalized training and test targets with normalization parameters
    """
    y_train_mean = y_train.mean()
    y_train_std = y_train.std()
    
    y_train_normalized = (y_train - y_train_mean) / y_train_std
    y_test_normalized = (y_test - y_train_mean) / y_train_std
    
    return y_train_normalized, y_test_normalized, y_train_mean, y_train_std
