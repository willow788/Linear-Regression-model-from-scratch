"""
Data preprocessing module for Linear Regression model
Handles data loading, polynomial feature engineering, and normalization
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_and_preprocess_data(file_path='Advertising.csv', test_size=0.2, random_state=42):
    """
    Load data from CSV and create polynomial features
    
    Args: 
        file_path:  Path to the CSV file
        test_size:  Fraction of data to use for testing
        random_state: Random seed for reproducibility
    
    Returns:
        X_train, X_test, y_train, y_test:  Preprocessed and normalized data splits
        X_mean, X_std, y_train_mean, y_train_std: Normalization parameters
    """
    # Load the dataset
    data = pd.read_csv(file_path)
    print("Missing values check:")
    print(data.isna().sum())
    
    # Extract features
    X = data[['TV', 'Radio', 'Newspaper']].values
    y = data['Sales']. values. reshape(-1, 1)
    
    # Create polynomial features
    X_poly = create_polynomial_features(X)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_poly, y, test_size=test_size, random_state=random_state
    )
    
    # Normalize the features
    X_train, X_test, X_mean, X_std = normalize_features(X_train, X_test)
    
    # Normalize the target variable
    y_train, y_test, y_train_mean, y_train_std = normalize_target(y_train, y_test)
    
    return X_train, X_test, y_train, y_test, X_mean, X_std, y_train_mean, y_train_std


def create_polynomial_features(X):
    """
    Create polynomial features including: 
    - Original features:  TV, Radio, Newspaper
    - Squared features: TV², Radio², Newspaper²
    - Interaction terms: TV*Radio, TV*Newspaper, Radio*Newspaper
    
    Args:
        X: Original feature matrix (n_samples, 3)
    
    Returns:
        X_poly: Polynomial feature matrix (n_samples, 9)
    """
    TV = X[:, 0]. reshape(-1, 1)
    Radio = X[:, 1]. reshape(-1, 1)
    Newspaper = X[:, 2].reshape(-1, 1)
    
    X_poly = np.hstack([
        TV,
        Radio,
        Newspaper,
        TV**2,
        Radio**2,
        Newspaper**2,
        TV*Radio,
        TV*Newspaper,
        Radio*Newspaper
    ])
    
    return X_poly


def normalize_features(X_train, X_test):
    """
    Normalize features using mean and standard deviation from training set
    
    Args:
        X_train: Training features
        X_test: Test features
    
    Returns:
        X_train_normalized, X_test_normalized, X_mean, X_std
    """
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)
    
    X_train_normalized = (X_train - X_mean) / X_std
    X_test_normalized = (X_test - X_mean) / X_std
    
    return X_train_normalized, X_test_normalized, X_mean, X_std


def normalize_target(y_train, y_test):
    """
    Normalize target variable using mean and standard deviation from training set
    
    Args:
        y_train: Training target values
        y_test: Test target values
    
    Returns: 
        y_train_normalized, y_test_normalized, y_train_mean, y_train_std
    """
    y_train_mean = y_train.mean()
    y_train_std = y_train.std()
    
    y_train_normalized = (y_train - y_train_mean) / y_train_std
    y_test_normalized = (y_test - y_train_mean) / y_train_std
    
    return y_train_normalized, y_test_normalized, y_train_mean, y_train_std
