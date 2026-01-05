"""
Data loading and preprocessing for Version 3.1

🎉 THE COMPLETE FIX! 🎉

Combines:
- Train/test split (from Version 3)
- Target normalization (from Version 2.3)
- Proper use of training statistics for normalization
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_and_split_data(filepath='Advertising.csv', test_size=0.2, random_state=42):
    """
    Load data from CSV, split into train/test, and normalize both features and target. 
    
    KEY FIX: Now normalizes the target variable using TRAINING statistics!
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file containing the data
    test_size :  float, default=0.2
        Proportion of dataset to include in test split
    random_state : int, default=42
        Random state for reproducibility
    
    Returns:
    --------
    tuple :  (X_train, X_test, y_train, y_test, X_mean, X_std, y_mean, y_std)
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
    
    # Normalize features using training statistics
    X_train = (X_train - X_mean) / X_std
    X_test = (X_test - X_mean) / X_std
    
    # 🔑 KEY FIX: Normalize target variable using TRAINING statistics
    y_train_mean = y_train.mean()
    y_train_std = y_train.std()
    y_train = (y_train - y_train_mean) / y_train_std
    y_test = (y_test - y_train_mean) / y_train_std  # Use TRAIN stats for test!
    
    return X_train, X_test, y_train, y_test, X_mean, X_std, y_train_mean, y_train_std


def denormalize_predictions(y_normalized, y_mean, y_std):
    """
    Convert normalized predictions back to original scale. 
    
    Parameters:
    -----------
    y_normalized : numpy.ndarray
        Normalized predictions
    y_mean : float
        Mean used for normalization
    y_std : float
        Standard deviation used for normalization
    
    Returns: 
    --------
    numpy. ndarray :  Predictions in original scale
    """
    return (y_normalized * y_std) + y_mean
