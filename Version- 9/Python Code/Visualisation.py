"""
Visualization utilities for linear regression analysis.

This module provides functions for creating various plots to analyze
model performance. 
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_loss_convergence(loss_history):
    """
    Plot the loss convergence over iterations.
    
    Parameters:
    -----------
    loss_history : list
        List of loss values during training
    """
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Loss Convergence")
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_residuals(y_true, y_pred):
    """
    Create a residual plot. 
    
    Parameters:
    -----------
    y_true : np.ndarray
        True target values
    y_pred : np. ndarray
        Predicted target values
    """
    residuals = y_true - y_pred
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_correlation_matrix(data):
    """
    Plot a correlation matrix heatmap.
    
    Parameters:
    -----------
    data :  pd.DataFrame
        DataFrame containing the features and target
    """
    correlation_matrix = data.corr()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', 
                center=0, square=True, linewidths=1)
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.show()


def plot_actual_vs_predicted(y_true, y_pred):
    """
    Plot actual vs predicted values.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True target values
    y_pred : np.ndarray
        Predicted target values
    """
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true_flat, y_pred_flat, alpha=0.6)
    
    # Plot diagonal line (perfect predictions)
    min_val = min(y_true_flat.min(), y_pred_flat.min())
    max_val = max(y_true_flat.max(), y_pred_flat.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    
    plt.xlabel("Actual Sales")
    plt.ylabel("Predicted Sales")
    plt.title("Actual vs Predicted Sales")
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_feature_importance(feature_names, weights):
    """
    Plot feature importance based on model weights.
    
    Parameters:
    -----------
    feature_names :  list
        List of feature names
    weights : np.ndarray
        Model weights
    """
    weights_flat = weights.flatten()
    
    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, np.abs(weights_flat))
    plt.xlabel("Absolute Weight Value")
    plt.ylabel("Features")
    plt.title("Feature Importance")
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.show()
