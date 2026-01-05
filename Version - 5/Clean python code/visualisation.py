"""
Visualization utilities for the linear regression model.
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_loss_convergence(loss_history, title="Loss Convergence", save_path=None):
    """
    Plot the loss convergence over iterations.
    
    Parameters:
    -----------
    loss_history : list
        List of loss values over iterations
    title : str, default="Loss Convergence"
        Plot title
    save_path : str, optional
        Path to save the figure
    """
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    if save_path: 
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_predictions_vs_actual(y_true, y_pred, dataset_name="", save_path=None):
    """
    Plot predictions vs actual values. 
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        True values
    y_pred : numpy.ndarray
        Predicted values
    dataset_name : str, default=""
        Name of the dataset
    save_path : str, optional
        Path to save the figure
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(f"Predictions vs Actual Values ({dataset_name})")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt. savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_residuals(y_true, y_pred, dataset_name="", save_path=None):
    """
    Plot residuals distribution.
    
    Parameters:
    -----------
    y_true :  numpy.ndarray
        True values
    y_pred : numpy. ndarray
        Predicted values
    dataset_name : str, default=""
        Name of the dataset
    save_path : str, optional
        Path to save the figure
    """
    residuals = y_true - y_pred
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--', lw=2)
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title(f"Residual Plot ({dataset_name})")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
