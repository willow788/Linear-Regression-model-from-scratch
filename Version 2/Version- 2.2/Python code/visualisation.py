"""
Visualization utilities for the linear regression model - Version 2.2
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_loss_history(loss_history, title="Loss Convergence", save_path=None):
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
    plt.plot(loss_history, linewidth=2)
    plt.xlabel("Iterations", fontsize=12)
    plt.ylabel("Loss (MSE)", fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add annotation for initial and final loss
    plt.annotate(f'Initial: {loss_history[0]:.2f}', 
                 xy=(0, loss_history[0]), 
                 xytext=(len(loss_history)*0.1, loss_history[0]),
                 arrowprops=dict(arrowstyle='->', color='red'),
                 fontsize=10, color='red')
    
    plt.annotate(f'Final: {loss_history[-1]:. 2f}', 
                 xy=(len(loss_history)-1, loss_history[-1]), 
                 xytext=(len(loss_history)*0.6, loss_history[-1]*1.2),
                 arrowprops=dict(arrowstyle='->', color='green'),
                 fontsize=10, color='green')
    
    if save_path:
        plt. savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_predictions_vs_actual(y_true, y_pred, save_path=None):
    """
    Plot predictions vs actual values. 
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        True values
    y_pred : numpy.ndarray
        Predicted values
    save_path : str, optional
        Path to save the figure
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolors='k', linewidth=0.5)
    
    # Plot ideal prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    plt.xlabel("Actual Sales", fontsize=12)
    plt.ylabel("Predicted Sales", fontsize=12)
    plt.title("Predictions vs Actual Values", fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    if save_path: 
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_residuals(y_true, y_pred, save_path=None):
    """
    Plot residuals distribution.
    
    Parameters:
    -----------
    y_true :  numpy.ndarray
        True values
    y_pred : numpy. ndarray
        Predicted values
    save_path : str, optional
        Path to save the figure
    """
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Residual scatter plot
    axes[0].scatter(y_pred, residuals, alpha=0.6, edgecolors='k', linewidth=0.5)
    axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0].set_xlabel("Predicted Values", fontsize=12)
    axes[0].set_ylabel("Residuals", fontsize=12)
    axes[0].set_title("Residual Plot", fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Residual histogram
    axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[1].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel("Residuals", fontsize=12)
    axes[1].set_ylabel("Frequency", fontsize=12)
    axes[1].set_title("Residual Distribution", fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt. savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
