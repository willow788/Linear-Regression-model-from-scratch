"""
Visualization utilities for Version 3
Includes train/test comparison plots
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_train_test_loss(model, save_path=None):
    """
    Plot training loss over iterations.
    
    Parameters:
    -----------
    model : LinearRegression
        Trained model with loss_history
    save_path : str, optional
        Path to save the figure
    """
    plt.figure(figsize=(12, 6))
    plt.plot(model.loss_history, linewidth=2, color='#2E86AB')
    plt.xlabel("Iterations", fontsize=13)
    plt.ylabel("Training Loss (MSE)", fontsize=13)
    plt.title("Training Loss Convergence", fontsize=15, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Add stats
    textstr = f'Initial:  {model.loss_history[0]:.2f}\nFinal: {model.loss_history[-1]:.2f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=11,
             verticalalignment='top', bbox=props)
    
    if save_path:
        plt. savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_train_test_predictions(y_train, train_pred, y_test, test_pred, save_path=None):
    """
    Plot predictions vs actual for both train and test sets.
    
    Parameters:
    -----------
    y_train : numpy.ndarray
        Training true values
    train_pred : numpy. ndarray
        Training predictions
    y_test : numpy.ndarray
        Test true values
    test_pred : numpy.ndarray
        Test predictions
    save_path : str, optional
        Path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Training set
    axes[0].scatter(y_train, train_pred, alpha=0.6, s=50, 
                    edgecolors='k', linewidth=0.5, color='#2E86AB', label='Training Data')
    min_val = min(y_train.min(), train_pred.min())
    max_val = max(y_train.max(), train_pred.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 
                 'r--', lw=2, label='Perfect Prediction')
    axes[0].set_xlabel("Actual Sales", fontsize=12)
    axes[0].set_ylabel("Predicted Sales", fontsize=12)
    axes[0].set_title("Training Set Predictions", fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Test set
    axes[1].scatter(y_test, test_pred, alpha=0.6, s=50, 
                    edgecolors='k', linewidth=0.5, color='#A23B72', label='Test Data')
    min_val = min(y_test.min(), test_pred.min())
    max_val = max(y_test.max(), test_pred.max())
    axes[1].plot([min_val, max_val], [min_val, max_val], 
                 'r--', lw=2, label='Perfect Prediction')
    axes[1].set_xlabel("Actual Sales", fontsize=12)
    axes[1].set_ylabel("Predicted Sales", fontsize=12)
    axes[1].set_title("Test Set Predictions", fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_train_test_residuals(y_train, train_pred, y_test, test_pred, save_path=None):
    """
    Plot residuals for both train and test sets. 
    
    Parameters:
    -----------
    y_train : numpy. ndarray
        Training true values
    train_pred : numpy.ndarray
        Training predictions
    y_test : numpy.ndarray
        Test true values
    test_pred : numpy.ndarray
        Test predictions
    save_path :  str, optional
        Path to save the figure
    """
    train_residuals = y_train - train_pred
    test_residuals = y_test - test_pred
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Training residual scatter
    axes[0, 0]. scatter(train_pred, train_residuals, alpha=0.6, s=50,
                       edgecolors='k', linewidth=0.5, color='#2E86AB')
    axes[0, 0].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0, 0].set_xlabel("Predicted Values", fontsize=11)
    axes[0, 0].set_ylabel("Residuals", fontsize=11)
    axes[0, 0]. set_title("Training Set Residuals", fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Test residual scatter
    axes[0, 1].scatter(test_pred, test_residuals, alpha=0.6, s=50,
                       edgecolors='k', linewidth=0.5, color='#A23B72')
    axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0, 1].set_xlabel("Predicted Values", fontsize=11)
    axes[0, 1]. set_ylabel("Residuals", fontsize=11)
    axes[0, 1].set_title("Test Set Residuals", fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Training residual histogram
    axes[1, 0].hist(train_residuals, bins=30, edgecolor='black', 
                    alpha=0.7, color='#2E86AB')
    axes[1, 0].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[1, 0]. set_xlabel("Residuals", fontsize=11)
    axes[1, 0].set_ylabel("Frequency", fontsize=11)
    axes[1, 0].set_title("Training Residual Distribution", fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Test residual histogram
    axes[1, 1].hist(test_residuals, bins=20, edgecolor='black', 
                    alpha=0.7, color='#A23B72')
    axes[1, 1].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[1, 1].set_xlabel("Residuals", fontsize=11)
    axes[1, 1].set_ylabel("Frequency", fontsize=11)
    axes[1, 1].set_title("Test Residual Distribution", fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_metrics_comparison(train_metrics, test_metrics, save_path=None):
    """
    Bar plot comparing train vs test metrics.
    
    Parameters:
    -----------
    train_metrics :  dict
        Training set metrics
    test_metrics : dict
        Test set metrics
    save_path : str, optional
        Path to save the figure
    """
    metrics_names = ['R² Score', 'MSE', 'RMSE', 'MAE']
    train_values = [train_metrics['r2'], train_metrics['mse'], 
                    train_metrics['rmse'], train_metrics['mae']]
    test_values = [test_metrics['r2'], test_metrics['mse'], 
                   test_metrics['rmse'], test_metrics['mae']]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, train_values, width, label='Train', 
                   color='#2E86AB', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, test_values, width, label='Test', 
                   color='#A23B72', alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Metrics', fontsize=13)
    ax.set_ylabel('Values', fontsize=13)
    ax.set_title('Train vs Test Metrics Comparison', fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
    
    autolabel(bars1)
    autolabel(bars2)
    
    if save_path:
        plt. savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
