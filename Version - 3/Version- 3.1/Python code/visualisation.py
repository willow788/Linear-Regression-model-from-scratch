"""
Visualization utilities for Version 3.1
Enhanced visualizations for the complete model
"""

import matplotlib.pyplot as plt
import numpy as np
from metrics import r2_score


def plot_loss_convergence(model, save_path=None):
    """
    Plot training loss convergence with annotations.
    
    Parameters:
    -----------
    model : LinearRegression
        Trained model with loss_history
    save_path : str, optional
        Path to save the figure
    """
    plt.figure(figsize=(12, 6))
    
    iterations = range(len(model.loss_history))
    plt.plot(iterations, model.loss_history, linewidth=2, color='#2E86AB', label='Training Loss')
    
    plt.xlabel("Iterations", fontsize=13)
    plt.ylabel("Loss (MSE)", fontsize=13)
    plt.title("Version 3.1 - Training Loss Convergence", fontsize=15, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=11)
    
    # Mark convergence point
    convergence_iter = 5000
    if len(model.loss_history) > convergence_iter:
        plt.axvline(x=convergence_iter, color='red', linestyle='--', 
                   alpha=0.5, label=f'Convergence (~{convergence_iter} iter)')
        plt.legend(fontsize=11)
    
    # Add stats box
    textstr = f'Initial Loss: {model.loss_history[0]:.6f}\n'
    textstr += f'Final Loss: {model.loss_history[-1]:. 6f}\n'
    textstr += f'Reduction: {model.loss_history[0] - model.loss_history[-1]:.6f}'
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
    plt.text(0.98, 0.97, textstr, transform=plt. gca().transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right', bbox=props)
    
    if save_path:
        plt. savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_train_test_predictions(y_train, train_pred, y_test, test_pred, save_path=None):
    """
    Side-by-side comparison of train and test predictions.
    
    Parameters:
    -----------
    y_train : numpy.ndarray
        Training true values
    train_pred : numpy. ndarray
        Training predictions
    y_test : numpy.ndarray
        Test true values
    test_pred : numpy. ndarray
        Test predictions
    save_path : str, optional
        Path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Training set
    train_r2 = r2_score(y_train, train_pred)
    axes[0].scatter(y_train, train_pred, alpha=0.6, s=60, 
                    edgecolors='k', linewidth=0.5, color='#2E86AB')
    
    min_val = min(y_train.min(), train_pred.min())
    max_val = max(y_train.max(), train_pred.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 
                 'r--', lw=2.5, label='Perfect Prediction', alpha=0.8)
    
    axes[0].set_xlabel("Actual Sales (Normalized)", fontsize=12)
    axes[0].set_ylabel("Predicted Sales (Normalized)", fontsize=12)
    axes[0].set_title(f"Training Set\nR² = {train_r2:.4f}", 
                     fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_aspect('equal', adjustable='box')
    
    # Test set
    test_r2 = r2_score(y_test, test_pred)
    axes[1].scatter(y_test, test_pred, alpha=0.6, s=60, 
                    edgecolors='k', linewidth=0.5, color='#A23B72')
    
    min_val = min(y_test.min(), test_pred.min())
    max_val = max(y_test.max(), test_pred.max())
    axes[1].plot([min_val, max_val], [min_val, max_val], 
                 'r--', lw=2.5, label='Perfect Prediction', alpha=0.8)
    
    axes[1].set_xlabel("Actual Sales (Normalized)", fontsize=12)
    axes[1].set_ylabel("Predicted Sales (Normalized)", fontsize=12)
    axes[1].set_title(f"Test Set (Unseen Data)\nR² = {test_r2:.4f}", 
                     fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_aspect('equal', adjustable='box')
    
    plt.suptitle("Predictions vs Actual Values - Train/Test Comparison", 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt. savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_residual_analysis(y_train, train_pred, y_test, test_pred, save_path=None):
    """
    Comprehensive residual analysis for both train and test sets.
    
    Parameters:
    -----------
    y_train : numpy.ndarray
        Training true values
    train_pred : numpy.ndarray
        Training predictions
    y_test :  numpy.ndarray
        Test true values
    test_pred : numpy.ndarray
        Test predictions
    save_path : str, optional
        Path to save the figure
    """
    train_residuals = y_train - train_pred
    test_residuals = y_test - test_pred
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Row 1: Training set residuals
    # Residual vs Predicted
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(train_pred, train_residuals, alpha=0.6, s=50,
               edgecolors='k', linewidth=0.5, color='#2E86AB')
    ax1.axhline(y=0, color='r', linestyle='--', lw=2)
    ax1.set_xlabel("Predicted Values", fontsize=11)
    ax1.set_ylabel("Residuals", fontsize=11)
    ax1.set_title("Training:  Residuals vs Predicted", fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Residual histogram
    ax2 = fig. add_subplot(gs[0, 1])
    ax2.hist(train_residuals, bins=30, edgecolor='black', 
            alpha=0.7, color='#2E86AB')
    ax2.axvline(x=0, color='r', linestyle='--', lw=2)
    ax2.set_xlabel("Residuals", fontsize=11)
    ax2.set_ylabel("Frequency", fontsize=11)
    ax2.set_title("Training: Residual Distribution", fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Q-Q plot
    ax3 = fig.add_subplot(gs[0, 2])
    from scipy import stats
    stats.probplot(train_residuals. flatten(), dist="norm", plot=ax3)
    ax3.set_title("Training: Q-Q Plot", fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Row 2: Test set residuals
    # Residual vs Predicted
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.scatter(test_pred, test_residuals, alpha=0.6, s=50,
               edgecolors='k', linewidth=0.5, color='#A23B72')
    ax4.axhline(y=0, color='r', linestyle='--', lw=2)
    ax4.set_xlabel("Predicted Values", fontsize=11)
    ax4.set_ylabel("Residuals", fontsize=11)
    ax4.set_title("Test: Residuals vs Predicted", fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Residual histogram
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.hist(test_residuals, bins=20, edgecolor='black', 
            alpha=0.7, color='#A23B72')
    ax5.axvline(x=0, color='r', linestyle='--', lw=2)
    ax5.set_xlabel("Residuals", fontsize=11)
    ax5.set_ylabel("Frequency", fontsize=11)
    ax5.set_title("Test: Residual Distribution", fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # Q-Q plot
    ax6 = fig.add_subplot(gs[1, 2])
    stats.probplot(test_residuals.flatten(), dist="norm", plot=ax6)
    ax6.set_title("Test: Q-Q Plot", fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle("Residual Analysis - Train vs Test", 
                 fontsize=16, fontweight='bold', y=0.995)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_comprehensive_dashboard(model, X_train, y_train, X_test, y_test, 
                                 train_metrics, test_metrics, save_path=None):
    """
    Create a comprehensive dashboard with all key visualizations.
    
    Parameters:
    -----------
    model : LinearRegression
        Trained model
    X_train, y_train : numpy.ndarray
        Training data
    X_test, y_test : numpy.ndarray
        Test data
    train_metrics, test_metrics : dict
        Computed metrics
    save_path : str, optional
        Path to save the figure
    """
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # 1. Loss convergence
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(model.loss_history, linewidth=2, color='#2E86AB')
    ax1.set_xlabel("Iterations", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.set_title("Training Loss Convergence", fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 2. Train predictions
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(y_train, train_pred, alpha=0.6, s=40, color='#2E86AB', edgecolors='k', linewidth=0.3)
    min_val, max_val = y_train.min(), y_train.max()
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    ax2.set_xlabel("Actual", fontsize=11)
    ax2.set_ylabel("Predicted", fontsize=11)
    ax2.set_title(f"Train:  R²={train_metrics['r2']:.4f}", fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Test predictions
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.scatter(y_test, test_pred, alpha=0.6, s=40, color='#A23B72', edgecolors='k', linewidth=0.3)
    min_val, max_val = y_test.min(), y_test.max()
    ax3.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    ax3.set_xlabel("Actual", fontsize=11)
    ax3.set_ylabel("Predicted", fontsize=11)
    ax3.set_title(f"Test:  R²={test_metrics['r2']:.4f}", fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Train residuals
    ax4 = fig.add_subplot(gs[1, 0])
    train_res = y_train - train_pred
    ax4.scatter(train_pred, train_res, alpha=0.6, s=40, color='#2E86AB', edgecolors='k', linewidth=0.3)
    ax4.axhline(y=0, color='r', linestyle='--', lw=2)
    ax4.set_xlabel("Predicted", fontsize=11)
    ax4.set_ylabel("Residuals", fontsize=11)
    ax4.set_title("Train: Residual Plot", fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # 5. Test residuals
    ax5 = fig.add_subplot(gs[1, 1])
    test_res = y_test - test_pred
    ax5.scatter(test_pred, test_res, alpha=0.6, s=40, color='#A23B72', edgecolors='k', linewidth=0.3)
    ax5.axhline(y=0, color='r', linestyle='--', lw=2)
    ax5.set_xlabel("Predicted", fontsize=11)
    ax5.set_ylabel("Residuals", fontsize=11)
    ax5.set_title("Test: Residual Plot", fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # 6. Metrics summary
    ax6 = fig. add_subplot(gs[1, 2])
    ax6.axis('off')
    
    summary_text = f"""
    VERSION 3.1 - MODEL SUMMARY
    {'='*35}
    
    TRAINING SET
    R² Score:      {train_metrics['r2']:.4f}
    MSE:          {train_metrics['mse']:.4f}
    RMSE:         {train_metrics['rmse']:.4f}
    MAE:          {train_metrics['mae']:.4f}
    
    TEST SET (Unseen Data)
    R² Score:     {test_metrics['r2']:.4f}
    MSE:          {test_metrics['mse']:.4f}
    RMSE:          {test_metrics['rmse']:.4f}
    MAE:          {test_metrics['mae']:.4f}
    
    GENERALIZATION
    R² Difference: {abs(train_metrics['r2'] - test_metrics['r2']):.4f}
    Status:       {'✅ Excellent' if abs(train_metrics['r2'] - test_metrics['r2']) < 0.05 else '✓ Good'}
    
    MODEL INFO
    Learning Rate: {model.lr}
    Iterations:    {model.iter}
    Final Loss:    {model.loss_history[-1]:.6f}
    """
    
    ax6.text(0.1, 0.5, summary_text, fontsize=9, 
             verticalalignment='center', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    plt.suptitle("Version 3.1 - Comprehensive Model Dashboard", 
                 fontsize=18, fontweight='bold', y=0.995)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
