"""
Visualization utilities for the linear regression model - Version 2.3
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
    plt.figure(figsize=(12, 6))
    plt.plot(loss_history, linewidth=2, color='#2E86AB')
    plt.xlabel("Iterations", fontsize=13)
    plt.ylabel("Loss (MSE)", fontsize=13)
    plt.title(title, fontsize=15, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Add annotation for convergence
    convergence_point = 5000
    if len(loss_history) > convergence_point:
        plt. axvline(x=convergence_point, color='red', linestyle='--', 
                    alpha=0.5, label=f'Convergence (~{convergence_point} iter)')
        plt.legend(fontsize=11)
    
    # Add text box with stats
    textstr = f'Initial Loss: {loss_history[0]:.4f}\nFinal Loss: {loss_history[-1]:.4f}\nReduction: {loss_history[0] - loss_history[-1]:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    if save_path:
        plt. savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_predictions_vs_actual(y_true, y_pred, save_path=None):
    """
    Plot predictions vs actual values. 
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        True values (normalized)
    y_pred : numpy.ndarray
        Predicted values (normalized)
    save_path : str, optional
        Path to save the figure
    """
    plt.figure(figsize=(10, 8))
    
    # Scatter plot
    plt.scatter(y_true, y_pred, alpha=0.6, s=50, 
                edgecolors='k', linewidth=0.5, color='#A23B72')
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 
             'r--', lw=2.5, label='Perfect Prediction', alpha=0.8)
    
    # Calculate R²
    from metrics import r2_score
    r2 = r2_score(y_true, y_pred)
    
    plt.xlabel("Actual Sales (Normalized)", fontsize=13)
    plt.ylabel("Predicted Sales (Normalized)", fontsize=13)
    plt.title(f"Predictions vs Actual Values\nR² = {r2:.4f}", 
              fontsize=15, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Make plot square
    plt.axis('equal')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_residuals(y_true, y_pred, save_path=None):
    """
    Plot residuals distribution and analysis.
    
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
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Residual scatter plot
    axes[0, 0].scatter(y_pred, residuals, alpha=0.6, s=50,
                       edgecolors='k', linewidth=0.5, color='#F18F01')
    axes[0, 0].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0, 0].set_xlabel("Predicted Values", fontsize=12)
    axes[0, 0].set_ylabel("Residuals", fontsize=12)
    axes[0, 0].set_title("Residual Plot", fontsize=13, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, linestyle='--')
    
    # 2. Residual histogram
    axes[0, 1].hist(residuals, bins=30, edgecolor='black', 
                    alpha=0.7, color='#C73E1D')
    axes[0, 1].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[0, 1].set_xlabel("Residuals", fontsize=12)
    axes[0, 1]. set_ylabel("Frequency", fontsize=12)
    axes[0, 1].set_title("Residual Distribution", fontsize=13, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, linestyle='--')
    
    # 3. Q-Q plot (normal probability plot)
    from scipy import stats
    stats.probplot(residuals. flatten(), dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title("Q-Q Plot (Normality Check)", fontsize=13, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, linestyle='--')
    
    # 4. Residuals vs actual
    axes[1, 1].scatter(y_true, residuals, alpha=0.6, s=50,
                       edgecolors='k', linewidth=0.5, color='#6A4C93')
    axes[1, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1, 1].set_xlabel("Actual Values", fontsize=12)
    axes[1, 1].set_ylabel("Residuals", fontsize=12)
    axes[1, 1].set_title("Residuals vs Actual Values", fontsize=13, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_comprehensive_analysis(model, X, y, save_path=None):
    """
    Create a comprehensive 4-panel analysis plot.
    
    Parameters:
    -----------
    model : LinearRegression
        Trained model
    X : numpy. ndarray
        Features
    y : numpy.ndarray
        Target values
    save_path : str, optional
        Path to save the figure
    """
    predictions = model.predict(X)
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # 1. Loss convergence
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(model.loss_history, linewidth=2, color='#2E86AB')
    ax1.set_xlabel("Iterations", fontsize=12)
    ax1.set_ylabel("Loss (MSE)", fontsize=12)
    ax1.set_title("Loss Convergence", fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # 2. Predictions vs Actual
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(y, predictions, alpha=0.6, s=50, 
                edgecolors='k', linewidth=0.5, color='#A23B72')
    min_val = min(y.min(), predictions.min())
    max_val = max(y.max(), predictions.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 
             'r--', lw=2, label='Perfect Prediction')
    ax2.set_xlabel("Actual Sales", fontsize=12)
    ax2.set_ylabel("Predicted Sales", fontsize=12)
    ax2.set_title("Predictions vs Actual", fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # 3. Residual plot
    ax3 = fig. add_subplot(gs[1, 0])
    residuals = y - predictions
    ax3.scatter(predictions, residuals, alpha=0.6, s=50,
                edgecolors='k', linewidth=0.5, color='#F18F01')
    ax3.axhline(y=0, color='r', linestyle='--', lw=2)
    ax3.set_xlabel("Predicted Values", fontsize=12)
    ax3.set_ylabel("Residuals", fontsize=12)
    ax3.set_title("Residual Plot", fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')
    
    # 4. Metrics summary
    ax4 = fig. add_subplot(gs[1, 1])
    ax4.axis('off')
    
    from metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error, r2_score
    mse = mean_squared_error(y, predictions)
    rmse = root_mean_squared_error(y, predictions)
    mae = mean_absolute_error(y, predictions)
    r2 = r2_score(y, predictions)
    
    metrics_text = f"""
    MODEL PERFORMANCE METRICS
    {'='*40}
    
    R² Score:            {r2:.4f}
    Mean Squared Error:  {mse:.4f}
    Root MSE:           {rmse:.4f}
    Mean Absolute Error:{mae:.4f}
    
    TRAINING INFO
    {'='*40}
    
    Learning Rate:       {model.lr}
    Iterations:         {model.iter}
    Final Loss:         {model.loss_history[-1]:.6f}
    Initial Loss:       {model.loss_history[0]:.6f}
    
    STATUS
    {'='*40}
    
    {'✅ EXCELLENT PERFORMANCE!' if r2 > 0.85 else '✓ Good Performance' if r2 > 0.7 else '⚠ Needs Improvement'}
    """
    
    ax4.text(0.1, 0.5, metrics_text, fontsize=11, 
             verticalalignment='center', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.suptitle("Version 2.3 - Comprehensive Model Analysis", 
                 fontsize=16, fontweight='bold', y=0.995)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
