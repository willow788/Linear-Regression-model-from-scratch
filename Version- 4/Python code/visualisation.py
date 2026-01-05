"""
Visualization utilities for Version 4
Compares different gradient descent methods
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_loss_comparison(models_dict, save_path=None):
    """
    Compare loss convergence of different gradient descent methods.
    
    Parameters:
    -----------
    models_dict : dict
        Dictionary with method names as keys and trained models as values
    save_path : str, optional
        Path to save the figure
    """
    plt.figure(figsize=(14, 6))
    
    colors = {'Batch GD': '#2E86AB', 'SGD': '#A23B72', 'Mini-Batch GD': '#F18F01'}
    
    for method_name, model in models_dict. items():
        color = colors. get(method_name, 'blue')
        plt.plot(model.loss_history, label=method_name, linewidth=2, color=color, alpha=0.8)
    
    plt.xlabel("Iterations/Epochs", fontsize=13)
    plt.ylabel("Loss (MSE)", fontsize=13)
    plt.title("Loss Convergence Comparison - Different Gradient Descent Methods", 
              fontsize=15, fontweight='bold')
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(True, alpha=0.3, linestyle='--')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_method_comparison_bars(results_dict, save_path=None):
    """
    Bar chart comparing performance metrics across methods.
    
    Parameters:
    -----------
    results_dict :  dict
        Dictionary with method names as keys and metrics as values
    save_path :  str, optional
        Path to save the figure
    """
    methods = list(results_dict.keys())
    train_r2 = [results_dict[m]['train']['r2'] for m in methods]
    test_r2 = [results_dict[m]['test']['r2'] for m in methods]
    test_mse = [results_dict[m]['test']['mse'] for m in methods]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # R² Score comparison
    x = np.arange(len(methods))
    width = 0.35
    
    axes[0].bar(x - width/2, train_r2, width, label='Train R²', 
                color='#2E86AB', alpha=0.8, edgecolor='black')
    axes[0].bar(x + width/2, test_r2, width, label='Test R²', 
                color='#A23B72', alpha=0.8, edgecolor='black')
    
    axes[0].set_xlabel('Gradient Descent Method', fontsize=12)
    axes[0].set_ylabel('R² Score', fontsize=12)
    axes[0].set_title('R² Score Comparison', fontsize=13, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].set_ylim([0.85, 0.92])
    
    # Add value labels
    for i, (tr, te) in enumerate(zip(train_r2, test_r2)):
        axes[0].text(i - width/2, tr + 0.002, f'{tr:.4f}', 
                    ha='center', va='bottom', fontsize=9)
        axes[0].text(i + width/2, te + 0.002, f'{te:.4f}', 
                    ha='center', va='bottom', fontsize=9)
    
    # MSE comparison
    axes[1].bar(methods, test_mse, color='#F18F01', alpha=0.8, edgecolor='black')
    axes[1].set_xlabel('Gradient Descent Method', fontsize=12)
    axes[1].set_ylabel('Test MSE', fontsize=12)
    axes[1].set_title('Test MSE Comparison (Lower is Better)', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, mse in enumerate(test_mse):
        axes[1].text(i, mse + 0.002, f'{mse:. 4f}', 
                    ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt. savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_comprehensive_comparison(models_dict, X_train, y_train, X_test, y_test, 
                                   results_dict, save_path=None):
    """
    Comprehensive dashboard comparing all methods.
    
    Parameters:
    -----------
    models_dict :  dict
        Dictionary with method names as keys and trained models as values
    X_train, y_train : numpy.ndarray
        Training data
    X_test, y_test : numpy.ndarray
        Test data
    results_dict :  dict
        Dictionary with method names as keys and metrics as values
    save_path : str, optional
        Path to save the figure
    """
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    colors = {'Batch GD': '#2E86AB', 'SGD': '#A23B72', 'Mini-Batch GD': '#F18F01'}
    
    # Row 1: Loss convergence for each method
    for idx, (method_name, model) in enumerate(models_dict.items()):
        ax = fig.add_subplot(gs[0, idx])
        ax.plot(model.loss_history, linewidth=2, color=colors[method_name])
        ax.set_xlabel("Iterations/Epochs", fontsize=10)
        ax.set_ylabel("Loss", fontsize=10)
        ax.set_title(f"{method_name} - Loss Convergence", fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # Row 2: Predictions vs Actual for each method
    for idx, (method_name, model) in enumerate(models_dict.items()):
        ax = fig.add_subplot(gs[1, idx])
        test_pred = model.predict(X_test)
        ax.scatter(y_test, test_pred, alpha=0.6, s=40, 
                  color=colors[method_name], edgecolors='k', linewidth=0.3)
        min_val, max_val = y_test.min(), y_test.max()
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        ax.set_xlabel("Actual", fontsize=10)
        ax.set_ylabel("Predicted", fontsize=10)
        test_r2 = results_dict[method_name]['test']['r2']
        ax. set_title(f"{method_name}\nTest R² = {test_r2:.4f}", 
                    fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # Row 3: Comparison charts
    # R² comparison
    ax1 = fig.add_subplot(gs[2, 0:2])
    methods = list(results_dict.keys())
    train_r2 = [results_dict[m]['train']['r2'] for m in methods]
    test_r2 = [results_dict[m]['test']['r2'] for m in methods]
    
    x = np.arange(len(methods))
    width = 0.35
    ax1.bar(x - width/2, train_r2, width, label='Train R²', color='#2E86AB', alpha=0.7)
    ax1.bar(x + width/2, test_r2, width, label='Test R²', color='#A23B72', alpha=0.7)
    ax1.set_ylabel('R² Score', fontsize=11)
    ax1.set_title('Performance Comparison', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=10)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Summary table
    ax2 = fig. add_subplot(gs[2, 2])
    ax2.axis('off')
    
    summary_text = "METHOD COMPARISON\n" + "="*30 + "\n\n"
    for method in methods:
        summary_text += f"{method}:\n"
        summary_text += f"  Train R²:  {results_dict[method]['train']['r2']:.4f}\n"
        summary_text += f"  Test R²:   {results_dict[method]['test']['r2']:.4f}\n"
        summary_text += f"  Test MSE:  {results_dict[method]['test']['mse']:.4f}\n\n"
    
    ax2.text(0.1, 0.5, summary_text, fontsize=9, 
             verticalalignment='center', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.suptitle("Version 4 - Gradient Descent Methods Comparison", 
                 fontsize=18, fontweight='bold', y=0.995)
    
    if save_path:
        plt. savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
