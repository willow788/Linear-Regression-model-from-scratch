"""
Evaluation metrics for regression models - Version 3.1
Enhanced with denormalization support for interpretable results
"""

import numpy as np


def mean_squared_error(y_true, y_pred):
    """Calculate Mean Squared Error."""
    return np.mean((y_pred - y_true) ** 2)


def root_mean_squared_error(y_true, y_pred):
    """Calculate Root Mean Squared Error."""
    mse = mean_squared_error(y_true, y_pred)
    return np.sqrt(mse)


def mean_absolute_error(y_true, y_pred):
    """Calculate Mean Absolute Error."""
    return np.mean(np.abs(y_pred - y_true))


def r2_score(y_true, y_pred):
    """Calculate R-squared score."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


def evaluate_model(y_true, y_pred, dataset_name=""):
    """
    Evaluate model and print all metrics.
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        True values
    y_pred : numpy.ndarray
        Predicted values
    dataset_name : str, default=""
        Name of the dataset (e.g., 'Train', 'Test')
    
    Returns:
    --------
    dict : Dictionary containing all metrics
    """
    prefix = f"{dataset_name} " if dataset_name else ""
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"{prefix}Mean Squared Error: {mse:.6f}")
    print(f"{prefix}Root Mean Squared Error:  {rmse:.6f}")
    print(f"{prefix}Mean Absolute Error: {mae:.6f}")
    print(f"{prefix}R² Score: {r2:.6f}")
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }


def compare_train_test_performance(train_metrics, test_metrics):
    """
    Compare training and test set performance.
    
    Parameters:
    -----------
    train_metrics : dict
        Metrics from training set
    test_metrics : dict
        Metrics from test set
    """
    print("\n" + "="*60)
    print("TRAIN vs TEST COMPARISON")
    print("="*60)
    
    print(f"\nR² Score:")
    print(f"  Train: {train_metrics['r2']:.6f}")
    print(f"  Test:   {test_metrics['r2']:.6f}")
    r2_diff = train_metrics['r2'] - test_metrics['r2']
    print(f"  Difference: {r2_diff:. 6f} (Train - Test)")
    
    print(f"\nMean Squared Error:")
    print(f"  Train: {train_metrics['mse']:.6f}")
    print(f"  Test:  {test_metrics['mse']:.6f}")
    
    print(f"\nMean Absolute Error:")
    print(f"  Train: {train_metrics['mae']:.6f}")
    print(f"  Test:  {test_metrics['mae']:.6f}")
    
    # Analyze generalization
    print("\n" + "="*60)
    print("GENERALIZATION ANALYSIS")
    print("="*60)
    
    if train_metrics['r2'] > 0 and test_metrics['r2'] > 0:
        if abs(r2_diff) <= 0.05: 
            print("✅ EXCELLENT:  Very similar train/test performance")
            print("   Model generalizes well to unseen data")
        elif abs(r2_diff) <= 0.10:
            print("✓ GOOD: Acceptable train/test performance difference")
            print("   Model generalizes reasonably well")
        elif r2_diff > 0.10:
            print("⚠️  WARNING: Train R² significantly higher than test")
            print("   Possible overfitting - model may not generalize well")
        else:
            print("⚠️  UNUSUAL: Test R² higher than train")
            print("   This is uncommon but can happen with small datasets")
    else:
        print("❌ CRITICAL:  Negative R² scores indicate severe underfitting")


def print_model_summary(model, train_metrics, test_metrics):
    """
    Print a comprehensive model summary.
    
    Parameters:
    -----------
    model :  LinearRegression
        Trained model
    train_metrics : dict
        Training metrics
    test_metrics :  dict
        Test metrics
    """
    print("\n" + "="*60)
    print("MODEL SUMMARY")
    print("="*60)
    
    print(f"\nModel Parameters:")
    print(f"  Learning Rate: {model.lr}")
    print(f"  Iterations: {model.iter}")
    print(f"  Feature Count: {model.weights.shape[0]}")
    
    print(f"\nTraining Results:")
    print(f"  Initial Loss: {model.loss_history[0]:. 6f}")
    print(f"  Final Loss: {model.loss_history[-1]:.6f}")
    print(f"  Loss Reduction: {model.loss_history[0] - model.loss_history[-1]:.6f}")
    
    print(f"\nPerformance:")
    print(f"  Train R²: {train_metrics['r2']:.4f} ({train_metrics['r2']*100:.2f}% variance explained)")
    print(f"  Test R²:   {test_metrics['r2']:.4f} ({test_metrics['r2']*100:. 2f}% variance explained)")
    
    if test_metrics['r2'] > 0.9:
        print("\n🌟 EXCELLENT MODEL PERFORMANCE!")
    elif test_metrics['r2'] > 0.8:
        print("\n✅ VERY GOOD MODEL PERFORMANCE")
    elif test_metrics['r2'] > 0.7:
        print("\n✓ GOOD MODEL PERFORMANCE")
    elif test_metrics['r2'] > 0.5:
        print("\n⚠️  MODERATE MODEL PERFORMANCE")
    else:
        print("\n❌ POOR MODEL PERFORMANCE")
