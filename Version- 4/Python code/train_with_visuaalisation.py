"""
Complete training and visualization pipeline for Version 4
Trains all three gradient descent methods and creates comprehensive visualizations
"""

import numpy as np
from data_preprocessing import load_and_split_data
from linear_regression import LinearRegression
from metrics import evaluate_model, compare_methods
from visualization import (plot_loss_comparison, plot_method_comparison_bars,
                           plot_comprehensive_comparison)


def main():
    """Train all methods and create comprehensive visualizations."""
    
    print("="*70)
    print(" " * 15 + "VERSION 4 - COMPLETE PIPELINE")
    print(" " * 10 + "Multiple Gradient Descent Methods")
    print("="*70)
    
    # Load data
    print("\n📁 Step 1: Loading data...")
    X_train, X_test, y_train, y_test, X_mean, X_std, y_mean, y_std = \
        load_and_split_data('Advertising. csv')
    
    print(f"✓ Training:  {X_train.shape[0]} samples")
    print(f"✓ Test: {X_test. shape[0]} samples")
    
    # Train all three methods
    print("\n🚀 Step 2: Training models...")
    
    models = {}
    results = {}
    
    # Batch GD
    print("\n" + "-"*70)
    print("Training Batch Gradient Descent...")
    models['Batch GD'] = LinearRegression(learn_rate=0.02, iter=50000, method='batch')
    models['Batch GD'].fit(X_train, y_train)
    
    train_pred = models['Batch GD'].predict(X_train)
    test_pred = models['Batch GD'].predict(X_test)
    results['Batch GD'] = {
        'train': evaluate_model(y_train, train_pred),
        'test': evaluate_model(y_test, test_pred)
    }
    
    # SGD
    print("\n" + "-"*70)
    print("Training Stochastic Gradient Descent...")
    models['SGD'] = LinearRegression(learn_rate=0.01, iter=50, method='stochastic')
    models['SGD'].fit(X_train, y_train)
    
    train_pred = models['SGD'].predict(X_train)
    test_pred = models['SGD'].predict(X_test)
    results['SGD'] = {
        'train': evaluate_model(y_train, train_pred),
        'test': evaluate_model(y_test, test_pred)
    }
    
    # Mini-Batch GD
    print("\n" + "-"*70)
    print("Training Mini-Batch Gradient Descent...")
    models['Mini-Batch GD'] = LinearRegression(learn_rate=0.01, iter=1000, 
                                                method='mini-batch', batch_size=16)
    models['Mini-Batch GD'].fit(X_train, y_train)
    
    train_pred = models['Mini-Batch GD'].predict(X_train)
    test_pred = models['Mini-Batch GD'].predict(X_test)
    results['Mini-Batch GD'] = {
        'train': evaluate_model(y_train, train_pred),
        'test': evaluate_model(y_test, test_pred)
    }
    
    # Compare
    print("\n📊 Step 3: Comparing methods...")
    compare_methods(results)
    
    # Visualize
    print("\n🎨 Step 4: Creating visualizations...")
    
    print("\n1️⃣  Loss convergence comparison...")
    plot_loss_comparison(models)
    
    print("\n2️⃣  Performance comparison bars...")
    plot_method_comparison_bars(results)
    
    print("\n3️⃣  Comprehensive dashboard...")
    plot_comprehensive_comparison(models, X_train, y_train, X_test, y_test, results)
    
    print("\n✅ Complete!")
    print("="*70)
    
    return models, results


if __name__ == "__main__":
    models, results = main()
