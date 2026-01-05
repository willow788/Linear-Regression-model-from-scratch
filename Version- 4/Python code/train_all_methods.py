"""
Training script for Version 4 - Compares all gradient descent methods

Trains three different models: 
1. Batch Gradient Descent
2. Stochastic Gradient Descent (SGD)
3. Mini-Batch Gradient Descent
"""

import numpy as np
from data_preprocessing import load_and_split_data
from linear_regression import LinearRegression
from metrics import evaluate_model, compare_methods


def train_batch_gd(X_train, y_train, X_test, y_test):
    """Train using Batch Gradient Descent."""
    print("\n" + "="*70)
    print("1️⃣  BATCH GRADIENT DESCENT")
    print("="*70)
    print("Updates:  After processing ALL training samples")
    print("Hyperparameters:  lr=0.02, iter=50000\n")
    
    model = LinearRegression(learn_rate=0.02, iter=50000, method='batch')
    model.fit(X_train, y_train)
    
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    print("\nEvaluation:")
    train_metrics = evaluate_model(y_train, train_pred, "Train", "Batch GD")
    test_metrics = evaluate_model(y_test, test_pred, "Test", "Batch GD")
    
    return model, {'train': train_metrics, 'test': test_metrics}


def train_sgd(X_train, y_train, X_test, y_test):
    """Train using Stochastic Gradient Descent."""
    print("\n" + "="*70)
    print("2️⃣  STOCHASTIC GRADIENT DESCENT (SGD)")
    print("="*70)
    print("Updates: After EACH individual sample")
    print("Hyperparameters: lr=0.01, epochs=50\n")
    
    model_sgd = LinearRegression(learn_rate=0.01, iter=50, method='stochastic')
    model_sgd.fit(X_train, y_train)
    
    train_pred = model_sgd.predict(X_train)
    test_pred = model_sgd. predict(X_test)
    
    print("\nEvaluation:")
    train_metrics = evaluate_model(y_train, train_pred, "Train", "SGD")
    test_metrics = evaluate_model(y_test, test_pred, "Test", "SGD")
    
    return model_sgd, {'train':  train_metrics, 'test':  test_metrics}


def train_mini_batch_gd(X_train, y_train, X_test, y_test):
    """Train using Mini-Batch Gradient Descent."""
    print("\n" + "="*70)
    print("3️⃣  MINI-BATCH GRADIENT DESCENT")
    print("="*70)
    print("Updates: After processing batches of 16 samples")
    print("Hyperparameters: lr=0.01, epochs=1000, batch_size=16\n")
    
    model_mini = LinearRegression(learn_rate=0.01, iter=1000, 
                                   method='mini-batch', batch_size=16)
    model_mini.fit(X_train, y_train)
    
    train_pred = model_mini. predict(X_train)
    test_pred = model_mini. predict(X_test)
    
    print("\nEvaluation:")
    train_metrics = evaluate_model(y_train, train_pred, "Train", "Mini-Batch GD")
    test_metrics = evaluate_model(y_test, test_pred, "Test", "Mini-Batch GD")
    
    return model_mini, {'train': train_metrics, 'test': test_metrics}


def main():
    """Train and compare all gradient descent methods."""
    
    print("="*70)
    print(" " * 20 + "VERSION 4")
    print(" " * 10 + "Multiple Gradient Descent Methods")
    print("="*70)
    
    # Load data
    print("\n📁 Loading and preparing data...")
    X_train, X_test, y_train, y_test, X_mean, X_std, y_mean, y_std = \
        load_and_split_data('Advertising.csv')
    
    print(f"\n✓ Training samples: {X_train.shape[0]}")
    print(f"✓ Test samples: {X_test.shape[0]}")
    print(f"✓ Features: {X_train.shape[1]}")
    
    # Train all three methods
    results = {}
    
    # 1. Batch GD
    model_batch, results['Batch GD'] = train_batch_gd(X_train, y_train, X_test, y_test)
    
    # 2. Stochastic GD
    model_sgd, results['SGD'] = train_sgd(X_train, y_train, X_test, y_test)
    
    # 3. Mini-Batch GD
    model_mini, results['Mini-Batch GD'] = train_mini_batch_gd(X_train, y_train, X_test, y_test)
    
    # Compare all methods
    compare_methods(results)
    
    # Analysis
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    
    print("\n📊 Batch Gradient Descent:")
    print("   ✅ Most stable convergence")
    print("   ✅ Smooth loss curve")
    print("   ❌ Slowest per epoch (uses all data)")
    print("   → Best for:  Small to medium datasets")
    
    print("\n📊 Stochastic Gradient Descent (SGD):")
    print("   ✅ Fastest updates (per sample)")
    print("   ✅ Can escape local minima")
    print("   ❌ Noisy convergence")
    print("   ❌ Requires careful learning rate tuning")
    print("   → Best for: Very large datasets, online learning")
    
    print("\n📊 Mini-Batch Gradient Descent:")
    print("   ✅ Good balance of speed and stability")
    print("   ✅ Works well with vectorization")
    print("   ✅ Moderate memory requirements")
    print("   → Best for: Most practical applications (most popular! )")
    
    print("\n" + "="*70)
    print("KEY INSIGHT:")
    print("All three methods achieve similar final R² scores (~0.89-0.90)")
    print("The choice depends on dataset size and computational constraints!")
    print("="*70)
    
    return {
        'batch':  model_batch,
        'sgd': model_sgd,
        'mini_batch': model_mini
    }, results


if __name__ == "__main__":
    models, results = main()
