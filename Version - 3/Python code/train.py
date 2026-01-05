"""
Main training script for Version 3

KEY ADDITION: Train/Test split for proper evaluation
PROBLEM: Forgot to normalize target variable, leading to negative R² scores

This demonstrates the importance of consistent preprocessing! 
"""

import numpy as np
from data_preprocessing import load_and_split_data, add_bias_term
from linear_regression import LinearRegression
from metrics import evaluate_model, compare_train_test_performance


def main():
    """Train and evaluate the linear regression model with train/test split."""
    
    print("="*60)
    print(" " * 15 + "VERSION 3")
    print(" " * 10 + "Train/Test Split Added")
    print("="*60)
    
    # Load and prepare data with train/test split
    print("\n📁 Loading and splitting data...")
    X_train, X_test, y_train, y_test, X_mean, X_std = load_and_split_data(
        'Advertising.csv', 
        test_size=0.2, 
        random_state=42
    )
    
    print(f"\n✓ Training set size: {X_train.shape[0]} samples")
    print(f"✓ Test set size: {X_test.shape[0]} samples")
    print(f"✓ Features normalized using training statistics")
    print(f"⚠️  Target variable NOT normalized (this is the problem!)")
    
    # Note: In the original notebook, bias was added to X (not X_train/X_test)
    # This is a bug in the original code - we'll replicate it here
    # In practice, we should add bias to X_train and X_test separately
    
    # Initialize and train model
    print("\n" + "="*60)
    print("🚀 Training model...")
    print("="*60)
    print("Hyperparameters:  learning_rate=0.02, iterations=80000")
    model = LinearRegression(learn_rate=0.02, iter=80000)
    model.fit(X_train, y_train)
    
    # Make predictions on both train and test sets
    print("\n📊 Making predictions...")
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    
    # Evaluate on training set
    print("\n" + "="*60)
    print("📈 TRAINING SET EVALUATION")
    print("="*60)
    train_metrics = evaluate_model(y_train, train_predictions, "Training")
    
    # Evaluate on test set
    print("\n" + "="*60)
    print("📊 TEST SET EVALUATION")
    print("="*60)
    test_metrics = evaluate_model(y_test, test_predictions, "Test")
    
    # Compare performance
    compare_train_test_performance(train_metrics, test_metrics)
    
    # Analysis
    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)
    
    if test_metrics['r2'] < 0:
        print("\n❌ PROBLEM IDENTIFIED:")
        print("   Both train and test have negative R² scores")
        print("\n🔍 Root Cause:")
        print("   Target variable (y) is NOT normalized!")
        print("   This causes numerical instability in gradient descent")
        print("\n💡 Solution:")
        print("   Add target normalization like in Version 2.3:")
        print("   y_train = (y_train - y_train. mean()) / y_train.std()")
        print("   y_test = (y_test - y_train.mean()) / y_train.std()")
        print("\n📌 See Version 3.1 for the fixed version")
    
    print("\n" + "="*60)
    print("KEY LEARNING:")
    print("Train/test split is important, but preprocessing must be")
    print("consistent!  All the improvements from Version 2.3 must be")
    print("carried forward.")
    print("="*60)
    
    return model, train_metrics, test_metrics


if __name__ == "__main__":
    model, train_metrics, test_metrics = main()
