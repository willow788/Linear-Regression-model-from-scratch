"""
Complete training script with visualizations for Version 3
"""

import numpy as np
from data_preprocessing import load_and_split_data
from linear_regression import LinearRegression
from metrics import evaluate_model, compare_train_test_performance
from visualization import (plot_train_test_loss, plot_train_test_predictions,
                           plot_train_test_residuals, plot_metrics_comparison)


def main():
    """Train, evaluate, and visualize with train/test split."""
    
    print("="*60)
    print(" " * 15 + "VERSION 3")
    print(" " * 5 + "Full Pipeline with Train/Test Split")
    print("="*60)
    
    # Load and split data
    print("\n📁 Loading and splitting data...")
    X_train, X_test, y_train, y_test, X_mean, X_std = load_and_split_data(
        'Advertising.csv'
    )
    
    print(f"\n✓ Training set:  {X_train.shape[0]} samples")
    print(f"✓ Test set: {X_test.shape[0]} samples")
    
    # Train model
    print("\n🚀 Training model...")
    model = LinearRegression(learn_rate=0.02, iter=80000)
    model.fit(X_train, y_train)
    
    # Predictions
    print("\n📊 Making predictions...")
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    
    # Evaluation
    print("\n" + "="*60)
    print("📈 EVALUATION")
    print("="*60)
    
    print("\nTraining Set:")
    train_metrics = evaluate_model(y_train, train_predictions, "Train")
    
    print("\nTest Set:")
    test_metrics = evaluate_model(y_test, test_predictions, "Test")
    
    compare_train_test_performance(train_metrics, test_metrics)
    
    # Visualizations
    print("\n" + "="*60)
    print("🎨 Generating visualizations...")
    print("="*60)
    
    print("\n1. Training loss convergence...")
    plot_train_test_loss(model)
    
    print("2. Train vs Test predictions...")
    plot_train_test_predictions(y_train, train_predictions, y_test, test_predictions)
    
    print("3. Train vs Test residuals...")
    plot_train_test_residuals(y_train, train_predictions, y_test, test_predictions)
    
    print("4. Metrics comparison...")
    plot_metrics_comparison(train_metrics, test_metrics)
    
    print("\n✅ Complete!")
    
    return model, train_metrics, test_metrics


if __name__ == "__main__":
    model, train_metrics, test_metrics = main()
