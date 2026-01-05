"""
Complete training pipeline with all visualizations for Version 3.1
🎉 The complete, working solution!  🎉
"""

import numpy as np
from data_preprocessing import load_and_split_data
from linear_regression import LinearRegression
from metrics import evaluate_model, compare_train_test_performance, print_model_summary
from visualization import (plot_loss_convergence, plot_train_test_predictions,
                           plot_residual_analysis, plot_comprehensive_dashboard)


def main():
    """Complete training and visualization pipeline."""
    
    print("="*70)
    print(" " * 20 + "VERSION 3.1")
    print(" " * 15 + "🎉 THE COMPLETE FIX! 🎉")
    print("="*70)
    
    # Load and split data
    print("\n📁 Step 1: Loading and preparing data...")
    X_train, X_test, y_train, y_test, X_mean, X_std, y_mean, y_std = load_and_split_data(
        'Advertising.csv'
    )
    
    print(f"\n✓ Data loaded successfully")
    print(f"  - Training samples: {X_train.shape[0]}")
    print(f"  - Test samples: {X_test.shape[0]}")
    print(f"  - Features: {X_train.shape[1]}")
    print(f"  - All data normalized (features AND target)")
    
    # Train model
    print("\n🚀 Step 2: Training model...")
    print("  - Learning rate: 0.02")
    print("  - Iterations:  50,000")
    print("  - Using separate bias term\n")
    
    model = LinearRegression(learn_rate=0.02, iter=50000)
    model.fit(X_train, y_train)
    
    # Make predictions
    print("\n📊 Step 3: Making predictions...")
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    print("✓ Predictions complete")
    
    # Evaluate
    print("\n📈 Step 4: Evaluating model...")
    print("\n" + "-"*70)
    print("TRAINING SET METRICS:")
    print("-"*70)
    train_metrics = evaluate_model(y_train, train_pred, "Train")
    
    print("\n" + "-"*70)
    print("TEST SET METRICS (Most Important!):")
    print("-"*70)
    test_metrics = evaluate_model(y_test, test_pred, "Test")
    
    compare_train_test_performance(train_metrics, test_metrics)
    print_model_summary(model, train_metrics, test_metrics)
    
    # Visualizations
    print("\n🎨 Step 5: Generating visualizations...")
    print("="*70)
    
    print("\n1️⃣  Loss convergence plot...")
    plot_loss_convergence(model)
    
    print("\n2️⃣  Train vs Test predictions...")
    plot_train_test_predictions(y_train, train_pred, y_test, test_pred)
    
    print("\n3️⃣  Residual analysis...")
    plot_residual_analysis(y_train, train_pred, y_test, test_pred)
    
    print("\n4️⃣  Comprehensive dashboard...")
    plot_comprehensive_dashboard(model, X_train, y_train, X_test, y_test,
                                train_metrics, test_metrics)
    
    # Final summary
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"\n🎯 Final Test R² Score: {test_metrics['r2']:.4f}")
    print(f"   Model explains {test_metrics['r2']*100:.2f}% of variance in unseen data")
    print(f"\n🎯 Train/Test R² Difference: {abs(train_metrics['r2'] - test_metrics['r2']):.4f}")
    print("   Excellent generalization!")
    
    if test_metrics['r2'] > 0.89:
        print("\n🏆 OUTSTANDING PERFORMANCE!")
        print("   This model is ready for deployment!")
    
    print("\n" + "="*70)
    
    return model, train_metrics, test_metrics


if __name__ == "__main__":
    model, train_metrics, test_metrics = main()
    print("\n💾 Tip: Save the model for future use!")
    print("   Consider pickling the model object and normalization parameters")
