"""
Training script with comprehensive visualizations for Version 2.3
🎉 Breakthrough version with positive R² score!
"""

import numpy as np
from data_preprocessing import load_and_prepare_data, add_bias_term
from linear_regression import LinearRegression
from metrics import evaluate_model
from visualization import (plot_loss_convergence, plot_predictions_vs_actual, 
                           plot_residuals, plot_comprehensive_analysis)


def main():
    """Train, evaluate, and visualize the linear regression model."""
    
    print("="*60)
    print(" " * 10 + "VERSION 2.3 - BREAKTHROUGH VERSION!")
    print("="*60)
    
    # Load and prepare data
    print("\n📁 Loading and preparing data...")
    X, y = load_and_prepare_data('Advertising.csv')
    
    # Add bias term
    X_b = add_bias_term(X)
    print(f"\n✓ Feature shape with bias: {X_b.shape}")
    print(f"✓ Target shape: {y. shape}")
    print(f"✓ Target normalized - mean:  {y.mean():.6f}, std: {y.std():.6f}")
    
    # Initialize and train model
    print("\n" + "="*60)
    print("🚀 Training model...")
    print("="*60)
    model = LinearRegression(learn_rate=0.02, iter=80000)
    model.fit(X_b, y)
    
    # Make predictions
    print("\n📊 Making predictions...")
    predictions = model.predict(X_b)
    
    # Evaluate model
    print("\n" + "="*60)
    print("📈 Model Evaluation:")
    print("="*60)
    metrics = evaluate_model(y, predictions)
    
    # Generate visualizations
    print("\n" + "="*60)
    print("🎨 Generating visualizations...")
    print("="*60)
    
    print("\n1. Loss convergence plot...")
    plot_loss_convergence(model.loss_history, 
                         "Version 2.3 - Loss Convergence (Breakthrough! )")
    
    print("2. Predictions vs Actual plot...")
    plot_predictions_vs_actual(y, predictions)
    
    print("3. Residual analysis plots...")
    plot_residuals(y, predictions)
    
    print("4. Comprehensive analysis dashboard...")
    plot_comprehensive_analysis(model, X_b, y)
    
    print("\n" + "="*60)
    print("✅ Analysis complete!")
    print("="*60)
    
    return model, metrics


if __name__ == "__main__":
    model, metrics = main()
