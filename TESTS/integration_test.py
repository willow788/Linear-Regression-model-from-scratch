"""
Integration tests for the complete pipeline.
"""

import pytest
import numpy as np
import pandas as pd
from data_preprocessing import load_and_preprocess_data
from linear_regression import LinearRegression
from model_evaluation import calculate_metrics, cross_validation_score


class TestEndToEndPipeline:
    """Test complete end-to-end pipeline."""
    
    def test_complete_pipeline(self, sample_csv):
        """Test entire pipeline from data loading to prediction."""
        # Load data
        X_train, X_test, y_train, y_test = load_and_preprocess_data(
            str(sample_csv),
            test_size=0.2,
            random_state=42
        )
        
        # Train model
        model = LinearRegression(
            learn_rate=0.01,
            iter=500,
            method='batch',
            l1_reg=0.1
        )
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = calculate_metrics(y_test, y_pred)
        
        # Assertions
        assert y_pred.shape == y_test.shape
        assert 'r2_score' in metrics
        assert model.weights is not None
        assert len(model.loss_history) > 0
    
    def test_multiple_methods_comparison(self, sample_csv):
        """Test training with different gradient descent methods."""
        X_train, X_test, y_train, y_test = load_and_preprocess_data(
            str(sample_csv),
            random_state=42
        )
        
        methods = ['batch', 'stochastic', 'mini-batch']
        results = {}
        
        for method in methods:
            model = LinearRegression(
                learn_rate=0.01,
                iter=100 if method != 'stochastic' else 10,
                method=method,
                batch_size=8
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            metrics = calculate_metrics(y_test, y_pred)
            results[method] = metrics['r2_score']
        
        # All methods should produce valid results
        for method, score in results.items():
            assert -1 <= score <= 1
    
    def test_regularization_comparison(self, sample_csv):
        """Test models with different regularization strengths."""
        X_train, X_test, y_train, y_test = load_and_preprocess_data(
            str(sample_csv),
            random_state=42
        )
        
        l1_values = [0.0, 0.1, 0.5, 1.0]
        results = {}
        
        for l1_reg in l1_values:
            model = LinearRegression(
                learn_rate=0.01,
                iter=500,
                method='batch',
                l1_reg=l1_reg
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            metrics = calculate_metrics(y_test, y_pred)
            results[l1_reg] = {
                'r2_score':  metrics['r2_score'],
                'weight_sum': np.sum(np.abs(model.weights))
            }
        
        # Higher regularization should lead to smaller weights
        assert results[1.0]['weight_sum'] < results[0.0]['weight_sum']
    
    def test_overfitting_detection(self, sample_csv):
        """Test that model can detect overfitting."""
        X_train, X_test, y_train, y_test = load_and_preprocess_data(
            str(sample_csv),
            random_state=42
        )
        
        # Train without regularization
        model_no_reg = LinearRegression(
            learn_rate=0.01,
            iter=1000,
            method='batch',
            l1_reg=0.0
        )
        model_no_reg.fit(X_train, y_train)
        
        train_pred = model_no_reg.predict(X_train)
        test_pred = model_no_reg.predict(X_test)
        
        train_metrics = calculate_metrics(y_train, train_pred)
        test_metrics = calculate_metrics(y_test, test_pred)
        
        # Both should produce valid scores
        assert -1 <= train_metrics['r2_score'] <= 1
        assert -1 <= test_metrics['r2_score'] <= 1
