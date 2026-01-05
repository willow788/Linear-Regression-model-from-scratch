"""
Unit tests for model evaluation module.
"""

import pytest
import numpy as np
from model_evaluation import (
    calculate_metrics,
    print_metrics,
    cross_validation_score
)


class TestCalculateMetrics:
    """Test metric calculation functions."""
    
    def test_calculate_metrics_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        y_true = np.array([[1], [2], [3], [4], [5]])
        y_pred = np.array([[1], [2], [3], [4], [5]])
        
        metrics = calculate_metrics(y_true, y_pred)
        
        assert metrics['mse'] == 0
        assert metrics['rmse'] == 0
        assert metrics['mae'] == 0
        assert np.isclose(metrics['r2_score'], 1.0)
    
    def test_calculate_metrics_values(self):
        """Test metric calculations with known values."""
        y_true = np.array([[10], [20], [30], [40], [50]])
        y_pred = np.array([[12], [19], [31], [38], [51]])
        
        metrics = calculate_metrics(y_true, y_pred)
        
        # MSE = mean of [4, 1, 1, 4, 1] = 2.2
        assert np.isclose(metrics['mse'], 2.2)
        # RMSE = sqrt(2.2) ≈ 1.483
        assert np.isclose(metrics['rmse'], np.sqrt(2.2))
        # MAE = mean of [2, 1, 1, 2, 1] = 1.4
        assert np.isclose(metrics['mae'], 1.4)
        # R² should be high (good predictions)
        assert metrics['r2_score'] > 0.9
    
    def test_calculate_metrics_worst_case(self):
        """Test metrics with predictions equal to mean."""
        y_true = np.array([[10], [20], [30], [40], [50]])
        y_pred = np.array([[30], [30], [30], [30], [30]])  # Always predict mean
        
        metrics = calculate_metrics(y_true, y_pred)
        
        # R² should be 0 when predicting mean
        assert np.isclose(metrics['r2_score'], 0.0, atol=1e-10)
    
    def test_calculate_metrics_negative_r2(self):
        """Test that R² can be negative for bad predictions."""
        y_true = np.array([[10], [20], [30], [40], [50]])
        y_pred = np.array([[50], [40], [30], [20], [10]])  # Opposite predictions
        
        metrics = calculate_metrics(y_true, y_pred)
        
        # R² should be negative (worse than predicting mean)
        assert metrics['r2_score'] < 0
    
    def test_calculate_metrics_return_type(self):
        """Test that calculate_metrics returns dictionary."""
        y_true = np.array([[1], [2], [3]])
        y_pred = np. array([[1.1], [2.1], [2.9]])
        
        metrics = calculate_metrics(y_true, y_pred)
        
        assert isinstance(metrics, dict)
        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2_score' in metrics


class TestPrintMetrics: 
    """Test metric printing function."""
    
    def test_print_metrics_executes(self, capsys):
        """Test that print_metrics executes without error."""
        metrics = {
            'mse': 2.5,
            'rmse': 1.58,
            'mae': 1.2,
            'r2_score': 0.95
        }
        
        print_metrics(metrics, 'Test')
        captured = capsys.readouterr()
        
        assert 'Test' in captured.out
        assert '2.5' in captured.out or '2.500000' in captured.out
        assert '0.95' in captured.out or '0.950000' in captured.out
    
    def test_print_metrics_no_dataset_name(self, capsys):
        """Test print_metrics without dataset name."""
        metrics = {
            'mse': 1.0,
            'rmse': 1.0,
            'mae':  1.0,
            'r2_score': 0.9
        }
        
        print_metrics(metrics)
        captured = capsys.readouterr()
        
        assert 'Mean Squared Error' in captured.out


class TestCrossValidationScore: 
    """Test cross-validation function."""
    
    def test_cross_validation_score_returns_float(self, sample_data):
        """Test that cross-validation returns a float."""
        score = cross_validation_score(
            sample_data['X'],
            sample_data['y'],
            k=3,
            iter=100
        )
        
        assert isinstance(score, (float, np.floating))
    
    def test_cross_validation_score_range(self, sample_data):
        """Test that cross-validation score is in reasonable range."""
        score = cross_validation_score(
            sample_data['X'],
            sample_data['y'],
            k=3,
            iter=500,
            learn_rate=0.01
        )
        
        # R² should be between -1 and 1 (typically positive for good models)
        assert -1 <= score <= 1
    
    def test_cross_validation_score_k_folds(self, sample_data):
        """Test cross-validation with different k values."""
        score_3_fold = cross_validation_score(
            sample_data['X'],
            sample_data['y'],
            k=3,
            iter=100
        )
        
        score_5_fold = cross_validation_score(
            sample_data['X'],
            sample_data['y'],
            k=5,
            iter=100
        )
        
        # Both should return valid scores
        assert isinstance(score_3_fold, (float, np.floating))
        assert isinstance(score_5_fold, (float, np.floating))
    
    def test_cross_validation_with_regularization(self, sample_data):
        """Test cross-validation with L1 regularization."""
        score = cross_validation_score(
            sample_data['X'],
            sample_data['y'],
            k=3,
            iter=500,
            l1_reg=0.1
        )
        
        assert isinstance(score, (float, np.floating))
    
    def test_cross_validation_reproducible(self, sample_data):
        """Test that cross-validation is reproducible."""
        # Cross-validation uses fixed random_state internally
        score1 = cross_validation_score(
            sample_data['X'],
            sample_data['y'],
            k=3,
            iter=100
        )
        
        score2 = cross_validation_score(
            sample_data['X'],
            sample_data['y'],
            k=3,
            iter=100
        )
        
        # Should get same score (within numerical precision)
        assert np.isclose(score1, score2)
