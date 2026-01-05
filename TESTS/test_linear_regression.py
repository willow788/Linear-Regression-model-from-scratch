"""
Unit tests for the Linear Regression model.
"""

import pytest
import numpy as np
from linear_regression import LinearRegression


class TestLinearRegressionInitialization:
    """Test LinearRegression initialization."""
    
    def test_default_initialization(self):
        """Test model initialization with default parameters."""
        model = LinearRegression()
        
        assert model.lr == 1e-7
        assert model.iter == 50000
        assert model.method == 'batch'
        assert model.batch_size == 32
        assert model.l1_reg == 0.0
        assert model.early_stopping == False
        assert model.patience == 1000
        assert model.weights is None
        assert model.loss_history == []
    
    def test_custom_initialization(self):
        """Test model initialization with custom parameters."""
        model = LinearRegression(
            learn_rate=0.01,
            iter=1000,
            method='stochastic',
            batch_size=16,
            l1_reg=0.1,
            early_stopping=True,
            patience=500
        )
        
        assert model.lr == 0.01
        assert model.iter == 1000
        assert model.method == 'stochastic'
        assert model.batch_size == 16
        assert model.l1_reg == 0.1
        assert model.early_stopping == True
        assert model.patience == 500


class TestLinearRegressionFit:
    """Test LinearRegression fitting methods."""
    
    def test_batch_gradient_descent_fit(self, normalized_data):
        """Test model fitting with batch gradient descent."""
        model = LinearRegression(
            learn_rate=0.01,
            iter=100,
            method='batch'
        )
        
        model.fit(normalized_data['X_normalized'], normalized_data['y_normalized'])
        
        assert model.weights is not None
        assert model.weights.shape == (normalized_data['X_normalized'].shape[1], 1)
        assert isinstance(model.bias, (int, float, np.number))
        assert len(model.loss_history) > 0
    
    def test_stochastic_gradient_descent_fit(self, normalized_data):
        """Test model fitting with stochastic gradient descent."""
        model = LinearRegression(
            learn_rate=0.01,
            iter=10,
            method='stochastic'
        )
        
        model.fit(normalized_data['X_normalized'], normalized_data['y_normalized'])
        
        assert model.weights is not None
        assert model.weights.shape == (normalized_data['X_normalized']. shape[1], 1)
        assert len(model.loss_history) == 10
    
    def test_mini_batch_gradient_descent_fit(self, normalized_data):
        """Test model fitting with mini-batch gradient descent."""
        model = LinearRegression(
            learn_rate=0.01,
            iter=50,
            method='mini-batch',
            batch_size=16
        )
        
        model.fit(normalized_data['X_normalized'], normalized_data['y_normalized'])
        
        assert model.weights is not None
        assert model.weights.shape == (normalized_data['X_normalized']. shape[1], 1)
        assert len(model.loss_history) == 50
    
    def test_loss_decreases(self, normalized_data):
        """Test that loss generally decreases during training."""
        model = LinearRegression(
            learn_rate=0.01,
            iter=1000,
            method='batch'
        )
        
        model.fit(normalized_data['X_normalized'], normalized_data['y_normalized'])
        
        # Loss at end should be less than loss at beginning
        assert model.loss_history[-1] < model. loss_history[0]
    
    def test_early_stopping(self, normalized_data):
        """Test early stopping functionality."""
        model = LinearRegression(
            learn_rate=0.01,
            iter=10000,
            method='batch',
            early_stopping=True,
            patience=100
        )
        
        model.fit(normalized_data['X_normalized'], normalized_data['y_normalized'])
        
        # Should stop before max iterations
        assert len(model.loss_history) < 10000
    
    def test_l1_regularization_effect(self, normalized_data):
        """Test that L1 regularization reduces weight magnitudes."""
        # Model without regularization
        model_no_reg = LinearRegression(
            learn_rate=0.01,
            iter=500,
            method='batch',
            l1_reg=0.0
        )
        model_no_reg.fit(normalized_data['X_normalized'], normalized_data['y_normalized'])
        
        # Model with regularization
        model_with_reg = LinearRegression(
            learn_rate=0.01,
            iter=500,
            method='batch',
            l1_reg=1.0
        )
        model_with_reg.fit(normalized_data['X_normalized'], normalized_data['y_normalized'])
        
        # Regularized model should have smaller weights
        assert np.sum(np.abs(model_with_reg.weights)) < np.sum(np.abs(model_no_reg.weights))


class TestLinearRegressionPredict:
    """Test LinearRegression prediction."""
    
    def test_predict_shape(self, normalized_data):
        """Test that predictions have correct shape."""
        model = LinearRegression(learn_rate=0.01, iter=100)
        model.fit(normalized_data['X_normalized'], normalized_data['y_normalized'])
        
        predictions = model.predict(normalized_data['X_normalized'])
        
        assert predictions.shape == normalized_data['y_normalized'].shape
    
    def test_predict_values(self, sample_data):
        """Test that predictions are reasonable for known data."""
        # Create simple case where we know the answer
        X = np.array([[1], [2], [3], [4]])
        y = np.array([[2], [4], [6], [8]])  # y = 2*x
        
        model = LinearRegression(learn_rate=0.01, iter=5000, method='batch')
        model.fit(X, y)
        
        predictions = model.predict(X)
        
        # Predictions should be close to actual values
        assert np.allclose(predictions, y, rtol=0.1)
    
    def test_predict_without_fit_raises_error(self, normalized_data):
        """Test that predicting before fitting raises an error."""
        model = LinearRegression()
        
        with pytest.raises((ValueError, AttributeError)):
            model.predict(normalized_data['X_normalized'])


class TestLinearRegressionEdgeCases: 
    """Test edge cases and error handling."""
    
    def test_single_sample(self):
        """Test model with single sample."""
        X = np.array([[1, 2, 3]])
        y = np.array([[5]])
        
        model = LinearRegression(learn_rate=0.01, iter=10)
        model.fit(X, y)
        
        assert model.weights is not None
    
    def test_single_feature(self):
        """Test model with single feature."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([[2], [4], [6], [8], [10]])
        
        model = LinearRegression(learn_rate=0.01, iter=100)
        model.fit(X, y)
        
        assert model.weights. shape == (1, 1)
    
    def test_zero_learning_rate(self, normalized_data):
        """Test that zero learning rate doesn't update weights."""
        model = LinearRegression(learn_rate=0.0, iter=100)
        model.fit(normalized_data['X_normalized'], normalized_data['y_normalized'])
        
        # Weights should remain zero
        assert np.allclose(model.weights, 0)
    
    def test_large_learning_rate_divergence(self, normalized_data):
        """Test that very large learning rate causes divergence."""
        model = LinearRegression(learn_rate=10.0, iter=100, method='batch')
        model.fit(normalized_data['X_normalized'], normalized_data['y_normalized'])
        
        # Loss should increase (diverge)
        assert model.loss_history[-1] > model.loss_history[0]
