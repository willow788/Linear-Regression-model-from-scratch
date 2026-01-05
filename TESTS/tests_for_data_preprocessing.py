"""
Unit tests for data preprocessing module.
"""

import pytest
import numpy as np
import pandas as pd
from data_preprocessing import (
    load_and_preprocess_data,
    create_polynomial_features,
    normalize_features,
    normalize_target
)


class TestCreatePolynomialFeatures: 
    """Test polynomial feature creation."""
    
    def test_polynomial_feature_shape(self):
        """Test that polynomial features have correct shape."""
        X = np.random.randn(50, 3)
        X_poly = create_polynomial_features(X)
        
        # Should create 9 features:  3 original + 3 squared + 3 interactions
        assert X_poly.shape == (50, 9)
    
    def test_polynomial_feature_values(self):
        """Test that polynomial features have correct values."""
        X = np.array([[2, 3, 4]])
        X_poly = create_polynomial_features(X)
        
        expected = np.array([[
            2,      # TV
            3,      # Radio
            4,      # Newspaper
            4,      # TV^2
            9,      # Radio^2
            16,     # Newspaper^2
            6,      # TV * Radio
            8,      # TV * Newspaper
            12      # Radio * Newspaper
        ]])
        
        assert np.allclose(X_poly, expected)
    
    def test_polynomial_features_single_sample(self):
        """Test polynomial features with single sample."""
        X = np.array([[1, 1, 1]])
        X_poly = create_polynomial_features(X)
        
        assert X_poly.shape == (1, 9)
        assert np.allclose(X_poly, np.ones((1, 9)))
    
    def test_polynomial_features_zeros(self):
        """Test polynomial features with zero values."""
        X = np. array([[0, 0, 0]])
        X_poly = create_polynomial_features(X)
        
        assert X_poly.shape == (1, 9)
        assert np.allclose(X_poly, np.zeros((1, 9)))


class TestNormalizeFeatures:
    """Test feature normalization."""
    
    def test_normalize_features_mean(self):
        """Test that normalized features have zero mean."""
        X_train = np.random.randn(100, 3) * 10 + 5
        X_test = np.random.randn(20, 3) * 10 + 5
        
        X_train_norm, X_test_norm, X_mean, X_std = normalize_features(X_train, X_test)
        
        # Training data should have approximately zero mean
        assert np.allclose(X_train_norm.mean(axis=0), 0, atol=1e-10)
    
    def test_normalize_features_std(self):
        """Test that normalized features have unit standard deviation."""
        X_train = np.random.randn(100, 3) * 10 + 5
        X_test = np. random.randn(20, 3) * 10 + 5
        
        X_train_norm, X_test_norm, X_mean, X_std = normalize_features(X_train, X_test)
        
        # Training data should have approximately unit std
        assert np.allclose(X_train_norm.std(axis=0), 1, atol=1e-10)
    
    def test_normalize_features_returns_parameters(self):
        """Test that normalization returns mean and std."""
        X_train = np. random.randn(100, 3)
        X_test = np. random.randn(20, 3)
        
        X_train_norm, X_test_norm, X_mean, X_std = normalize_features(X_train, X_test)
        
        assert X_mean.shape == (3,)
        assert X_std. shape == (3,)
    
    def test_normalize_features_test_uses_train_stats(self):
        """Test that test set uses training set statistics."""
        X_train = np.ones((100, 3)) * 5
        X_test = np. ones((20, 3)) * 10
        
        X_train_norm, X_test_norm, X_mean, X_std = normalize_features(X_train, X_test)
        
        # Mean should be from training set (5)
        assert np.allclose(X_mean, 5)


class TestNormalizeTarget: 
    """Test target normalization."""
    
    def test_normalize_target_mean(self):
        """Test that normalized target has zero mean."""
        y_train = np.random.randn(100, 1) * 10 + 20
        y_test = np. random.randn(20, 1) * 10 + 20
        
        y_train_norm, y_test_norm, y_mean, y_std = normalize_target(y_train, y_test)
        
        # Training target should have approximately zero mean
        assert np.allclose(y_train_norm.mean(), 0, atol=1e-10)
    
    def test_normalize_target_std(self):
        """Test that normalized target has unit standard deviation."""
        y_train = np.random. randn(100, 1) * 10 + 20
        y_test = np.random.randn(20, 1) * 10 + 20
        
        y_train_norm, y_test_norm, y_mean, y_std = normalize_target(y_train, y_test)
        
        # Training target should have approximately unit std
        assert np.allclose(y_train_norm.std(), 1, atol=1e-10)
    
    def test_normalize_target_returns_scalars(self):
        """Test that target normalization returns scalar mean and std."""
        y_train = np.random.randn(100, 1)
        y_test = np.random.randn(20, 1)
        
        y_train_norm, y_test_norm, y_mean, y_std = normalize_target(y_train, y_test)
        
        assert isinstance(y_mean, (int, float, np.number))
        assert isinstance(y_std, (int, float, np.number))


class TestLoadAndPreprocessData:
    """Test complete data loading and preprocessing pipeline."""
    
    def test_load_and_preprocess_data_shapes(self, sample_csv):
        """Test that loaded data has correct shapes."""
        X_train, X_test, y_train, y_test = load_and_preprocess_data(
            str(sample_csv),
            test_size=0.2,
            random_state=42
        )
        
        # Total samples = 50, test_size = 0.2 -> 40 train, 10 test
        assert X_train.shape[0] == 40
        assert X_test.shape[0] == 10
        assert X_train.shape[1] == 9  # Polynomial features
        assert y_train.shape == (40, 1)
        assert y_test.shape == (10, 1)
    
    def test_load_and_preprocess_data_normalized(self, sample_csv):
        """Test that loaded data is normalized."""
        X_train, X_test, y_train, y_test = load_and_preprocess_data(
            str(sample_csv),
            test_size=0.2,
            random_state=42
        )
        
        # Training data should be normalized
        assert np.allclose(X_train. mean(axis=0), 0, atol=1e-10)
        assert np.allclose(X_train.std(axis=0), 1, atol=1e-10)
        assert np.allclose(y_train.mean(), 0, atol=1e-10)
        assert np.allclose(y_train.std(), 1, atol=1e-10)
    
    def test_load_and_preprocess_data_reproducible(self, sample_csv):
        """Test that data loading is reproducible with same random_state."""
        X_train1, X_test1, y_train1, y_test1 = load_and_preprocess_data(
            str(sample_csv),
            random_state=42
        )
        
        X_train2, X_test2, y_train2, y_test2 = load_and_preprocess_data(
            str(sample_csv),
            random_state=42
        )
        
        assert np.allclose(X_train1, X_train2)
        assert np.allclose(y_train1, y_train2)
    
    def test_load_and_preprocess_data_missing_file(self):
        """Test that loading non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_and_preprocess_data('nonexistent_file.csv')
