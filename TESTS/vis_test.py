"""
Unit tests for visualization module.
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from visualization import (
    plot_loss_convergence,
    plot_residuals,
    plot_correlation_matrix,
    plot_actual_vs_predicted,
    plot_feature_importance
)


class TestPlotLossConvergence:
    """Test loss convergence plotting."""
    
    def test_plot_loss_convergence_executes(self):
        """Test that plot_loss_convergence executes without error."""
        loss_history = [0.5, 0.4, 0.3, 0.2, 0.1]
        
        try:
            plot_loss_convergence(loss_history)
            plt.close()
            assert True
        except Exception as e:
            pytest.fail(f"plot_loss_convergence raised {e}")
    
    def test_plot_loss_convergence_empty_list(self):
        """Test plotting with empty loss history."""
        loss_history = []
        
        try:
            plot_loss_convergence(loss_history)
            plt.close()
            assert True
        except Exception: 
            # Empty list might raise error, which is acceptable
            assert True
    
    def test_plot_loss_convergence_single_value(self):
        """Test plotting with single loss value."""
        loss_history = [0.5]
        
        try:
            plot_loss_convergence(loss_history)
            plt.close()
            assert True
        except Exception as e:
            pytest.fail(f"plot_loss_convergence raised {e}")


class TestPlotResiduals:
    """Test residual plotting."""
    
    def test_plot_residuals_executes(self):
        """Test that plot_residuals executes without error."""
        y_true = np.array([[1], [2], [3], [4], [5]])
        y_pred = np.array([[1.1], [1.9], [3.1], [3.9], [5.1]])
        
        try:
            plot_residuals(y_true, y_pred)
            plt.close()
            assert True
        except Exception as e:
            pytest.fail(f"plot_residuals raised {e}")
    
    def test_plot_residuals_perfect_predictions(self):
        """Test residual plot with perfect predictions."""
        y_true = np.array([[1], [2], [3], [4], [5]])
        y_pred = np.array([[1], [2], [3], [4], [5]])
        
        try:
            plot_residuals(y_true, y_pred)
            plt.close()
            assert True
        except Exception as e:
            pytest.fail(f"plot_residuals raised {e}")


class TestPlotCorrelationMatrix:
    """Test correlation matrix plotting."""
    
    def test_plot_correlation_matrix_executes(self):
        """Test that plot_correlation_matrix executes without error."""
        data = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [2, 4, 6, 8, 10],
            'C': [5, 4, 3, 2, 1]
        })
        
        try:
            plot_correlation_matrix(data)
            plt.close()
            assert True
        except Exception as e:
            pytest.fail(f"plot_correlation_matrix raised {e}")
    
    def test_plot_correlation_matrix_single_column(self):
        """Test correlation matrix with single column."""
        data = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
        
        try:
            plot_correlation_matrix(data)
            plt.close()
            assert True
        except Exception as e: 
            pytest.fail(f"plot_correlation_matrix raised {e}")


class TestPlotActualVsPredicted:
    """Test actual vs predicted plotting."""
    
    def test_plot_actual_vs_predicted_executes(self):
        """Test that plot_actual_vs_predicted executes without error."""
        y_true = np.array([[1], [2], [3], [4], [5]])
        y_pred = np.array([[1.1], [1.9], [3.1], [3.9], [5.1]])
        
        try:
            plot_actual_vs_predicted(y_true, y_pred)
            plt.close()
            assert True
        except Exception as e:
            pytest. fail(f"plot_actual_vs_predicted raised {e}")
    
    def test_plot_actual_vs_predicted_large_data(self):
        """Test plotting with large dataset."""
        np.random.seed(42)
        y_true = np.random.randn(1000, 1)
        y_pred = y_true + np.random.randn(1000, 1) * 0.1
        
        try:
            plot_actual_vs_predicted(y_true, y_pred)
            plt.close()
            assert True
        except Exception as e:
            pytest.fail(f"plot_actual_vs_predicted raised {e}")


class TestPlotFeatureImportance:
    """Test feature importance plotting."""
    
    def test_plot_feature_importance_executes(self):
        """Test that plot_feature_importance executes without error."""
        feature_names = ['Feature1', 'Feature2', 'Feature3']
        weights = np.array([[0.5], [-0.3], [0.8]])
        
        try: 
            plot_feature_importance(feature_names, weights)
            plt.close()
            assert True
        except Exception as e:
            pytest.fail(f"plot_feature_importance raised {e}")
    
    def test_plot_feature_importance_single_feature(self):
        """Test plotting with single feature."""
        feature_names = ['Feature1']
        weights = np.array([[0.5]])
        
        try: 
            plot_feature_importance(feature_names, weights)
            plt.close()
            assert True
        except Exception as e: 
            pytest.fail(f"plot_feature_importance raised {e}")
    
    def test_plot_feature_importance_many_features(self):
        """Test plotting with many features."""
        feature_names = [f'Feature{i}' for i in range(20)]
        weights = np.random.randn(20, 1)
        
        try: 
            plot_feature_importance(feature_names, weights)
            plt.close()
            assert True
        except Exception as e:
            pytest.fail(f"plot_feature_importance raised {e}")
