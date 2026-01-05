"""
Linear Regression model implemented from scratch.

This module contains a LinearRegression class that supports: 
- Batch, Stochastic, and Mini-batch gradient descent
- L1 regularization (Lasso)
- Early stopping
"""

import numpy as np


class LinearRegression:
    """
    Linear Regression model with various gradient descent methods.
    
    Parameters:
    -----------
    learn_rate : float, default=1e-7
        Learning rate for gradient descent
    iter :  int, default=50000
        Maximum number of iterations
    method : str, default='batch'
        Gradient descent method:  'batch', 'stochastic', or 'mini-batch'
    batch_size : int, default=32
        Batch size for mini-batch gradient descent
    l1_reg : float, default=0.0
        L1 regularization parameter (Lasso)
    early_stopping : bool, default=False
        Whether to use early stopping
    patience : int, default=1000
        Number of iterations to wait before stopping if loss doesn't improve
        
    Attributes:
    -----------
    weights : np.ndarray
        Model weights/coefficients
    bias : float
        Model bias/intercept
    loss_history : list
        History of loss values during training
    """
    
    def __init__(self, learn_rate=1e-7, iter=50000, method='batch', 
                 batch_size=32, l1_reg=0.0, early_stopping=False, patience=1000):
        self.l1_reg = l1_reg
        self.method = method
        self.batch_size = batch_size
        self.lr = float(learn_rate)
        self.iter = int(iter)
        self.weights = None
        self.early_stopping = early_stopping
        self.patience = patience
        self.loss_history = []

    def fit(self, X, y):
        """
        Fit the linear regression model to the training data.
        
        Parameters:
        -----------
        X :  np.ndarray
            Training feature matrix of shape (n_samples, n_features)
        y : np.ndarray
            Training target values of shape (n_samples, 1)
        """
        m, n = X.shape
        # m -- number of samples (rows)
        # n -- number of features (columns)

        # Initialize weights to zero including bias term
        self.weights = np. zeros((n, 1))
        self.bias = 0

        for iteration in range(self.iter):
            if self.method == 'batch':
                loss = self._batch_gradient_descent(X, y, m)
                
            elif self.method == 'stochastic':
                loss = self._stochastic_gradient_descent(X, y, m)
                
            elif self.method == 'mini-batch':
                loss = self._mini_batch_gradient_descent(X, y, m)
        
            self.loss_history. append(loss)

            # Early stopping implementation
            if self.early_stopping and iteration > 0:
                if loss > self.loss_history[-2]:
                    if iteration - self.loss_history. index(min(self.loss_history)) >= self.patience:
                        print(f"Early stopping at iteration {iteration} with loss {loss}")
                        break

            if iteration % 5000 == 0:
                print(f"Loss at iteration {iteration}: {loss}")

    def _batch_gradient_descent(self, X, y, m):
        """
        Perform one iteration of batch gradient descent.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target values
        m : int
            Number of samples
            
        Returns:
        --------
        loss : float
            Current loss value
        """
        y_pred = X @ self.weights + self.bias
        error = y_pred - y
        loss = (1/(2*m)) * np.sum(error ** 2) + (self.l1_reg/2) * np.sum(np.abs(self.weights))

        grad_w = (1/m) * (X.T @ error) + self.l1_reg * np.sign(self.weights)
        grad_b = (1/m) * np.sum(error)

        self.weights -= self.lr * grad_w
        self.bias -= self.lr * grad_b
        
        return loss

    def _stochastic_gradient_descent(self, X, y, m):
        """
        Perform one epoch of stochastic gradient descent. 
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y :  np.ndarray
            Target values
        m : int
            Number of samples
            
        Returns: 
        --------
        loss :  float
            Loss after one epoch
        """
        for i in range(m):
            xi = X[i]. reshape(1, -1)
            yi = y[i].reshape(1, -1)
            y_pred = xi @ self.weights + self.bias
            error = y_pred - yi

            gradient_w = xi.T @ error
            gradient_b = error. item()

            self.weights -= self.lr * gradient_w
            self.bias -= self.lr * gradient_b

        # Calculate loss after one epoch
        y_pred = X @ self.weights + self.bias
        loss = (1/(2*m)) * np.sum((y_pred - y) ** 2) + (self.l1_reg/2) * np.sum(np.abs(self. weights))
        
        return loss

    def _mini_batch_gradient_descent(self, X, y, m):
        """
        Perform one epoch of mini-batch gradient descent. 
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y :  np.ndarray
            Target values
        m : int
            Number of samples
            
        Returns: 
        --------
        loss :  float
            Loss after one epoch
        """
        perm = np.random.permutation(m)
        X_shuffle = X[perm]
        y_shuffle = y[perm]

        for start in range(0, m, self.batch_size):
            end = start + self.batch_size
            xb = X_shuffle[start:end]
            yb = y_shuffle[start:end]

            y_pred = xb @ self.weights + self.bias
            error = y_pred - yb

            gradient_w = (1/self.batch_size) * (xb.T @ error)
            gradient_b = (1/self.batch_size) * np.sum(error)

            self.weights -= self.lr * gradient_w
            self.bias -= self.lr * gradient_b
        
        # Calculate loss after one epoch
        y_pred = X @ self.weights + self.bias
        error = y_pred - y
        loss = (1/(2*m)) * np.sum(error ** 2) + (self.l1_reg/2) * np.sum(np.abs(self.weights))
        
        return loss

    def predict(self, X):
        """
        Make predictions using the trained model.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix for prediction
            
        Returns: 
        --------
        predictions : np.ndarray
            Predicted target values
        """
        return X @ self.weights + self. bias
