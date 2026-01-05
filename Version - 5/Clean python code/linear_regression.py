"""
Linear Regression implementation from scratch with multiple gradient descent methods.
"""

import numpy as np


class LinearRegression:
    """
    Linear Regression model with support for batch, stochastic, and mini-batch gradient descent.
    
    Parameters:
    -----------
    learn_rate : float, default=1e-7
        Learning rate for gradient descent
    iter : int, default=50000
        Number of iterations for training
    method : str, default='batch'
        Gradient descent method:  'batch', 'stochastic', or 'mini-batch'
    batch_size : int, default=32
        Batch size for mini-batch gradient descent
    l2_reg : float, default=0.0
        L2 regularization parameter
    """
    
    def __init__(self, learn_rate=1e-7, iter=50000, method='batch', batch_size=32, l2_reg=0.0):
        self.l2_reg = l2_reg
        self.method = method
        self.batch_size = batch_size
        self.lr = float(learn_rate)
        self.iter = int(iter)
        self.weights = None
        self.bias = 0
        self.loss_history = []
    
    def fit(self, X, y):
        """
        Fit the linear regression model.
        
        Parameters:
        -----------
        X :  numpy.ndarray
            Training features of shape (m, n)
        y : numpy.ndarray
            Training targets of shape (m, 1)
        """
        m, n = X.shape
        # m -- number of samples
        # n -- number of features
        
        # Initialize weights to zero
        self.weights = np.zeros((n, 1))
        self.bias = 0
        
        for iteration in range(self.iter):
            if self.method == 'batch': 
                loss = self._batch_gradient_descent(X, y, m)
                
            elif self.method == 'stochastic':
                loss = self._stochastic_gradient_descent(X, y, m)
                
            elif self.method == 'mini-batch':
                loss = self._mini_batch_gradient_descent(X, y, m)
            
            else:
                raise ValueError(f"Unknown method: {self.method}. Use 'batch', 'stochastic', or 'mini-batch'.")
            
            self.loss_history.append(loss)
            
            if iteration % 5000 == 0:
                print(f"Loss at iteration {iteration}: {loss}")
    
    def _batch_gradient_descent(self, X, y, m):
        """Perform one iteration of batch gradient descent."""
        y_pred = X @ self.weights + self.bias
        error = y_pred - y
        loss = (1/(2*m)) * np.sum(error ** 2) + (self.l2_reg/2) * np.sum(self.weights ** 2)
        
        grad_w = (1/m) * (X.T @ error) + self.l2_reg * self. weights
        grad_b = (1/m) * np.sum(error)
        
        self.weights -= self.lr * grad_w
        self.bias -= self.lr * grad_b
        
        return loss
    
    def _stochastic_gradient_descent(self, X, y, m):
        """Perform one epoch of stochastic gradient descent."""
        for i in range(m):
            xi = X[i]. reshape(1, -1)
            yi = y[i]. reshape(1, -1)
            y_pred = xi @ self.weights + self.bias
            error = y_pred - yi
            
            gradient_w = xi.T @ error
            gradient_b = error. item()
            
            self.weights -= self.lr * gradient_w
            self.bias -= self. lr * gradient_b
        
        # Calculate loss after one epoch
        y_pred = X @ self.weights + self.bias
        loss = (1/(2*m)) * np.sum((y_pred - y) ** 2) + (self.l2_reg/2) * np.sum(self.weights ** 2)
        
        return loss
    
    def _mini_batch_gradient_descent(self, X, y, m):
        """Perform one epoch of mini-batch gradient descent."""
        # Shuffle data
        perm = np.random.permutation(m)
        X_shuffle = X[perm]
        y_shuffle = y[perm]
        
        # Process mini-batches
        for start in range(0, m, self.batch_size):
            end = start + self.batch_size
            xb = X_shuffle[start: end]
            yb = y_shuffle[start:end]
            
            y_pred = xb @ self.weights + self. bias
            error = y_pred - yb
            
            gradient_w = (1/self.batch_size) * (xb.T @ error)
            gradient_b = (1/self.batch_size) * np.sum(error)
            
            self.weights -= self.lr * gradient_w
            self. bias -= self.lr * gradient_b
        
        # Calculate loss after one epoch
        y_pred = X @ self.weights + self.bias
        error = y_pred - y
        loss = (1/(2*m)) * np.sum(error ** 2) + (self.l2_reg/2) * np.sum(self.weights ** 2)
        
        return loss
    
    def predict(self, X):
        """
        Make predictions using the trained model.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Features to predict on
        
        Returns:
        --------
        numpy.ndarray : Predictions
        """
        return X @ self.weights + self. bias
