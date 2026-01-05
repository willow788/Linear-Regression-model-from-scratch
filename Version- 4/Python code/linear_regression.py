"""
Linear Regression implementation from scratch - Version 4

🚀 NEW FEATURE: Multiple Gradient Descent Methods! 

Supports three optimization methods:
1. Batch Gradient Descent (default) - uses all samples
2. Stochastic Gradient Descent (SGD) - updates after each sample
3. Mini-Batch Gradient Descent - updates after small batches
"""

import numpy as np


class LinearRegression:
    """
    Linear Regression model with multiple gradient descent methods.
    
    Parameters:
    -----------
    learn_rate : float, default=1e-7
        Learning rate for gradient descent
    iter :  int, default=50000
        Number of iterations (epochs) for training
    method : str, default='batch'
        Gradient descent method:  'batch', 'stochastic', or 'mini-batch'
    batch_size : int, default=32
        Batch size for mini-batch gradient descent (ignored for other methods)
    """
    
    def __init__(self, learn_rate=1e-7, iter=50000, method='batch', batch_size=32):
        self.method = method
        self.batch_size = batch_size
        self.lr = float(learn_rate)
        self.iter = int(iter)
        self.weights = None
        self.bias = 0
        self.loss_history = []
    
    def fit(self, X, y):
        """
        Fit the linear regression model using the specified gradient descent method.
        
        Parameters:
        -----------
        X :  numpy.ndarray
            Training features of shape (m, n)
        y : numpy.ndarray
            Training targets of shape (m, 1)
        """
        m, n = X.shape
        # m -- no of samples (north to south)
        # n -- no of features (west to east)
        
        # Initialize weights and bias to zero
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
                raise ValueError(f"Unknown method: {self.method}. "
                               "Use 'batch', 'stochastic', or 'mini-batch'.")
            
            self. loss_history.append(loss)
            
            # Print loss every 5000 iterations
            if iteration % 5000 == 0:
                print(f"Loss at iteration {iteration}: {loss}")
    
    def _batch_gradient_descent(self, X, y, m):
        """
        Perform one iteration of batch gradient descent.
        
        Uses ALL training samples to compute gradient.
        Most stable but can be slow for large datasets.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Training features
        y : numpy.ndarray
            Training targets
        m : int
            Number of samples
        
        Returns:
        --------
        float :  Loss value after update
        """
        # Forward pass on all data
        y_pred = X @ self.weights + self.bias
        error = y_pred - y
        loss = (1/(2*m)) * np.sum(error ** 2)
        
        # Compute gradients using all samples
        grad_w = (1/m) * (X.T @ error)
        grad_b = (1/m) * np.sum(error)
        
        # Update parameters
        self.weights -= self. lr * grad_w
        self.bias -= self.lr * grad_b
        
        return loss
    
    def _stochastic_gradient_descent(self, X, y, m):
        """
        Perform one epoch of stochastic gradient descent (SGD).
        
        Updates parameters after EACH individual sample.
        Faster updates but noisier convergence.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Training features
        y : numpy.ndarray
            Training targets
        m : int
            Number of samples
        
        Returns:
        --------
        float : Loss value after one complete epoch
        """
        # Iterate through each sample
        for i in range(m):
            # Get single sample
            xi = X[i]. reshape(1, -1)
            yi = y[i].reshape(1, -1)
            
            # Forward pass
            y_pred = xi @ self.weights + self.bias
            error = y_pred - yi
            
            # Compute gradients from single sample
            gradient_w = xi.T @ error
            gradient_b = error. item()
            
            # Update parameters immediately
            self.weights -= self. lr * gradient_w
            self.bias -= self.lr * gradient_b
        
        # Calculate loss after one complete epoch
        y_pred = X @ self.weights + self.bias
        loss = (1/(2*m)) * np.sum((y_pred - y) ** 2)
        
        return loss
    
    def _mini_batch_gradient_descent(self, X, y, m):
        """
        Perform one epoch of mini-batch gradient descent. 
        
        Updates parameters after processing small batches of data.
        Balance between batch GD (stable) and SGD (fast).
        
        Parameters:
        -----------
        X : numpy. ndarray
            Training features
        y : numpy.ndarray
            Training targets
        m : int
            Number of samples
        
        Returns:
        --------
        float : Loss value after one complete epoch
        """
        # Shuffle data at the start of each epoch
        perm = np.random.permutation(m)
        X_shuffle = X[perm]
        y_shuffle = y[perm]
        
        # Process data in mini-batches
        for start in range(0, m, self. batch_size):
            end = start + self.batch_size
            xb = X_shuffle[start:end]
            yb = y_shuffle[start:end]
            
            # Forward pass on mini-batch
            y_pred = xb @ self.weights + self.bias
            error = y_pred - yb
            
            # Compute gradients from mini-batch
            gradient_w = (1/self.batch_size) * (xb.T @ error)
            gradient_b = (1/self.batch_size) * np.sum(error)
            
            # Update parameters
            self.weights -= self.lr * gradient_w
            self.bias -= self. lr * gradient_b
        
        # Calculate loss after one complete epoch
        y_pred = X @ self.weights + self.bias
        error = y_pred - y
        loss = (1/(2*m)) * np.sum(error ** 2)
        
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
