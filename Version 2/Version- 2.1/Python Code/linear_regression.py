"""
Linear Regression implementation from scratch - Version 2.1
Basic implementation with bias term included in weight matrix
"""

import numpy as np


class LinearRegression:
    """
    Linear Regression model using gradient descent. 
    
    Parameters:
    -----------
    learn_rate : float, default=1e-7
        Learning rate for gradient descent
    iter : int, default=50000
        Number of iterations for training
    """
    
    def __init__(self, learn_rate=1e-7, iter=50000):
        self.lr = learn_rate
        self. iter = iter
        self.weights = None
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
        # m -- no of samples (north to south)
        # n -- no of features (west to east)
        
        # Initializing weights to zero including bias term
        self.weights = np.zeros((n, 1))
        
        for _ in range(self.iter):
            # y_pred = Xw
            y_pred = np.dot(X, self.weights)
            
            # error term
            error = y_pred - y
            
            # Calculate loss
            loss = (1/m) * np.sum(error ** 2)
            self.loss_history.append(loss)
            
            # gradient of loss with respect to weights
            gradient_loss = (2/m) * np.dot(X.T, error)
            
            # updating weights
            self.weights -= self.lr * gradient_loss
    
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
        return np.dot(X, self.weights)
