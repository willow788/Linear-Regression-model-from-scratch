"""
Linear Regression implementation from scratch - Version 3.1

🎉 KEY IMPROVEMENT: Truly separate bias term! 

The predict() method now uses: X @ weights + bias
Instead of concatenating bias to the weight matrix. 
"""

import numpy as np


class LinearRegression:
    """
    Linear Regression model using gradient descent with separate bias term.
    
    Parameters:
    -----------
    learn_rate : float, default=1e-7
        Learning rate for gradient descent
    iter :  int, default=50000
        Number of iterations for training
    """
    
    def __init__(self, learn_rate=1e-7, iter=50000):
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
            Training features of shape (m, n) - WITHOUT bias column
        y : numpy.ndarray
            Training targets of shape (m, 1) - normalized
        """
        m, n = X.shape
        # m -- no of samples (north to south)
        # n -- no of features (west to east)
        
        # Initialize weights and bias to zero
        self.weights = np.zeros((n, 1))
        self.bias = 0
        
        for iteration in range(self.iter):
            # Forward pass:  y_pred = Xw (bias NOT included in weight matrix)
            y_pred = np.dot(X, self.weights)
            
            # Error term
            error = y_pred - y
            
            # Calculate loss (MSE)
            loss = (1/(2*m)) * np.sum(error ** 2)
            self.loss_history.append(loss)
            
            # Gradient of loss with respect to weights
            gradient_loss = (1/m) * np.dot(X.T, error)
            
            # Gradient of loss with respect to bias
            gradient_bias = (1/m) * np.sum(error)
            
            # Update weights
            self.weights -= self.lr * gradient_loss
            
            # Update bias (separate parameter, but still computed and tracked)
            self.bias -= self.lr * gradient_bias
            
            # Print loss every 5000 iterations
            if iteration % 5000 == 0:
                print(f"Loss at iteration {iteration}: {loss}")
    
    def predict(self, X):
        """
        Make predictions using the trained model.
        
        🔑 KEY CHANGE: Now explicitly adds bias term!
        
        Parameters:
        -----------
        X :  numpy.ndarray
            Features to predict on (WITHOUT bias column)
        
        Returns:
        --------
        numpy.ndarray : Predictions
        """
        # X @ weights + bias (truly separate bias!)
        return np.dot(X, self.weights) + self.bias
