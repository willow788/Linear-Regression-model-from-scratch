"""
Linear Regression model implementation from scratch
Supports batch, stochastic, and mini-batch gradient descent with L1 regularization
"""

import numpy as np


class LinearRegression:
    """
    Linear Regression model with gradient descent optimization
    
    Supports three training methods:
    - 'batch':  Batch Gradient Descent (uses all samples)
    - 'stochastic': Stochastic Gradient Descent (uses one sample at a time)
    - 'mini-batch': Mini-batch Gradient Descent (uses batches of samples)
    
    Includes L1 regularization (Lasso) to prevent overfitting
    """
    
    def __init__(self, learn_rate=1e-7, iter=50000, method='batch', batch_size=32, l1_reg=0.0):
        """
        Initialize the Linear Regression model
        
        Args:
            learn_rate: Learning rate for gradient descent
            iter:  Number of iterations (epochs)
            method: Training method ('batch', 'stochastic', or 'mini-batch')
            batch_size: Size of mini-batches (only used when method='mini-batch')
            l1_reg: L1 regularization parameter (lambda)
        """
        self. l1_reg = l1_reg
        self.method = method
        self.batch_size = batch_size
        self.lr = float(learn_rate)
        self.iter = int(iter)
        self.weights = None
        self.bias = 0
        self.loss_history = []

    def fit(self, X, y):
        """
        Train the model using gradient descent
        
        Args: 
            X: Training features (m samples, n features)
            y: Training target values (m samples, 1)
        """
        m, n = X.shape
        # m -- number of samples (rows)
        # n -- number of features (columns)

        # Initialize weights to zero
        self.weights = np. zeros((n, 1))
        self.bias = 0

        for epoch in range(self.iter):
            if self.method == 'batch': 
                loss = self._batch_gradient_descent(X, y, m)
                
            elif self.method == 'stochastic':
                loss = self._stochastic_gradient_descent(X, y, m)
                
            elif self.method == 'mini-batch': 
                loss = self._mini_batch_gradient_descent(X, y, m)
            
            self.loss_history.append(loss)
            
            if epoch % 5000 == 0:
                print(f"Loss at iteration {epoch}: {loss}")

    def _batch_gradient_descent(self, X, y, m):
        """
        Perform one iteration of batch gradient descent
        
        Args:
            X: Training features
            y: Training target values
            m: Number of samples
        
        Returns:
            loss: Current loss value
        """
        y_pred = X @ self.weights + self.bias
        error = y_pred - y
        loss = (1/(2*m)) * np.sum(error ** 2) + (self.l1_reg/2) * np.sum(np.abs(self.weights))

        # Compute gradients
        grad_w = (1/m) * (X.T @ error) + self.l1_reg * np.sign(self.weights)
        grad_b = (1/m) * np.sum(error)

        # Update parameters
        self.weights -= self.lr * grad_w
        self.bias -= self.lr * grad_b
        
        return loss

    def _stochastic_gradient_descent(self, X, y, m):
        """
        Perform one epoch of stochastic gradient descent
        
        Args:
            X:  Training features
            y: Training target values
            m: Number of samples
        
        Returns:
            loss: Loss value after the epoch
        """
        for i in range(m):
            xi = X[i].reshape(1, -1)
            yi = y[i].reshape(1, -1)
            y_pred = xi @ self.weights + self.bias
            error = y_pred - yi

            gradient_w = xi.T @ error + self.l1_reg * np.sign(self.weights)
            gradient_b = error. item()

            self.weights -= self.lr * gradient_w
            self.bias -= self.lr * gradient_b

        # Calculate loss after one epoch
        y_pred = X @ self.weights + self.bias
        loss = (1/(2*m)) * np.sum((y_pred - y) ** 2) + (self.l1_reg/2) * np.sum(np.abs(self. weights))
        
        return loss

    def _mini_batch_gradient_descent(self, X, y, m):
        """
        Perform one epoch of mini-batch gradient descent
        
        Args:
            X:  Training features
            y: Training target values
            m: Number of samples
        
        Returns:
            loss: Loss value after the epoch
        """
        # Shuffle the data
        perm = np.random.permutation(m)
        X_shuffle = X[perm]
        y_shuffle = y[perm]

        # Process mini-batches
        for start in range(0, m, self.batch_size):
            end = start + self.batch_size
            xb = X_shuffle[start: end]
            yb = y_shuffle[start:end]

            y_pred = xb @ self. weights + self.bias
            error = y_pred - yb

            gradient_w = (1/self.batch_size) * (xb.T @ error) + self.l1_reg * np. sign(self.weights)
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
        Make predictions using the trained model
        
        Args:
            X: Feature matrix for prediction
        
        Returns:
            Predicted values
        """
        return X @ self.weights + self. bias
