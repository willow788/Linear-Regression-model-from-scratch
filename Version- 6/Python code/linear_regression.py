"""
Linear Regression implementation from scratch - Version 6

🎯 NEW FEATURE:   L1 Regularization (Lasso Regression)!

L1 regularization uses the absolute value of weights: 
- Loss = MSE + (λ/2) * sum(|weights|)
- Gradient = normal_gradient + λ * sign(weights)

Key property: CAN SHRINK WEIGHTS TO EXACTLY ZERO! 
This provides automatic feature selection. 
"""

import numpy as np


class LinearRegression:
    """
    Linear Regression model with L1 regularization (Lasso Regression).
    
    Parameters:
    -----------
    learn_rate : float, default=1e-7
        Learning rate for gradient descent
    iter :  int, default=50000
        Number of iterations (epochs) for training
    method : str, default='batch'
        Gradient descent method: 'batch', 'stochastic', or 'mini-batch'
    batch_size : int, default=32
        Batch size for mini-batch gradient descent
    l1_reg : float, default=0.0
        L1 regularization parameter (lambda).
        - 0.0 = no regularization
        - Higher values = stronger regularization + more sparsity
    """
    
    def __init__(self, learn_rate=1e-7, iter=50000, method='batch', 
                 batch_size=32, l1_reg=0.0):
        self.l1_reg = l1_reg
        self.method = method
        self.batch_size = batch_size
        self.lr = float(learn_rate)
        self.iter = int(iter)
        self.weights = None
        self.bias = 0
        self.loss_history = []
    
    def fit(self, X, y):
        """
        Fit the linear regression model with L1 regularization.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Training features of shape (m, n)
        y : numpy.ndarray
            Training targets of shape (m, 1)
        """
        m, n = X.shape
        
        # Initialize weights and bias to zero
        self.weights = np. zeros((n, 1))
        self.bias = 0
        
        for iteration in range(self.iter):
            if self.method == 'batch':
                loss = self._batch_gradient_descent(X, y, m)
                
            elif self.method == 'stochastic':
                loss = self._stochastic_gradient_descent(X, y, m)
                
            elif self.method == 'mini-batch':
                loss = self._mini_batch_gradient_descent(X, y, m)
            
            else:
                raise ValueError(f"Unknown method: {self.method}")
            
            self.loss_history.append(loss)
            
            if iteration % 5000 == 0:
                print(f"Loss at iteration {iteration}: {loss}")
    
    def _batch_gradient_descent(self, X, y, m):
        """
        Batch gradient descent with L1 regularization.
        
        L1 uses sign(weights) instead of weights in gradient. 
        sign(w) = +1 if w > 0, -1 if w < 0, 0 if w = 0
        
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
        float :   Loss value (including regularization term)
        """
        # Forward pass
        y_pred = X @ self.weights + self.bias
        error = y_pred - y
        
        # Loss with L1 regularization
        # L = (1/2m) * sum(error²) + (λ/2) * sum(|weights|)
        mse_loss = (1/(2*m)) * np.sum(error ** 2)
        reg_loss = (self.l1_reg/2) * np.sum(np.abs(self. weights))
        loss = mse_loss + reg_loss
        
        # Gradients with L1 regularization
        # ∂L/∂w = (1/m) * X.T @ error + λ * sign(weights)
        grad_w = (1/m) * (X.T @ error) + self.l1_reg * np.sign(self.weights)
        grad_b = (1/m) * np.sum(error)  # Bias is NOT regularized
        
        # Update parameters
        self.weights -= self. lr * grad_w
        self.bias -= self.lr * grad_b
        
        return loss
    
    def _stochastic_gradient_descent(self, X, y, m):
        """
        Stochastic gradient descent (L1 regularization in epoch loss only).
        
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
        float :  Loss value after one complete epoch
        """
        # Iterate through each sample
        for i in range(m):
            xi = X[i]. reshape(1, -1)
            yi = y[i]. reshape(1, -1)
            
            y_pred = xi @ self.weights + self.bias
            error = y_pred - yi
            
            # Gradients (no regularization in SGD updates)
            gradient_w = xi.T @ error
            gradient_b = error. item()
            
            self.weights -= self.lr * gradient_w
            self.bias -= self.lr * gradient_b
        
        # Calculate loss after epoch (with regularization)
        y_pred = X @ self.weights + self.bias
        mse_loss = (1/(2*m)) * np.sum((y_pred - y) ** 2)
        reg_loss = (self.l1_reg/2) * np.sum(np.abs(self.weights))
        loss = mse_loss + reg_loss
        
        return loss
    
    def _mini_batch_gradient_descent(self, X, y, m):
        """
        Mini-batch gradient descent (L1 regularization in epoch loss only).
        
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
        # Shuffle data
        perm = np.random.permutation(m)
        X_shuffle = X[perm]
        y_shuffle = y[perm]
        
        # Process mini-batches
        for start in range(0, m, self. batch_size):
            end = start + self.batch_size
            xb = X_shuffle[start: end]
            yb = y_shuffle[start:end]
            
            y_pred = xb @ self. weights + self.bias
            error = y_pred - yb
            
            # Gradients (no regularization in batch updates)
            gradient_w = (1/self.batch_size) * (xb.T @ error)
            gradient_b = (1/self.batch_size) * np.sum(error)
            
            self.weights -= self.lr * gradient_w
            self. bias -= self.lr * gradient_b
        
        # Calculate loss after epoch (with regularization)
        y_pred = X @ self.weights + self.bias
        error = y_pred - y
        mse_loss = (1/(2*m)) * np.sum(error ** 2)
        reg_loss = (self.l1_reg/2) * np.sum(np.abs(self. weights))
        loss = mse_loss + reg_loss
        
        return loss
    
    def predict(self, X):
        """Make predictions using the trained model."""
        return X @ self.weights + self. bias
    
    def get_non_zero_features(self, feature_names=None, threshold=1e-10):
        """
        Get features with non-zero weights (selected by Lasso).
        
        Parameters:
        -----------
        feature_names : list, optional
            Names of features
        threshold : float, default=1e-10
            Threshold below which weights are considered zero
        
        Returns:
        --------
        dict :  Dictionary with feature indices/names and their weights
        """
        non_zero_mask = np.abs(self.weights. flatten()) > threshold
        non_zero_indices = np.where(non_zero_mask)[0]
        
        if feature_names is not None:
            selected = {feature_names[i]: self.weights[i, 0] 
                       for i in non_zero_indices}
        else: 
            selected = {f"Feature_{i}": self.weights[i, 0] 
                       for i in non_zero_indices}
        
        return selected
