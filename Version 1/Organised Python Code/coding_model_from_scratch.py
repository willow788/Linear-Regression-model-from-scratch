#coding the  linear regression model from scratch

class LinearRegression:

    def __init__(self, learn_rate = 1e-7, iter = 50000):
        self.lr = learn_rate
        self.iter = iter
        self.weights = None
        self.loss_history = []

    def fit(self, X, y):
        m, n = X.shape
        #m -- no of samples (north to south)
        #n -- no of features (west to east)

        self.weights = np.zeros((n, 1))
        #initializing weights to zero including bias term


        for _ in range(self.iter):
            y_pred = np.dot(X, self.weights)
            #y_pred = Xw

            error = y_pred - y
            #error term

            loss = (1/(m)) * np.sum(error ** 2)

            self.loss_history.append(loss)

            gradient_loss = (2/m) * np.dot(X.T, error)
            #gradient of loss with respect to weights

            self.weights -= self.lr * gradient_loss
            #updating weights

    def predict(self, X):
        return np.dot(X, self.weights)
