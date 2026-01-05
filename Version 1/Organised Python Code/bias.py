#adding the bias term

import numpy as np

expanded_X = np.ones((X.shape[0], 1))
X_b = np.c_[expanded_X, X]
#X_b will have the bias term added to the features
