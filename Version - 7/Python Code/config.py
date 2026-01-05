"""
Configuration file for model hyperparameters
"""

# Data configuration
DATA_FILE = 'Advertising.csv'
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Batch Gradient Descent configuration
BATCH_CONFIG = {
    'learn_rate': 0.02,
    'iter': 50000,
    'method': 'batch',
    'l1_reg': 0.1
}

# Stochastic Gradient Descent configuration
SGD_CONFIG = {
    'learn_rate': 0.01,
    'iter': 50,
    'method': 'stochastic',
    'l1_reg': 0.2
}

# Mini-Batch Gradient Descent configuration
MINI_BATCH_CONFIG = {
    'learn_rate': 0.01,
    'iter': 1000,
    'method':  'mini-batch',
    'batch_size': 16,
    'l1_reg':  0.15
}
