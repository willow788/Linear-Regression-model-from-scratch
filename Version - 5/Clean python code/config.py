"""
Configuration file for hyperparameters and paths.
"""

# Data paths
DATA_PATH = 'Advertising.csv'

# Training configuration
BATCH_GD_CONFIG = {
    'learn_rate': 0.02,
    'iter': 50000,
    'method': 'batch',
    'l2_reg': 0.1
}

SGD_CONFIG = {
    'learn_rate': 0.01,
    'iter': 50,
    'method': 'stochastic',
    'l2_reg': 0.2
}

MINI_BATCH_GD_CONFIG = {
    'learn_rate': 0.01,
    'iter': 1000,
    'method': 'mini-batch',
    'batch_size': 16,
    'l2_reg': 0.15
}

# Data preprocessing
TEST_SIZE = 0.2
RANDOM_STATE = 42
NORMALIZE_Y = True
