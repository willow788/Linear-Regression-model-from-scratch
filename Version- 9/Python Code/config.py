"""
Configuration file for model hyperparameters and paths. 
"""

# Data paths
DATA_PATH = 'Advertising.csv'

# Train-test split
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Model hyperparameters
BATCH_GD_CONFIG = {
    'learn_rate': 0.02,
    'iter': 50000,
    'method': 'batch',
    'l1_reg': 0.1
}

SGD_CONFIG = {
    'learn_rate': 0.01,
    'iter': 50,
    'method': 'stochastic',
    'l1_reg': 0.2
}

MINI_BATCH_CONFIG = {
    'learn_rate': 0.01,
    'iter': 1000,
    'method': 'mini-batch',
    'batch_size': 16,
    'l1_reg': 0.15
}

# Cross-validation
CV_FOLDS = 5

# Feature names
FEATURE_NAMES = [
    "TV",
    "Radio",
    "Newspaper",
    "TV²",
    "Radio²",
    "Newspaper²",
    "TV × Radio",
    "TV × Newspaper",
    "Radio × Newspaper"
]
