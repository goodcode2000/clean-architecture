"""
Configuration for CSV Data Provider Service
Add new projects here to expand the service
"""

# Server configuration
SERVER_HOST = '0.0.0.0'
SERVER_PORT = 8001

# Project configurations
PROJECTS = {
    'btc-main72': {
        'name': 'BTC Main72 Project',
        'data_path': '/home/ubuntu/BTC/main72_origin/clean-architecture/data',
        'files': {
            'historical_real': 'historical_real.csv',
            'predictions': 'predictions.csv'
        }
    }
    
    # Add more projects here:
    # 'eth-project': {
    #     'name': 'ETH Prediction Project',
    #     'data_path': '/home/ubuntu/ETH/project/data',
    #     'files': {
    #         'historical_real': 'eth_historical.csv',
    #         'predictions': 'eth_predictions.csv'
    #     }
    # },
    # 'ltc-project': {
    #     'name': 'LTC Prediction Project', 
    #     'data_path': '/home/ubuntu/LTC/project/data',
    #     'files': {
    #         'historical_real': 'ltc_real.csv',
    #         'predictions': 'ltc_pred.csv'
    #     }
    # }
}

# API configuration
API_PREFIX = '/api'
CORS_ORIGINS = '*'  # Allow all origins, restrict in production

# File reading settings
DEFAULT_LIMIT = 1000  # Default number of records to return
MAX_LIMIT = 10000     # Maximum records allowed per request