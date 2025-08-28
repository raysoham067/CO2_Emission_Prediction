"""
Configuration file for CO2 Emission Prediction project
Contains all configurable parameters and settings
"""

import os

# Flask Configuration
class Config:
    """Base configuration class"""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    DEBUG = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    HOST = os.environ.get('FLASK_HOST', '0.0.0.0')
    PORT = int(os.environ.get('FLASK_PORT', 5000))

# Machine Learning Configuration
class MLConfig:
    """Machine Learning configuration"""
    
    # Dataset settings
    DATASET_PATH = 'FuelConsumptionCo2.csv'
    FEATURE_COLUMNS = [
        'ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 
        'FUELCONSUMPTION_HWY', 'FUELCONSUMPTION_COMB'
    ]
    TARGET_COLUMN = 'CO2EMISSIONS'
    
    # Alternative column names (if dataset has different column names)
    ALTERNATIVE_FEATURES = [
        'Engine Size', 'Cylinders', 'City Fuel Consumption', 
        'Highway Fuel Consumption', 'Combined Fuel Consumption'
    ]
    ALTERNATIVE_TARGET = 'CO2 Emissions'
    
    # Model training settings
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    CROSS_VALIDATION_FOLDS = 5
    
    # Model parameters
    RANDOM_FOREST_PARAMS = {
        'n_estimators': 100,
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'random_state': 42
    }
    
    GRADIENT_BOOSTING_PARAMS = {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 5,
        'subsample': 0.9,
        'random_state': 42
    }
    
    SVR_PARAMS = {
        'kernel': 'rbf',
        'C': 100,
        'gamma': 'scale'
    }

# Web Interface Configuration
class UIConfig:
    """User Interface configuration"""
    
    # Form validation ranges
    ENGINE_SIZE_RANGE = (0.5, 8.0)
    CYLINDERS_RANGE = (3, 12)
    FUEL_CONSUMPTION_RANGE = (3.0, 25.0)
    
    # Environmental impact thresholds (g/km CO2)
    LOW_IMPACT_THRESHOLD = 100
    MEDIUM_IMPACT_THRESHOLD = 150
    
    # Chart comparison values
    AVERAGE_VEHICLE_EMISSIONS = 120
    ECO_FRIENDLY_EMISSIONS = 80
    MAX_CHART_EMISSIONS = 200

# File Paths
class Paths:
    """File and directory paths"""
    
    # Model storage
    MODELS_DIR = 'models'
    LINEAR_REGRESSION_MODEL = 'models/linear_regression.pkl'
    RANDOM_FOREST_MODEL = 'models/random_forest.pkl'
    GRADIENT_BOOSTING_MODEL = 'models/gradient_boosting.pkl'
    SVR_MODEL = 'models/svr.pkl'
    SCALER_FILE = 'models/scaler.pkl'
    
    # Templates
    TEMPLATES_DIR = 'templates'
    INDEX_TEMPLATE = 'templates/index.html'
    RESULT_TEMPLATE = 'templates/result.html'
    
    # Output files
    MODEL_COMPARISON_FILE = 'models/model_comparison.txt'
    PERFORMANCE_PLOT = 'models/model_performance.png'
    PREDICTION_PLOT = 'models/actual_vs_predicted.png'

# API Configuration
class APIConfig:
    """API configuration"""
    
    # Rate limiting
    RATE_LIMIT = '100 per minute'
    
    # Response format
    DEFAULT_RESPONSE_FORMAT = 'json'
    
    # Error messages
    ERROR_MESSAGES = {
        'invalid_input': 'Invalid input parameters provided',
        'model_not_found': 'Specified model not found',
        'prediction_failed': 'Failed to generate prediction',
        'server_error': 'Internal server error occurred'
    }

# Logging Configuration
class LogConfig:
    """Logging configuration"""
    
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_FILE = 'co2_prediction.log'
    MAX_LOG_SIZE = 10 * 1024 * 1024  # 10MB
    BACKUP_COUNT = 5

# Development Configuration
class DevelopmentConfig(Config):
    """Development environment configuration"""
    DEBUG = True
    TESTING = False

# Production Configuration
class ProductionConfig(Config):
    """Production environment configuration"""
    DEBUG = False
    TESTING = False
    HOST = '0.0.0.0'
    PORT = 5000

# Testing Configuration
class TestingConfig(Config):
    """Testing environment configuration"""
    TESTING = True
    DEBUG = False

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

def get_config(config_name=None):
    """Get configuration class by name"""
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'default')
    return config.get(config_name, config['default'])
