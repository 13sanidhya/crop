# Configuration file for Crop Recommendation System

class Config:
    """Base configuration"""
    SECRET_KEY = 'your-secret-key-change-this-in-production'
    DEBUG = True
    TESTING = False
    
    # Model paths
    MODEL_PATH = 'model.pkl'
    STANDARD_SCALER_PATH = 'standscaler.pkl'
    MINMAX_SCALER_PATH = 'minmaxscaler.pkl'
    
    # Application settings
    MAX_HISTORY_SIZE = 100  # Maximum number of predictions to store
    
    # Valid parameter ranges
    PARAMETER_RANGES = {
        'Nitrogen': (0, 140),
        'Phosphorus': (5, 145),
        'Potassium': (5, 205),
        'Temperature': (8.825, 43.675),
        'Humidity': (14.258, 99.981),
        'Ph': (3.505, 9.935),
        'Rainfall': (20.211, 298.56)
    }

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    ENV = 'development'

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    ENV = 'production'
    # In production, use environment variables for sensitive data
    # SECRET_KEY = os.environ.get('SECRET_KEY')

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
