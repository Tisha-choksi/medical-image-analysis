"""
Configuration Module for Medical Image Analysis System
=====================================================

This module centralizes all configuration settings including:
- Project paths and directories
- Model hyperparameters
- Database connections
- API settings
- Training configurations

Usage:
    from config import config, ModelConfig, PineconeConfig
    
    # Access main configuration
    data_path = config.DATA_PATHS['raw_data']
    
    # Use model configuration
    model_cfg = ModelConfig('chest_xray')
    print(model_cfg.input_shape)
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Dictionary containing configuration settings
    """
    try:
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        logger.info(f"Successfully loaded config from {config_path}")
        return config_data
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}")
        raise

# Load main configuration
CONFIG_FILE = PROJECT_ROOT / 'config' / 'config.yaml'
config = load_yaml_config(str(CONFIG_FILE)) if CONFIG_FILE.exists() else {}

# Import configuration classes
from .model_config import (
    ModelConfig,
    ChestXrayConfig,
    SkinLesionConfig,
    BrainTumorConfig,
    get_model_config
)

from .pinecone_config import (
    PineconeConfig,
    EmbeddingConfig,
    initialize_pinecone,
    get_pinecone_index
)

# Export commonly used configurations
__all__ = [
    'config',
    'PROJECT_ROOT',
    'load_yaml_config',
    'ModelConfig',
    'ChestXrayConfig',
    'SkinLesionConfig',
    'BrainTumorConfig',
    'get_model_config',
    'PineconeConfig',
    'EmbeddingConfig',
    'initialize_pinecone',
    'get_pinecone_index',
]

# Version information
__version__ = '1.0.0'
__author__ = 'Medical AI Team'

# Configuration validation
def validate_config() -> bool:
    """
    Validate that all required configuration settings are present
    
    Returns:
        bool: True if configuration is valid, False otherwise
    """
    required_keys = [
        'project',
        'data_paths',
        'model_settings',
        'training',
        'api'
    ]
    
    missing_keys = [key for key in required_keys if key not in config]
    
    if missing_keys:
        logger.warning(f"Missing configuration keys: {missing_keys}")
        return False
    
    logger.info("Configuration validation passed")
    return True

# Validate configuration on import
if config:
    validate_config()

# Environment-specific settings
ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')
DEBUG = os.getenv('DEBUG_MODE', 'True').lower() == 'true'

logger.info(f"Configuration module initialized - Environment: {ENVIRONMENT}")
logger.info(f"Project root: {PROJECT_ROOT}")
