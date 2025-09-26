"""
Preprocessing Module for Medical Image Analysis
=============================================

This module provides comprehensive preprocessing utilities for medical imaging data including:
- Image preprocessing and normalization
- Data augmentation for medical images
- Medical literature text processing
- PDF extraction from research papers

Key Components:
- ImagePreprocessor: Medical image preprocessing pipeline
- MedicalDataAugmentation: Specialized augmentation for medical images
- LiteratureProcessor: Process medical literature and extract text
- PDFExtractor: Extract and process text from medical PDFs

Usage:
    from preprocessing import ImagePreprocessor, MedicalDataAugmentation
    
    # Image preprocessing
    preprocessor = ImagePreprocessor('chest_xray')
    processed_image = preprocessor.preprocess(image)
    
    # Data augmentation
    augmenter = MedicalDataAugmentation('skin_lesion')
    augmented_images = augmenter.augment_batch(images, labels)
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import warnings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

# Import core preprocessing classes
from .image_preprocessing import (
    ImagePreprocessor,
    ChestXrayPreprocessor,
    SkinLesionPreprocessor,
    BrainTumorPreprocessor,
    get_preprocessor,
    validate_image,
    normalize_image,
    resize_image,
    apply_clahe,
    remove_noise,
    enhance_contrast
)

from .data_augmentation import (
    MedicalDataAugmentation,
    ChestXrayAugmentation,
    SkinLesionAugmentation,
    BrainTumorAugmentation,
    get_augmentation_pipeline,
    create_balanced_dataset,
    apply_augmentation_strategy
)

from .literature_processor import (
    LiteratureProcessor,
    TextCleaner,
    MedicalTextProcessor,
    process_medical_literature,
    extract_medical_entities,
    chunk_medical_text,
    create_literature_embeddings
)

from .pdf_extractor import (
    PDFExtractor,
    MedicalPDFProcessor,
    extract_pdf_text,
    extract_pdf_metadata,
    process_medical_papers,
    batch_process_pdfs,
    validate_pdf_structure
)

# Version information
__version__ = '1.0.0'
__author__ = 'Medical AI Team'

# Supported medical imaging modalities
SUPPORTED_MODALITIES = [
    'chest_xray',
    'skin_lesion', 
    'brain_tumor'
]

# Default preprocessing configurations
DEFAULT_CONFIGS = {
    'chest_xray': {
        'target_size': (224, 224),
        'color_mode': 'rgb',
        'apply_clahe': True,
        'normalize': True,
        'remove_artifacts': True
    },
    'skin_lesion': {
        'target_size': (224, 224),
        'color_mode': 'rgb',
        'apply_hair_removal': True,
        'enhance_contrast': True,
        'normalize': True
    },
    'brain_tumor': {
        'target_size': (224, 224),
        'color_mode': 'rgb',
        'skull_stripping': False,
        'intensity_normalization': True,
        'bias_correction': False
    }
}

# Export all public classes and functions
__all__ = [
    # Image Processing
    'ImagePreprocessor',
    'ChestXrayPreprocessor',
    'SkinLesionPreprocessor', 
    'BrainTumorPreprocessor',
    'get_preprocessor',
    'validate_image',
    'normalize_image',
    'resize_image',
    'apply_clahe',
    'remove_noise',
    'enhance_contrast',
    
    # Data Augmentation
    'MedicalDataAugmentation',
    'ChestXrayAugmentation',
    'SkinLesionAugmentation',
    'BrainTumorAugmentation',
    'get_augmentation_pipeline',
    'create_balanced_dataset',
    'apply_augmentation_strategy',
    
    # Literature Processing
    'LiteratureProcessor',
    'TextCleaner',
    'MedicalTextProcessor',
    'process_medical_literature',
    'extract_medical_entities',
    'chunk_medical_text',
    'create_literature_embeddings',
    
    # PDF Processing
    'PDFExtractor',
    'MedicalPDFProcessor',
    'extract_pdf_text',
    'extract_pdf_metadata',
    'process_medical_papers',
    'batch_process_pdfs',
    'validate_pdf_structure',
    
    # Constants
    'SUPPORTED_MODALITIES',
    'DEFAULT_CONFIGS'
]

# Module initialization
logger.info(f"Medical Image Preprocessing Module v{__version__} initialized")
logger.info(f"Supported modalities: {', '.join(SUPPORTED_MODALITIES)}")

# Utility functions for quick access
def preprocess_image(image_path: str, modality: str, **kwargs) -> Any:
    """
    Quick image preprocessing function
    
    Args:
        image_path: Path to image file
        modality: Medical imaging modality
        **kwargs: Additional preprocessing parameters
        
    Returns:
        Preprocessed image array
    """
    preprocessor = get_preprocessor(modality)
    return preprocessor.preprocess_from_path(image_path, **kwargs)

def augment_medical_data(images: List[Any], labels: List[Any], modality: str, **kwargs) -> Tuple[List[Any], List[Any]]:
    """
    Quick data augmentation function
    
    Args:
        images: List of image arrays
        labels: Corresponding labels
        modality: Medical imaging modality
        **kwargs: Augmentation parameters
        
    Returns:
        Tuple of (augmented_images, augmented_labels)
    """
    augmenter = get_augmentation_pipeline(modality)
    return augmenter.augment_batch(images, labels, **kwargs)

def process_literature_file(file_path: str, **kwargs) -> Dict[str, Any]:
    """
    Quick literature processing function
    
    Args:
        file_path: Path to literature file (PDF, TXT, etc.)
        **kwargs: Processing parameters
        
    Returns:
        Processed literature data dictionary
    """
    processor = LiteratureProcessor()
    return processor.process_file(file_path, **kwargs)

# Configuration validation
def validate_preprocessing_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate preprocessing configuration
    
    Args:
        config: Configuration dictionary
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    required_keys = ['target_size', 'color_mode']
    for key in required_keys:
        if key not in config:
            errors.append(f"Missing required configuration key: {key}")
    
    if 'target_size' in config:
        size = config['target_size']
        if not isinstance(size, (tuple, list)) or len(size) != 2:
            errors.append("target_size must be tuple/list of 2 integers")
        elif not all(isinstance(x, int) and x > 0 for x in size):
            errors.append("target_size values must be positive integers")
    
    if 'color_mode' in config:
        valid_modes = ['rgb', 'grayscale', 'rgba']
        if config['color_mode'] not in valid_modes:
            errors.append(f"color_mode must be one of: {valid_modes}")
    
    return errors

# Environment setup check
def check_dependencies() -> Dict[str, bool]:
    """
    Check if all required dependencies are available
    
    Returns:
        Dictionary of dependency availability status
    """
    dependencies = {
        'opencv': False,
        'pillow': False,
        'numpy': False,
        'scikit_image': False,
        'albumentations': False,
        'pydicom': False,
        'pypdf2': False,
        'nltk': False,
        'spacy': False
    }
    
    try:
        import cv2
        dependencies['opencv'] = True
    except ImportError:
        pass
    
    try:
        import PIL
        dependencies['pillow'] = True
    except ImportError:
        pass
    
    try:
        import numpy
        dependencies['numpy'] = True
    except ImportError:
        pass
    
    try:
        import skimage
        dependencies['scikit_image'] = True
    except ImportError:
        pass
    
    try:
        import albumentations
        dependencies['albumentations'] = True
    except ImportError:
        pass
    
    try:
        import pydicom
        dependencies['pydicom'] = True
    except ImportError:
        pass
    
    try:
        import PyPDF2
        dependencies['pypdf2'] = True
    except ImportError:
        pass
    
    try:
        import nltk
        dependencies['nltk'] = True
    except ImportError:
        pass
    
    try:
        import spacy
        dependencies['spacy'] = True
    except ImportError:
        pass
    
    return dependencies

# Log dependency status
deps = check_dependencies()
missing_deps = [name for name, available in deps.items() if not available]
if missing_deps:
    logger.warning(f"Missing optional dependencies: {', '.join(missing_deps)}")
    logger.warning("Some functionality may be limited. Install with: pip install -r requirements.txt")
else:
    logger.info("All dependencies are available")

# Create default directories
def setup_preprocessing_directories():
    """Create necessary directories for preprocessing operations"""
    from pathlib import Path
    
    base_dirs = [
        'data/processed',
        'data/augmented', 
        'data/embeddings',
        'temp/preprocessing',
        'logs/preprocessing'
    ]
    
    for dir_path in base_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    logger.info("Preprocessing directories created")

# Initialize directories on import
setup_preprocessing_directories()
