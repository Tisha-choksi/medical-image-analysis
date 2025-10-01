"""
Medical Image Preprocessing Module
=================================

This module provides specialized preprocessing functions for different medical imaging modalities.
Each modality has specific requirements for optimal preprocessing.

Classes:
- ImagePreprocessor: Base class for image preprocessing
- ChestXrayPreprocessor: Specialized for chest X-ray images
- SkinLesionPreprocessor: Specialized for dermatoscopic images  
- BrainTumorPreprocessor: Specialized for brain MRI images

Usage:
    from preprocessing.image_preprocessing import ChestXrayPreprocessor
    
    preprocessor = ChestXrayPreprocessor()
    processed_image = preprocessor.preprocess(image)
"""

import os
import cv2
import numpy as np
from typing import Tuple, Optional, Union, Dict, Any
from pathlib import Path
import logging
from abc import ABC, abstractmethod

try:
    from PIL import Image, ImageEnhance
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    from skimage import exposure, filters, morphology, segmentation
    from skimage.restoration import denoise_nl_means
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

try:
    import pydicom
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False

logger = logging.getLogger(__name__)


class ImagePreprocessor(ABC):
    """
    Abstract base class for medical image preprocessing
    """
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224), normalize: bool = True):
        """
        Initialize preprocessor
        
        Args:
            target_size: Target image size (height, width)
            normalize: Whether to normalize pixel values
        """
        self.target_size = target_size
        self.normalize = normalize
        self.modality = "general"
        
    @abstractmethod
    def preprocess(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Preprocess image - to be implemented by subclasses
        
        Args:
            image: Input image array
            **kwargs: Additional preprocessing parameters
            
        Returns:
            Preprocessed image array
        """
        pass
    
    def preprocess_from_path(self, image_path: str, **kwargs) -> np.ndarray:
        """
        Load and preprocess image from file path
        
        Args:
            image_path: Path to image file
            **kwargs: Additional preprocessing parameters
            
        Returns:
            Preprocessed image array
        """
        image = self.load_image(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        return self.preprocess(image, **kwargs)
    
    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Load image from various formats
        
        Args:
            image_path: Path to image file
            
        Returns:
            Image array or None if loading failed
        """
        try:
            path = Path(image_path)
            
            if not path.exists():
                logger.error(f"Image file not found: {image_path}")
                return None
            
            # Handle DICOM files
            if path.suffix.lower() in ['.dcm', '.dicom']:
                if PYDICOM_AVAILABLE:
                    return self._load_dicom(image_path)
                else:
                    logger.warning("pydicom not available for DICOM file loading")
                    return None
            
            # Handle standard image formats
            image = cv2.imread(str(image_path))
            if image is not None:
                # Convert BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                return image
            
            # Try with PIL as fallback
            if PIL_AVAILABLE:
                pil_image = Image.open(image_path)
                return np.array(pil_image)
            
            logger.error(f"Could not load image: {image_path}")
            return None
            
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {str(e)}")
            return None
    
    def _load_dicom(self, dicom_path: str) -> Optional[np.ndarray]:
        """
        Load DICOM image file
        
        Args:
            dicom_path: Path to DICOM file
            
        Returns:
            Image array or None if loading failed
        """
        try:
            ds = pydicom.dcmread(dicom_path)
            
            # Get pixel array
            image = ds.pixel_array
            
            # Apply DICOM transformations
            if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                image = image * ds.RescaleSlope + ds.RescaleIntercept
            
            # Convert to uint8 if needed
            if image.dtype != np.uint8:
                image = self._normalize_to_uint8(image)
            
            # Handle grayscale to RGB conversion
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            return image
            
        except Exception as e:
            logger.error(f"Error loading DICOM file {dicom_path}: {str(e)}")
            return None
    
    def _normalize_to_uint8(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image to uint8 range [0, 255]
        
        Args:
            image: Input image array
            
        Returns:
            Normalized uint8 image
        """
        # Clip extreme values
        p2, p98 = np.percentile(image, (2, 98))
        image = np.clip(image, p2, p98)
        
        # Normalize to 0-255
        image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
        
        return image
    
    def resize_image(self, image: np.ndarray, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Resize image to target size
        
        Args:
            image: Input image array
            target_size: Target size (height, width), uses self.target_size if None
            
        Returns:
            Resized image
        """
        if target_size is None:
            target_size = self.target_size
        
        return cv2.resize(image, (target_size[1], target_size[0]), interpolation=cv2.INTER_LANCZOS4)
    
    def normalize_image(self, image: np.ndarray, method: str = 'minmax') -> np.ndarray:
        """
        Normalize image pixel values
        
        Args:
            image: Input image array
            method: Normalization method ('minmax', 'zscore', 'unit')
            
        Returns:
            Normalized image
        """
        image = image.astype(np.float32)
        
        if method == 'minmax':
            # Min-max normalization to [0, 1]
            image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        elif method == 'zscore':
            # Z-score normalization
            image = (image - image.mean()) / (image.std() + 1e-8)
        elif method == 'unit':
            # Unit vector normalization
            norm = np.linalg.norm(image)
            image = image / (norm + 1e-8)
        else:
            logger.warning(f"Unknown normalization method: {method}")
        
        return image


class ChestXrayPreprocessor(ImagePreprocessor):
    """
    Specialized preprocessor for chest X-ray images
    """
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        super().__init__(target_size, normalize=True)
        self.modality = "chest_xray"
    
    def preprocess(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Preprocess chest X-ray image
        
        Args:
            image: Input chest X-ray image
            **kwargs: Additional parameters
                - apply_clahe: Apply CLAHE (default: True)
                - remove_noise: Apply denoising (default: True)
                - enhance_contrast: Enhance contrast (default: True)
                - normalize_method: Normalization method (default: 'minmax')
            
        Returns:
            Preprocessed chest X-ray image
        """
        # Get parameters
        apply_clahe = kwargs.get('apply_clahe', True)
        remove_noise = kwargs.get('remove_noise', True)
        enhance_contrast = kwargs.get('enhance_contrast', True)
        normalize_method = kwargs.get('normalize_method', 'minmax')
        
        # Ensure RGB format
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # Convert to grayscale for processing, then back to RGB
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        if apply_clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
        
        # Remove noise
        if remove_noise and SKIMAGE_AVAILABLE:
            gray = denoise_nl_means(gray, h=10, fast_mode=True, preserve_range=True).astype(np.uint8)
        elif remove_noise:
            # Fallback to Gaussian blur
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Enhance contrast
        if enhance_contrast:
            gray = self._enhance_chest_contrast(gray)
        
        # Convert back to RGB
        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        
        # Resize
        image = self.resize_image(image)
        
        # Normalize
        if self.normalize:
            image = self.normalize_image(image, method=normalize_method)
        
        return image
    
    def _enhance_chest_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance contrast specifically for chest X-rays
        
        Args:
            image: Grayscale chest X-ray image
            
        Returns:
            Contrast-enhanced image
        """
        # Apply gamma correction
        gamma = 0.8
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
        image = cv2.LUT(image, table)
        
        # Apply adaptive histogram equalization
        if SKIMAGE_AVAILABLE:
            image = exposure.equalize_adapthist(image, clip_limit=0.02)
            image = (image * 255).astype(np.uint8)
        
        return image


class SkinLesionPreprocessor(ImagePreprocessor):
    """
    Specialized preprocessor for dermatoscopic skin lesion images
    """
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        super().__init__(target_size, normalize=True)
        self.modality = "skin_lesion"
    
    def preprocess(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Preprocess skin lesion image
        
        Args:
            image: Input skin lesion image
            **kwargs: Additional parameters
                - remove_hair: Remove hair artifacts (default: True)
                - enhance_colors: Enhance color contrast (default: True)
                - remove_artifacts: Remove imaging artifacts (default: True)
                - normalize_method: Normalization method (default: 'minmax')
            
        Returns:
            Preprocessed skin lesion image
        """
        # Get parameters
        remove_hair = kwargs.get('remove_hair', True)
        enhance_colors = kwargs.get('enhance_colors', True)
        remove_artifacts = kwargs.get('remove_artifacts', True)
        normalize_method = kwargs.get('normalize_method', 'minmax')
        
        # Ensure RGB format
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # Remove hair artifacts
        if remove_hair:
            image = self._remove_hair_artifacts(image)
        
        # Remove other artifacts
        if remove_artifacts:
            image = self._remove_skin_artifacts(image)
        
        # Enhance colors
        if enhance_colors:
            image = self._enhance_skin_colors(image)
        
        # Resize
        image = self.resize_image(image)
        
        # Normalize
        if self.normalize:
            image = self.normalize_image(image, method=normalize_method)
        
        return image
    
    def _remove_hair_artifacts(self, image: np.ndarray) -> np.ndarray:
        """
        Remove hair artifacts from dermatoscopic images
        
        Args:
            image: RGB skin lesion image
            
        Returns:
            Image with hair artifacts removed
        """
        # Convert to grayscale for hair detection
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Create hair mask using morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
        
        # Black-hat transform to detect dark hair
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        
        # Create mask for hair regions
        _, hair_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
        
        # Dilate mask to cover hair completely
        hair_mask = cv2.dilate(hair_mask, kernel, iterations=1)
        
        # Inpaint hair regions
        result = cv2.inpaint(image, hair_mask, 6, cv2.INPAINT_TELEA)
        
        return result
    
    def _remove_skin_artifacts(self, image: np.ndarray) -> np.ndarray:
        """
        Remove imaging artifacts from skin lesion images
        
        Args:
            image: RGB skin lesion image
            
        Returns:
            Image with artifacts removed
        """
        # Apply bilateral filter to reduce noise while preserving edges
        filtered = cv2.bilateralFilter(image, 9, 75, 75)
        
        # Remove small bright spots (imaging artifacts)
        gray = cv2.cvtColor(filtered, cv2.COLOR_RGB2GRAY)
        
        # Morphological opening to remove small bright spots
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        
        # Convert back to RGB
        result = cv2.cvtColor(opened, cv2.COLOR_GRAY2RGB)
        
        # Blend with original to preserve color information
        result = cv2.addWeighted(filtered, 0.7, result, 0.3, 0)
        
        return result
    
    def _enhance_skin_colors(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance color contrast for better lesion visibility
        
        Args:
            image: RGB skin lesion image
            
        Returns:
            Color-enhanced image
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Enhance L channel (lightness)
        l_channel = lab[:, :, 0]
        if SKIMAGE_AVAILABLE:
            l_enhanced = exposure.equalize_adapthist(l_channel, clip_limit=0.03)
            lab[:, :, 0] = (l_enhanced * 255).astype(np.uint8)
        else:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(l_channel)
        
        # Convert back to RGB
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Enhance saturation
        hsv = cv2.cvtColor(enhanced, cv2.COLOR_RGB2HSV)
        hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], 1.2)  # Increase saturation
        enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        return enhanced


class BrainTumorPreprocessor(ImagePreprocessor):
    """
    Specialized preprocessor for brain MRI images
    """
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        super().__init__(target_size, normalize=True)
        self.modality = "brain_tumor"
    
    def preprocess(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Preprocess brain MRI image
        
        Args:
            image: Input brain MRI image
            **kwargs: Additional parameters
                - intensity_correction: Apply intensity normalization (default: True)
                - noise_reduction: Apply noise reduction (default: True)
                - enhance_contrast: Enhance brain tissue contrast (default: True)
                - normalize_method: Normalization method (default: 'zscore')
            
        Returns:
            Preprocessed brain MRI image
        """
        # Get parameters
        intensity_correction = kwargs.get('intensity_correction', True)
        noise_reduction = kwargs.get('noise_reduction', True)
        enhance_contrast = kwargs.get('enhance_contrast', True)
        normalize_method = kwargs.get('normalize_method', 'zscore')
        
        # Ensure RGB format
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # Convert to grayscale for MRI processing
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply intensity correction
        if intensity_correction:
            gray = self._correct_mri_intensity(gray)
        
        # Reduce noise
        if noise_reduction:
            gray = self._reduce_mri_noise(gray)
        
        # Enhance brain tissue contrast
        if enhance_contrast:
            gray = self._enhance_brain_contrast(gray)
        
        # Convert back to RGB
        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        
        # Resize
        image = self.resize_image(image)
        
        # Normalize (Z-score is often better for MRI)
        if self.normalize:
            image = self.normalize_image(image, method=normalize_method)
        
        return image
    
    def _correct_mri_intensity(self, image: np.ndarray) -> np.ndarray:
        """
        Correct intensity non-uniformity in MRI images
        
        Args:
            image: Grayscale MRI image
            
        Returns:
            Intensity-corrected image
        """
        # Simple intensity correction using percentile normalization
        p2, p98 = np.percentile(image[image > 0], (2, 98))
        image = np.clip(image, p2, p98)
        
        # Normalize within brain region
        brain_mask = image > np.mean(image) * 0.1
        if np.sum(brain_mask) > 0:
            brain_pixels = image[brain_mask]
            mean_intensity = np.mean(brain_pixels)
            std_intensity = np.std(brain_pixels)
            
            # Apply intensity correction
            image = (image - mean_intensity) / (std_intensity + 1e-8) * 50 + 128
            image = np.clip(image, 0, 255).astype(np.uint8)
        
        return image
    
    def _reduce_mri_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Reduce noise in MRI images while preserving edges
        
        Args:
            image: Grayscale MRI image
            
        Returns:
            Denoised image
        """
        if SKIMAGE_AVAILABLE:
            # Use Non-local means denoising
            denoised = denoise_nl_means(image, h=0.1, fast_mode=True, preserve_range=True)
            return denoised.astype(np.uint8)
        else:
            # Fallback to bilateral filter
            return cv2.bilateralFilter(image, 9, 75, 75)
    
    def _enhance_brain_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance contrast specifically for brain tissue
        
        Args:
            image: Grayscale MRI image
            
        Returns:
            Contrast-enhanced image
        """
        # Apply CLAHE with smaller tile size for brain images
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(image)
        
        # Apply gamma correction for brain tissue
        gamma = 1.2
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
        enhanced = cv2.LUT(enhanced, table)
        
        return enhanced


# Utility functions
def validate_image(image: np.ndarray) -> bool:
    """
    Validate image array for preprocessing
    
    Args:
        image: Image array to validate
        
    Returns:
        True if image is valid for preprocessing
    """
    if image is None:
        return False
    
    if not isinstance(image, np.ndarray):
        return False
    
    if len(image.shape) not in [2, 3]:
        return False
    
    if len(image.shape) == 3 and image.shape[2] not in [1, 3, 4]:
        return False
    
    if image.size == 0:
        return False
    
    return True


def normalize_image(image: np.ndarray, method: str = 'minmax') -> np.ndarray:
    """
    Standalone image normalization function
    
    Args:
        image: Input image array
        method: Normalization method ('minmax', 'zscore', 'unit')
        
    Returns:
        Normalized image
    """
    preprocessor = ImagePreprocessor()
    return preprocessor.normalize_image(image, method)


def resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Standalone image resizing function
    
    Args:
        image: Input image array
        target_size: Target size (height, width)
        
    Returns:
        Resized image
    """
    return cv2.resize(image, (target_size[1], target_size[0]), interpolation=cv2.INTER_LANCZOS4)


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization
    
    Args:
        image: Input grayscale image
        clip_limit: Threshold for contrast limiting
        tile_grid_size: Size of grid for histogram equalization
        
    Returns:
        CLAHE-enhanced image
    """
    if len(image.shape) == 3:
        # Convert to grayscale if needed
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)


def remove_noise(image: np.ndarray, method: str = 'bilateral') -> np.ndarray:
    """
    Remove noise from image using specified method
    
    Args:
        image: Input image array
        method: Denoising method ('bilateral', 'gaussian', 'nlmeans')
        
    Returns:
        Denoised image
    """
    if method == 'bilateral':
        return cv2.bilateralFilter(image, 9, 75, 75)
    elif method == 'gaussian':
        return cv2.GaussianBlur(image, (5, 5), 0)
    elif method == 'nlmeans' and SKIMAGE_AVAILABLE:
        if len(image.shape) == 3:
            denoised = denoise_nl_means(image, h=0.1, fast_mode=True, preserve_range=True, channel_axis=-1)
        else:
            denoised = denoise_nl_means(image, h=0.1, fast_mode=True, preserve_range=True)
        return denoised.astype(image.dtype)
    else:
        logger.warning(f"Unknown denoising method: {method}, using bilateral filter")
        return cv2.bilateralFilter(image, 9, 75, 75)


def enhance_contrast(image: np.ndarray, method: str = 'clahe') -> np.ndarray:
    """
    Enhance image contrast using specified method
    
    Args:
        image: Input image array
        method: Contrast enhancement method ('clahe', 'histogram_eq', 'gamma')
        
    Returns:
        Contrast-enhanced image
    """
    if len(image.shape) == 3:
        # Process each channel separately for color images
        result = np.zeros_like(image)
        for i in range(image.shape[2]):
            result[:, :, i] = enhance_contrast(image[:, :, i], method)
        return result
    
    if method == 'clahe':
        return apply_clahe(image)
    elif method == 'histogram_eq':
        return cv2.equalizeHist(image)
    elif method == 'gamma':
        # Apply gamma correction (gamma = 0.8)
        gamma = 0.8
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
        return cv2.LUT(image, table)
    else:
        logger.warning(f"Unknown contrast enhancement method: {method}, using CLAHE")
        return apply_clahe(image)


def get_preprocessor(modality: str) -> ImagePreprocessor:
    """
    Factory function to get preprocessor for specific modality
    
    Args:
        modality: Medical imaging modality
        
    Returns:
        Appropriate preprocessor instance
        
    Raises:
        ValueError: If modality is not supported
    """
    modality_lower = modality.lower()
    
    if modality_lower == 'chest_xray':
        return ChestXrayPreprocessor()
    elif modality_lower == 'skin_lesion':
        return SkinLesionPreprocessor()
    elif modality_lower == 'brain_tumor':
        return BrainTumorPreprocessor()
    else:
        raise ValueError(f"Unsupported modality: {modality}. Supported: chest_xray, skin_lesion, brain_tumor")


def batch_preprocess(image_paths: list, modality: str, output_dir: str = None, **kwargs) -> list:
    """
    Preprocess a batch of images
    
    Args:
        image_paths: List of image file paths
        modality: Medical imaging modality
        output_dir: Optional directory to save processed images
        **kwargs: Additional preprocessing parameters
        
    Returns:
        List of preprocessed image arrays
    """
    preprocessor = get_preprocessor(modality)
    processed_images = []
    
    for i, image_path in enumerate(image_paths):
        try:
            logger.info(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
            
            # Preprocess image
            processed_image = preprocessor.preprocess_from_path(image_path, **kwargs)
            processed_images.append(processed_image)
            
            # Save if output directory specified
            if output_dir:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                
                filename = Path(image_path).stem + '_processed.png'
                save_path = output_path / filename
                
                # Convert to uint8 for saving
                if processed_image.max() <= 1.0:
                    save_image = (processed_image * 255).astype(np.uint8)
                else:
                    save_image = processed_image.astype(np.uint8)
                
                # Save as RGB
                save_image_bgr = cv2.cvtColor(save_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(save_path), save_image_bgr)
                
        except Exception as e:
            logger.error(f"Failed to process {image_path}: {str(e)}")
            processed_images.append(None)
    
    logger.info(f"Batch processing completed. {len([img for img in processed_images if img is not None])}/{len(image_paths)} images processed successfully")
    
    return processed_images


class PreprocessingPipeline:
    """
    Complete preprocessing pipeline for medical images
    """
    
    def __init__(self, modality: str, config: Dict[str, Any] = None):
        """
        Initialize preprocessing pipeline
        
        Args:
            modality: Medical imaging modality
            config: Preprocessing configuration dictionary
        """
        self.modality = modality
        self.preprocessor = get_preprocessor(modality)
        self.config = config or {}
        
        # Default pipeline steps
        self.pipeline_steps = [
            'validate',
            'resize', 
            'denoise',
            'enhance_contrast',
            'normalize'
        ]
        
        if 'pipeline_steps' in self.config:
            self.pipeline_steps = self.config['pipeline_steps']
    
    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Process image through complete pipeline
        
        Args:
            image: Input image array
            
        Returns:
            Fully processed image
        """
        if not validate_image(image):
            raise ValueError("Invalid image for preprocessing")
        
        processed_image = image.copy()
        
        for step in self.pipeline_steps:
            if step == 'validate':
                if not validate_image(processed_image):
                    raise ValueError("Image validation failed during pipeline")
            
            elif step == 'resize':
                target_size = self.config.get('target_size', (224, 224))
                processed_image = resize_image(processed_image, target_size)
            
            elif step == 'denoise':
                method = self.config.get('denoise_method', 'bilateral')
                processed_image = remove_noise(processed_image, method)
            
            elif step == 'enhance_contrast':
                method = self.config.get('contrast_method', 'clahe')
                processed_image = enhance_contrast(processed_image, method)
            
            elif step == 'normalize':
                method = self.config.get('normalize_method', 'minmax')
                processed_image = normalize_image(processed_image, method)
            
            elif step == 'modality_specific':
                # Use modality-specific preprocessing
                processed_image = self.preprocessor.preprocess(processed_image, **self.config)
        
        return processed_image
    
    def process_batch(self, images: list) -> list:
        """
        Process batch of images through pipeline
        
        Args:
            images: List of image arrays
            
        Returns:
            List of processed images
        """
        return [self.process(image) for image in images]
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """
        Get information about the preprocessing pipeline
        
        Returns:
            Dictionary with pipeline information
        """
        return {
            'modality': self.modality,
            'preprocessor_type': type(self.preprocessor).__name__,
            'pipeline_steps': self.pipeline_steps,
            'configuration': self.config
        }