"""
Medical Data Augmentation Module
===============================

This module provides specialized data augmentation techniques for different medical imaging modalities.
Each augmentation strategy is tailored to preserve medical relevance while improving model generalization.

Classes:
- MedicalDataAugmentation: Base class for medical image augmentation
- ChestXrayAugmentation: Specialized for chest X-ray images
- SkinLesionAugmentation: Specialized for dermatoscopic images
- BrainTumorAugmentation: Specialized for brain MRI images

Usage:
    from preprocessing.data_augmentation import ChestXrayAugmentation
    
    augmenter = ChestXrayAugmentation()
    augmented_images, augmented_labels = augmenter.augment_batch(images, labels)
"""

import os
import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict, Any, Union
from abc import ABC, abstractmethod
import random
import logging
from pathlib import Path

try:
    import albumentations as A
    from albumentations import pytorch as AT
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False

try:
    from imgaug import augmenters as iaa
    import imgaug as ia
    IMGAUG_AVAILABLE = True
except ImportError:
    IMGAUG_AVAILABLE = False

try:
    from scipy import ndimage
    from scipy.ndimage import gaussian_filter, uniform_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from sklearn.utils.class_weight import compute_class_weight
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)


class MedicalDataAugmentation(ABC):
    """
    Abstract base class for medical image augmentation
    """
    
    def __init__(self, 
                 augmentation_probability: float = 0.8,
                 preserve_aspect_ratio: bool = True,
                 maintain_medical_integrity: bool = True):
        """
        Initialize augmentation pipeline
        
        Args:
            augmentation_probability: Probability of applying augmentation
            preserve_aspect_ratio: Whether to preserve image aspect ratio
            maintain_medical_integrity: Whether to maintain medical relevance
        """
        self.aug_prob = augmentation_probability
        self.preserve_aspect = preserve_aspect_ratio
        self.maintain_integrity = maintain_medical_integrity
        self.modality = "general"
        self.pipeline = None
        self._setup_pipeline()
    
    @abstractmethod
    def _setup_pipeline(self):
        """Setup augmentation pipeline - to be implemented by subclasses"""
        pass
    
    def augment_image(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Augment single image
        
        Args:
            image: Input image array
            **kwargs: Additional parameters
            
        Returns:
            Augmented image array
        """
        if self.pipeline is None:
            logger.warning("Augmentation pipeline not initialized")
            return image
        
        if random.random() > self.aug_prob:
            return image
        
        try:
            if ALBUMENTATIONS_AVAILABLE and hasattr(self.pipeline, 'apply'):
                # Albumentations pipeline
                augmented = self.pipeline(image=image)
                return augmented['image']
            elif IMGAUG_AVAILABLE:
                # imgaug pipeline
                return self.pipeline(image=image)
            else:
                # Fallback to basic augmentation
                return self._basic_augment(image)
        except Exception as e:
            logger.error(f"Augmentation failed: {str(e)}")
            return image
    
    def augment_batch(self, 
                     images: List[np.ndarray], 
                     labels: Optional[List[Any]] = None,
                     **kwargs) -> Tuple[List[np.ndarray], Optional[List[Any]]]:
        """
        Augment batch of images
        
        Args:
            images: List of image arrays
            labels: Optional list of labels
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (augmented_images, augmented_labels)
        """
        augmented_images = []
        augmented_labels = [] if labels is not None else None
        
        for i, image in enumerate(images):
            aug_image = self.augment_image(image, **kwargs)
            augmented_images.append(aug_image)
            
            if labels is not None:
                augmented_labels.append(labels[i])
        
        return augmented_images, augmented_labels
    
    def _basic_augment(self, image: np.ndarray) -> np.ndarray:
        """Basic augmentation fallback when advanced libraries unavailable"""
        # Simple rotation
        if random.random() < 0.5:
            angle = random.uniform(-10, 10)
            center = (image.shape[1] // 2, image.shape[0] // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))
        
        # Simple brightness adjustment
        if random.random() < 0.5:
            brightness = random.uniform(0.8, 1.2)
            image = np.clip(image * brightness, 0, 255).astype(np.uint8)
        
        return image
    
    def balance_dataset(self, 
                       images: List[np.ndarray], 
                       labels: List[Any],
                       target_size_per_class: Optional[int] = None) -> Tuple[List[np.ndarray], List[Any]]:
        """
        Balance dataset using augmentation
        
        Args:
            images: List of image arrays
            labels: List of corresponding labels
            target_size_per_class: Target number of samples per class
            
        Returns:
            Balanced dataset
        """
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available for class balancing")
            return images, labels
        
        # Count samples per class
        unique_labels, counts = np.unique(labels, return_counts=True)
        class_counts = dict(zip(unique_labels, counts))
        
        if target_size_per_class is None:
            target_size_per_class = max(counts)
        
        balanced_images = []
        balanced_labels = []
        
        for label in unique_labels:
            # Get images for this class
            class_indices = [i for i, l in enumerate(labels) if l == label]
            class_images = [images[i] for i in class_indices]
            
            current_count = len(class_images)
            needed = target_size_per_class - current_count
            
            # Add original images
            balanced_images.extend(class_images)
            balanced_labels.extend([label] * current_count)
            
            # Generate additional samples if needed
            if needed > 0:
                for _ in range(needed):
                    # Randomly select image to augment
                    source_img = random.choice(class_images)
                    aug_img = self.augment_image(source_img)
                    
                    balanced_images.append(aug_img)
                    balanced_labels.append(label)
        
        logger.info(f"Balanced dataset: {len(balanced_images)} total samples")
        return balanced_images, balanced_labels


class ChestXrayAugmentation(MedicalDataAugmentation):
    """
    Specialized augmentation for chest X-ray images
    Conservative augmentations to maintain anatomical accuracy
    """
    
    def __init__(self, **kwargs):
        self.modality = "chest_xray"
        super().__init__(**kwargs)
    
    def _setup_pipeline(self):
        """Setup chest X-ray specific augmentation pipeline"""
        if ALBUMENTATIONS_AVAILABLE:
            self.pipeline = A.Compose([
                # Geometric transformations (conservative)
                A.Rotate(limit=10, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0),
                A.ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=0.1,
                    rotate_limit=5,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=0.5
                ),
                
                # Intensity transformations
                A.RandomBrightnessContrast(
                    brightness_limit=0.1,
                    contrast_limit=0.1,
                    p=0.5
                ),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
                
                # Noise and blur (minimal)
                A.GaussNoise(var_limit=(5.0, 20.0), p=0.3),
                A.GaussianBlur(blur_limit=(1, 3), p=0.3),
                
                # Elastic deformation (very subtle)
                A.ElasticTransform(
                    alpha=50,
                    sigma=5,
                    alpha_affine=5,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=0.2
                ),
                
                # Ensure proper normalization
                A.Normalize(mean=[0.485], std=[0.229], p=1.0),
            ], p=self.aug_prob)
        
        elif IMGAUG_AVAILABLE:
            self.pipeline = iaa.Sequential([
                iaa.Sometimes(0.5, iaa.Affine(rotate=(-10, 10), mode='constant')),
                iaa.Sometimes(0.5, iaa.LinearContrast((0.9, 1.1))),
                iaa.Sometimes(0.5, iaa.Add((-20, 20))),
                iaa.Sometimes(0.3, iaa.GaussianBlur(sigma=(0.5, 1.5))),
            ])
        
        else:
            logger.warning("Advanced augmentation libraries not available")
    
    def augment_with_masks(self, 
                          image: np.ndarray, 
                          lung_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Augment chest X-ray with lung segmentation mask
        
        Args:
            image: Chest X-ray image
            lung_mask: Optional lung segmentation mask
            
        Returns:
            Augmented image and mask
        """
        if lung_mask is not None and ALBUMENTATIONS_AVAILABLE:
            # Use mask-aware augmentation
            pipeline = A.Compose([
                A.Rotate(limit=8, p=0.5),
                A.ShiftScaleRotate(shift_limit=0.03, scale_limit=0.08, rotate_limit=3, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.08, contrast_limit=0.08, p=0.5),
            ])
            
            augmented = pipeline(image=image, mask=lung_mask)
            return augmented['image'], augmented['mask']
        else:
            return self.augment_image(image), lung_mask


class SkinLesionAugmentation(MedicalDataAugmentation):
    """
    Specialized augmentation for dermatoscopic skin lesion images
    More aggressive augmentations suitable for skin images
    """
    
    def __init__(self, **kwargs):
        self.modality = "skin_lesion"
        super().__init__(**kwargs)
    
    def _setup_pipeline(self):
        """Setup skin lesion specific augmentation pipeline"""
        if ALBUMENTATIONS_AVAILABLE:
            self.pipeline = A.Compose([
                # Geometric transformations (more aggressive for skin)
                A.Rotate(limit=90, p=0.7),  # Full rotation for skin lesions
                A.RandomResizedCrop(height=224, width=224, scale=(0.8, 1.0), p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.2,
                    rotate_limit=45,
                    p=0.6
                ),
                
                # Color augmentations (important for skin lesions)
                A.HueSaturationValue(
                    hue_shift_limit=20,
                    sat_shift_limit=30,
                    val_shift_limit=20,
                    p=0.6
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.6
                ),
                A.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1,
                    p=0.5
                ),
                
                # Advanced augmentations
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.4),
                A.RandomGamma(gamma_limit=(80, 120), p=0.4),
                
                # Noise and artifacts
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=0.3),
                
                # Blur effects
                A.OneOf([
                    A.MotionBlur(blur_limit=3, p=1.0),
                    A.GaussianBlur(blur_limit=3, p=1.0),
                    A.MedianBlur(blur_limit=3, p=1.0),
                ], p=0.3),
                
                # Elastic deformation
                A.ElasticTransform(
                    alpha=120,
                    sigma=6,
                    alpha_affine=10,
                    p=0.3
                ),
                
                # Grid distortion
                A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.3),
                
                # Normalize
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ], p=self.aug_prob)
        
        elif IMGAUG_AVAILABLE:
            self.pipeline = iaa.Sequential([
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
                iaa.Sometimes(0.7, iaa.Affine(rotate=(-90, 90))),
                iaa.Sometimes(0.5, iaa.Affine(scale=(0.8, 1.2))),
                iaa.Sometimes(0.6, iaa.LinearContrast((0.8, 1.2))),
                iaa.Sometimes(0.6, iaa.Add((-30, 30))),
                iaa.Sometimes(0.5, iaa.AddToHueAndSaturation((-20, 20))),
                iaa.Sometimes(0.3, iaa.GaussianBlur(sigma=(0.5, 2.0))),
            ])
    
    def augment_with_hair_artifacts(self, image: np.ndarray) -> np.ndarray:
        """
        Add synthetic hair artifacts to simulate real dermatoscopic conditions
        
        Args:
            image: Skin lesion image
            
        Returns:
            Image with synthetic hair artifacts
        """
        if random.random() < 0.3:  # 30% chance to add hair
            hair_image = self._generate_hair_pattern(image.shape[:2])
            # Blend hair pattern with image
            alpha = random.uniform(0.1, 0.3)
            image = cv2.addWeighted(image, 1-alpha, hair_image, alpha, 0)
        
        return image
    
    def _generate_hair_pattern(self, shape: Tuple[int, int]) -> np.ndarray:
        """Generate synthetic hair pattern"""
        hair_image = np.zeros((*shape, 3), dtype=np.uint8)
        
        # Generate random hair strands
        num_hairs = random.randint(5, 15)
        
        for _ in range(num_hairs):
            # Random hair parameters
            thickness = random.randint(1, 3)
            color = random.randint(0, 100)  # Dark hair
            
            # Random hair path
            start_x = random.randint(0, shape[1])
            start_y = random.randint(0, shape[0])
            
            length = random.randint(20, min(shape[0], shape[1]) // 2)
            angle = random.uniform(0, 2 * np.pi)
            
            end_x = int(start_x + length * np.cos(angle))
            end_y = int(start_y + length * np.sin(angle))
            
            # Ensure coordinates are within bounds
            end_x = max(0, min(end_x, shape[1] - 1))
            end_y = max(0, min(end_y, shape[0] - 1))
            
            # Draw hair strand
            cv2.line(hair_image, (start_x, start_y), (end_x, end_y), 
                    (color, color, color), thickness)
        
        return hair_image


class BrainTumorAugmentation(MedicalDataAugmentation):
    """
    Specialized augmentation for brain MRI images
    Conservative augmentations to preserve neuroanatomical structures
    """
    
    def __init__(self, **kwargs):
        self.modality = "brain_tumor"
        super().__init__(**kwargs)
    
    def _setup_pipeline(self):
        """Setup brain MRI specific augmentation pipeline"""
        if ALBUMENTATIONS_AVAILABLE:
            self.pipeline = A.Compose([
                # Very conservative geometric transformations
                A.Rotate(limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0),
                A.ShiftScaleRotate(
                    shift_limit=0.08,
                    scale_limit=0.15,
                    rotate_limit=10,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=0.5
                ),
                
                # No horizontal flip (brain anatomy is not symmetric)
                # Intensity transformations suitable for MRI
                A.RandomBrightnessContrast(
                    brightness_limit=0.15,
                    contrast_limit=0.15,
                    p=0.6
                ),
                A.RandomGamma(gamma_limit=(85, 115), p=0.4),
                
                # MRI-specific augmentations
                A.GaussNoise(var_limit=(5.0, 25.0), p=0.3),
                A.GaussianBlur(blur_limit=(1, 2), p=0.2),
                
                # Subtle elastic deformation
                A.ElasticTransform(
                    alpha=30,
                    sigma=4,
                    alpha_affine=3,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=0.2
                ),
                
                # Grid distortion (minimal)
                A.GridDistortion(num_steps=3, distort_limit=0.05, p=0.2),
                
                # Normalize for brain MRI
                A.Normalize(mean=[0.5], std=[0.5]),
            ], p=self.aug_prob)
        
        elif IMGAUG_AVAILABLE:
            self.pipeline = iaa.Sequential([
                iaa.Sometimes(0.5, iaa.Affine(
                    rotate=(-15, 15),
                    scale=(0.9, 1.1),
                    mode='constant'
                )),
                iaa.Sometimes(0.6, iaa.LinearContrast((0.85, 1.15))),
                iaa.Sometimes(0.4, iaa.Add((-15, 15))),
                iaa.Sometimes(0.3, iaa.GaussianBlur(sigma=(0.5, 1.0))),
            ])
    
    def augment_with_intensity_correction(self, image: np.ndarray) -> np.ndarray:
        """
        Augment with MRI-specific intensity variations
        
        Args:
            image: Brain MRI image
            
        Returns:
            Augmented image with intensity corrections
        """
        if random.random() < 0.4:
            # Simulate bias field inhomogeneity
            if SCIPY_AVAILABLE:
                # Create smooth bias field
                bias_field = self._generate_bias_field(image.shape[:2])
                image = image * bias_field
                image = np.clip(image, 0, 255).astype(np.uint8)
        
        return image
    
    def _generate_bias_field(self, shape: Tuple[int, int]) -> np.ndarray:
        """Generate smooth bias field for MRI simulation"""
        # Create Gaussian field
        center_x, center_y = shape[1] // 2, shape[0] // 2
        y, x = np.ogrid[:shape[0], :shape[1]]
        
        # Random bias parameters
        intensity = random.uniform(0.8, 1.2)
        sigma_x = random.uniform(shape[1] * 0.3, shape[1] * 0.7)
        sigma_y = random.uniform(shape[0] * 0.3, shape[0] * 0.7)
        
        bias_field = intensity * np.exp(
            -((x - center_x) ** 2) / (2 * sigma_x ** 2) -
            ((y - center_y) ** 2) / (2 * sigma_y ** 2)
        )
        
        # Normalize to reasonable range
        bias_field = 0.7 + 0.6 * (bias_field - bias_field.min()) / (bias_field.max() - bias_field.min())
        
        return bias_field


# Factory function
def get_augmentation_pipeline(modality: str, **kwargs) -> MedicalDataAugmentation:
    """
    Factory function to get augmentation pipeline for specific modality
    
    Args:
        modality: Medical imaging modality
        **kwargs: Additional parameters for augmentation
        
    Returns:
        Appropriate augmentation pipeline
        
    Raises:
        ValueError: If modality is not supported
    """
    modality_lower = modality.lower()
    
    if modality_lower == 'chest_xray':
        return ChestXrayAugmentation(**kwargs)
    elif modality_lower == 'skin_lesion':
        return SkinLesionAugmentation(**kwargs)
    elif modality_lower == 'brain_tumor':
        return BrainTumorAugmentation(**kwargs)
    else:
        raise ValueError(f"Unsupported modality: {modality}. Supported: chest_xray, skin_lesion, brain_tumor")


def create_balanced_dataset(
    images: List[np.ndarray],
    labels: List[Any],
    modality: str,
    augmentation_factor: int = 2,
    **kwargs
) -> Tuple[List[np.ndarray], List[Any]]:
    """
    Create balanced dataset using modality-specific augmentation
    
    Args:
        images: List of image arrays
        labels: List of corresponding labels
        modality: Medical imaging modality
        augmentation_factor: Factor by which to increase minority classes
        **kwargs: Additional parameters
        
    Returns:
        Balanced dataset
    """
    augmenter = get_augmentation_pipeline(modality, **kwargs)
    
    # Get class distribution
    unique_labels, counts = np.unique(labels, return_counts=True)
    max_count = max(counts)
    target_count = max_count * augmentation_factor
    
    return augmenter.balance_dataset(images, labels, target_count)


def apply_augmentation_strategy(
    images: List[np.ndarray],
    labels: List[Any],
    strategy: str,
    modality: str,
    **kwargs
) -> Tuple[List[np.ndarray], List[Any]]:
    """
    Apply specific augmentation strategy
    
    Args:
        images: List of image arrays
        labels: List of corresponding labels
        strategy: Augmentation strategy ('balanced', 'aggressive', 'conservative')
        modality: Medical imaging modality
        **kwargs: Additional parameters
        
    Returns:
        Augmented dataset
    """
    # Adjust augmentation parameters based on strategy
    if strategy == 'conservative':
        kwargs['augmentation_probability'] = 0.3
    elif strategy == 'aggressive':
        kwargs['augmentation_probability'] = 0.9
    else:  # balanced
        kwargs['augmentation_probability'] = 0.6
    
    augmenter = get_augmentation_pipeline(modality, **kwargs)
    
    if strategy == 'balanced':
        return augmenter.balance_dataset(images, labels)
    else:
        return augmenter.augment_batch(images, labels)


class AugmentationPipeline:
    """
    Complete augmentation pipeline with configuration management
    """
    
    def __init__(self, 
                 modality: str,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize augmentation pipeline
        
        Args:
            modality: Medical imaging modality
            config: Augmentation configuration
        """
        self.modality = modality
        self.config = config or self._get_default_config()
        self.augmenter = get_augmentation_pipeline(modality, **self.config)
        
        # Statistics tracking
        self.original_count = 0
        self.augmented_count = 0
        self.class_distribution = {}
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for modality"""
        default_configs = {
            'chest_xray': {
                'augmentation_probability': 0.5,
                'preserve_aspect_ratio': True,
                'maintain_medical_integrity': True
            },
            'skin_lesion': {
                'augmentation_probability': 0.7,
                'preserve_aspect_ratio': False,
                'maintain_medical_integrity': True
            },
            'brain_tumor': {
                'augmentation_probability': 0.4,
                'preserve_aspect_ratio': True,
                'maintain_medical_integrity': True
            }
        }
        
        return default_configs.get(self.modality, {})
    
    def process_dataset(self, 
                       images: List[np.ndarray],
                       labels: List[Any],
                       balance_classes: bool = True,
                       save_examples: bool = False,
                       examples_dir: Optional[str] = None) -> Tuple[List[np.ndarray], List[Any]]:
        """
        Process complete dataset with augmentation
        
        Args:
            images: List of image arrays
            labels: List of corresponding labels
            balance_classes: Whether to balance class distribution
            save_examples: Whether to save augmentation examples
            examples_dir: Directory to save examples
            
        Returns:
            Processed dataset
        """
        self.original_count = len(images)
        
        # Track original class distribution
        unique_labels, counts = np.unique(labels, return_counts=True)
        self.class_distribution['original'] = dict(zip(unique_labels, counts))
        
        # Apply augmentation strategy
        if balance_classes:
            processed_images, processed_labels = self.augmenter.balance_dataset(images, labels)
        else:
            processed_images, processed_labels = self.augmenter.augment_batch(images, labels)
        
        self.augmented_count = len(processed_images)
        
        # Track final class distribution
        unique_labels, counts = np.unique(processed_labels, return_counts=True)
        self.class_distribution['final'] = dict(zip(unique_labels, counts))
        
        # Save examples if requested
        if save_examples and examples_dir:
            self._save_examples(images[:5], processed_images[:5], examples_dir)
        
        logger.info(f"Augmentation complete: {self.original_count} -> {self.augmented_count} samples")
        
        return processed_images, processed_labels
    
    def _save_examples(self, 
                      original_images: List[np.ndarray],
                      augmented_images: List[np.ndarray],
                      save_dir: str):
        """Save augmentation examples for visualization"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        for i, (orig, aug) in enumerate(zip(original_images, augmented_images)):
            # Save original
            orig_path = save_path / f"{self.modality}_original_{i}.png"
            cv2.imwrite(str(orig_path), cv2.cvtColor(orig, cv2.COLOR_RGB2BGR))
            
            # Save augmented
            aug_path = save_path / f"{self.modality}_augmented_{i}.png"
            cv2.imwrite(str(aug_path), cv2.cvtColor(aug, cv2.COLOR_RGB2BGR))
        
        logger.info(f"Saved augmentation examples to {save_dir}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get augmentation statistics"""
        return {
            'modality': self.modality,
            'original_count': self.original_count,
            'augmented_count': self.augmented_count,
            'augmentation_ratio': self.augmented_count / self.original_count if self.original_count > 0 else 0,
            'class_distribution': self.class_distribution,
            'configuration': self.config
        }


# Utility functions for testing and validation
def validate_augmentation(
    original_image: np.ndarray,
    augmented_image: np.ndarray,
    tolerance: float = 0.1
) -> Dict[str, bool]:
    """
    Validate that augmentation preserves important image properties
    
    Args:
        original_image: Original image
        augmented_image: Augmented image
        tolerance: Tolerance for property differences
        
    Returns:
        Validation results
    """
    results = {
        'shape_preserved': original_image.shape == augmented_image.shape,
        'intensity_range_reasonable': True,
        'not_identical': True,
        'quality_maintained': True
    }
    
    # Check intensity range
    orig_mean = np.mean(original_image)
    aug_mean = np.mean(augmented_image)
    intensity_diff = abs(aug_mean - orig_mean) / orig_mean
    results['intensity_range_reasonable'] = intensity_diff < tolerance * 2
    
    # Check if images are not identical
    diff = np.mean(np.abs(original_image.astype(float) - augmented_image.astype(float)))
    results['not_identical'] = diff > 1.0  # Some difference should exist
    
    # Check quality maintenance (contrast preservation)
    orig_std = np.std(original_image)
    aug_std = np.std(augmented_image)
    if orig_std > 0:
        contrast_ratio = aug_std / orig_std
        results['quality_maintained'] = 0.5 < contrast_ratio < 2.0
    
    return results


def test_augmentation_pipeline(modality: str, test_images: List[np.ndarray]) -> Dict[str, Any]:
    """
    Test augmentation pipeline with sample images
    
    Args:
        modality: Medical imaging modality
        test_images: Sample images for testing
        
    Returns:
        Test results
    """
    logger.info(f"Testing {modality} augmentation pipeline...")
    
    try:
        augmenter = get_augmentation_pipeline(modality)
        
        results = {
            'modality': modality,
            'pipeline_created': True,
            'augmentation_results': [],
            'overall_success': True,
            'errors': []
        }
        
        for i, image in enumerate(test_images[:3]):  # Test first 3 images
            try:
                aug_image = augmenter.augment_image(image)
                validation = validate_augmentation(image, aug_image)
                
                results['augmentation_results'].append({
                    'image_index': i,
                    'original_shape': image.shape,
                    'augmented_shape': aug_image.shape,
                    'validation': validation
                })
                
            except Exception as e:
                error_msg = f"Failed to augment image {i}: {str(e)}"
                results['errors'].append(error_msg)
                results['overall_success'] = False
        
        return results
        
    except Exception as e:
        return {
            'modality': modality,
            'pipeline_created': False,
            'overall_success': False,
            'errors': [f"Failed to create pipeline: {str(e)}"]
        }


# Export all classes and functions
__all__ = [
    'MedicalDataAugmentation',
    'ChestXrayAugmentation',
    'SkinLesionAugmentation',
    'BrainTumorAugmentation',
    'get_augmentation_pipeline',
    'create_balanced_dataset',
    'apply_augmentation_strategy',
    'AugmentationPipeline',
    'validate_augmentation',
    'test_augmentation_pipeline'
]


# Configuration examples
class AugmentationConfig:
    """Pre-defined augmentation configurations"""
    
    CONSERVATIVE = {
        'augmentation_probability': 0.3,
        'preserve_aspect_ratio': True,
        'maintain_medical_integrity': True
    }
    
    BALANCED = {
        'augmentation_probability': 0.6,
        'preserve_aspect_ratio': True,
        'maintain_medical_integrity': True
    }
    
    AGGRESSIVE = {
        'augmentation_probability': 0.9,
        'preserve_aspect_ratio': False,
        'maintain_medical_integrity': True
    }
    
    CHEST_XRAY_OPTIMAL = {
        'augmentation_probability': 0.5,
        'preserve_aspect_ratio': True,
        'maintain_medical_integrity': True,
        'rotation_limit': 10,
        'brightness_limit': 0.1,
        'contrast_limit': 0.1
    }
    
    SKIN_LESION_OPTIMAL = {
        'augmentation_probability': 0.7,
        'preserve_aspect_ratio': False,
        'maintain_medical_integrity': True,
        'rotation_limit': 90,
        'color_augmentation': True,
        'hair_artifacts': True
    }
    
    BRAIN_MRI_OPTIMAL = {
        'augmentation_probability': 0.4,
        'preserve_aspect_ratio': True,
        'maintain_medical_integrity': True,
        'rotation_limit': 15,
        'no_flip': True,
        'intensity_correction': True
    }