"""
Model Configuration Module
========================

This module contains model-specific configurations for different medical imaging modalities.
Each configuration class handles hyperparameters, architecture settings, and training configs
for chest X-ray, skin lesion, and brain tumor classification models.

Usage:
    from config.model_config import ChestXrayConfig, get_model_config
    
    # Get specific model configuration
    config = ChestXrayConfig()
    print(config.input_shape)
    
    # Or use factory function
    config = get_model_config('chest_xray')
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
from enum import Enum
import tensorflow as tf
from pathlib import Path


class ModelArchitecture(Enum):
    """Supported CNN architectures"""
    RESNET50 = "ResNet50"
    RESNET101 = "ResNet101"
    EFFICIENTNET_B0 = "EfficientNetB0"
    EFFICIENTNET_B3 = "EfficientNetB3"
    DENSENET121 = "DenseNet121"
    DENSENET169 = "DenseNet169"
    INCEPTION_V3 = "InceptionV3"
    MOBILENET_V2 = "MobileNetV2"
    VGG16 = "VGG16"
    XCEPTION = "Xception"


class Optimizer(Enum):
    """Supported optimizers"""
    ADAM = "Adam"
    ADAMW = "AdamW"
    SGD = "SGD"
    RMSPROP = "RMSprop"
    ADAGRAD = "Adagrad"


@dataclass
class BaseModelConfig:
    """Base configuration for all medical imaging models"""
    
    # Model identification
    model_name: str = "base_model"
    modality: str = "general"
    
    # Architecture settings
    architecture: str = "ResNet50"
    input_shape: Tuple[int, int, int] = (224, 224, 3)
    num_classes: int = 2
    class_names: List[str] = field(default_factory=list)
    
    # Pre-trained weights
    weights: str = "imagenet"  # imagenet, None, or path to weights
    include_top: bool = False
    pooling: str = "avg"  # avg, max, or None
    
    # Transfer learning
    freeze_base_model: bool = True
    freeze_layers: int = 100  # Number of layers to freeze
    unfreeze_after_epoch: int = 10  # Unfreeze layers after this epoch
    
    # Regularization
    dropout_rate: float = 0.5
    l2_regularization: float = 0.0001
    
    # Training hyperparameters
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    optimizer: str = "Adam"
    loss_function: str = "categorical_crossentropy"
    
    # Learning rate schedule
    use_lr_schedule: bool = True
    lr_schedule_type: str = "ReduceLROnPlateau"  # ReduceLROnPlateau, CosineDecay, ExponentialDecay
    lr_factor: float = 0.5
    lr_patience: int = 5
    min_lr: float = 1e-7
    
    # Data augmentation
    augmentation_enabled: bool = True
    rotation_range: int = 20
    width_shift_range: float = 0.2
    height_shift_range: float = 0.2
    horizontal_flip: bool = True
    vertical_flip: bool = False
    zoom_range: float = 0.2
    shear_range: float = 0.1
    brightness_range: Tuple[float, float] = (0.8, 1.2)
    
    # Image preprocessing
    rescale: float = 1.0/255.0
    preprocessing_function: Optional[str] = None
    featurewise_center: bool = False
    featurewise_std_normalization: bool = False
    
    # Callbacks
    early_stopping_patience: int = 15
    reduce_lr_patience: int = 5
    save_best_only: bool = True
    monitor_metric: str = "val_accuracy"
    
    # Model paths
    model_dir: str = "./models/saved_models"
    checkpoint_dir: str = "./models/checkpoints"
    logs_dir: str = "./logs/training"
    
    # Performance optimization
    use_mixed_precision: bool = True
    use_multi_gpu: bool = False
    num_workers: int = 4
    prefetch_factor: int = 2
    
    def __post_init__(self):
        """Validate and setup configuration after initialization"""
        # Create directories if they don't exist
        Path(self.model_dir).mkdir(parents=True, exist_ok=True)
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.logs_dir).mkdir(parents=True, exist_ok=True)
        
        # Set default class names if not provided
        if not self.class_names:
            self.class_names = [f"class_{i}" for i in range(self.num_classes)]
    
    def get_model_path(self) -> str:
        """Get the full path for saving the model"""
        return os.path.join(self.model_dir, f"{self.model_name}_best.h5")
    
    def get_checkpoint_path(self) -> str:
        """Get the checkpoint path"""
        return os.path.join(self.checkpoint_dir, f"{self.model_name}_checkpoint.h5")
    
    def get_optimizer(self):
        """Get configured optimizer instance"""
        if self.optimizer == "Adam":
            return tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        elif self.optimizer == "AdamW":
            return tf.keras.optimizers.AdamW(learning_rate=self.learning_rate)
        elif self.optimizer == "SGD":
            return tf.keras.optimizers.SGD(
                learning_rate=self.learning_rate,
                momentum=0.9,
                nesterov=True
            )
        elif self.optimizer == "RMSprop":
            return tf.keras.optimizers.RMSprop(learning_rate=self.learning_rate)
        else:
            return tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
    
    def get_callbacks(self):
        """Get list of Keras callbacks"""
        callbacks = []
        
        # Early stopping
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor=self.monitor_metric,
                patience=self.early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            )
        )
        
        # Model checkpoint
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=self.get_checkpoint_path(),
                monitor=self.monitor_metric,
                save_best_only=self.save_best_only,
                verbose=1
            )
        )
        
        # Reduce learning rate on plateau
        if self.use_lr_schedule:
            callbacks.append(
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor=self.monitor_metric,
                    factor=self.lr_factor,
                    patience=self.reduce_lr_patience,
                    min_lr=self.min_lr,
                    verbose=1
                )
            )
        
        # TensorBoard
        callbacks.append(
            tf.keras.callbacks.TensorBoard(
                log_dir=os.path.join(self.logs_dir, self.model_name),
                histogram_freq=1,
                write_graph=True,
                update_freq='epoch'
            )
        )
        
        # CSV Logger
        callbacks.append(
            tf.keras.callbacks.CSVLogger(
                filename=os.path.join(self.logs_dir, f"{self.model_name}_training.csv"),
                separator=',',
                append=True
            )
        )
        
        return callbacks
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'model_name': self.model_name,
            'modality': self.modality,
            'architecture': self.architecture,
            'input_shape': self.input_shape,
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'learning_rate': self.learning_rate,
            'optimizer': self.optimizer,
        }


@dataclass
class ChestXrayConfig(BaseModelConfig):
    """Configuration for Chest X-ray classification (Pneumonia detection)"""
    
    model_name: str = "chest_xray_classifier"
    modality: str = "chest_xray"
    
    # Architecture
    architecture: str = "ResNet50"
    input_shape: Tuple[int, int, int] = (224, 224, 3)
    num_classes: int = 2
    class_names: List[str] = field(default_factory=lambda: ["NORMAL", "PNEUMONIA"])
    
    # Model settings
    freeze_layers: int = 100
    dropout_rate: float = 0.5
    
    # Training
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 0.0001
    
    # Augmentation (chest X-rays need careful augmentation)
    rotation_range: int = 10  # Limited rotation for medical images
    horizontal_flip: bool = False  # No flipping for X-rays
    vertical_flip: bool = False
    zoom_range: float = 0.1
    
    # Class imbalance handling
    use_class_weights: bool = True
    class_weight_method: str = "balanced"  # balanced or custom
    
    # Preprocessing
    apply_clahe: bool = True  # Contrast Limited Adaptive Histogram Equalization
    normalize_method: str = "z_score"  # z_score, min_max, or None


@dataclass
class SkinLesionConfig(BaseModelConfig):
    """Configuration for Skin Lesion classification (HAM10000 dataset)"""
    
    model_name: str = "skin_lesion_classifier"
    modality: str = "skin_lesion"
    
    # Architecture
    architecture: str = "EfficientNetB0"
    input_shape: Tuple[int, int, int] = (224, 224, 3)
    num_classes: int = 7
    class_names: List[str] = field(default_factory=lambda: [
        "MEL",    # Melanoma
        "NV",     # Melanocytic nevi
        "BCC",    # Basal cell carcinoma
        "AKIEC",  # Actinic keratoses
        "BKL",    # Benign keratosis
        "DF",     # Dermatofibroma
        "VASC"    # Vascular lesions
    ])
    
    # Model settings
    freeze_layers: int = 150
    dropout_rate: float = 0.3
    
    # Training
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    
    # Augmentation (dermatoscopic images can handle more augmentation)
    rotation_range: int = 90  # Full rotation for skin lesions
    horizontal_flip: bool = True
    vertical_flip: bool = True
    zoom_range: float = 0.2
    brightness_range: Tuple[float, float] = (0.7, 1.3)
    
    # Color augmentation
    apply_color_jitter: bool = True
    hue_shift_range: float = 0.1
    saturation_shift_range: float = 0.1
    
    # Class imbalance (HAM10000 is highly imbalanced)
    use_class_weights: bool = True
    use_oversampling: bool = True
    
    # Hair removal preprocessing
    apply_hair_removal: bool = True
    
    # Preprocessing
    normalize_method: str = "imagenet"  # Use ImageNet stats


@dataclass
class BrainTumorConfig(BaseModelConfig):
    """Configuration for Brain Tumor MRI classification"""
    
    model_name: str = "brain_tumor_classifier"
    modality: str = "brain_tumor"
    
    # Architecture
    architecture: str = "DenseNet121"
    input_shape: Tuple[int, int, int] = (224, 224, 3)
    num_classes: int = 4
    class_names: List[str] = field(default_factory=lambda: [
        "glioma",
        "meningioma",
        "pituitary",
        "no_tumor"
    ])
    
    # Model settings
    freeze_layers: int = 200
    dropout_rate: float = 0.4
    
    # Training
    batch_size: int = 16  # Smaller batch for brain MRIs
    epochs: int = 75
    learning_rate: float = 0.0005
    
    # Augmentation (conservative for brain MRIs)
    rotation_range: int = 15
    horizontal_flip: bool = False  # Brain anatomy is not symmetric
    vertical_flip: bool = False
    zoom_range: float = 0.15
    width_shift_range: float = 0.1
    height_shift_range: float = 0.1
    
    # MRI-specific preprocessing
    apply_skull_stripping: bool = False  # Usually pre-processed
    apply_bias_field_correction: bool = False
    normalize_method: str = "brain_mri"  # Custom MRI normalization
    
    # Intensity normalization
    intensity_clipping: bool = True
    clip_percentile: Tuple[float, float] = (1.0, 99.0)
    
    # Class balance
    use_class_weights: bool = True


# Factory function to get model configuration
def get_model_config(modality: str) -> BaseModelConfig:
    """
    Factory function to get model configuration based on modality
    
    Args:
        modality: Type of medical imaging modality
        
    Returns:
        Appropriate model configuration instance
        
    Raises:
        ValueError: If modality is not supported
    """
    modality_configs = {
        'chest_xray': ChestXrayConfig,
        'skin_lesion': SkinLesionConfig,
        'brain_tumor': BrainTumorConfig,
    }
    
    if modality not in modality_configs:
        available_modalities = list(modality_configs.keys())
        raise ValueError(
            f"Unsupported modality: {modality}. "
            f"Available modalities: {available_modalities}"
        )
    
    return modality_configs[modality]()


# Model registry for architecture mapping
MODEL_REGISTRY = {
    "ResNet50": tf.keras.applications.ResNet50,
    "ResNet101": tf.keras.applications.ResNet101,
    "EfficientNetB0": tf.keras.applications.EfficientNetB0,
    "EfficientNetB3": tf.keras.applications.EfficientNetB3,
    "DenseNet121": tf.keras.applications.DenseNet121,
    "DenseNet169": tf.keras.applications.DenseNet169,
    "InceptionV3": tf.keras.applications.InceptionV3,
    "MobileNetV2": tf.keras.applications.MobileNetV2,
    "VGG16": tf.keras.applications.VGG16,
    "Xception": tf.keras.applications.Xception,
}


def get_base_model(config: BaseModelConfig):
    """
    Get the base model architecture based on configuration
    
    Args:
        config: Model configuration instance
        
    Returns:
        Keras base model
        
    Raises:
        ValueError: If architecture is not supported
    """
    if config.architecture not in MODEL_REGISTRY:
        available_models = list(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unsupported architecture: {config.architecture}. "
            f"Available architectures: {available_models}"
        )
    
    model_class = MODEL_REGISTRY[config.architecture]
    
    # Create base model
    base_model = model_class(
        weights=config.weights if config.weights != "None" else None,
        include_top=config.include_top,
        input_shape=config.input_shape,
        pooling=config.pooling
    )
    
    # Freeze layers if specified
    if config.freeze_base_model:
        for layer in base_model.layers[:config.freeze_layers]:
            layer.trainable = False
    
    return base_model


def create_model(config: BaseModelConfig):
    """
    Create complete model with classification head
    
    Args:
        config: Model configuration instance
        
    Returns:
        Compiled Keras model
    """
    # Get base model
    base_model = get_base_model(config)
    
    # Add classification head
    x = base_model.output
    
    # Global average pooling if not already applied
    if config.pooling is None:
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    # Dense layers with dropout
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(config.dropout_rate)(x)
    
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(config.dropout_rate / 2)(x)
    
    # Output layer
    if config.num_classes == 2:
        # Binary classification
        predictions = tf.keras.layers.Dense(
            1, activation='sigmoid', name='predictions'
        )(x)
        loss = 'binary_crossentropy'
        metrics = ['accuracy', 'precision', 'recall']
    else:
        # Multi-class classification
        predictions = tf.keras.layers.Dense(
            config.num_classes, activation='softmax', name='predictions'
        )(x)
        loss = 'categorical_crossentropy'
        metrics = ['accuracy', 'top_2_accuracy']
    
    # Create model
    model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
    
    # Compile model
    model.compile(
        optimizer=config.get_optimizer(),
        loss=loss,
        metrics=metrics
    )
    
    return model


# Preprocessing functions for different modalities
class ImagePreprocessor:
    """Image preprocessing utilities for medical images"""
    
    @staticmethod
    def chest_xray_preprocess(image, config: ChestXrayConfig):
        """Preprocess chest X-ray images"""
        import cv2
        import numpy as np
        
        # Convert to grayscale if needed, then back to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        
        # Apply CLAHE if specified
        if config.apply_clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            if len(image.shape) == 3:
                for i in range(3):
                    image[:, :, i] = clahe.apply(image[:, :, i])
            else:
                image = clahe.apply(image)
        
        # Normalize
        if config.normalize_method == "z_score":
            image = (image - np.mean(image)) / (np.std(image) + 1e-8)
        elif config.normalize_method == "min_max":
            image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)
        
        return image
    
    @staticmethod
    def skin_lesion_preprocess(image, config: SkinLesionConfig):
        """Preprocess dermatoscopic images"""
        import cv2
        import numpy as np
        
        # Hair removal if specified
        if config.apply_hair_removal:
            # Simple hair removal using morphological operations
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
            _, mask = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            image = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
        
        # Normalize using ImageNet statistics if specified
        if config.normalize_method == "imagenet":
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = (image / 255.0 - mean) / std
        
        return image
    
    @staticmethod
    def brain_tumor_preprocess(image, config: BrainTumorConfig):
        """Preprocess brain MRI images"""
        import numpy as np
        
        # Intensity clipping
        if config.intensity_clipping:
            lower_percentile, upper_percentile = config.clip_percentile
            lower_bound = np.percentile(image, lower_percentile)
            upper_bound = np.percentile(image, upper_percentile)
            image = np.clip(image, lower_bound, upper_bound)
        
        # Brain MRI specific normalization
        if config.normalize_method == "brain_mri":
            # Zero-mean unit variance within brain region
            brain_mask = image > np.mean(image) * 0.1  # Simple brain mask
            if np.sum(brain_mask) > 0:
                brain_pixels = image[brain_mask]
                mean_val = np.mean(brain_pixels)
                std_val = np.std(brain_pixels)
                image = (image - mean_val) / (std_val + 1e-8)
        
        return image


# Configuration validation
def validate_config(config: BaseModelConfig) -> List[str]:
    """
    Validate model configuration
    
    Args:
        config: Model configuration to validate
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    # Check required fields
    if not config.model_name:
        errors.append("Model name is required")
    
    if config.num_classes < 2:
        errors.append("Number of classes must be >= 2")
    
    if len(config.class_names) != config.num_classes:
        errors.append(f"Number of class names ({len(config.class_names)}) "
                     f"doesn't match num_classes ({config.num_classes})")
    
    # Check hyperparameters
    if config.learning_rate <= 0:
        errors.append("Learning rate must be positive")
    
    if config.batch_size <= 0:
        errors.append("Batch size must be positive")
    
    if config.epochs <= 0:
        errors.append("Number of epochs must be positive")
    
    if not 0 <= config.dropout_rate <= 1:
        errors.append("Dropout rate must be between 0 and 1")
    
    # Check architecture
    if config.architecture not in MODEL_REGISTRY:
        errors.append(f"Unsupported architecture: {config.architecture}")
    
    # Check input shape
    if len(config.input_shape) != 3:
        errors.append("Input shape must be 3D (height, width, channels)")
    
    return errors


# Export all classes and functions
__all__ = [
    'BaseModelConfig',
    'ChestXrayConfig',
    'SkinLesionConfig',
    'BrainTumorConfig',
    'get_model_config',
    'get_base_model',
    'create_model',
    'ImagePreprocessor',
    'validate_config',
    'MODEL_REGISTRY',
    'ModelArchitecture',
    'Optimizer'
]