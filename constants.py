"""
Project Constants and Configuration Settings

This module defines all the constants, configuration parameters, and global settings
used throughout the FRCNN Utils project. It centralizes configuration management
for dataset paths, model parameters, training hyperparameters, and device settings.

The constants are organized into several categories:
- Dataset and file path configurations
- Pest classification definitions  
- Image processing parameters
- Model training hyperparameters
- Device and optimization settings

Key Features:
    - Centralized configuration management
    - Automatic device detection and setup
    - Pest class definitions for agricultural applications
    - Optimized hyperparameters for object detection training
    - Path resolution relative to project structure

Constants Exported:
    PROJECT_DIR_PATH: Root directory of the project
    DATASET_DIR_PATH: Directory containing datasets
    DATA_YAML_PATH: Path to dataset configuration file
    PEST_CLASSES: List of pest category names
    IMAGE_SIZE: Standard image dimensions for model input
    IMAGE_RESIZE_FACTOR: Factor for resizing images during processing
    TRAIN_BATCH_SIZE: Batch size for training
    VALID_BATCH_SIZE: Batch size for validation
    LEARNING_RATE: Learning rate for optimization
    MOMENTUM: Momentum for SGD optimizer
    WEIGHT_DECAY: Weight decay for regularization
    NUM_EPOCHS: Number of training epochs
    OPTIMIZER_TYPE: Optimizer class to use
    DEVICE: Automatically detected compute device

Dependencies:
    - os: For path operations
    - torch.optim: For optimizer definitions
    - device_check: For automatic device detection

Author: Dhruv Salot
Date: November 2025
License: MIT

Usage:
    >>> from constants import DEVICE, PEST_CLASSES, LEARNING_RATE
    >>> print(f"Training on {DEVICE} with {len(PEST_CLASSES)} classes")
    >>> print(f"Using learning rate: {LEARNING_RATE}")
"""

import os
from torch.optim import SGD
from .device_check import check_set_gpu

# Defining constants
# Dataset paths

__all__ = [
    'PROJECT_DIR_PATH',
    'DATASET_DIR_PATH', 
    'DATA_YAML_PATH',
    'PEST_CLASSES',
    'IMAGE_SIZE',
    'IMAGE_RESIZE_FACTOR',
    'TRAIN_BATCH_SIZE',
    'VALID_BATCH_SIZE',
    'LEARNING_RATE',
    'MOMENTUM',
    'WEIGHT_DECAY',
    'NUM_EPOCHS',
    'OPTIMIZER_TYPE',
    'DEVICE'
]

# Path Configuration
# Automatically determine project structure relative to this file
_this_file_path = os.path.abspath(__file__)
_this_dir_path = os.path.dirname(_this_file_path)

PROJECT_DIR_PATH = os.path.dirname(_this_dir_path)
DATASET_DIR_PATH = os.path.join(PROJECT_DIR_PATH, 'dataset')

# Data yaml file path
DATA_YAML_PATH = os.path.join(_this_dir_path, 'dataset/data.yaml')

# Pest Classification Configuration
# Pests classes mapping for agricultural object detection
# Categories based on AgroPest-12 dataset for crop pest identification
PEST_CLASSES = [
    'ant',         # Small social insects, can damage plants
    'bee',         # Beneficial pollinators, sometimes considered pests in certain contexts  
    'beetle',      # Hard-bodied insects, many species are crop pests
    'caterpillar', # Larval stage of moths/butterflies, major crop damage
    'earthworm',   # Soil invertebrates, generally beneficial but can indicate issues
    'earwig',      # Nocturnal insects, can damage soft plant tissues
    'grasshopper', # Jumping insects, can cause significant crop damage in swarms
    'moth',        # Adult stage, many species are agricultural pests
    'slug',        # Soft-bodied mollusks, damage leaves and fruits
    'snail',       # Shelled mollusks, similar damage patterns to slugs
    'wasp',        # Flying insects, can be both beneficial and harmful
    'weevil'       # Specialized beetles, significant stored grain pests
]

# Image Processing Configuration
# IMAGE_SIZE for model input - standard dimensions for Faster R-CNN
# Using square images for consistent aspect ratios during training
IMAGE_SIZE = (640, 640) 
# Image resize factor for preprocessing - reduces computational load while maintaining quality
IMAGE_RESIZE_FACTOR = 0.5

# Training Configuration 
# Optimized hyperparameters for Faster R-CNN object detection training

# Batch size for training - balanced for memory usage and gradient stability
TRAIN_BATCH_SIZE = 16

# Batch size for validation - can be larger since no gradients computed
VALID_BATCH_SIZE = 12

# Learning rate - tuned for SGD with momentum on detection tasks  
LEARNING_RATE = 5e-3

# SGD momentum factor - helps accelerate convergence and reduce oscillations
MOMENTUM = 0.9

# L2 regularization weight decay - prevents overfitting
WEIGHT_DECAY = 5e-4

# Total number of training epochs - sufficient for convergence on pest detection
NUM_EPOCHS = 50

# Optimizer class - SGD with momentum is proven effective for detection models
OPTIMIZER_TYPE = SGD

# Device Configuration
# Automatically detect and configure the best available compute device
# Priority order: CUDA GPU > Apple MPS > CPU
DEVICE = check_set_gpu()

# Additional Configuration Notes:
# - IMAGE_SIZE chosen for balance between detail preservation and computational efficiency
# - Batch sizes optimized for typical GPU memory constraints (8-16GB)
# - Learning rate and weight decay values follow proven practices from detection literature
# - Number of epochs sufficient for convergence on agricultural pest datasets
# - Device selection handles cross-platform compatibility (NVIDIA, Apple Silicon, CPU)