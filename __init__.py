"""
FRCNN Utils Package

A comprehensive utility package for Faster R-CNN object detection models, specifically designed
for crop pest detection tasks. This package provides utilities for COCO dataset handling,
model training, evaluation, visualization, and various helper functions.

Modules:
    coco: COCO dataset utilities and converters
    coco_eval: COCO evaluation metrics and evaluators  
    coco_utils: COCO data processing utilities
    constants: Project-wide constants and configuration
    device_check: Device detection and configuration utilities
    engine: Training and evaluation engine
    ploter: Visualization utilities for images and predictions
    req_utils: Request utilities and distributed training helpers

Dependencies:
    - torch: PyTorch deep learning framework
    - torchvision: Computer vision utilities for PyTorch
    - pycocotools: COCO dataset API
    - PIL: Python Imaging Library
    - matplotlib: Plotting and visualization
    - numpy: Numerical computing

Author: Dhruv Salot
Date: November 2025
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Dhruv Salot"
__email__ = "dhruv@example.com"

# Package metadata
__all__ = [
    'coco',
    'coco_eval', 
    'coco_utils',
    'constants',
    'device_check',
    'engine',
    'ploter',
    'req_utils'
]