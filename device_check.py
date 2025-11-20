"""
Device Detection and Configuration Utilities

This module provides automatic detection and configuration of the best available
compute device for PyTorch operations. It handles cross-platform compatibility
across NVIDIA CUDA GPUs, Apple Silicon MPS, and CPU fallback scenarios.

The module prioritizes devices in the following order:
1. Apple Metal Performance Shaders (MPS) - for Apple Silicon Macs
2. NVIDIA CUDA - for NVIDIA GPUs  
3. CPU - fallback for all other systems

Key Features:
    - Automatic device detection with intelligent prioritization
    - Cross-platform compatibility (macOS, Linux, Windows)
    - Informative logging of selected device
    - Override capability for manual device selection
    - Graceful fallback to CPU when GPU acceleration unavailable

Functions:
    check_set_gpu: Main function for device detection and configuration

Dependencies:
    - torch: For device detection and configuration

Author: Dhruv Salot
Date: November 2025
License: MIT

Usage:
    >>> from device_check import check_set_gpu
    >>> device = check_set_gpu()
    >>> print(f"Using device: {device}")
    
    # Manual override example
    >>> device = check_set_gpu(override='cuda')
    >>> print(f"Forced device: {device}")

Note:
    - MPS is prioritized over CUDA for optimal performance on Apple Silicon
    - Device selection considers both availability and performance characteristics
    - The function provides clear feedback about the selected device
"""

import torch

def check_set_gpu(override=None):
    """
    Automatically detect and configure the best available compute device.
    
    This function intelligently selects the most appropriate device for PyTorch
    computations based on hardware availability and platform-specific optimizations.
    It prioritizes Apple MPS over CUDA for better performance on Apple Silicon systems.
    
    Args:
        override (str, optional): Force the use of a specific device type.
                                 Supported values: 'cuda', 'mps', 'cpu'
                                 If None, automatic detection is performed.
                                 
    Returns:
        torch.device: The selected PyTorch device object ready for use in model
                     training and inference operations.
                     
    Device Priority Order:
        1. Apple MPS (Metal Performance Shaders) - optimized for Apple Silicon
        2. NVIDIA CUDA - optimized for NVIDIA GPUs
        3. CPU - universal fallback with broad compatibility
        
    Examples:
        >>> # Automatic detection
        >>> device = check_set_gpu()
        Using GPU: Apple Metal Performance Shaders (MPS)
        
        >>> # Manual override
        >>> device = check_set_gpu(override='cpu') 
        Using CPU
        
        >>> # Use in model training
        >>> model = model.to(device)
        >>> data = data.to(device)
        
    Note:
        - MPS provides superior performance on Apple Silicon compared to CPU
        - CUDA is highly optimized for NVIDIA GPU architectures
        - CPU fallback ensures compatibility across all systems
        - Override parameter useful for debugging and testing scenarios
        - Device selection affects memory allocation and computation speed
    """
    if override is None:
        # Automatic device detection with intelligent prioritization
        if torch.backends.mps.is_available():
            device = torch.device('mps')
            print("Using GPU: Apple Metal Performance Shaders (MPS)")
            return device
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            return device
        device = torch.device('cpu')
        print("Using CPU")
        return device
    else:
        # Manual device override
        device = torch.device(override)
        print(f"Using overridden device: {device}")
    return device
