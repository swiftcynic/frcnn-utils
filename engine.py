"""
Training and Evaluation Engine for Object Detection Models

This module provides comprehensive training and evaluation functionality for object detection
models, particularly Faster R-CNN architectures. It includes utilities for model creation,
training loops, evaluation metrics, and cross-platform device compatibility.

The engine handles:
- Model creation and configuration with custom prediction heads
- Training loops with mixed precision support and learning rate scheduling
- Evaluation using COCO metrics and comprehensive performance analysis  
- Cross-platform device support (CUDA, MPS, CPU) with automatic fallbacks
- Progress tracking and logging with memory usage monitoring
- Prediction generation and confidence filtering

Key Features:
    - Automatic mixed precision training for improved performance and memory efficiency
    - Device-agnostic operations with intelligent fallbacks for MPS/CUDA compatibility
    - COCO evaluation metrics integration for standardized performance assessment
    - Configurable confidence thresholds for prediction filtering
    - Comprehensive logging and progress tracking during training
    - Support for learning rate scheduling and warmup phases

Functions:
    get_model: Create and configure Faster R-CNN models
    train: Complete training workflow with validation
    train_one_epoch: Single epoch training with detailed logging
    evaluate: Model evaluation with COCO metrics
    get_predictions: Generate filtered predictions with confidence thresholds

Dependencies:
    - torch: Core deep learning framework
    - torchvision: Computer vision models and utilities
    - Custom modules: req_utils, coco_eval, coco_utils, constants

Author: Dhruv Salot
Date: November 2025
License: MIT

Usage:
    >>> model = get_model(backbone, weights, train_dataset, predictor)
    >>> train(model, optimizer, train_loader, valid_loader, num_epochs=50)
    >>> results = evaluate(model, test_loader)
    >>> predictions = get_predictions(model, images, threshold=0.8)

Note:
    - Mixed precision training requires compatible hardware and PyTorch version
    - MPS support includes fallbacks for operations not yet fully optimized
    - Memory usage tracking is device-specific and may not be available on all platforms
"""

import math
import sys
import time

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from .req_utils import MetricLogger, SmoothedValue
from .req_utils import reduce_dict
from .coco_eval import CocoEvaluator
from .coco_utils import get_coco_api_from_dataset
from .constants import DEVICE

# Cross-Platform Compatibility Notes:
# - torch.amp.autocast currently supports 'cuda' and 'cpu' devices only
# - MPS (Metal Performance Shaders) operations automatically fall back to CPU autocast
# - Some PyTorch operations may not be fully optimized for MPS and fall back to CPU
# - Memory tracking varies by platform and may not be available on all devices

def get_model(backbone, weights, train_dataset, num_classes, device=DEVICE):
    """
    Create and configure a Faster R-CNN model with a custom predictor head.
    
    This function sets up a complete Faster R-CNN model by loading a pre-trained
    backbone, determining the number of classes from the dataset, and replacing
    the classification head with a custom predictor for the target task.
    
    Args:
        backbone (callable): Faster R-CNN backbone constructor (e.g., torchvision.models.detection.fasterrcnn_resnet50_fpn)
        weights (str or torchvision.models.WeightsEnum): Pre-trained weights to load
        train_dataset: Training dataset with COCO API for class information
        predictor (callable): Custom predictor head constructor  
        device (torch.device, optional): Target device for the model. Defaults to DEVICE constant.
        
    Returns:
        torch.nn.Module: Configured Faster R-CNN model ready for training, moved to the specified device
        
    Model Configuration:
        - Loads pre-trained backbone with specified weights
        - Automatically detects number of classes from dataset (+1 for background)
        - Replaces roi_heads.box_predictor with custom predictor
        - Configures input features based on backbone architecture
        
    Example:
        >>> from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
        >>> from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        >>> 
        >>> model = get_model(
        ...     backbone=fasterrcnn_resnet50_fpn,
        ...     weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1,
        ...     train_dataset=train_dataset,
        ...     device=device
        ... )
        
    Note:
        - The model is automatically moved to the specified device
        - Number of classes includes background class (hence +1)
        - Custom predictor must be compatible with the backbone's feature dimensions
    """
    # Load a pre-trained Faster R-CNN model with specified backbone and weights
    model = backbone(weights=weights)

    # Get the number of classes in the dataset (including background)
    num_classes = num_classes + 1

    # Get the number of input features for the classifier head from the backbone
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the pre-trained classification head with a custom predictor
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Move the model to the specified device (GPU or CPU)
    model.to(device)
    
    return model


def _get_autocast_device(device):
    """
    Get the appropriate device string for torch.amp.autocast based on the device type.
    
    This helper function provides cross-platform compatibility for mixed precision training.
    Since torch.amp.autocast currently only supports 'cuda' and 'cpu' devices, MPS and
    other devices are mapped to 'cpu' for autocast operations.
    
    Args:
        device (torch.device): The target device for training
        
    Returns:
        str: Device string compatible with torch.amp.autocast ('cuda' or 'cpu')
        
    Compatibility Notes:
        - CUDA devices return 'cuda' for full mixed precision support
        - MPS and other devices fall back to 'cpu' autocast
        - CPU autocast provides some benefits even on non-CUDA devices
        - Future PyTorch versions may expand autocast device support
        
    Example:
        >>> device = torch.device('mps')
        >>> autocast_device = _get_autocast_device(device)
        >>> print(autocast_device)  # 'cpu'
        >>> 
        >>> with torch.amp.autocast(autocast_device):
        ...     output = model(input)
    """
    if device.type == 'cuda':
        return 'cuda'
    else:
        # For MPS and other devices, fall back to CPU autocast
        # This ensures compatibility while maintaining some mixed precision benefits
        return 'cpu'


def _synchronize_device(device):
    """
    Synchronize operations on the appropriate device based on its type.
    
    This function ensures that all pending operations on the specified device
    are completed before proceeding. It provides cross-platform compatibility
    for device synchronization across CUDA, MPS, and CPU devices.
    
    Args:
        device (torch.device): The device to synchronize
        
    Device Synchronization:
        - CUDA: Uses torch.cuda.synchronize() for GPU synchronization
        - MPS: Uses torch.mps.synchronize() for Apple Silicon GPU synchronization  
        - CPU: No synchronization needed, operations are inherently synchronous
        
    Error Handling:
        - Gracefully handles cases where synchronization is not supported
        - Prints warning message if synchronization fails
        - Continues execution without synchronization as fallback
        
    Note:
        - Synchronization is important for accurate timing measurements
        - Some PyTorch versions may not support all synchronization methods
        - The function is designed to be safe and non-blocking in case of errors
        
    Example:
        >>> device = torch.device('cuda')
        >>> _synchronize_device(device)  # Ensures all CUDA operations complete
        >>> timing_start = time.time()  # Now accurate for timing
    """
    try:
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type == 'mps':
            torch.mps.synchronize()
        # CPU operations are synchronous by default, no synchronization needed
    except (RuntimeError, AttributeError):
        # Handle cases where synchronization is not available or fails
        print(f"Warning: Device synchronization failed or is not supported on {device}.")
        pass


def train(
    model,
    optimizer,
    train_loader,
    valid_loader,
    device=DEVICE,
    num_epochs=10,
    print_freq=250,
    scaler=None,
    output_dir=None,
    lr_scheduler=None
):
    """
    Complete training workflow for object detection models.
    
    This function orchestrates the entire training process including multiple epochs,
    validation evaluation, and optional model checkpointing. It provides a high-level
    interface for training object detection models with comprehensive logging and monitoring.
    
    Args:
        model (torch.nn.Module): The object detection model to train
        optimizer (torch.optim.Optimizer): Optimizer for training (e.g., SGD, Adam)
        train_loader (torch.utils.data.DataLoader): Training data loader
        valid_loader (torch.utils.data.DataLoader): Validation data loader  
        device (torch.device, optional): Training device. Defaults to DEVICE constant.
        num_epochs (int, optional): Number of training epochs. Defaults to 10.
        print_freq (int, optional): Frequency of progress logging. Defaults to 250.
        scaler (torch.cuda.amp.GradScaler, optional): Mixed precision scaler. Defaults to None.
        output_dir (str, optional): Directory to save model checkpoints. Defaults to None.
        lr_scheduler (torch.optim.lr_scheduler, optional): Learning rate scheduler. Defaults to None.
        
    Training Process:
        1. Train for one epoch using train_one_epoch()
        2. Evaluate on validation set using evaluate()
        3. Save model checkpoint if output_dir is specified
        4. Repeat for specified number of epochs
        
    Features:
        - Automatic evaluation after each epoch
        - Optional model checkpointing with epoch-specific filenames
        - Progress tracking and logging throughout training
        - Mixed precision support for improved performance
        - Learning rate scheduling support
        
    Example:
        >>> train(
        ...     model=faster_rcnn_model,
        ...     optimizer=sgd_optimizer, 
        ...     train_loader=train_dataloader,
        ...     valid_loader=val_dataloader,
        ...     device=device,
        ...     num_epochs=50,
        ...     output_dir='./checkpoints'
        ... )
        
    Note:
        - Model checkpoints are saved as 'model_epoch_{epoch}.pth'
        - Validation evaluation provides COCO metrics for performance monitoring
        - The function expects data loaders to return (images, targets) tuples
    """
    for epoch in range(1, num_epochs+1):
        # Train for one epoch with detailed logging
        train_one_epoch(
            model,
            optimizer,
            train_loader=train_loader,
            device=device,
            epoch=epoch,
            print_freq=print_freq,
            scaler=scaler,
            lr_scheduler=lr_scheduler
        )

        # Evaluate on validation set after each epoch
        evaluate(model, valid_loader, device=device)
        
        # Save model checkpoint if output directory is specified
        if output_dir:
            checkpoint_path = f'{output_dir}/model_epoch_{epoch}.pth'
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model checkpoint saved: {checkpoint_path}")


def train_one_epoch(
    model,
    optimizer,
    data_loader,
    device=DEVICE,
    epoch=1,
    print_freq=250,
    lr_scheduler=None,
    scaler=None
):
    """
    Train the model for one complete epoch with detailed logging and monitoring.
    
    This function handles the core training loop for a single epoch, including
    forward and backward passes, loss computation, gradient updates, and progress
    logging. It supports mixed precision training and learning rate scheduling.
    
    Args:
        model (torch.nn.Module): The model to train
        optimizer (torch.optim.Optimizer): Optimizer for parameter updates
        data_loader (torch.utils.data.DataLoader): Training data loader
        device (torch.device, optional): Training device. Defaults to DEVICE constant.
        epoch (int, optional): Current epoch number for logging. Defaults to 1.
        print_freq (int, optional): Frequency of progress updates. Defaults to 250.
        lr_scheduler (torch.optim.lr_scheduler, optional): Learning rate scheduler. Defaults to None.
        scaler (torch.cuda.amp.GradScaler, optional): Mixed precision scaler. Defaults to None.
        
    Returns:
        MetricLogger: Comprehensive metrics from the training epoch including:
            - loss: Overall training loss
            - Individual loss components (classification, bbox regression, etc.)
            - lr: Current learning rate
            - Timing information
            
    Training Features:
        - Mixed precision training support for improved performance and memory usage
        - Automatic learning rate warmup for the first epoch
        - Comprehensive loss tracking and reporting
        - Progress logging with timing and memory information
        - Gradient scaling for mixed precision stability
        - Loss validation to detect training instabilities
        
    Learning Rate Warmup:
        - Applied automatically during the first epoch
        - Linear warmup from 1/1000 of base LR to full LR
        - Warmup duration: min(1000 iterations, total iterations - 1)
        
    Example:
        >>> scaler = torch.cuda.amp.GradScaler()
        >>> metrics = train_one_epoch(
        ...     model=model,
        ...     optimizer=optimizer,
        ...     data_loader=train_loader, 
        ...     device=device,
        ...     epoch=1,
        ...     scaler=scaler
        ... )
        >>> print(f"Training loss: {metrics.loss.global_avg}")
        
    Note:
        - Mixed precision requires compatible hardware and PyTorch version
        - The function will exit if non-finite loss values are detected
        - Progress is logged every print_freq batches
        - Memory usage is tracked on compatible devices
    """
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    # Set up learning rate warmup for the first epoch
    lr_scheduler = lr_scheduler
    if epoch == 1:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        # Move data to the training device
        images = list(image.to(device) for image in images)
        targets = [{'boxes': t['boxes'].to(device), 'labels': t['labels'].to(device)} for t in targets]
        
        # Use device-compatible autocast for mixed precision training
        autocast_device = _get_autocast_device(device)
        with torch.amp.autocast(autocast_device, enabled=scaler is not None):
            # Forward pass - compute model predictions and losses
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # Reduce losses across all processes for distributed training compatibility
        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        # Check for training instabilities (infinite or NaN losses)
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        # Backward pass and optimizer step
        optimizer.zero_grad()
        if scaler is not None:
            # Mixed precision backward pass
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard precision backward pass
            losses.backward()
            optimizer.step()

        # Update learning rate if scheduler is provided
        if lr_scheduler is not None:
            lr_scheduler.step()

        # Update metrics for logging
        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


def _get_iou_types(model):
    """
    Determine the IoU types supported by the given model.
    
    This function analyzes the model architecture to determine what types of
    evaluation metrics should be computed based on the model's capabilities.
    
    Args:
        model (torch.nn.Module): The object detection model to analyze
        
    Returns:
        list: List of IoU types supported by the model:
            - "bbox": Bounding box detection (always included)
            - "segm": Segmentation masks (for Mask R-CNN models)  
            - "keypoints": Keypoint detection (for Keypoint R-CNN models)
            
    Model Support:
        - All models support bounding box detection ("bbox")
        - Mask R-CNN models additionally support segmentation ("segm")
        - Keypoint R-CNN models additionally support keypoints ("keypoints")
        - Distributed models are unwrapped to check the underlying model type
        
    Example:
        >>> iou_types = _get_iou_types(faster_rcnn_model)
        >>> print(iou_types)  # ['bbox']
        >>> 
        >>> iou_types = _get_iou_types(mask_rcnn_model) 
        >>> print(iou_types)  # ['bbox', 'segm']
        
    Note:
        - This function is used internally by the evaluation system
        - IoU types determine which COCO evaluation metrics are computed
        - Distributed Data Parallel wrappers are automatically handled
    """
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
        
    iou_types = ["bbox"]  # All models support bounding box detection
    
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
        
    return iou_types


@torch.inference_mode()
def evaluate(model, data_loader, device=DEVICE):
    """
    Comprehensive evaluation of object detection models using COCO metrics.
    
    This function performs thorough evaluation of trained models using the COCO
    evaluation protocol. It computes standard detection metrics including AP at
    different IoU thresholds and provides detailed performance analysis.
    
    Args:
        model (torch.nn.Module): The trained model to evaluate
        data_loader (torch.utils.data.DataLoader): Evaluation data loader
        device (torch.device, optional): Evaluation device. Defaults to DEVICE constant.
        
    Returns:
        CocoEvaluator: Comprehensive evaluation results including:
            - AP (Average Precision) at IoU=0.5:0.95
            - AP at IoU=0.5 and IoU=0.75
            - AP for small, medium, and large objects
            - AR (Average Recall) metrics
            - Per-category performance breakdown
            
    Evaluation Process:
        1. Set model to evaluation mode
        2. Run inference on all evaluation samples
        3. Collect predictions and ground truth
        4. Compute COCO evaluation metrics
        5. Display comprehensive results summary
        
    Features:
        - Standard COCO evaluation protocol
        - Automatic IoU type detection based on model capabilities
        - Device synchronization for accurate timing
        - Memory usage monitoring during evaluation
        - Progress tracking with detailed logging
        - Support for various model types (Faster R-CNN, Mask R-CNN, etc.)
        
    Performance Metrics:
        - AP: Average Precision across IoU thresholds 0.5:0.95
        - AP50: Average Precision at IoU threshold 0.5  
        - AP75: Average Precision at IoU threshold 0.75
        - APs, APm, APl: AP for small, medium, large objects
        - AR: Average Recall at different detection counts
        
    Example:
        >>> evaluator = evaluate(model, test_loader, device)
        >>> print("Evaluation completed:")
        >>> # COCO evaluation results are automatically printed
        
    Note:
        - Model is automatically set to evaluation mode
        - Thread count is temporarily adjusted for optimal performance
        - All predictions are moved to CPU for COCO evaluation
        - Device synchronization ensures accurate timing measurements
    """
    # Store current thread count and optimize for evaluation
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)  # Optimize for COCO evaluation performance
    
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"

    # Set up COCO evaluation with appropriate IoU types
    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        # Move images to evaluation device
        images = list(img.to(device) for img in images)

        # Synchronize device for accurate timing measurements
        _synchronize_device(device)
        
        model_time = time.time()
        
        # Run model inference
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        # Move outputs to CPU for COCO evaluation
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

        # Prepare results for evaluation
        res = {target["image_id"]: output for target, output in zip(targets, outputs)}
        
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        
        # Update timing metrics
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # Synchronize metrics across all processes (for distributed evaluation)
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # Compute and display comprehensive evaluation results
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    
    # Restore original thread count
    torch.set_num_threads(n_threads)
    
    return coco_evaluator


def get_predictions(model, images, threshold=0.75):
    """
    Generate filtered predictions from the model with confidence thresholding.
    
    This function runs inference on input images and filters predictions based on
    confidence scores. It provides an easy interface for getting high-quality
    predictions suitable for visualization or downstream processing.
    
    Args:
        model (torch.nn.Module): Trained object detection model
        images (list): List of input images as tensors
        threshold (float, optional): Confidence threshold for filtering predictions. 
                                   Defaults to 0.75.
                                   
    Returns:
        list: List of prediction dictionaries, one per image. Each dictionary contains:
            - 'boxes': List of bounding boxes for confident predictions
            - 'labels': List of class labels for confident predictions  
            - 'scores': List of confidence scores for confident predictions
            
    Filtering Logic:
        - Predictions above threshold are included
        - If no predictions above threshold, includes the highest confidence prediction
        - If no predictions at all, returns empty lists
        - Ensures at least one prediction per image when possible
        
    Prediction Format:
        - Bounding boxes in [xmin, ymin, xmax, ymax] format
        - Labels as integer class IDs
        - Scores as confidence values between 0 and 1
        
    Example:
        >>> model.eval()
        >>> with torch.no_grad():
        ...     predictions = get_predictions(model, test_images, threshold=0.8)
        >>> 
        >>> for pred in predictions:
        ...     print(f"Found {len(pred['boxes'])} confident detections")
        ...     for box, label, score in zip(pred['boxes'], pred['labels'], pred['scores']):
        ...         print(f"Class {label}: {score:.3f} confidence")
        
    Note:
        - Model should be in evaluation mode before calling this function
        - Images should be preprocessed and on the appropriate device
        - The function ensures graceful handling of images with no detections
        - Confidence threshold can be adjusted based on application requirements
    """
    model.eval()
    
    with torch.no_grad():
        # Run model inference
        output_dict = model(images)

    predictions = []

    for output in output_dict:
        scores = output['scores']
        labels = output['labels'] 
        boxes = output['boxes']

        confident_predictions = {
            'boxes': [],
            'labels': [],
            'scores': []
        }

        # Filter predictions by confidence threshold
        for i in range(len(scores)):
            if scores[i] <= threshold:
                continue
            confident_predictions['boxes'].append(boxes[i])
            confident_predictions['labels'].append(labels[i])
            confident_predictions['scores'].append(scores[i])

        # Fallback: if no confident predictions, use the highest confidence prediction
        if len(confident_predictions['boxes']) == 0:
            if len(scores) != 0:
                # Include the highest confidence prediction as fallback
                confident_predictions = {
                    'boxes': [boxes[0]],
                    'labels': [labels[0]], 
                    'scores': [scores[0]]
                }
            else:
                # No predictions at all - return empty prediction
                confident_predictions = {
                    'boxes': [],
                    'labels': [],
                    'scores': []
                }
                
        predictions.append(confident_predictions)
        
    return predictions