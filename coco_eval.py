"""
COCO Evaluation Metrics and Evaluators

This module provides comprehensive evaluation functionality for object detection models
using the COCO (Common Objects in Context) evaluation protocol. It implements the
standard COCO evaluation metrics including Average Precision (AP) and Average Recall (AR)
across different IoU thresholds and object scales.

The module includes:
- CocoEvaluator class for managing evaluation workflows
- Support for multiple IoU types (bbox, segmentation, keypoints)
- Distributed training compatibility with process synchronization
- Automatic conversion between different bounding box formats
- Comprehensive evaluation result aggregation and summarization

Key Features:
    - Standard COCO evaluation protocol implementation
    - Multi-IoU threshold evaluation (0.5:0.95)
    - Object scale-based evaluation (small, medium, large)
    - Support for different annotation types (detection, segmentation, keypoints)
    - Distributed training synchronization for consistent results
    - Automatic coordinate format conversion for COCO compatibility

Classes:
    CocoEvaluator: Main evaluation coordinator for COCO metrics

Functions:
    convert_to_xywh: Convert bounding box format from [xmin, ymin, xmax, ymax] to [x, y, width, height]
    merge: Merge evaluation results from distributed processes
    create_common_coco_eval: Create unified evaluation from distributed results
    evaluate: Run COCO evaluation on image set

Dependencies:
    - pycocotools: COCO API for evaluation metrics
    - torch: For tensor operations
    - numpy: For numerical operations
    - Custom modules: req_utils for distributed training support

Author: Dhruv Salot
Date: November 2025
License: MIT

Usage:
    >>> # Basic evaluation setup
    >>> evaluator = CocoEvaluator(coco_gt, ['bbox'])
    >>> evaluator.update(predictions)
    >>> evaluator.accumulate()
    >>> evaluator.summarize()
    
    >>> # Distributed training evaluation
    >>> evaluator.synchronize_between_processes()
    >>> evaluator.accumulate()
    >>> evaluator.summarize()

Note:
    - Follows standard COCO evaluation protocol for fair comparison
    - Compatible with distributed training environments
    - Supports evaluation of detection, segmentation, and keypoint models
    - Results are consistent with official COCO evaluation tools
"""

import copy
import io
from contextlib import redirect_stdout

import numpy as np
import pycocotools.mask as mask_util
import torch
from .req_utils import all_gather
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class CocoEvaluator:
    """
    COCO evaluation coordinator for comprehensive object detection assessment.
    
    This class manages the evaluation workflow for object detection models using the
    COCO evaluation protocol. It supports multiple IoU types, handles distributed
    training scenarios, and provides comprehensive metric computation including
    Average Precision and Average Recall across different conditions.
    
    Args:
        coco_gt (COCO): Ground truth COCO API object containing annotations
        iou_types (list or tuple): List of IoU types to evaluate. Supported types:
                                  - "bbox": Bounding box detection
                                  - "segm": Instance segmentation  
                                  - "keypoints": Keypoint detection
                                  
    Attributes:
        coco_gt (COCO): Ground truth COCO object (deep copied for safety)
        iou_types (list): IoU evaluation types
        coco_eval (dict): Dictionary of COCOeval objects for each IoU type
        img_ids (list): List of processed image IDs
        eval_imgs (dict): Evaluation results for each IoU type
        
    Evaluation Metrics Computed:
        - AP (Average Precision) at IoU 0.5:0.95 (primary metric)
        - AP50: Average Precision at IoU 0.5
        - AP75: Average Precision at IoU 0.75  
        - APs, APm, APl: AP for small, medium, large objects
        - AR: Average Recall at different detection counts
        - ARs, ARm, ARl: AR for small, medium, large objects
        
    Example:
        >>> # Initialize evaluator for bounding box detection
        >>> evaluator = CocoEvaluator(coco_gt, ["bbox"])
        >>> 
        >>> # Update with model predictions
        >>> predictions = {img_id: {"boxes": boxes, "scores": scores, "labels": labels}}
        >>> evaluator.update(predictions)
        >>> 
        >>> # Compute and display results
        >>> evaluator.accumulate()
        >>> evaluator.summarize()
        
    Distributed Training:
        >>> # Synchronize across processes before final evaluation
        >>> evaluator.synchronize_between_processes()
        >>> evaluator.accumulate()
        >>> evaluator.summarize()
        
    Note:
        - Ground truth object is deep copied to prevent modification
        - Supports evaluation of multiple annotation types simultaneously
        - Compatible with distributed training through process synchronization
        - Results follow standard COCO evaluation protocol for reproducibility
    """
    
    def __init__(self, coco_gt, iou_types):
        """
        Initialize the COCO evaluator.
        
        Args:
            coco_gt (COCO): Ground truth COCO API object
            iou_types (list or tuple): IoU types to evaluate
            
        Raises:
            TypeError: If iou_types is not a list or tuple
        """
        if not isinstance(iou_types, (list, tuple)):
            raise TypeError(f"This constructor expects iou_types of type list or tuple, instead  got {type(iou_types)}")
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt = coco_gt

        self.iou_types = iou_types
        self.coco_eval = {}
        for iou_type in iou_types:
            self.coco_eval[iou_type] = COCOeval(coco_gt, iouType=iou_type)

        self.img_ids = []
        self.eval_imgs = {k: [] for k in iou_types}

    def update(self, predictions):
        """
        Update the evaluator with new model predictions.
        
        This method processes model predictions for a batch of images and prepares
        them for COCO evaluation. It handles format conversion, creates detection
        results, and stores evaluation data for later aggregation.
        
        Args:
            predictions (dict): Dictionary mapping image IDs to prediction dictionaries.
                              Each prediction should contain:
                              - "boxes": Bounding boxes in [xmin, ymin, xmax, ymax] format
                              - "scores": Confidence scores for each detection
                              - "labels": Class labels for each detection
                              - "masks": Segmentation masks (for segmentation evaluation)
                              - "keypoints": Keypoint coordinates (for keypoint evaluation)
                              
        Processing Steps:
            1. Extract unique image IDs from predictions
            2. For each IoU type, convert predictions to COCO format
            3. Create COCO detection object from predictions
            4. Run evaluation for the current batch
            5. Store evaluation results for later aggregation
            
        Example:
            >>> predictions = {
            ...     12345: {
            ...         "boxes": torch.tensor([[100, 100, 200, 200]]),
            ...         "scores": torch.tensor([0.95]),
            ...         "labels": torch.tensor([1])
            ...     },
            ...     12346: {
            ...         "boxes": torch.tensor([[50, 50, 150, 150], [300, 300, 400, 400]]),
            ...         "scores": torch.tensor([0.85, 0.75]),
            ...         "labels": torch.tensor([2, 1])
            ...     }
            ... }
            >>> evaluator.update(predictions)
            
        Note:
            - Image IDs should match those in the ground truth COCO object
            - Bounding boxes are automatically converted to COCO format
            - Empty predictions are handled gracefully
            - Evaluation results are stored for batch processing efficiency
        """
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        for iou_type in self.iou_types:
            results = self.prepare(predictions, iou_type)
            with redirect_stdout(io.StringIO()):
                coco_dt = COCO.loadRes(self.coco_gt, results) if results else COCO()
            coco_eval = self.coco_eval[iou_type]

            coco_eval.cocoDt = coco_dt
            coco_eval.params.imgIds = list(img_ids)
            img_ids, eval_imgs = evaluate(coco_eval)

            self.eval_imgs[iou_type].append(eval_imgs)

    def synchronize_between_processes(self):
        """
        Synchronize evaluation results across distributed training processes.
        
        This method aggregates evaluation results from all processes in distributed
        training, ensuring consistent and complete evaluation across the entire
        dataset. It combines evaluation images and creates unified COCO evaluation
        objects for final metric computation.
        
        Synchronization Steps:
            1. Concatenate evaluation images from all processes
            2. Create common COCO evaluation objects with merged results
            3. Prepare evaluators for final accumulation and summarization
            
        Note:
            - Must be called before accumulate() in distributed training
            - Has no effect in single-process training
            - Ensures evaluation covers the complete dataset across all processes
            - Required for accurate distributed training evaluation
            
        Example:
            >>> # In distributed training workflow
            >>> for batch in data_loader:
            ...     predictions = model(batch)
            ...     evaluator.update(predictions)
            >>> 
            >>> evaluator.synchronize_between_processes()  # Sync across processes
            >>> evaluator.accumulate()  # Compute final metrics
            >>> evaluator.summarize()   # Display results
        """
        for iou_type in self.iou_types:
            self.eval_imgs[iou_type] = np.concatenate(self.eval_imgs[iou_type], 2)
            create_common_coco_eval(self.coco_eval[iou_type], self.img_ids, self.eval_imgs[iou_type])

    def accumulate(self):
        """
        Accumulate evaluation results across all images and compute final metrics.
        
        This method processes all stored evaluation results and computes the final
        COCO metrics including Average Precision and Average Recall. It should be
        called after all predictions have been processed via update().
        
        Accumulation Process:
            1. Process evaluation results for all IoU types
            2. Compute Average Precision across IoU thresholds and object scales
            3. Compute Average Recall for different detection counts
            4. Prepare metrics for summarization and display
            
        Metrics Computed:
            - AP: Average Precision (IoU 0.5:0.95, all object scales)
            - AP50, AP75: Average Precision at IoU 0.5 and 0.75
            - APs, APm, APl: AP for small, medium, large objects
            - AR1, AR10, AR100: Average Recall at 1, 10, 100 detections
            - ARs, ARm, ARl: AR for small, medium, large objects
            
        Example:
            >>> evaluator.accumulate()
            >>> # Metrics are now computed and ready for summarization
            
        Note:
            - Must be called after all update() calls are complete
            - In distributed training, call after synchronize_between_processes()
            - Prepares metrics for display via summarize()
            - Computationally intensive step that processes all evaluation data
        """
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

    def summarize(self):
        """
        Display comprehensive evaluation results in a formatted table.
        
        This method prints detailed evaluation metrics for each IoU type in a
        standardized format compatible with COCO evaluation conventions. It
        displays Average Precision and Average Recall metrics across different
        IoU thresholds and object scales.
        
        Output Format:
            - IoU type header (e.g., "IoU metric: bbox")
            - Average Precision metrics:
              * AP (IoU=0.50:0.95) - Primary metric
              * AP (IoU=0.50) - Loose threshold
              * AP (IoU=0.75) - Strict threshold  
              * AP (small) - Small objects
              * AP (medium) - Medium objects
              * AP (large) - Large objects
            - Average Recall metrics:
              * AR (max=1) - Single detection
              * AR (max=10) - Up to 10 detections
              * AR (max=100) - Up to 100 detections
              * AR (small/medium/large) - By object scale
              
        Example Output:
            IoU metric: bbox
            Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.503
            Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.847
            Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.545
            ...
            
        Example:
            >>> evaluator.summarize()
            # Displays formatted evaluation table
            
        Note:
            - Must be called after accumulate()
            - Results are printed to standard output
            - Format matches official COCO evaluation output
            - Useful for comparing with published results
        """
        for iou_type, coco_eval in self.coco_eval.items():
            print(f"IoU metric: {iou_type}")
            coco_eval.summarize()

    def prepare(self, predictions, iou_type):
        """
        Convert model predictions to COCO evaluation format.
        
        This method transforms model predictions into the format required by COCO
        evaluation tools. It handles different annotation types (bounding boxes,
        segmentation masks, keypoints) and performs necessary coordinate conversions.
        
        Args:
            predictions (dict): Raw model predictions mapped by image ID
            iou_type (str): Type of evaluation ("bbox", "segm", or "keypoints")
            
        Returns:
            list: List of COCO-format result dictionaries ready for evaluation
            
        Supported IoU Types:
            - "bbox": Bounding box detection results
            - "segm": Instance segmentation results with masks
            - "keypoints": Keypoint detection results
            
        Format Conversion:
            - Bounding boxes: [xmin, ymin, xmax, ymax] → [x, y, width, height]
            - Segmentation masks: Binary masks → RLE-encoded format
            - Keypoints: Structured coordinates → Flattened coordinate arrays
            
        Example:
            >>> bbox_results = evaluator.prepare(predictions, "bbox")
            >>> # Returns COCO-format bounding box results
            
        Raises:
            ValueError: If iou_type is not supported
            
        Note:
            - Each result includes image_id, category_id, score, and annotation data
            - Coordinate systems are converted to match COCO conventions
            - Results are ready for direct use with COCO evaluation tools
        """
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        if iou_type == "segm":
            return self.prepare_for_coco_segmentation(predictions)
        if iou_type == "keypoints":
            return self.prepare_for_coco_keypoint(predictions)
        raise ValueError(f"Unknown iou type {iou_type}")

    def prepare_for_coco_detection(self, predictions):
        """
        Convert detection predictions to COCO bounding box format.
        
        This method transforms bounding box predictions into COCO evaluation format,
        performing coordinate conversion and structuring results for evaluation.
        
        Args:
            predictions (dict): Dictionary mapping image IDs to prediction dictionaries
                              containing "boxes", "scores", and "labels"
                              
        Returns:
            list: COCO-format detection results with entries containing:
                 - image_id: Original image identifier
                 - category_id: Object class label
                 - bbox: Bounding box in [x, y, width, height] format
                 - score: Detection confidence score
                 
        Coordinate Conversion:
            - Input: [xmin, ymin, xmax, ymax] (PyTorch/torchvision format)
            - Output: [x, y, width, height] (COCO format)
            
        Example:
            >>> predictions = {
            ...     123: {
            ...         "boxes": torch.tensor([[10, 20, 30, 40]]),
            ...         "scores": torch.tensor([0.95]),
            ...         "labels": torch.tensor([1])
            ...     }
            ... }
            >>> results = evaluator.prepare_for_coco_detection(predictions)
            >>> print(results[0])
            {'image_id': 123, 'category_id': 1, 'bbox': [10, 20, 20, 20], 'score': 0.95}
            
        Note:
            - Empty predictions (no detections) are skipped gracefully
            - Coordinate conversion ensures compatibility with COCO evaluation
            - Results maintain association between boxes, scores, and labels
        """
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results

    def prepare_for_coco_segmentation(self, predictions):
        """
        Convert segmentation predictions to COCO mask format.
        
        This method transforms instance segmentation predictions into COCO evaluation
        format, converting binary masks to RLE (Run-Length Encoding) format and
        structuring results for segmentation evaluation.
        
        Args:
            predictions (dict): Dictionary mapping image IDs to prediction dictionaries
                              containing "masks", "scores", and "labels"
                              
        Returns:
            list: COCO-format segmentation results with entries containing:
                 - image_id: Original image identifier
                 - category_id: Object class label
                 - segmentation: RLE-encoded mask
                 - score: Detection confidence score
                 
        Mask Processing:
            1. Apply threshold (> 0.5) to convert to binary masks
            2. Encode masks using RLE format for efficient storage
            3. Convert RLE byte strings to UTF-8 for JSON compatibility
            
        Example:
            >>> predictions = {
            ...     123: {
            ...         "masks": torch.tensor([[[0.8, 0.2], [0.9, 0.1]]]),
            ...         "scores": torch.tensor([0.95]),
            ...         "labels": torch.tensor([1])
            ...     }
            ... }
            >>> results = evaluator.prepare_for_coco_segmentation(predictions)
            
        Note:
            - Masks are thresholded at 0.5 to create binary segmentations
            - RLE encoding provides efficient representation for sparse masks
            - Results are compatible with COCO segmentation evaluation tools
        """
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            scores = prediction["scores"]
            labels = prediction["labels"]
            masks = prediction["masks"]

            masks = masks > 0.5

            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            rles = [
                mask_util.encode(np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F"))[0] for mask in masks
            ]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "segmentation": rle,
                        "score": scores[k],
                    }
                    for k, rle in enumerate(rles)
                ]
            )
        return coco_results

    def prepare_for_coco_keypoint(self, predictions):
        """
        Convert keypoint predictions to COCO keypoint format.
        
        This method transforms keypoint detection predictions into COCO evaluation
        format, flattening keypoint coordinates and structuring results for
        keypoint evaluation.
        
        Args:
            predictions (dict): Dictionary mapping image IDs to prediction dictionaries
                              containing "keypoints", "boxes", "scores", and "labels"
                              
        Returns:
            list: COCO-format keypoint results with entries containing:
                 - image_id: Original image identifier  
                 - category_id: Object class label
                 - keypoints: Flattened keypoint coordinates
                 - score: Detection confidence score
                 
        Keypoint Processing:
            1. Convert bounding boxes to COCO format [x, y, width, height]
            2. Flatten keypoint tensors from structured format to coordinate arrays
            3. Maintain correspondence between keypoints and detection boxes
            
        Keypoint Format:
            - Input: Structured keypoint tensors (e.g., [num_keypoints, 3] for [x, y, visibility])
            - Output: Flattened coordinate arrays compatible with COCO format
            
        Example:
            >>> predictions = {
            ...     123: {
            ...         "keypoints": torch.tensor([[[10, 20, 2], [30, 40, 2]]]),  # 2 keypoints
            ...         "boxes": torch.tensor([[5, 15, 35, 45]]),
            ...         "scores": torch.tensor([0.95]),
            ...         "labels": torch.tensor([1])
            ...     }
            ... }
            >>> results = evaluator.prepare_for_coco_keypoint(predictions)
            
        Note:
            - Keypoint visibility flags are preserved in the flattened format
            - Bounding boxes provide spatial context for keypoint evaluation
            - Results are compatible with COCO keypoint evaluation protocols
        """
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            keypoints = prediction["keypoints"]
            keypoints = keypoints.flatten(start_dim=1).tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "keypoints": keypoint,
                        "score": scores[k],
                    }
                    for k, keypoint in enumerate(keypoints)
                ]
            )
        return coco_results


def convert_to_xywh(boxes):
    """
    Convert bounding boxes from [xmin, ymin, xmax, ymax] to [x, y, width, height] format.
    
    This function performs coordinate conversion from the corner-based format used
    by PyTorch and torchvision models to the center-width-height format expected
    by COCO evaluation tools.
    
    Args:
        boxes (torch.Tensor): Tensor of bounding boxes in [xmin, ymin, xmax, ymax] format
                             Shape: (N, 4) where N is the number of boxes
                             
    Returns:
        torch.Tensor: Converted bounding boxes in [x, y, width, height] format
                     Shape: (N, 4) with same number of boxes
                     
    Conversion Formula:
        - x = xmin (left coordinate remains the same)
        - y = ymin (top coordinate remains the same)
        - width = xmax - xmin
        - height = ymax - ymin
        
    Example:
        >>> boxes = torch.tensor([[10, 20, 50, 80], [100, 150, 200, 250]])
        >>> converted = convert_to_xywh(boxes)
        >>> print(converted)
        tensor([[ 10,  20,  40,  60],    # [x=10, y=20, w=40, h=60]
                [100, 150, 100, 100]])   # [x=100, y=150, w=100, h=100]
        
    Note:
        - Input boxes must be in [xmin, ymin, xmax, ymax] format
        - Output format is compatible with COCO evaluation tools
        - Conversion preserves the spatial information while changing representation
        - Essential for proper COCO evaluation compliance
    """
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


def merge(img_ids, eval_imgs):
    """
    Merge evaluation results from multiple distributed processes.
    
    This function aggregates evaluation results across all processes in distributed
    training, combining image IDs and evaluation images while removing duplicates
    and maintaining proper ordering.
    
    Args:
        img_ids: Image IDs from the current process
        eval_imgs: Evaluation images from the current process
        
    Returns:
        tuple: (merged_img_ids, merged_eval_imgs) containing:
              - merged_img_ids: Unique image IDs across all processes
              - merged_eval_imgs: Corresponding evaluation images
              
    Merging Process:
        1. Gather image IDs from all processes using all_gather
        2. Gather evaluation images from all processes
        3. Combine results from all processes
        4. Remove duplicate image IDs while preserving order
        5. Filter evaluation images to match unique image IDs
        
    Example:
        >>> merged_ids, merged_imgs = merge(local_img_ids, local_eval_imgs)
        >>> # Results now contain data from all distributed processes
        
    Note:
        - Essential for distributed evaluation to ensure complete dataset coverage
        - Removes duplicates that may occur due to data loader distribution
        - Maintains consistent ordering for reproducible evaluation
        - Used internally by synchronize_between_processes()
    """
    all_img_ids = all_gather(img_ids)
    all_eval_imgs = all_gather(eval_imgs)

    merged_img_ids = []
    for p in all_img_ids:
        merged_img_ids.extend(p)

    merged_eval_imgs = []
    for p in all_eval_imgs:
        merged_eval_imgs.append(p)

    merged_img_ids = np.array(merged_img_ids)
    merged_eval_imgs = np.concatenate(merged_eval_imgs, 2)

    # keep only unique (and in sorted order) images
    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)
    merged_eval_imgs = merged_eval_imgs[..., idx]

    return merged_img_ids, merged_eval_imgs


def create_common_coco_eval(coco_eval, img_ids, eval_imgs):
    """
    Create a unified COCO evaluation object from distributed results.
    
    This function configures a COCO evaluation object with merged results from
    distributed training, setting up the evaluation images and parameters for
    final metric computation.
    
    Args:
        coco_eval (COCOeval): COCO evaluation object to configure
        img_ids: Merged image IDs from all processes
        eval_imgs: Merged evaluation images from all processes
        
    Side Effects:
        - Sets coco_eval.evalImgs with flattened evaluation images
        - Sets coco_eval.params.imgIds with merged image IDs
        - Creates deep copy of parameters for evaluation
        
    Configuration Process:
        1. Merge image IDs and evaluation images using merge()
        2. Flatten evaluation images for COCO eval compatibility
        3. Set evaluation object parameters
        4. Create parameter copy for evaluation state preservation
        
    Example:
        >>> create_common_coco_eval(evaluator, merged_ids, merged_eval_imgs)
        >>> # COCO evaluator is now ready for accumulation and summarization
        
    Note:
        - Called internally by synchronize_between_processes()
        - Prepares evaluation object for final metric computation
        - Essential for distributed training evaluation consistency
        - Maintains COCO evaluation protocol compliance
    """
    img_ids, eval_imgs = merge(img_ids, eval_imgs)
    img_ids = list(img_ids)
    eval_imgs = list(eval_imgs.flatten())

    coco_eval.evalImgs = eval_imgs
    coco_eval.params.imgIds = img_ids
    coco_eval._paramsEval = copy.deepcopy(coco_eval.params)


def evaluate(imgs):
    """
    Run COCO evaluation on a set of images.
    
    This function executes the core COCO evaluation process on a prepared set of
    images, computing detection metrics and returning the results in a format
    suitable for aggregation.
    
    Args:
        imgs (COCOeval): COCO evaluation object with loaded predictions and ground truth
        
    Returns:
        tuple: (img_ids, eval_imgs) containing:
              - img_ids: List of evaluated image IDs
              - eval_imgs: Evaluation results reshaped for processing
              
    Evaluation Process:
        1. Run COCO evaluation with output suppression
        2. Extract image IDs from evaluation parameters
        3. Reshape evaluation results for consistent processing
        4. Return structured results for aggregation
        
    Output Suppression:
        - Uses redirect_stdout to suppress verbose COCO evaluation output
        - Maintains clean logging while preserving evaluation functionality
        
    Example:
        >>> img_ids, eval_results = evaluate(coco_evaluator)
        >>> # Results ready for storage and later aggregation
        
    Note:
        - Core evaluation function called by CocoEvaluator.update()
        - Output shape depends on area ranges and image counts from COCO parameters
        - Results are structured for efficient batch processing and storage
    """
    with redirect_stdout(io.StringIO()):
        imgs.evaluate()
    return imgs.params.imgIds, np.asarray(imgs.evalImgs).reshape(-1, len(imgs.params.areaRng), len(imgs.params.imgIds))
