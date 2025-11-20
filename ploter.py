"""
Image plotting utilities for object detection visualization.

This module provides functions for adding bounding boxes, labels, and scores to images,
as well as plotting multiple images with their annotations in a grid layout.
Used primarily for visualizing object detection results and ground truth annotations.

Dependencies:
    - PIL (Pillow): Image processing and drawing
    - matplotlib: Plotting and visualization
    - torch: Tensor operations
    - torchvision: Image transformations

Classes:
    None

Functions:
    add_boxes_labels: Add bounding boxes and labels to a single image
    plot_images: Display multiple images with annotations in a grid layout
"""

from math import ceil
import PIL
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as T
from .constants import PEST_CLASSES, DEVICE


def add_boxes_labels(image, boxes, labels, scores=None, threshold=0.75):
    """
    Add bounding boxes and labels to an image for object detection visualization.
    
    This function draws bounding boxes around detected objects, adds class labels,
    and optionally includes confidence scores. Boxes are color-coded based on
    confidence scores (red for low confidence, green for high confidence).
    
    Args:
        image (torch.Tensor): Input image tensor in format (C, H, W)
        boxes (torch.Tensor): Bounding box coordinates in format (N, 4) where
                             each box is [xmin, ymin, xmax, ymax]
        labels (torch.Tensor): Class labels for each box (N,) with 1-based indexing
        scores (torch.Tensor, optional): Confidence scores for each box (N,).
                                       If None, creates zero tensor. Defaults to None.
        threshold (float, optional): Confidence threshold for color coding.
                                   Scores below threshold are colored red,
                                   above are colored green. Defaults to 0.75.
    
    Returns:
        PIL.Image: Image with bounding boxes, labels, and scores drawn on it
        
    Color Coding:
        - Green boxes: High confidence (score >= threshold) or ground truth
        - Red boxes: Low confidence (score < threshold)
        - White text: Labels and scores on colored backgrounds
        
    Text Positioning:
        - Ground truth (no scores): Above the box when possible, inside if no space
        - Predictions (with scores): Bottom-right of box when possible, adjusted for edge cases
        
    Note:
        - Labels use 1-based indexing (PEST_CLASSES[label-1])
        - Text backgrounds are drawn to ensure readability
        - Handles edge cases where text would go outside image boundaries
        
    Example:
        >>> image_with_boxes = add_boxes_labels(
        ...     image_tensor, 
        ...     predicted_boxes, 
        ...     predicted_labels,
        ...     predicted_scores,
        ...     threshold=0.8
        ... )
    """
    # Convert tensor to PIL Image for drawing operations
    img = T.ToPILImage()(image).convert("RGB")
    
    # Create zero scores tensor if none provided (for ground truth visualization)
    scores = torch.zeros(
        len(boxes),
        device=DEVICE,
        dtype=torch.float32,
        requires_grad=False
    ) if not scores else scores

    # Process each detection/annotation
    for box, label, score in zip(boxes, labels, scores):
        # Convert box coordinates to integers
        xmin, ymin, xmax, ymax = map(int, box)
        
        # Determine box color based on confidence score
        fill_color = (0, 70, 0)  # Default green color for boxes
        if score:
            if score < threshold:
                fill_color = (150, 0, 0)  # Red for low confidence
            else:
                fill_color = (0, 70, 0)   # Green for high confidence

        # Draw bounding box outline
        PIL.ImageDraw.Draw(img).rectangle([(xmin, ymin), (xmax, ymax)], outline=fill_color, width=3)

        # Set up font for text rendering
        font = PIL.ImageFont.load_default(size=14)
        ascent, descent = font.getmetrics()

        # Create label text (with or without confidence score)
        if score:
            # Include confidence score for predictions
            annotation_label = f"{PEST_CLASSES[label-1]} {score:.2f}"
        else:
            # Just class name for ground truth
            annotation_label = f"{PEST_CLASSES[label-1]}"
        
        # Calculate text dimensions for background rectangle
        (width, baseline), (offset_x, offset_y) = font.font.getsize(annotation_label)
        text_height = ascent + descent
        text_width = width + 20  # Add padding

        # Determine text and background rectangle position based on detection type
        if not score:
            # Ground truth positioning: prefer above the box
            if ymin - text_height - baseline >= 0:
                # Space available above box
                text_position = (xmin + 10, ymin - text_height)
                back_box_position = [(xmin, ymin - text_height), (xmin + text_width, ymin)]
            else:
                # No space above, place inside box at top
                text_position = (xmin + 10, ymin)
                back_box_position = [(xmin, ymin), (xmin + text_width, ymin + text_height)]
        else:
            # Prediction positioning: prefer bottom-right of box
            if xmax - text_width < 0:
                # Box too narrow, place to the right
                text_position = (xmax + 10, ymin)
                back_box_position = [(xmax, ymin), (xmax + text_width, ymin + text_height)]
            else:
                # Standard position: bottom-right corner
                text_position = (xmax - text_width + 10, ymax - text_height)
                back_box_position = [(xmax - text_width, ymax - text_height), (xmax, ymax)]
        
        # Draw background rectangle first (behind text)
        PIL.ImageDraw.Draw(img).rectangle(back_box_position, fill=fill_color)
        # Draw text on top of background
        PIL.ImageDraw.Draw(img).text(text_position, annotation_label, fill="white", font=font)

    return img


def plot_images(images, targets, cmap=None, suptitle=None, title_y=1.0, fontsize=14, threshold=0.75):
    """
    Display multiple images with object detection annotations in a grid layout.
    
    This function creates a matplotlib figure showing images with their bounding boxes,
    labels, and scores. Handles both single image display and multi-image grids.
    Images without annotations are highlighted with red borders.
    
    Args:
        images (list): List of image tensors in format (C, H, W)
        targets (list): List of dictionaries containing annotation data.
                       Each dict should have:
                       - 'boxes': bounding box coordinates
                       - 'labels': class labels  
                       - 'scores': confidence scores (optional)
        cmap (str, optional): Matplotlib colormap for image display. Defaults to None.
        suptitle (str, optional): Main title for the entire figure. Defaults to None.
        title_y (float, optional): Vertical position of suptitle. Defaults to 1.0.
        fontsize (int, optional): Font size for subplot titles. Defaults to 14.
        threshold (float, optional): Confidence threshold for box coloring. Defaults to 0.75.
    
    Display Layout:
        - Single image: 5x4 figure
        - Multiple images: Grid layout with max 3 columns
        - Empty annotations: Red border highlight
        - Automatic subplot sizing and spacing
        
    Target Dictionary Format:
        {
            'boxes': torch.Tensor,     # (N, 4) bounding boxes
            'labels': torch.Tensor,    # (N,) class labels
            'scores': torch.Tensor     # (N,) confidence scores (optional)
        }
        
    Returns:
        None: Displays the plot using matplotlib
        
    Note:
        - Automatically adjusts grid size based on number of images
        - Handles edge cases like empty annotation lists
        - Uses tight layout for optimal spacing
        - All axes are turned off for clean visualization
        
    Example:
        >>> plot_images(
        ...     [img1, img2, img3],
        ...     [target1, target2, target3],
        ...     suptitle="Detection Results",
        ...     threshold=0.8
        ... )
    """
    images_count = len(images)

    # Handle single image case with specific layout
    if images_count == 1:
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        if suptitle:
            plt.suptitle(suptitle, fontsize=16, fontweight='bold', y=title_y)

        # Add annotations to the single image
        img = add_boxes_labels(
            images[0],
            targets[0]['boxes'],
            targets[0]['labels'],
            targets[0]['scores'] if 'scores' in targets[0] else None
        )

        # Display the annotated image
        ax.imshow(img, cmap=cmap)
        ax.set_title(f'Sample 1', fontsize=fontsize)
        ax.axis('off')
        plt.tight_layout()
        return
    
    # Calculate grid dimensions for multiple images
    row_count = ceil(images_count / 3)  # Maximum 3 columns
    col_count = min(images_count, 3)
    width = 4 * col_count   # 4 inches per column
    height = 4 * row_count  # 4 inches per row

    # Create subplot grid
    fig, axs = plt.subplots(
        row_count,
        col_count,
        figsize=(width, height),
        squeeze=False,              # Always return 2D array
        layout='constrained'        # Better spacing algorithm
    )

    # Add main title if provided
    if suptitle:
        plt.suptitle(suptitle, fontsize=16, fontweight='bold', y=title_y)

    # Process each image and display in grid
    for (i, image), ax in zip(enumerate(images), axs.flat[:images_count]):
        # Check if image has any annotations
        if len(targets[i]['labels']) == 0:
            # No annotations: convert to PIL and add red border
            img = T.ToPILImage()(image).convert("RGB")
            img = PIL.ImageOps.expand(img, border=5, fill='red')
        else:
            # Has annotations: add bounding boxes and labels
            img = add_boxes_labels(
                image,
                targets[i]['boxes'],
                targets[i]['labels'],
                targets[i]['scores'] if 'scores' in targets[i] else None,
                threshold=threshold
            )
        
        # Display image in subplot
        ax.imshow(img, cmap=cmap)
        ax.set_title(f'Sample {i+1}', fontsize=14)

    # Turn off axes for all subplots (clean look)
    for ax in axs.flat: 
        ax.axis('off')

    # Optimize layout spacing
    plt.tight_layout()