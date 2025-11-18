"""
COCO Dataset Utilities for Computer Vision Projects

This module provides utilities for working with COCO (Common Objects in Context) formatted datasets,
specifically designed for crop pest detection tasks. It includes a PyTorch Dataset class for loading
COCO-formatted data and a converter class to transform YOLO-format datasets to COCO format.

Classes:
    CocoDetectionDataset: PyTorch Dataset for loading COCO detection data
    COCOConverter: Utility class for converting YOLO format to COCO format

Dependencies:
    - torch: For tensor operations and dataset functionality
    - pycocotools: For COCO API operations
    - PIL: For image processing
    - Custom constants from .constants module

Author: Dhruv Salot
Date: November 2025
"""

import os
import json
from datetime import date
from typing import Dict, Tuple, Optional, Any

import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from PIL import Image

from .constants import PEST_CLASSES, IMAGE_SIZE, IMAGE_RESIZE_FACTOR


class COCODetectionDataset(Dataset):
    """
    PyTorch Dataset class for loading COCO-formatted object detection data.
    
    This dataset class is designed to work with COCO-formatted annotation files and provides
    data loading functionality for training object detection models. It loads images and their
    corresponding annotations (bounding boxes, labels, etc.) and applies optional transforms.
    
    Args:
        image_dir (str): Path to the directory containing images
        annotation_path (str): Path to the COCO annotation JSON file
        transforms (callable, optional): Optional transform to be applied to images
    
    Attributes:
        image_dir (str): Directory path containing images
        coco (COCO): COCO API instance for annotation handling
        image_ids (list): List of image IDs from the COCO dataset
        transforms (callable): Image transformation function
    
    Example:
        >>> dataset = CocoDetectionDataset(
        ...     image_dir="/path/to/images",
        ...     annotation_path="/path/to/annotations.json",
        ...     transforms=some_transform_function
        ... )
        >>> image, target = dataset[0]
    """

    def __init__(self, image_dir: str, annotation_path: str, transforms: Optional[callable] = None):
        """
        Initialize the COCO detection dataset.
        
        Args:
            image_dir (str): Path to the directory containing images
            annotation_path (str): Path to the COCO annotation JSON file
            transforms (callable, optional): Optional transform to be applied to images
        """
        self.image_dir = image_dir
        self.coco = COCO(annotation_path)
        self.image_ids = list(self.coco.imgs.keys())
        self.transforms = transforms

    def __len__(self) -> int:
        """
        Return the total number of images in the dataset.
        
        Returns:
            int: Number of images in the dataset
        """
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Tuple[Any, Dict[str, torch.Tensor]]:
        """
        Get a single item from the dataset.
        
        Loads an image and its corresponding annotations, converts bounding boxes to
        the appropriate format, and applies transforms if specified.
        
        Args:
            idx (int): Index of the item to retrieve
            
        Returns:
            tuple: A tuple containing:
                - image: PIL Image or transformed image (depending on transforms)
                - target (dict): Dictionary containing:
                    - boxes (torch.Tensor): Bounding boxes in [xmin, ymin, xmax, ymax] format
                    - labels (torch.Tensor): Class labels for each box
                    - image_id (int): Original image ID from COCO dataset
                    - area (torch.Tensor): Area of each bounding box
                    - iscrowd (torch.Tensor): Flag indicating if annotation represents a crowd
        
        Note:
            Bounding boxes are converted from COCO format [x, y, width, height] to
            [xmin, ymin, xmax, ymax] format for compatibility with PyTorch models.
        """
        # Get image information and load the image
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.image_dir, image_info['file_name'])
        image = Image.open(image_path).convert("RGB")

        # Load annotations for this image
        annotation_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(annotation_ids)

        # Process bounding boxes and labels
        boxes = []
        labels = []
        for obj in annotations:
            # Convert from COCO format [x, y, width, height] to [xmin, ymin, xmax, ymax]
            xmin, ymin, width, height = obj['bbox']
            xmax = xmin + width
            ymax = ymin + height
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(obj['category_id'])

        # Convert to PyTorch tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        area = torch.as_tensor([obj['area'] for obj in annotations], dtype=torch.float32)
        iscrowd = torch.as_tensor([obj.get('iscrowd', 0) for obj in annotations], dtype=torch.int64)

        # Create target dictionary
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd
        }

        # Apply transforms if provided
        if self.transforms:
            image = self.transforms(image)

        return image, target
    
class COCOConverter:
    """
    Utility class for converting YOLO-format datasets to COCO format.
    
    This class provides functionality to convert datasets from YOLO format (with .txt annotation files)
    to COCO format JSON files. It's specifically designed for the AgroPest-12 dataset but can be
    adapted for other similar datasets.
    
    The converter handles:
    - Converting YOLO bounding box format (normalized center coordinates) to COCO format (absolute coordinates)
    - Creating proper COCO JSON structure with metadata, categories, images, and annotations
    - Processing multiple dataset splits (train, valid, test)
    - Generating dataset statistics and summaries
    
    Args:
        dataset_path (str): Path to the root directory containing dataset splits
        output_path (str): Path where COCO annotation files will be saved
        image_size (tuple, optional): Original image dimensions. Defaults to IMAGE_SIZE from constants
        image_resize_factor (float, optional): Factor to resize images. Defaults to IMAGE_RESIZE_FACTOR
        keep_one_bbox_per_image (bool, optional): If True, keeps only one bounding box per image. Defaults to False.
    
    Attributes:
        dataset_dir_path (str): Path to dataset directory
        output_path (str): Output directory for COCO annotations
        image_size (tuple): Image dimensions (width, height)
        image_resize_factor (float): Image resize factor
    
    Example:
        >>> converter = COCOConverter(
        ...     dataset_path="/path/to/yolo/dataset",
        ...     output_path="/path/to/output",
        ...     image_size=(640, 640),
        ...     image_resize_factor=1.0
        ... )
        >>> converter.convert_to_coco()
    """
    
    def __init__(
        self,
        dataset_path: str,
        output_path: str,
        image_size: Tuple[int, int] = IMAGE_SIZE,
        image_resize_factor: float = IMAGE_RESIZE_FACTOR,
        keep_one_bbox_per_image: bool = False
    ):
        """
        Initialize the COCO converter.
        
        Args:
            dataset_path (str): Path to the root directory containing dataset splits
            output_path (str): Path where COCO annotation files will be saved
            image_size (tuple, optional): Original image dimensions (width, height)
            image_resize_factor (float, optional): Factor to resize images
        """
        self.dataset_dir_path = dataset_path
        self.output_path = os.path.join(output_path, 'coco_annotations')
        self.image_size = image_size
        self.image_resize_factor = image_resize_factor
        self.keep_one_bbox_per_image = keep_one_bbox_per_image

    def convert_to_coco(self) -> None:
        """
        Convert YOLO-format dataset to COCO format.
        
        This method processes the dataset splits (train, valid, test) and converts each split
        from YOLO format to COCO format. It creates JSON annotation files containing:
        - Dataset metadata and licensing information
        - Category definitions based on PEST_CLASSES
        - Image information with resized dimensions
        - Annotation data with converted bounding boxes
        
        The conversion process:
        1. Reads YOLO format .txt files with normalized coordinates
        2. Converts normalized coordinates to absolute pixel coordinates
        3. Applies image resize factor if specified
        4. Creates COCO-formatted JSON structure
        5. Saves annotation files for each split
        
        Raises:
            FileNotFoundError: If a corresponding label file is not found for an image
        
        Note:
            - Images without corresponding label files are skipped
            - Empty label files trigger a warning and the image is skipped
            - Category IDs are incremented by 1 (YOLO uses 0-based, COCO uses 1-based indexing)
        """
        data_splits = ['train', 'valid', 'test']
        
        # Define the base COCO structure with metadata
        main_contents = {
            "info": {
                "description": "AgroPest-12 Dataset",
                "version": "1.0",
                "year": 2025,
                "contributor": "Dhruv Salot",
                "date_created": str(date.today())
            },
            "licenses": [
                {
                    "id": 1,
                    "name": "MIT License",
                    "url": "https://www.mit.edu/~amini/LICENSE.md"
                }
            ],
            "categories": [
                {
                    "id": class_id,
                    "name": class_name,
                    "supercategory": "obj_classes"
                } for class_id, class_name in enumerate(PEST_CLASSES, start=1)
            ],
        }

        # Process each dataset split
        for split in data_splits:
            print(f"Processing {split} split...")

            images_path = os.path.join(self.dataset_dir_path, split, 'images')
            labels_path = os.path.join(self.dataset_dir_path, split, 'labels')

            contents = main_contents.copy()
            
            images = []
            annotations = []

            ann_id = 1  # Annotation ID counter
            img_id = 1  # Image ID counter

            # Helper function to apply resize factor
            image_resizer = lambda x: int(x * self.image_resize_factor)

            images_folders = os.listdir(images_path)

            for img_file in images_folders:
                # Skip non-image files
                if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue

                # Find corresponding label file
                label_file = img_file \
                    .replace('.jpg', '.txt') \
                    .replace('.png', '.txt') \
                    .replace('.jpeg', '.txt')
                
                label_path = os.path.join(labels_path, label_file)
                
                # Check if label file exists
                if not os.path.exists(label_path):
                    raise FileNotFoundError(f"Label file {label_file} not found for image {img_file}")
                
                # Skip images with empty label files
                if os.path.getsize(label_path) == 0:
                    print(f"Warning: Empty label file for image {img_file}, skipping image.")
                    continue

                # Process annotations from YOLO format
                with open(label_path, 'r') as lf:
                    lines = lf.readlines()
                    
                    if self.keep_one_bbox_per_image and lines:
                        lines = [lines[0]]  # Keep only the first bounding box
                    
                    for line in lines:
                        parts = line.strip().split()
                        
                        # Extract YOLO format data
                        category_id = int(parts[0]) + 1  # Convert from 0-based to 1-based indexing
                        x_center = float(parts[1]) * image_resizer(self.image_size[0])
                        y_center = float(parts[2]) * image_resizer(self.image_size[1])
                        width = round(float(parts[3]) * image_resizer(self.image_size[0]))
                        height = round(float(parts[4]) * image_resizer(self.image_size[1]))

                        # Convert to COCO format (top-left corner + width/height)
                        x_min = round(x_center - width / 2)
                        y_min = round(y_center - height / 2)

                        # Check positive area
                        if width <= 0 or height <= 0:
                            width = max(1, width)
                            height = max(1, height)
                            print(f"Warning: Non-positive area for annotation in image {img_file}. Adjusting width/height to minimum 1 pixel.")

                        # Create annotation entry
                        annotation_info = {
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": category_id,
                            "bbox": [x_min, y_min, width, height],
                            "area": width * height,
                            "iscrowd": 0
                        }
                        annotations.append(annotation_info)
                        ann_id += 1

                # Create image information entry
                image_info = {
                    "id": img_id,
                    "file_name": img_file,
                    "width": image_resizer(self.image_size[0]),
                    "height": image_resizer(self.image_size[1]),
                    "license": 1,
                    "date_captured": str(date.today())
                }

                images.append(image_info)
                
                img_id += 1

            # Add processed data to contents
            contents["images"] = images
            contents["annotations"] = annotations

            # Save COCO annotation file
            output_file = os.path.join(self.output_path, f"agropest_coco_{split}.json")
            os.makedirs(self.output_path, exist_ok=True)
            with open(output_file, 'w') as of:
                json.dump(contents, of, indent=4)
            print(f"COCO annotation file created at: {output_file}")
            print(f"  - Images: {len(images)}")
            print(f"  - Annotations: {len(annotations)}")
            print()

    def get_coco_summary(self, coco_json_path: str) -> Tuple[int, int, Dict[str, int], Dict[str, int]]:
        """
        Generate a comprehensive summary of a COCO dataset.
        
        This method analyzes a COCO annotation file and provides detailed statistics about
        the dataset including image counts, category distribution, and annotation distribution.
        
        Args:
            coco_json_path (str): Path to the COCO annotation JSON file to analyze
            
        Returns:
            tuple: A tuple containing:
                - num_images (int): Total number of images in the dataset
                - num_categories (int): Total number of categories/classes
                - image_distribution (dict): Number of images per category
                - category_distribution (dict): Number of annotations per category
        
        Example:
            >>> converter = COCOConverter(dataset_path, output_path)
            >>> num_imgs, num_cats, img_dist, cat_dist = converter.get_coco_summary("annotations.json")
            >>> print(f"Dataset has {num_imgs} images with {num_cats} categories")
        
        Note:
            - image_distribution counts unique images per category
            - category_distribution counts total annotations per category
            - A single image may contain multiple categories
        """
        # Load COCO annotation file
        with open(coco_json_path, 'r') as f:
            coco_data = json.load(f)

        # Basic dataset statistics
        num_images = len(coco_data['images'])
        num_categories = len(coco_data['categories'])

        # Initialize counters for each category
        category_counts = {cat['id']: 0 for cat in coco_data['categories']}
        image_category_counts = {cat['id']: set() for cat in coco_data['categories']}

        # Count annotations and unique images per category
        for ann in coco_data['annotations']:
            category_counts[ann['category_id']] += 1
            image_category_counts[ann['category_id']].add(ann['image_id'])

        # Map category IDs to their names for better readability
        cat_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}

        # Create distribution dictionaries with category names
        category_distribution = {
            cat_id_to_name[cat_id]: count for cat_id, count in category_counts.items()
        }
        
        image_distribution = {
            cat_id_to_name[cat_id]: len(image_ids) for cat_id, image_ids in image_category_counts.items()
        }

        return num_images, num_categories, image_distribution, category_distribution