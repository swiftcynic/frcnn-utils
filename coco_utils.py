"""
COCO Dataset Utilities and Data Processing

This module provides comprehensive utilities for processing COCO-formatted datasets,
including polygon-to-mask conversion, data transformations, and dataset creation.
It contains helper functions and classes for working with COCO annotations in
PyTorch-based object detection workflows.

The module includes:
- Polygon-to-mask conversion utilities for segmentation tasks
- COCO dataset transformation classes for PyTorch compatibility
- Dataset creation and filtering functions
- API conversion utilities for COCO format compliance

Key Features:
    - Convert COCO polygon annotations to binary masks
    - Filter and validate COCO annotations 
    - Transform datasets for PyTorch training pipelines
    - Handle various annotation formats (bbox, segmentation, keypoints)
    - Remove invalid or empty annotations automatically

Classes:
    ConvertCocoPolysToMask: Transforms COCO polygon annotations to masks
    CocoDetection: Custom COCO detection dataset for PyTorch

Functions:
    convert_coco_poly_to_mask: Convert polygon annotations to binary masks
    convert_to_coco_api: Convert dataset to COCO API format
    get_coco_api_from_dataset: Extract COCO API from PyTorch dataset
    get_coco: Create COCO dataset with specified configurations
    _coco_remove_images_without_annotations: Filter out empty annotations

Dependencies:
    - torch: For tensor operations and dataset handling
    - torchvision: For vision-specific transformations and datasets
    - pycocotools: For COCO API operations and mask utilities
    - os: For file system operations

Author: Dhruv Salot  
Date: November 2025
License: MIT
"""

import os

import torch
import torch.utils.data
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms as T
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO

from .req_utils import collate_fn
from .coco import COCODetectionDataset


def get_data_loaders(image_dir, annotation_path, transforms, batch_size, shuffle=True, collate_fn=collate_fn):
    """
    Create a DataLoader for the given dataset.

    This function initializes a PyTorch DataLoader for the specified dataset,
    applying the provided transformations and batching configurations.

    Args:
        image_dir (str): Directory containing the images
        annotation_path (str): Path to the COCO annotation file
        transform (callable): Transformations to apply to the data
        batch_size (int): Number of samples per batch
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
        collate_fn (callable, optional): Custom collate function for batching. Defaults to None.

    Returns:
        DataLoader: PyTorch DataLoader object
    """
    dataset = COCODetectionDataset(
        image_dir=image_dir,
        annotation_path=annotation_path,
        transforms=transforms
    )

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn
    )

    return data_loader


def convert_coco_poly_to_mask(segmentations, height, width):
    """
    Convert COCO polygon segmentations to binary masks.
    
    This function takes polygon segmentations in COCO format and converts them
    to binary masks. It handles multiple polygons per annotation and ensures
    proper tensor formatting for PyTorch workflows.
    
    Args:
        segmentations (list): List of polygon segmentations, where each segmentation
                             is a list of polygons (each polygon is a flat list of coordinates)
        height (int): Height of the target mask in pixels
        width (int): Width of the target mask in pixels
        
    Returns:
        torch.Tensor: Binary masks tensor of shape (N, height, width) where N is the
                     number of annotations. Each mask contains 0s and 1s indicating
                     background and foreground pixels respectively.
                     
    Note:
        - If no valid masks are found, returns a zero tensor of shape (0, height, width)
        - Multiple polygons for the same annotation are combined using logical OR
        - Uses pycocotools for efficient polygon-to-mask conversion
        
    Example:
        >>> segmentations = [[[x1, y1, x2, y2, ...]], [[x3, y3, x4, y4, ...]]]
        >>> masks = convert_coco_poly_to_mask(segmentations, 480, 640)
        >>> print(masks.shape)  # torch.Size([2, 480, 640])
    """
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask:
    """
    Transform class to convert COCO polygon annotations to masks and process targets.
    
    This callable class processes COCO-style annotations and converts them into a format
    suitable for PyTorch object detection models. It handles bounding box validation,
    polygon-to-mask conversion, and target dictionary creation.
    
    The transformation performs several operations:
    - Filters out crowd annotations (iscrowd=1)
    - Validates and clamps bounding boxes to image boundaries  
    - Converts polygon segmentations to binary masks
    - Creates properly formatted target dictionaries
    
    Attributes:
        None (stateless transform)
        
    Returns:
        tuple: (image, target) where target contains:
            - boxes (torch.Tensor): Bounding boxes in [xmin, ymin, xmax, ymax] format
            - labels (torch.Tensor): Class labels for each annotation
            - masks (torch.Tensor): Binary segmentation masks
            - image_id (int): Unique identifier for the image
            - area (torch.Tensor): Area of each annotation
            - iscrowd (torch.Tensor): Crowd flag for each annotation
            - keypoints (torch.Tensor, optional): Keypoint annotations if present
            
    Example:
        >>> transform = ConvertCocoPolysToMask()
        >>> image, target = transform(image, raw_target)
        >>> print(target.keys())
        dict_keys(['boxes', 'labels', 'masks', 'image_id', 'area', 'iscrowd'])
        
    Note:
        - Annotations with invalid bounding boxes (width/height <= 0) are filtered out
        - Bounding boxes are converted from COCO format [x, y, w, h] to [xmin, ymin, xmax, ymax]
        - All crowd annotations are automatically excluded from processing
    """
    
    def __call__(self, image, target):
        """
        Apply the transformation to an image and its target annotations.
        
        Args:
            image (PIL.Image): Input image
            target (dict): Raw COCO target containing image_id and annotations
            
        Returns:
            tuple: Transformed (image, target) pair with processed annotations
        """
        w, h = image.size

        image_id = target["image_id"]

        anno = target["annotations"]

        anno = [obj for obj in anno if obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        segmentations = [obj["segmentation"] for obj in anno]
        masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] for obj in anno])
        target["area"] = area
        target["iscrowd"] = iscrowd

        return image, target


def _coco_remove_images_without_annotations(dataset, cat_list=None):
    """
    Remove images without valid annotations from a COCO dataset.
    
    This function filters out images that don't have valid annotations based on
    several criteria including empty annotations, invalid bounding boxes, and
    insufficient keypoints for keypoint detection tasks.
    
    Args:
        dataset: COCO dataset object with .ids and .coco attributes
        cat_list (list, optional): List of category IDs to filter by. If provided,
                                  only annotations belonging to these categories are considered.
                                  
    Returns:
        torch.utils.data.Subset: Filtered dataset containing only images with valid annotations
        
    Validation Criteria:
        - Images with no annotations are removed
        - Images where all bounding boxes have near-zero area are removed  
        - For keypoint tasks: Images with fewer than min_keypoints_per_image visible keypoints are removed
        - Crowd annotations (iscrowd=1) are ignored in validation
        
    Note:
        - min_keypoints_per_image is set to 10 for keypoint detection tasks
        - Bounding boxes with width or height <= 1 are considered invalid
        - This function helps ensure training data quality by removing problematic samples
        
    Example:
        >>> filtered_dataset = _coco_remove_images_without_annotations(train_dataset)
        >>> print(f"Filtered from {len(train_dataset)} to {len(filtered_dataset)} images")
    """
    def _has_only_empty_bbox(anno):
        return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)

    def _count_visible_keypoints(anno):
        return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)

    min_keypoints_per_image = 10

    def _has_valid_annotation(anno):
        # if it's empty, there is no annotation
        if len(anno) == 0:
            return False
        # if all boxes have close to zero area, there is no annotation
        if _has_only_empty_bbox(anno):
            return False
        # keypoints task have a slight different criteria for considering
        # if an annotation is valid
        if "keypoints" not in anno[0]:
            return True
        # for keypoint detection tasks, only consider valid images those
        # containing at least min_keypoints_per_image
        if _count_visible_keypoints(anno) >= min_keypoints_per_image:
            return True
        return False

    ids = []
    for ds_idx, img_id in enumerate(dataset.ids):
        ann_ids = dataset.coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anno = dataset.coco.loadAnns(ann_ids)
        if cat_list:
            anno = [obj for obj in anno if obj["category_id"] in cat_list]
        if _has_valid_annotation(anno):
            ids.append(ds_idx)

    dataset = torch.utils.data.Subset(dataset, ids)
    return dataset


def convert_to_coco_api(ds):
    """
    Convert a PyTorch dataset to COCO API format.
    
    This function takes a PyTorch dataset and converts it to a format compatible
    with the COCO API. It processes all images and annotations to create a valid
    COCO dataset structure that can be used with pycocotools.
    
    Args:
        ds: PyTorch dataset that returns (image, target) tuples where target
            contains COCO-style annotations
            
    Returns:
        COCO: COCO API object with the converted dataset
        
    Dataset Structure Created:
        - images: List of image metadata dictionaries
        - categories: List of category definitions  
        - annotations: List of annotation dictionaries
        - info: Dataset information metadata
        
    Supported Annotation Types:
        - Bounding boxes (required)
        - Segmentation masks (optional)
        - Keypoints (optional)
        
    Note:
        - Annotation IDs start at 1 (COCO requirement)
        - Bounding boxes are converted from [xmin, ymin, xmax, ymax] to [x, y, width, height] format
        - Segmentation masks are encoded using COCO mask utilities
        - Categories are automatically detected from the dataset labels
        
    Example:
        >>> coco_api = convert_to_coco_api(pytorch_dataset)
        >>> print(f"Converted dataset with {len(coco_api.getImgIds())} images")
    """
    coco_ds = COCO()
    # annotation IDs need to start at 1, not 0, see torchvision issue #1530
    ann_id = 1
    dataset = {"images": [], "categories": [], "annotations": [], "info": {}}
    categories = set()
    for img_idx in range(len(ds)):
        # find better way to get target
        # targets = ds.get_annotations(img_idx)
        img, targets = ds[img_idx]
        image_id = targets["image_id"]
        img_dict = {}
        img_dict["id"] = image_id
        img_dict["height"] = img.shape[-2]
        img_dict["width"] = img.shape[-1]
        dataset["images"].append(img_dict)
        bboxes = targets["boxes"].clone()
        bboxes[:, 2:] -= bboxes[:, :2]
        bboxes = bboxes.tolist()
        labels = targets["labels"].tolist()
        areas = targets["area"].tolist()
        iscrowd = targets["iscrowd"].tolist()
        if "masks" in targets:
            masks = targets["masks"]
            # make masks Fortran contiguous for coco_mask
            masks = masks.permute(0, 2, 1).contiguous().permute(0, 2, 1)
        if "keypoints" in targets:
            keypoints = targets["keypoints"]
            keypoints = keypoints.reshape(keypoints.shape[0], -1).tolist()
        num_objs = len(bboxes)
        for i in range(num_objs):
            ann = {}
            ann["image_id"] = image_id
            ann["bbox"] = bboxes[i]
            ann["category_id"] = labels[i]
            categories.add(labels[i])
            ann["area"] = areas[i]
            ann["iscrowd"] = iscrowd[i]
            ann["id"] = ann_id
            if "masks" in targets:
                ann["segmentation"] = coco_mask.encode(masks[i].numpy())
            if "keypoints" in targets:
                ann["keypoints"] = keypoints[i]
                ann["num_keypoints"] = sum(k != 0 for k in keypoints[i][2::3])
            dataset["annotations"].append(ann)
            ann_id += 1
    dataset["categories"] = [{"id": i} for i in sorted(categories)]
    coco_ds.dataset = dataset
    coco_ds.createIndex()
    return coco_ds


def get_coco_api_from_dataset(dataset):
    """
    Extract or create a COCO API object from a PyTorch dataset.
    
    This function attempts to extract a COCO API object from various types of
    PyTorch datasets. If the dataset is already a COCO dataset, it returns the
    existing COCO API. Otherwise, it converts the dataset to COCO format.
    
    Args:
        dataset: PyTorch dataset object (CocoDetection, Subset, or custom dataset)
        
    Returns:
        COCO: COCO API object for accessing dataset annotations
        
    Supported Dataset Types:
        - torchvision.datasets.CocoDetection: Returns existing .coco attribute
        - torch.utils.data.Subset: Recursively unwraps to find underlying dataset
        - Custom datasets: Converts using convert_to_coco_api()
        
    Note:
        - The function attempts up to 10 levels of dataset unwrapping for Subset datasets
        - This is useful for evaluation and analysis tasks that require COCO API access
        - The returned COCO object can be used with pycocotools for evaluation
        
    Example:
        >>> coco_api = get_coco_api_from_dataset(train_loader.dataset)
        >>> num_categories = len(coco_api.getCatIds())
        >>> print(f"Dataset has {num_categories} categories")
    """
    # FIXME: This is... awful?
    for _ in range(10):
        if isinstance(dataset, torchvision.datasets.CocoDetection):
            break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco
    return convert_to_coco_api(dataset)


class CocoDetection(torchvision.datasets.CocoDetection):
    """
    Extended COCO detection dataset with transform support.
    
    This class extends the standard torchvision COCO detection dataset to provide
    better integration with custom transforms and preprocessing pipelines. It maintains
    compatibility with the base CocoDetection while adding support for more complex
    transformation workflows.
    
    Args:
        img_folder (str): Path to the directory containing images
        ann_file (str): Path to the COCO annotation JSON file  
        transforms (callable, optional): Transform to be applied to (image, target) pairs
        
    Attributes:
        _transforms (callable): The transformation function to apply
        
    Note:
        - Unlike the base class, transforms are applied to both image and target
        - The target dictionary includes image_id and annotations for transform compatibility
        - This class is designed to work with transforms that expect (image, target) tuples
        
    Example:
        >>> dataset = CocoDetection(
        ...     img_folder="/path/to/images",
        ...     ann_file="/path/to/annotations.json", 
        ...     transforms=custom_transform
        ... )
        >>> image, target = dataset[0]
    """
    
    def __init__(self, img_folder, ann_file, transforms):
        """
        Initialize the extended COCO detection dataset.
        
        Args:
            img_folder (str): Directory containing images
            ann_file (str): COCO annotation file path
            transforms (callable): Transform function for (image, target) pairs
        """
        super().__init__(img_folder, ann_file)
        self._transforms = transforms

    def __getitem__(self, idx):
        """
        Get an item from the dataset with transforms applied.
        
        Args:
            idx (int): Index of the item to retrieve
            
        Returns:
            tuple: (image, target) pair after applying transforms
        """
        img, target = super().__getitem__(idx)
        image_id = self.ids[idx]
        target = dict(image_id=image_id, annotations=target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


def get_coco(root, image_set, transforms, mode="instances", use_v2=False, with_masks=False):
    """
    Create a COCO dataset with specified configuration.
    
    This function creates a COCO dataset instance with the specified parameters,
    handling both v1 and v2 transform APIs. It automatically applies appropriate
    transforms and filters out images without valid annotations for training sets.
    
    Args:
        root (str): Root directory containing COCO dataset
        image_set (str): Dataset split to load ("train" or "val")
        transforms (callable): Transform to apply to the dataset
        mode (str, optional): Annotation mode ("instances", "captions", etc.). Defaults to "instances"
        use_v2 (bool, optional): Whether to use torchvision v2 transforms. Defaults to False
        with_masks (bool, optional): Whether to include segmentation masks. Defaults to False
        
    Returns:
        Dataset: Configured COCO dataset ready for training or evaluation
        
    Dataset Structure:
        - Images are loaded from {root}/{split}2017/ directories
        - Annotations are loaded from {root}/annotations/{mode}_{split}2017.json files
        
    Features:
        - Automatic annotation filtering for training sets
        - Support for both v1 and v2 transform APIs
        - Optional mask inclusion for segmentation tasks
        - Proper target key configuration for v2 transforms
        
    Note:
        - Training sets automatically remove images without valid annotations
        - Validation sets keep all images for proper evaluation
        - The function handles the complexity of transform API differences
        
    Example:
        >>> train_dataset = get_coco(
        ...     root="/path/to/coco",
        ...     image_set="train", 
        ...     transforms=train_transforms,
        ...     with_masks=True
        ... )
        >>> print(f"Training dataset has {len(train_dataset)} images")
    """
    anno_file_template = "{}_{}2017.json"
    PATHS = {
        "train": ("train2017", os.path.join("annotations", anno_file_template.format(mode, "train"))),
        "val": ("val2017", os.path.join("annotations", anno_file_template.format(mode, "val"))),
        # "train": ("val2017", os.path.join("annotations", anno_file_template.format(mode, "val")))
    }

    img_folder, ann_file = PATHS[image_set]
    img_folder = os.path.join(root, img_folder)
    ann_file = os.path.join(root, ann_file)

    if use_v2:
        from torchvision.datasets import wrap_dataset_for_transforms_v2

        dataset = torchvision.datasets.CocoDetection(img_folder, ann_file, transforms=transforms)
        target_keys = ["boxes", "labels", "image_id"]
        if with_masks:
            target_keys += ["masks"]
        dataset = wrap_dataset_for_transforms_v2(dataset, target_keys=target_keys)
    else:
        # TODO: handle with_masks for V1?
        t = [ConvertCocoPolysToMask()]
        if transforms is not None:
            t.append(transforms)
        transforms = T.Compose(t)

        dataset = CocoDetection(img_folder, ann_file, transforms=transforms)

    if image_set == "train":
        dataset = _coco_remove_images_without_annotations(dataset)

    # dataset = torch.utils.data.Subset(dataset, [i for i in range(500)])

    return dataset
