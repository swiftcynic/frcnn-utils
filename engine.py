import math
import sys
import time

import torch
import torchvision

from .req_utils import MetricLogger, SmoothedValue
from .req_utils import reduce_dict
from .coco_eval import CocoEvaluator
from .coco_utils import get_coco_api_from_dataset
from .constants import DEVICE

# Note: MPS (Metal Performance Shaders) support considerations:
# - torch.amp.autocast currently only supports 'cuda' and 'cpu' devices
# - MPS operations fall back to CPU autocast for mixed precision training
# - Some operations might not be fully optimized for MPS and may fall back to CPU

def get_model(backbone, weights, train_dataset, predictor, device=DEVICE):
    """
    Create and return a Faster R-CNN model with a custom predictor head.
    """
    # Load a pre-trained Faster R-CNN model with ResNet50 backbone
    model = backbone(
        weights=weights
    )

    # Get the number of classes in the dataset
    num_classes = len(train_dataset.coco.getCatIds()) + 1

    # Get the number of input features for the classifier head
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = predictor(in_features, num_classes)

    # Move the model to the specified device (GPU or CPU)
    model.to(device)

def _get_autocast_device(device):
    """
    Get the appropriate device string for torch.amp.autocast based on the device type.
    Currently, torch.amp.autocast only supports 'cuda' and 'cpu'.
    MPS operations will fall back to CPU autocast.
    """
    if device.type == 'cuda':
        return 'cuda'
    else:
        # For MPS and other devices, fall back to CPU autocast
        return 'cpu'


def _synchronize_device(device):
    """
    Synchronize the appropriate device based on its type.
    """
    try:
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type == 'mps':
            torch.mps.synchronize()
    except (RuntimeError, AttributeError):
        # If synchronization fails, continue without it
        print(f"Warning: Device synchronization failed or is not supported on {device}.")
        pass

def train(model, optimizer, train_loader, valid_loader, device=DEVICE, num_epochs=10, print_freq=250, scaler=None, output_dir=None, lr_scheduler=None):
    """
    Train a model for object detection.
    
    Args:
        device: torch.device or None. If None, automatically detects best available device.
    """
        
    for epoch in range(1, num_epochs+1):
        train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq, scaler)
        evaluate(model, valid_loader, device=device)
        if output_dir:
            torch.save(model.state_dict(), f'{output_dir}/model_epoch_{epoch}.pth')

def train_one_epoch(model, optimizer, data_loader, device=DEVICE, epoch=1, print_freq=250, lr_scheduler=None, scaler=None):
    """
    Train the model for one epoch.
    
    Args:
        device: torch.device or None. If None, automatically detects best available device.
    """
        
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    lr_scheduler = lr_scheduler
    if epoch == 1:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{'boxes': t['boxes'].to(device), 'labels': t['labels'].to(device)} for t in targets]
        
        # Use helper function to get appropriate autocast device
        autocast_device = _get_autocast_device(device)
        with torch.amp.autocast(autocast_device, enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.inference_mode()
def evaluate(model, data_loader, device=DEVICE):
    """
    Evaluate the model.
    
    Args:
        device: torch.device or None. If None, automatically detects best available device.
    """
        
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)

        # Synchronize device before timing
        _synchronize_device(device)
        
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"]: output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator

def get_predictions(model, images, threshold=0.75):
    threshold = 0.75

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

        for i in range(len(scores)):
            if scores[i] <= threshold:
                continue
            confident_predictions['boxes'].append(boxes[i])
            confident_predictions['labels'].append(labels[i])
            confident_predictions['scores'].append(scores[i])

        
        if len(confident_predictions['boxes']) == 0:
            if len(scores) != 0:
                confident_predictions = {
                    'boxes': [boxes[0]],
                    'labels': [labels[0]],
                    'scores': [scores[0]]
                }
            else:
                confident_predictions = {
                    'boxes': [],
                    'labels': [],
                    'scores': []
                }
        predictions.append(confident_predictions)
    return (predictions)