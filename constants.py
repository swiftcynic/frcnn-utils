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

_this_file_path = os.path.abspath(__file__)
_this_dir_path = os.path.dirname(_this_file_path)

PROJECT_DIR_PATH = os.path.dirname(_this_dir_path)
DATASET_DIR_PATH = os.path.join(PROJECT_DIR_PATH, 'dataset')

# Data yaml file path
DATA_YAML_PATH = os.path.join(_this_dir_path, 'dataset/data.yaml')

# Pests classes mapping
PEST_CLASSES = [
    'ant',
    'bee',
    'beetle',
    'caterpillar',
    'earthworm',
    'earwig',
    'grasshopper',
    'moth',
    'slug',
    'snail',
    'wasp',
    'weevil'
]

# IMAGE_SIZE for model input
IMAGE_SIZE = (640, 640) 
IMAGE_RESIZE_FACTOR = 0.5

# Model parameters
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 12
LEARNING_RATE = 5e-3
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
NUM_EPOCHS = 50
OPTIMIZER_TYPE = SGD
DEVICE = check_set_gpu()