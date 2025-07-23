"""This package contains utility modules for the FungID mushroom classification project.

Modules:
- dataset_acquisition: Download and scrape mushroom images from MushroomObserver
- dataset_preparation: Split datasets and prepare data for training
- image_dataset: PyTorch dataset class and data transformations
- model_training: Training utilities and model architecture
- model_inference: Model inference and prediction functions
- logger: Logging configuration
"""

from .dataset_acquisition import (
    count_species,
    download_google_sheets_tsv,
    scrape_species_from_list,
)
from .dataset_preparation import split_dataset
from .image_dataset import ImageDataset, get_transforms
from .model_inference import image_classification
from .model_training import get_device, get_resnet_model, train_model

__all__ = [
    "download_google_sheets_tsv",
    "count_species",
    "scrape_species_from_list",
    "split_dataset",
    "ImageDataset",
    "get_transforms",
    "get_resnet_model",
    "train_model",
    "get_device",
    "image_classification",
]
