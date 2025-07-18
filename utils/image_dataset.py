"""Custom PyTorch Dataset for loading and preprocessing images.

This module provides a dataset class for loading images from file paths with support for data
transformations and augmentations.
"""

import os
from typing import Any

import cv2
import pandas as pd
from albumentations import (
    Compose,
    HorizontalFlip,
    Normalize,
    RandomBrightnessContrast,
    Resize,
    VerticalFlip,
)
from albumentations.pytorch import ToTensorV2
from torch import Tensor
from torch.utils.data import Dataset

# Default image dimensions for preprocessing
DEFAULT_WIDTH, DEFAULT_HEIGHT = 299, 299


class ImageDataset(Dataset):
    """Custom PyTorch Dataset for loading mushroom images with labels.

    This dataset loads images from file paths using OpenCV, converts them to RGB format,
    and applies optional transformations. It can work with either pandas DataFrames or CSV files.

    Args:
        split_df (pd.DataFrame, optional): DataFrame with 'image_path' and 'label_id' columns
        csv_file (str, optional): Path to CSV file containing image paths and labels
        data_dir (str, optional): Base directory for image paths (used with CSV files)
        transform (Callable, optional): Optional transform to be applied to images
        target_transform (Callable, optional): Optional transform to be applied to labels

    Raises:
        FileNotFoundError: If an image file cannot be read from the specified path
        ValueError: If neither split_df nor csv_file is provided
    """

    def __init__(
        self,
        split_df: pd.DataFrame | None = None,
        csv_file: str | None = None,
        data_dir: str | None = None,
        transform: Any = None,
        target_transform: Any = None,
    ) -> None:
        if split_df is not None:
            self.image_labels = split_df["label_id"]
            self.image_paths = split_df["image_path"]
        elif csv_file is not None:
            df = pd.read_csv(csv_file)
            if "label" in df.columns:
                self.image_labels = df["label"]
            elif "label_id" in df.columns:
                self.image_labels = df["label_id"]
            else:
                raise ValueError(
                    "CSV file must contain either 'label' or 'label_id' column"
                )

            if "filename" in df.columns and data_dir is not None:
                self.image_paths = df["filename"].apply(
                    lambda x: os.path.join(data_dir, x)
                )
            elif "image_path" in df.columns:
                self.image_paths = df["image_path"]
            elif "filename" in df.columns:
                self.image_paths = df["filename"]
            else:
                raise ValueError(
                    "CSV file must contain either 'image_path' or 'filename' column"
                )
        else:
            raise ValueError("Either split_df or csv_file must be provided")

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset
        """
        return len(self.image_labels)

    def __getitem__(self, index: int) -> tuple[Tensor | Any, int | Any]:
        """Get a single sample from the dataset.

        Args:
            index (int): Index of the sample to retrieve

        Returns:
            tuple[Tensor | Any, int | Any]: Tuple containing the image (tensor or array) and label

        Raises:
            FileNotFoundError: If the image file cannot be read
        """
        image_path = self.image_paths.iloc[index]
        image = cv2.imread(image_path)  # pylint: disable=E1101
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # pylint: disable=E1101

        label = self.image_labels.iloc[index]

        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]

        if self.target_transform:
            label = self.target_transform(label)

        return image, label


def get_transforms(
    image_size: tuple[int, int] | None = None, transform_type: str = "testing"
) -> Compose:
    """Get image transformation pipeline.

    Args:
        image_size (tuple[int, int] | None): Target image size as (width, height).
            Defaults to (299, 299) if not provided.
        transform_type (str): Type of transforms - "training" or "testing".
            Defaults to "testing".

    Returns:
        Compose: Albumentations composition of transforms

    Raises:
        ValueError: If transform_type is not "training" or "testing"
    """
    # Use default size if not provided
    if image_size is None:
        width, height = DEFAULT_WIDTH, DEFAULT_HEIGHT
    else:
        width, height = image_size

    if transform_type == "training":
        return Compose(
            [
                Resize(width, height),
                HorizontalFlip(p=0.5),
                VerticalFlip(p=0.5),
                RandomBrightnessContrast(p=0.5),
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

    if transform_type == "testing":
        return Compose(
            [
                Resize(width, height),
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

    raise ValueError(f"Unknown transform type: {transform_type}")
