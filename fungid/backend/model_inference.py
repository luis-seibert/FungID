"""Utility functions for performing image classification inference using pre-trained ResNet models.

This module provides helper functions to load images, initialize the model, and classify images.

Example:
```python
result = image_classification(
    image_path="path/to/image.jpg",
    model_path="checkpoints/ResNet18_best_model.pth",
    model_name="ResNet18",
    number_classes=2,
    class_labels={0: "Non-bitter Bolete (Edible)", 1: "Bitter Bolete (Unpalatable)"}
)
print(result)
```"""

import os
from typing import Any

import cv2
import torch
from albumentations import (
    Compose,
    HorizontalFlip,
    Normalize,
    RandomBrightnessContrast,
    Resize,
    VerticalFlip,
)
from albumentations.pytorch import ToTensorV2
from torch import nn
from torchvision import models


def image_classification(
    image_path: str,
    model_path: str,
    model_name: str,
    number_classes: int,
    class_labels: dict[int, str],
) -> dict[str, Any]:
    """Run image classification from selected file.

    Args:
        image_path (str): Path to the image file.
        model_path (str): Path to the pre-trained model file.
        model_name (str): Name of the model architecture to load.
        number_classes (int): Number of classes for the model.
        class_labels (dict[int, str]): Mapping of class indices to class names.

    Returns:
        dict[str, Any]: Classification results.

    Raises:
        ValueError: If the predicted class is not found in the class labels.
    """
    image = _read_image(image_path)
    tensor = _get_tensor(image)
    model = _initialize_model(model_path, model_name, number_classes)

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(outputs, dim=1).item()
        confidence = probabilities[0][int(predicted_class)].item()

    if predicted_class not in class_labels:
        raise ValueError(
            f"Predicted class '{predicted_class}' not found in class labels."
        )

    return {
        "filename": os.path.basename(image_path),
        "predicted_class": predicted_class,
        "class_name": class_labels[int(predicted_class)],
        "confidence": confidence,
        "is_bitter": predicted_class == 1,
        "probabilities": {
            "non_bitter": probabilities[0][0].item(),
            "bitter": probabilities[0][1].item(),
        },
    }


def _get_tensor(image: Any) -> Any:
    """Transform the image to a tensor suitable for model input.

    Args:
        image (Any): The input image.

    Returns:
        Any: The transformed image.
    """
    transform = _get_transforms((299, 299), "testing")
    return transform(image=image)["image"].unsqueeze(0).to(_get_device())


def _read_image(file_path: str) -> Any:
    """Read an image from the specified file path.

    Args:
        file_path (str): Path to the image file.

    Returns:
        Any: The loaded image.

    Raises:
        ValueError: If the file is not a valid image format or cannot be loaded.
    """
    if not file_path.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
        raise ValueError("Selected file is not a known image format.")

    image = cv2.imread(file_path)

    if image is None:
        raise ValueError(f"Failed to load image: {file_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def _initialize_model(
    model_path: str, model_name: str, number_classes: int
) -> nn.Module:
    """Load the pre-trained ResNet model.

    Args:
        model_path (str): Path to the model file.
        model_name (str): Name of the model architecture to load.
        number_classes (int): Number of classes for the model.

    Returns:
        nn.Module: The loaded model.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = _get_resnet_model(model_name)
    model.fc = nn.Linear(model.fc.in_features, number_classes)
    device = _get_device()
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    return model


def _get_transforms(
    image_size: tuple[int, int] = (299, 299), transform_type: str = "testing"
) -> Compose:
    """Get image transformation pipeline.

    Args:
        image_size (tuple[int, int]): Target image size as (width, height).
            Defaults to (299, 299) if not provided.
        transform_type (str): Type of transforms - "training" or "testing".
            Defaults to "testing".

    Returns:
        Compose: Albumentations composition of transforms

    Raises:
        ValueError: If transform_type is not "training" or "testing"
    """
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


def _get_resnet_model(model_name: str) -> models.ResNet:
    """Returns a pre-trained ResNet model based on the specified model name.

    The models selection is case-insensitive.

    Args:
        model_name (str): Name of the ResNet model ('ResNet18', 'ResNet34', 'ResNet50')

    Returns:
        models.ResNet: Pre-trained ResNet model with ImageNet weights

    Raises:
        ValueError: If the model name is not supported
    """
    name = model_name.lower()
    if name == "resnet18":
        return models.resnet18(weights="IMAGENET1K_V1")
    if name == "resnet34":
        return models.resnet34(weights="IMAGENET1K_V1")
    if name == "resnet50":
        return models.resnet50(weights="IMAGENET1K_V1")
    raise ValueError(f"Model {model_name} is not supported.")


def _get_device() -> torch.device:
    """Get the device for PyTorch.

    Returns:
        torch.device: The device to use (CPU or GPU).
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
