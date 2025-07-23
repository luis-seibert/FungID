"""Utility functions for performing image classification inference using pre-trained ResNet models.

This module provides helper functions to load images, initialize the model, and classify images.

Example:
```python
from utils.model_inference import image_classification
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
from torch import nn

from utils.image_dataset import get_transforms
from utils.model_training import get_device, get_resnet_model


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
    model = _initialize_model(model_path, model_name, number_classes)

    # Transform the image
    transform = get_transforms((299, 299), "testing")
    transformed = transform(image=image)
    image_tensor = transformed["image"].unsqueeze(0).to(get_device())

    # Predict the class
    with torch.no_grad():
        outputs = model(image_tensor)
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

    model = get_resnet_model(model_name)
    model.fc = nn.Linear(model.fc.in_features, number_classes)

    checkpoint = torch.load(model_path, map_location=get_device())

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()
    model.to(get_device())

    return model
