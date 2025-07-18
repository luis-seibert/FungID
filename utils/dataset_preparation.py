"""Module for preparing and splitting datasets for machine learning training.

This module provides functions to split image datasets into training, validation,
and testing sets with binary classification labels. It handles directory-based
image datasets and generates CSV files with the splits.

Example usage:
```python
from utils.dataset_preparation import split_dataset

split_dataset(
    data_path="data/images",
    training_ids_path="data/splits/training.csv",
    validation_ids_path="data/splits/validation.csv",
    testing_ids_path="data/splits/testing.csv",
    training_split_ratio=0.7,
    validation_split_ratio=0.2,
    positive_class="Amanita muscaria",
    random_seed=42
)
```
"""

import csv
import os
import random

from utils.logger import logger


def split_dataset(
    data_path: str,
    training_ids_path: str,
    validation_ids_path: str,
    testing_ids_path: str,
    training_split_ratio: float,
    validation_split_ratio: float,
    positive_class: str,
    random_seed: int = 0,
) -> None:
    """Generates CSV files with splits for training, validation, and testing datasets.

    Args:
        data_path (str): Path to the directory containing class directories.
        training_ids_path (str): Path to the training CSV file.
        validation_ids_path (str): Path to the validation CSV file.
        testing_ids_path (str): Path to the testing CSV file.
        training_split_ratio (float): Ratio for training split.
        validation_split_ratio (float): Ratio for validation split.
        positive_class (str): Name of the positive class.
        random_seed (int): Seed for random number generator for reproducibility.
    """
    # Set random seed for reproducible splits
    random.seed(random_seed)

    # Validate data path
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"The specified data path does not exist: {data_path}")

    # Avoid overwriting existing split
    if (
        os.path.exists(training_ids_path)
        or os.path.exists(validation_ids_path)
        or os.path.exists(testing_ids_path)
    ):
        logger.info("Dataset split files already exist, skipping generation.")
        return

    class_directories = os.listdir(data_path)

    # Filter out non-directory entries
    if not class_directories:
        raise ValueError("No class directories found in the specified path.")

    for file_path in [training_ids_path, validation_ids_path, testing_ids_path]:
        with open(file_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["species_name", "image_path", "label_id"])

    for class_name in class_directories:
        class_path = os.path.join(data_path, class_name)
        if not os.path.isdir(class_path):
            logger.warning("Skipping non-directory: %s", class_name)
            continue

        split_class_dataset(
            class_path,
            training_ids_path,
            validation_ids_path,
            testing_ids_path,
            training_split_ratio,
            validation_split_ratio,
            positive_class,
        )

    logger.info(
        "Dataset created with %s as positive class (randomized 3-way split)",
        positive_class,
    )
    test_split_ratio = 1 - training_split_ratio - validation_split_ratio
    logger.info(
        "Split ratios: Train %.0f%%, Val %.0f%%, Test %.0f%%",
        training_split_ratio * 100,
        validation_split_ratio * 100,
        test_split_ratio * 100,
    )


def split_class_dataset(
    class_path: str,
    training_ids_path: str,
    validation_ids_path: str,
    testing_ids_path: str,
    training_split_ratio: float,
    validation_split_ratio: float,
    positive_class: str,
) -> None:
    """Splits the dataset into training, validation, and testing sets with binary labels.

    Args:
        class_path (str): Path to the directory containing files for a specific class.
        training_ids_path (str): Path to the training CSV file.
        validation_ids_path (str): Path to the validation CSV file.
        testing_ids_path (str): Path to the testing CSV file.
        training_split_ratio (float): Ratio for training split.
        validation_split_ratio (float): Ratio for validation split.
        positive_class (str): Name of the positive class.
    """
    class_name = os.path.basename(class_path)

    file_ids = [
        f
        for f in os.listdir(class_path)
        if os.path.isfile(os.path.join(class_path, f))
        and f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    # Check if there are any image files in the class directory
    if not file_ids:
        logger.warning("No image files found in: %s", class_path)
        return

    # Randomize the file list to avoid biases from file ordering
    random.shuffle(file_ids)

    label_id = 1 if class_name == positive_class else 0

    # Calculate split sizes
    total_files = len(file_ids)
    train_size = int(total_files * training_split_ratio)
    val_size = int(total_files * validation_split_ratio)

    # Split the files
    train_files = file_ids[:train_size]
    val_files = file_ids[train_size : train_size + val_size]
    test_files = file_ids[train_size + val_size :]

    # Write training data
    with open(training_ids_path, "a", newline="", encoding="utf-8") as train_file:
        writer = csv.writer(train_file)
        for file_name in train_files:
            image_path = os.path.join("data", "images", class_name, file_name)
            writer.writerow([class_name, image_path, label_id])

    # Write validation data
    with open(validation_ids_path, "a", newline="", encoding="utf-8") as val_file:
        writer = csv.writer(val_file)
        for file_name in val_files:
            image_path = os.path.join("data", "images", class_name, file_name)
            writer.writerow([class_name, image_path, label_id])

    # Write testing data
    with open(testing_ids_path, "a", newline="", encoding="utf-8") as test_file:
        writer = csv.writer(test_file)
        for file_name in test_files:
            image_path = os.path.join("data", "images", class_name, file_name)
            writer.writerow([class_name, image_path, label_id])

    logger.info(
        "Processed %s: %d train, %d val, %d test images (label_id: %d)",
        class_name,
        len(train_files),
        len(val_files),
        len(test_files),
        label_id,
    )
