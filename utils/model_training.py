"""Model training utilities for training with checkpointing and recall-based model selection.

This module provides functions for training, validating, and evaluating PyTorch models, including
gradient accumulation, learning rate scheduling, and comprehensive metrics calculation for binary
classification. Model selection is based on recall performance, with AUROC computed for monitoring.

The module is organized into several key areas:
- Model preparation: get_resnet_model()
- Image preprocessing and metrics calculation: _prepare_images(), _calculate_batch_metrics()
- Training utilities: _train_epoch(), _validate_epoch()
- Metrics calculation: _calculate_auroc(), _calculate_recall()
- Model selection: _update_best_model_recall() (primary), _update_best_model() (legacy)
- Checkpoint management: _save_epoch_checkpoint(), _save_best_model()
- Progress tracking and logging: _update_progress_bar(), _log_epoch_results()
- Main training orchestration: train_model()

Example usage:
```python
import torch

from utils.model_training import get_resnet_model, train_model

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(
        torch.randn(100, 3, 224, 224), torch.randint(0, 10, (100,))
    ),
    batch_size=32,
    shuffle=True,
)
valid_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(
        torch.randn(50, 3, 224, 224), torch.randint(0, 10, (50,))
    ),
    batch_size=32,
    shuffle=False,
)

model = get_resnet_model("Resnet18")
train_model(
    model=model,
    train_loader=train_loader,
    valid_loader=valid_loader,
    criterion=torch.nn.CrossEntropyLoss(),
    optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=5
    ),
    device=get_device(),
    epochs=50,
    accumulation_steps=4,
    checkpoint_dir="checkpoints",
    model_name="ResNet18",
)
```
"""

import os
import time
from typing import Any

import torch
from sklearn.metrics import recall_score, roc_auc_score
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

from utils.logger import logger


def get_resnet_model(model_name: str) -> models.ResNet:
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


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    scheduler: Any,
    device: torch.device | str,
    epochs: int,
    accumulation_steps: int,
    checkpoint_dir: str = "checkpoints",
    model_name: str = "ResNet18",
) -> dict[str, Any]:
    """Complete training loop with recall-based model selection and comprehensive metrics.

    Model selection is based on highest recall score. Checkpoints are saved after each epoch, and the
    best model is loaded into the provided model at the end of training. AUROC is also computed for
    monitoring model discrimination capability.

    Args:
        model (nn.Module): PyTorch neural network model to train.
        train_loader (DataLoader): DataLoader for training data.
        valid_loader (DataLoader): DataLoader for validation data.
        criterion (nn.Module): Loss function (e.g., CrossEntropyLoss).
        optimizer (Optimizer): Optimizer for parameter updates (e.g., Adam, SGD).
        scheduler (Any): Learning rate scheduler (e.g., ReduceLROnPlateau).
        device (torch.device): Device to run computations on ('cuda', 'cpu', or torch.device).
        epochs (int): Number of training epochs.
        accumulation_steps (int): Number of mini-batches to accumulate before optimizer step.
        checkpoint_dir (str, optional): Directory to save checkpoints. Defaults to "checkpoints".
        model_name (str): Name prefix for saved model files.

    Returns:
        dict[str, Any]: Dictionary containing training history and best model information:
        - train_losses: List of training losses per epoch
        - train_accuracies: List of training accuracies per epoch
        - val_losses: List of validation losses per epoch
        - val_accuracies: List of validation accuracies per epoch
        - val_aurocs: List of validation AUROC scores per epoch
        - val_recalls: List of validation recall scores per epoch
        - best_val_acc: Best validation accuracy achieved
        - best_val_auroc: Best validation AUROC achieved
        - best_val_recall: Best validation recall achieved
        - best_model_state: State dict of the best model
        - best_epoch: Epoch number where best model was found
        - checkpoint_dir: Directory containing saved checkpoints
    """
    # Initialize training
    logger.info("Starting training...")
    logger.info(
        "Training for %d epochs with accumulation steps %d", epochs, accumulation_steps
    )
    logger.info("Best model selection based on: Recall")

    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize tracking variables
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies, val_aurocs, val_recalls = [], [], [], []
    best_val_acc, best_val_auroc, best_val_recall, best_epoch = 0.0, 0.0, 0.0, 0
    best_model_state = None
    previous_lr = optimizer.param_groups[0]["lr"]

    # Training loop
    for epoch in range(epochs):
        start_time = time.time()

        # Train and validate
        train_loss, train_acc = _train_epoch(
            model, train_loader, criterion, optimizer, device, accumulation_steps
        )
        val_loss, val_acc = _validate_epoch(model, valid_loader, criterion, device)
        val_auroc = _calculate_auroc(model, valid_loader, device)
        val_recall = _calculate_recall(model, valid_loader, device)

        # Update scheduler and check learning rate
        scheduler.step(val_loss)
        previous_lr = _check_learning_rate_reduction(optimizer, previous_lr)

        # Update metrics lists
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        val_aurocs.append(val_auroc)
        val_recalls.append(val_recall)

        # Update best model if necessary
        best_val_recall, best_val_acc, best_val_auroc, best_model_state, best_epoch = (
            _update_best_model_recall(
                val_recall,
                val_acc,
                val_auroc,
                epoch + 1,
                model,
                best_val_recall,
                best_val_acc,
                best_val_auroc,
                best_epoch,
                best_model_state,
            )
        )

        # Save epoch checkpoint
        checkpoint_data = _create_checkpoint_data(
            epoch + 1,
            model,
            optimizer,
            scheduler,
            train_losses,
            train_accuracies,
            val_losses,
            val_accuracies,
            val_aurocs,
            val_recalls,
            best_val_acc,
            best_val_auroc,
            best_val_recall,
            best_epoch,
            train_acc,
            val_acc,
            val_auroc,
            val_recall,
        )
        _save_epoch_checkpoint(checkpoint_dir, model_name, epoch + 1, checkpoint_data)

        # Log epoch results
        epoch_time = time.time() - start_time
        current_lr = optimizer.param_groups[0]["lr"]
        _log_epoch_results(
            epoch + 1,
            epochs,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            val_auroc,
            val_recall,
            current_lr,
            epoch_time,
        )

    # Finalize training
    _finalize_training(
        checkpoint_dir,
        model_name,
        best_model_state,
        train_losses,
        train_accuracies,
        val_losses,
        val_accuracies,
        val_aurocs,
        val_recalls,
        best_val_acc,
        best_val_auroc,
        best_val_recall,
        best_epoch,
        epochs,
        model,
    )

    return {
        "train_losses": train_losses,
        "train_accuracies": train_accuracies,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies,
        "val_aurocs": val_aurocs,
        "val_recalls": val_recalls,
        "best_val_acc": best_val_acc,
        "best_val_auroc": best_val_auroc,
        "best_val_recall": best_val_recall,
        "best_model_state": best_model_state,
        "best_epoch": best_epoch,
        "checkpoint_dir": checkpoint_dir,
    }


def get_device() -> torch.device:
    """Get the device for PyTorch.

    Returns:
        torch.device: The device to use (CPU or GPU).
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _prepare_images(images: torch.Tensor, device: torch.device | str) -> torch.Tensor:
    """Prepare images for model input by moving to device and converting to float.

    Args:
        images: Input image tensor
        device: Target device for computation

    Returns:
        torch.Tensor: Preprocessed images ready for model input
    """
    images = images.to(device)
    if images.dtype != torch.float32:
        images = images.float()
    return images


def _calculate_batch_metrics(
    outputs: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, int, int]:
    """Calculate batch-level prediction metrics.

    Args:
        outputs: Model output logits
        labels: True labels

    Returns:
        tuple: (predicted_labels, correct_predictions_count, total_samples)
    """
    _, predicted = torch.max(outputs.data, 1)
    total_samples = labels.size(0)
    correct_predictions = int((predicted == labels).sum().item())
    return predicted, correct_predictions, total_samples


def _update_progress_bar(
    progress_bar: tqdm,
    running_loss: float,
    batch_idx: int,
    correct_predictions: int,
    total_samples: int,
) -> None:
    """Update progress bar with current training/validation metrics.

    Args:
        progress_bar: tqdm progress bar instance
        running_loss: Accumulated loss
        batch_idx: Current batch index
        correct_predictions: Number of correct predictions so far
        total_samples: Total number of samples processed
    """
    current_acc = 100 * correct_predictions / total_samples if total_samples > 0 else 0
    progress_bar.set_postfix(
        {
            "Loss": f"{running_loss / (batch_idx + 1):.4f}",
            "Acc": f"{current_acc:.2f}%",
        }
    )


def _handle_gradient_accumulation(
    optimizer: Optimizer, batch_idx: int, accumulation_steps: int
) -> None:
    """Handle gradient accumulation and optimizer steps.

    Args:
        optimizer: Model optimizer
        batch_idx: Current batch index
        accumulation_steps: Number of steps to accumulate gradients
    """
    if (batch_idx + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()


def _finalize_gradient_accumulation(
    optimizer: Optimizer, batch_idx: int, accumulation_steps: int
) -> None:
    """Handle any remaining gradients after training loop.

    Args:
        optimizer: Model optimizer
        batch_idx: Final batch index
        accumulation_steps: Number of steps to accumulate gradients
    """
    if (batch_idx + 1) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()


def _train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device | str,
    accumulation_steps: int,
) -> tuple[float, float]:
    """Train the model for one epoch with gradient accumulation.

    Args:
        model (nn.Module): PyTorch neural network model to train
        train_loader (DataLoader): DataLoader for training data
        criterion (nn.Module): Loss function (e.g., CrossEntropyLoss)
        optimizer (Optimizer): Optimizer for parameter updates (e.g., Adam, SGD)
        device (torch.device): Device to run computations on ('cuda', 'cpu', or torch.device)
        accumulation_steps (int): Number of mini-batches to accumulate before optimizer step

    Returns:
        tuple[float, float]: Tuple of (average_loss, accuracy_percentage) for the epoch
    """
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    # Initialize gradient accumulation
    optimizer.zero_grad()
    batch_idx = -1  # Initialize batch_idx in case of empty loader

    progress_bar = tqdm(train_loader, desc="Training")
    for batch_idx, (images, labels) in enumerate(progress_bar):
        images = _prepare_images(images, device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Scale loss by accumulation steps
        loss = loss / accumulation_steps

        # Backward pass
        loss.backward()

        # Handle gradient accumulation
        _handle_gradient_accumulation(optimizer, batch_idx, accumulation_steps)

        # Update statistics
        running_loss += loss.item() * accumulation_steps
        _, batch_correct, batch_total = _calculate_batch_metrics(outputs, labels)
        total_samples += batch_total
        correct_predictions += batch_correct

        # Update progress bar
        _update_progress_bar(
            progress_bar, running_loss, batch_idx, correct_predictions, total_samples
        )

    # Handle any remaining gradients
    _finalize_gradient_accumulation(optimizer, batch_idx, accumulation_steps)

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct_predictions / total_samples
    return epoch_loss, epoch_acc


def _validate_epoch(
    model: nn.Module,
    valid_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device | str,
) -> tuple[float, float]:
    """Validate the model for one epoch without updating parameters.

    Args:
        model (nn.Module): PyTorch neural network model to validate
        valid_loader (DataLoader): DataLoader for validation data
        criterion (nn.Module): Loss function (e.g., CrossEntropyLoss)
        device (torch.device | str): Device to run computations on ('cuda', 'cpu', or torch.device)

    Returns:
        tuple[float, float]: Tuple of (average_loss, accuracy_percentage) for the validation epoch
    """
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        progress_bar = tqdm(valid_loader, desc="Validation")
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images = _prepare_images(images, device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Update statistics
            running_loss += loss.item()
            _, batch_correct, batch_total = _calculate_batch_metrics(outputs, labels)
            total_samples += batch_total
            correct_predictions += batch_correct

            # Update progress bar
            _update_progress_bar(
                progress_bar,
                running_loss,
                batch_idx,
                correct_predictions,
                total_samples,
            )

    epoch_loss = running_loss / len(valid_loader)
    epoch_acc = 100 * correct_predictions / total_samples
    return epoch_loss, epoch_acc


def _calculate_auroc(
    model: nn.Module,
    valid_loader: DataLoader,
    device: torch.device | str,
) -> float:
    """Calculate AUROC for binary classification validation set.

    Args:
        model (nn.Module): PyTorch neural network model for evaluation
        valid_loader (DataLoader): DataLoader for validation data
        device (torch.device): Device to run computations on ('cuda', 'cpu', or torch.device)

    Returns:
        float: AUROC score as decimal (0.0-1.0)

    Note:
        Uses positive class (class 1) probabilities for binary AUROC calculation.
        Assumes binary classification with class labels 0 and 1.
    """
    model.eval()
    all_probabilities = []
    all_labels = []

    with torch.no_grad():
        for images, labels in valid_loader:
            images = _prepare_images(images, device)
            labels = labels.to(device)

            outputs = model(images)
            # Get probabilities for positive class (class 1) - binary classification
            probabilities = torch.softmax(outputs, dim=1)[:, 1]

            all_probabilities.extend(probabilities.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate binary AUROC (no multiplication by 100)
    auroc = float(roc_auc_score(all_labels, all_probabilities))

    return auroc


def _calculate_recall(
    model: nn.Module,
    valid_loader: DataLoader,
    device: torch.device | str,
) -> float:
    """Calculate recall for binary classification validation set.

    Calculates recall for the positive class (class 1).
    Assumes binary classification with class labels 0 and 1.

    Args:
        model (nn.Module): PyTorch neural network model for evaluation
        valid_loader (DataLoader): DataLoader for validation data
        device (torch.device): Device to run computations on ('cuda', 'cpu', or torch.device)

    Returns:
        float: Recall score as decimal (0.0-1.0)
    """
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in valid_loader:
            images = _prepare_images(images, device)
            labels = labels.to(device)

            outputs = model(images)
            # Get predicted class
            _, predicted = torch.max(outputs.data, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate recall for positive class (class 1)
    recall = float(recall_score(all_labels, all_predictions, pos_label=1))

    return recall


def _create_checkpoint_data(
    epoch: int,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Any,
    train_losses: list[float],
    train_accuracies: list[float],
    val_losses: list[float],
    val_accuracies: list[float],
    val_aurocs: list[float],
    val_recalls: list[float],
    best_val_acc: float,
    best_val_auroc: float,
    best_val_recall: float,
    best_epoch: int,
    current_train_acc: float,
    current_val_acc: float,
    current_val_auroc: float,
    current_val_recall: float,
) -> dict[str, Any]:
    """Create checkpoint data dictionary.

    Args:
        epoch: Current epoch number
        model: PyTorch model
        optimizer: Model optimizer
        scheduler: Learning rate scheduler
        train_losses: List of training losses
        train_accuracies: List of training accuracies
        val_losses: List of validation losses
        val_accuracies: List of validation accuracies
        val_aurocs: List of validation AUROC scores
        val_recalls: List of validation recall scores
        best_val_acc: Best validation accuracy so far
        best_val_auroc: Best validation AUROC so far
        best_val_recall: Best validation recall so far
        best_epoch: Epoch with best performance
        current_train_acc: Current epoch training accuracy
        current_val_acc: Current epoch validation accuracy
        current_val_auroc: Current epoch validation AUROC
        current_val_recall: Current epoch validation recall

    Returns:
        dict: Checkpoint data dictionary
    """
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "train_losses": train_losses,
        "train_accuracies": train_accuracies,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies,
        "val_aurocs": val_aurocs,
        "val_recalls": val_recalls,
        "best_val_acc": best_val_acc,
        "best_val_auroc": best_val_auroc,
        "best_val_recall": best_val_recall,
        "best_epoch": best_epoch,
        "current_val_acc": current_val_acc,
        "current_train_acc": current_train_acc,
        "current_val_auroc": current_val_auroc,
        "current_val_recall": current_val_recall,
    }


def _save_epoch_checkpoint(
    checkpoint_dir: str,
    model_name: str,
    epoch: int,
    checkpoint_data: dict[str, Any],
) -> None:
    """Save checkpoint for a specific epoch.

    Args:
        checkpoint_dir: Directory to save checkpoint
        model_name: Name of the model for filename
        epoch: Current epoch number
        checkpoint_data: Checkpoint data to save
    """
    checkpoint_path = os.path.join(
        checkpoint_dir, f"{model_name}_epoch_{epoch:02d}.pth"
    )
    torch.save(checkpoint_data, checkpoint_path)
    logger.info("Checkpoint saved: %s", checkpoint_path)


def _save_best_model(
    checkpoint_dir: str,
    model_name: str,
    best_model_state: dict[str, Any] | None,
    train_losses: list[float],
    train_accuracies: list[float],
    val_losses: list[float],
    val_accuracies: list[float],
    val_aurocs: list[float],
    val_recalls: list[float],
    best_val_acc: float,
    best_val_auroc: float,
    best_val_recall: float,
    best_epoch: int,
    epochs: int,
) -> None:
    """Save the best model checkpoint.

    Args:
        checkpoint_dir: Directory to save checkpoint
        model_name: Name of the model for filename
        best_model_state: State dictionary of the best model
        train_losses: List of training losses
        train_accuracies: List of training accuracies
        val_losses: List of validation losses
        val_accuracies: List of validation accuracies
        val_aurocs: List of validation AUROC scores
        val_recalls: List of validation recall scores
        best_val_acc: Best validation accuracy
        best_val_auroc: Best validation AUROC
        best_val_recall: Best validation recall
        best_epoch: Epoch with best performance
        epochs: Total number of epochs trained
    """
    if best_model_state is None:
        logger.warning("No best model state to save")
        return

    best_model_path = os.path.join(checkpoint_dir, f"{model_name}_best_model.pth")
    torch.save(
        {
            "model_state_dict": best_model_state,
            "train_losses": train_losses,
            "train_accuracies": train_accuracies,
            "val_losses": val_losses,
            "val_accuracies": val_accuracies,
            "val_aurocs": val_aurocs,
            "val_recalls": val_recalls,
            "best_val_acc": best_val_acc,
            "best_val_auroc": best_val_auroc,
            "best_val_recall": best_val_recall,
            "best_epoch": best_epoch,
            "epochs_trained": epochs,
        },
        best_model_path,
    )
    logger.info("Best model saved: %s", best_model_path)


def _log_epoch_results(
    epoch: int,
    epochs: int,
    train_loss: float,
    train_acc: float,
    val_loss: float,
    val_acc: float,
    val_auroc: float,
    val_recall: float,
    current_lr: float,
    epoch_time: float,
) -> None:
    """Log the results of a training epoch.

    Args:
        epoch: Current epoch number (1-indexed)
        epochs: Total number of epochs
        train_loss: Training loss for the epoch
        train_acc: Training accuracy for the epoch
        val_loss: Validation loss for the epoch
        val_acc: Validation accuracy for the epoch
        val_auroc: Validation AUROC for the epoch
        val_recall: Validation recall for the epoch
        current_lr: Current learning rate
        epoch_time: Time taken for the epoch in seconds
    """
    logger.info("Epoch %d/%d", epoch, epochs)
    logger.info("-" * 40)
    logger.info("Train Loss: %.4f, Train Acc: %.2f%%", train_loss, train_acc)
    logger.info(
        "Val Loss: %.4f, Val Acc: %.2f%%, Val AUROC: %.4f, Val Recall: %.4f",
        val_loss,
        val_acc,
        val_auroc,
        val_recall,
    )
    logger.info("Learning Rate: %.6f", current_lr)
    logger.info("Epoch Time: %.2fs", epoch_time)


def _check_learning_rate_reduction(optimizer: Optimizer, previous_lr: float) -> float:
    """Check if learning rate was reduced and log if so.

    Args:
        optimizer: Model optimizer
        previous_lr: Previous learning rate value

    Returns:
        float: Current learning rate
    """
    current_lr = optimizer.param_groups[0]["lr"]
    if current_lr < previous_lr:
        logger.info("Learning rate reduced from %.6f to %.6f", previous_lr, current_lr)
    return current_lr


def _update_best_model_recall(
    val_recall: float,
    val_acc: float,
    val_auroc: float,
    epoch: int,
    model: nn.Module,
    best_val_recall: float,
    best_val_acc: float,
    best_val_auroc: float,
    best_epoch: int,
    best_model_state: dict[str, Any] | None,
) -> tuple[float, float, float, dict[str, Any] | None, int]:
    """Update best model information if current recall performance is better.

    Args:
        val_recall: Current validation recall
        val_acc: Current validation accuracy
        val_auroc: Current validation AUROC
        epoch: Current epoch number (1-indexed)
        model: Current model
        best_val_recall: Best recall so far
        best_val_acc: Best accuracy so far
        best_val_auroc: Best AUROC so far
        best_epoch: Best epoch so far
        best_model_state: Best model state so far

    Returns:
        tuple: (new_best_recall, new_best_acc, new_best_auroc, best_model_state, best_epoch)
    """
    if val_recall > best_val_recall:
        logger.info(
            "New best Recall: %.4f (Acc: %.2f%%, AUROC: %.4f)",
            val_recall,
            val_acc,
            val_auroc,
        )
        return val_recall, val_acc, val_auroc, model.state_dict().copy(), epoch
    return best_val_recall, best_val_acc, best_val_auroc, best_model_state, best_epoch


def _finalize_training(
    checkpoint_dir: str,
    model_name: str,
    best_model_state: dict[str, Any] | None,
    train_losses: list[float],
    train_accuracies: list[float],
    val_losses: list[float],
    val_accuracies: list[float],
    val_aurocs: list[float],
    val_recalls: list[float],
    best_val_acc: float,
    best_val_auroc: float,
    best_val_recall: float,
    best_epoch: int,
    epochs: int,
    model: nn.Module,
) -> None:
    """Finalize training by logging results, saving best model, and loading best weights.

    Args:
        checkpoint_dir: Directory to save checkpoints
        model_name: Name of the model
        best_model_state: State dictionary of the best model
        train_losses: List of training losses
        train_accuracies: List of training accuracies
        val_losses: List of validation losses
        val_accuracies: List of validation accuracies
        val_aurocs: List of validation AUROC scores
        val_recalls: List of validation recall scores
        best_val_acc: Best validation accuracy
        best_val_auroc: Best validation AUROC
        best_val_recall: Best validation recall
        best_epoch: Epoch with best performance
        epochs: Total number of epochs
        model: Model to load best weights into
    """
    logger.info("Training completed!")
    logger.info("Best Recall: %.4f at epoch %d", best_val_recall, best_epoch)
    logger.info(
        "Best model metrics - Acc: %.2f%%, AUROC: %.4f, Recall: %.4f",
        best_val_acc,
        best_val_auroc,
        best_val_recall,
    )

    # Save best model
    _save_best_model(
        checkpoint_dir,
        model_name,
        best_model_state,
        train_losses,
        train_accuracies,
        val_losses,
        val_accuracies,
        val_aurocs,
        val_recalls,
        best_val_acc,
        best_val_auroc,
        best_val_recall,
        best_epoch,
        epochs,
    )

    # Load best model weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info("Loaded best model weights")
