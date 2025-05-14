import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.transforms import v2
from early_stopping import EarlyStopping
from model_architecture import DiabeticRetinopathyNet
from constants import DATASET_PATH, DATA_PATH, ITERATION, NUM_EPOCHS, START_EPOCH
from hyperparameters import (
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_WORKERS,
    log_hyperparameters,
)
from image_dataset import AttributesDataset, RetinalImageDataset
from validate import validate, create_confusion_matrix
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import get_device, make_path, setup_logger, calculate_accuracy

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

progress_trackerS = ["Loss", "Accuracy"]
GRAPH_PATH = f"{DATA_PATH}/{ITERATION.replace(' ', '_')}/graphs"

TRAINING_ANNOTATIONS_PATH = Path(f"{DATASET_PATH}/trainLabels.csv")
VALIDATION_ANNOTATIONS_PATH = Path(f"{DATASET_PATH}/testLabels.csv")

SAVE_PROGRESS_INTERVAL = 5

logger = setup_logger(
    "train", Path(f"{DATA_PATH}/{ITERATION.replace(' ', '_')}/logs/training.log")
)


def get_progress_tracker() -> dict[str, list]:
    """Initialises variable to track model performance metrics"""
    return {
        "Validation Loss": [],
        "Train Loss": [],
        "Validation Accuracy": [],
        "Train Accuracy": [],
    }


def setup_dataset(
    transforms_to_apply: transforms.Compose,
    path_to_annotations: Path,
) -> tuple[RetinalImageDataset, AttributesDataset]:
    """Initialises RetinalImageDataset using given annotations file"""
    dataset_attributes: AttributesDataset = AttributesDataset()
    dataset: RetinalImageDataset = RetinalImageDataset(
        path_to_annotations, dataset_attributes, transforms_to_apply
    )
    return dataset, dataset_attributes


def setup_dataloader(dataset: RetinalImageDataset) -> DataLoader:
    """Returns data loader from given dataset for batching"""
    return DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )


def setup_model(
    attributes: AttributesDataset,
) -> tuple[DiabeticRetinopathyNet, torch.device]:
    """Initialises model and device for training process"""
    device: torch.device = get_device()
    model: DiabeticRetinopathyNet = DiabeticRetinopathyNet(
        n_diabetic_retinopathy_levels=attributes.num_classes
    ).to(device)
    return model, device


def standardizing_transforms() -> v2.Compose:
    """Transforms to apply to images during training"""
    return v2.Compose(
        [
            v2.ToImage(),
            v2.Resize((224, 224), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean, std),
        ]
    )


def save_checkpoint(model: DiabeticRetinopathyNet, epoch: int) -> None:
    """Saves current model weights"""
    checkpoint_name: str = f"checkpoint-epoch-{epoch}.pth"
    checkpoint_dir: Path = Path(
        f"{DATA_PATH}/{ITERATION.replace(' ', '_')}/checkpoints"
    )
    make_path(checkpoint_dir)
    checkpoint_path: Path = checkpoint_dir / checkpoint_name
    try:
        torch.save(model.state_dict(), checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
    except Exception as e:
        print(f"Failed to save model: {e}")
    return


def plot_graph(
    progress_tracker: dict[str, list],
    train_label: str,
    val_label: str,
    x_axis: list[int],
    graph_dir: Path,
) -> None:
    """Plots and saves individual performance metric graph"""
    graph_title: str = f"{train_label} VS {val_label}"
    plt.plot(x_axis, progress_tracker[train_label], "-o", label=train_label)
    plt.plot(x_axis, progress_tracker[val_label], "-o", label=val_label)
    plt.legend([train_label, val_label], loc="upper left")
    plt.title(graph_title)
    plt.savefig(f"{graph_dir}/{graph_title}.jpeg")
    plt.clf()
    plt.close()
    return


def save_progress_graph(
    progress_tracker: dict[str, list], epoch: int, fold: int | None = None
) -> None:
    """Creates graphs to track model performance"""
    x_axis: list[int] = list(range(1, epoch + 1))
    for performance_measure in progress_trackerS:
        if fold is None:
            graph_dir = Path(f"{GRAPH_PATH}/{epoch}_epochs")
        else:
            graph_dir = Path(f"{GRAPH_PATH}/{fold}_fold/{epoch}_epochs")
        make_path(graph_dir)
        plot_graph(
            progress_tracker,
            f"Train {performance_measure}",
            f"Validation {performance_measure}",
            x_axis,
            graph_dir,
        )
    return


def run_batch(
    model: DiabeticRetinopathyNet,
    optimizer: SGD,
    current_loss: float,
    accuracy: float,
    batch_data: Any,
    device: torch.device,
) -> tuple[float, float]:
    """Get predictions, accuracy and loss for individual batch of training data"""
    inputs = batch_data["img"]
    targets = batch_data["levels"].to(device)
    optimizer.zero_grad()
    model_output: dict[str, torch.Tensor] = model(inputs.to(device))
    predicted_labels: np.ndarray = (
        model_output["level"].argmax(dim=1).detach().cpu().numpy()
    )
    true_labels = targets.data.cpu().numpy()
    accuracy += calculate_accuracy(predicted_labels, batch_data["levels"])
    training_loss: torch.Tensor = model.get_loss(model_output["level"], targets)
    training_loss.backward()
    optimizer.step()
    current_loss += training_loss.item()
    return current_loss, accuracy, predicted_labels, true_labels


def run_epoch(
    progress_tracker: dict[str, list],
    train_dataloader: DataLoader,
    model: DiabeticRetinopathyNet,
    optimizer: SGD,
    epoch: int,
    device,
) -> tuple[dict[str, list], DiabeticRetinopathyNet, SGD]:
    """Runs one training iteration, using accuracy and loss of iteration to update model weights"""
    current_loss: float = 0.0
    accuracy: float = 0.0
    n_train_samples: int = len(train_dataloader)
    y_predictions = []
    y_true_labels = []
    for batch_data in train_dataloader:
        current_loss, accuracy, y_pred, y_true = run_batch(
            model, optimizer, current_loss, accuracy, batch_data, device
        )
        y_predictions.extend(y_pred)
        y_true_labels.extend(y_true)
    create_confusion_matrix(y_true_labels, y_predictions, epoch, False)
    total_training_loss: float = round((current_loss / n_train_samples), 3)
    total_training_accuracy: float = round(100 * (accuracy / n_train_samples), 3)
    progress_tracker["Train Loss"].append(total_training_loss)
    progress_tracker["Train Accuracy"].append(total_training_accuracy)
    logger.info(
        f"Epoch {epoch}, loss: {total_training_loss}, accuracy: {total_training_accuracy}%"
    )
    return progress_tracker, model, optimizer


def train_model() -> None:
    """Initialises variables for training and runs training epochs, tracking progress"""
    train_dataset, attributes = setup_dataset(
        standardizing_transforms(), TRAINING_ANNOTATIONS_PATH
    )
    val_dataset, _ = setup_dataset(
        standardizing_transforms(), VALIDATION_ANNOTATIONS_PATH
    )
    train_loader: DataLoader = setup_dataloader(train_dataset)
    val_loader: DataLoader = setup_dataloader(val_dataset)
    model_to_train, device = setup_model(attributes)
    optimizer: SGD = SGD(
        model_to_train.parameters(),
        lr=LEARNING_RATE,
        momentum=0.9,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler: ReduceLROnPlateau = ReduceLROnPlateau(optimizer, "min")
    progress_tracker: dict[str, list] = get_progress_tracker()
    early_stopping = EarlyStopping(patience=15)
    for epoch in tqdm(range(START_EPOCH, NUM_EPOCHS + 1)):
        model_to_train.train(True)
        progress_tracker, model_to_train, optimizer = run_epoch(
            progress_tracker, train_loader, model_to_train, optimizer, epoch, device
        )
        progress_tracker, _, val_loss = validate(
            progress_tracker, model_to_train, val_loader, epoch, device
        )

        scheduler.step(val_loss)
        if epoch % SAVE_PROGRESS_INTERVAL == 0:
            save_progress_graph(progress_tracker, epoch)
            save_checkpoint(model_to_train, epoch)
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break


def training_process() -> None:
    """Runs full model training process"""
    print(f"Beginning training process for {ITERATION}")
    logger.info(f"\n{ITERATION}")
    torch.manual_seed(42)
    start: float = time.time()
    log_hyperparameters()
    train_model()
    logger.info(
        f"Took {round(time.time() - start, 3)} seconds to train {NUM_EPOCHS} epochs, which is {round(((time.time() - start) / 60), 3)} minutes"
    )


if __name__ == "__main__":
    training_process()
