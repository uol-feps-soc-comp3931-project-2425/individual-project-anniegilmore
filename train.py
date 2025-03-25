import time
from pathlib import Path
from typing import Any
import torch.nn.functional as F

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from focal_loss import FocalLoss
import torchvision.models as models
import torch.nn as nn
from model_architecture import DiabeticRetinopathyNet
from constants import DATASET_PATH, DATA_PATH, ITERATION, NUM_EPOCHS, START_EPOCH
from hyperparameters import (
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_WORKERS,
    log_hyperparameters,
)
from image_dataset import AttributesDataset, RetinalImageDataset, mean, std
from validate import calculate_metrics, validate, get_confusion_matrix
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import get_device, make_path, setup_logger

progress_trackerS = ["Loss", "Accuracy"]
GRAPH_PATH = f"{DATA_PATH}/{ITERATION.replace(' ', '_')}/graphs"

TRAINING_ANNOTATIONS_PATH = Path(f"{DATASET_PATH}/trainLabels.csv")
VALIDATION_ANNOTATIONS_PATH = Path(f"{DATASET_PATH}/validateLabels.csv")

SAVE_PROGRESS_INTERVAL = 2

logger = setup_logger(
    "train", Path(f"{DATA_PATH}/{ITERATION.replace(' ', '_')}/logs/training.log")
)


def get_progress_tracker() -> dict[str, list]:
    return {
        "Validation Loss": [],
        "Train Loss": [],
        "Validation Accuracy": [],
        "Train Accuracy": [],
    }


def save_checkpoint(
    model: DiabeticRetinopathyNet,
    epoch: int
) -> None:
    checkpoint_name: str = f"checkpoint-epoch-{epoch}.pth"
    checkpoint_dir: Path = Path(f"{DATA_PATH}/{ITERATION.replace(' ', '_')}/checkpoints")
    make_path(checkpoint_dir)
    checkpoint_path: Path = Path(f"{checkpoint_dir}/{checkpoint_name}")
    torch.save(model, checkpoint_path)
    logger.info(f"Saved checkpoint: {checkpoint_path}")


def image_transforms() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # transforms.Normalize(mean, std),
        ]
    )


def build_dataloader(
    path: Path, attributes: AttributesDataset, transform: transforms.Compose
) -> DataLoader:
    dataset: RetinalImageDataset = RetinalImageDataset(path, attributes, transform)
    return DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )


def run_epoch(
    progress_tracker: dict[str, list],
    train_dataloader: DataLoader,
    model: DiabeticRetinopathyNet,
    optimizer: Adam,
    epoch: int,
) -> tuple[dict[str, list], DiabeticRetinopathyNet, Adam]:
    current_loss: float = 0.0
    accuracy: float = 0.0
    class_weights: torch.FloatTensor = torch.FloatTensor(train_dataloader.dataset.class_weights)
    # print(np.bincount(train_dataloader))
    n_train_samples: int = len(train_dataloader)
    print(f"There are {n_train_samples} samples\n")
    y_predictions = []
    y_true_labels = []
    for batch_data in train_dataloader:
        current_loss, accuracy, y_pred, y_true = run_batch(
            model, optimizer, current_loss, accuracy, batch_data, class_weights
        )
        y_predictions.extend(y_pred)
        y_true_labels.extend(y_true)
    get_confusion_matrix(y_true_labels, y_predictions, epoch)
    total_training_loss: float = round((current_loss / n_train_samples), 3)
    total_training_accuracy: float = round(100 * (accuracy / n_train_samples), 3)
    progress_tracker["Train Loss"].append(total_training_loss)
    progress_tracker["Train Accuracy"].append(total_training_accuracy)
    logger.info(
        f"Epoch {epoch}, loss: {total_training_loss}, accuracy: {total_training_accuracy}%"
    )
    return progress_tracker, model, optimizer


def run_batch(
    model: DiabeticRetinopathyNet,
    optimizer: Adam,
    current_loss: float,
    accuracy: float,
    batch_data: Any,
    class_weights: torch.FloatTensor,
) -> tuple[float, float]:
    inputs = batch_data["img"]
    targets = batch_data["levels"].to(get_device())
    optimizer.zero_grad()
    output: dict[str, torch.Tensor] = model(inputs.to(get_device()))
    predicted_labels = output["level"].argmax(dim=1).numpy()
    true_labels = targets.data.cpu().numpy()
    accuracy += calculate_metrics(output["level"], targets)
    training_loss: torch.Tensor = F.cross_entropy(output["level"], targets)
    # training_loss: torch.Tensor = model.get_loss(output, targets)
    # criterion = FocalLoss(alpha=class_weights, gamma=3)
    # training_loss = criterion(output, targets)
    training_loss.backward()
    optimizer.step()
    current_loss += training_loss.item()
    return current_loss, accuracy, predicted_labels, true_labels


def setup_dataset(
    transforms_to_apply: transforms.Compose,
    path_to_annotations: Path,
) -> tuple[RetinalImageDataset, AttributesDataset]:
    dataset_attributes: AttributesDataset = AttributesDataset()
    dataset: RetinalImageDataset = RetinalImageDataset(
        path_to_annotations, dataset_attributes, transforms_to_apply
    )
    return dataset, dataset_attributes


def setup_dataloader(dataset: RetinalImageDataset) -> DataLoader:
    return DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )


def plot_graph(
    progress_tracker: dict[str, list],
    train_label: str,
    val_label: str,
    x_axis: list[int],
    graph_dir: Path,
) -> None:
    graph_title: str = f"{train_label} VS {val_label}"
    plt.plot(x_axis, progress_tracker[train_label], "-o", label=train_label)
    plt.plot(x_axis, progress_tracker[val_label], "-o", label=val_label)
    plt.legend([train_label, val_label], loc="upper left")
    plt.title(graph_title)
    plt.savefig(f"{graph_dir}/{graph_title}.jpeg")
    plt.clf()


def setup_model(attributes: AttributesDataset) -> DiabeticRetinopathyNet:
    model: DiabeticRetinopathyNet = DiabeticRetinopathyNet(
        n_diabetic_retinopathy_levels=attributes.num_classes
    ).to(get_device())
    # resnet50 = models.resnet50()
    # for param in model.parameters():
    #     param.requires_grad = False
    return model


def save_progress_graph(
    progress_tracker: dict[str, list], epoch: int, fold: int | None = None
) -> None:
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
    

def train_model() -> None:
    train_dataset, attributes = setup_dataset(
        image_transforms(), TRAINING_ANNOTATIONS_PATH
    )
    print(train_dataset.class_weights)
    val_dataset, _ = setup_dataset(image_transforms(), VALIDATION_ANNOTATIONS_PATH)
    train_loader: DataLoader = setup_dataloader(train_dataset)
    print(train_loader.dataset.class_weights)
    val_loader: DataLoader = setup_dataloader(val_dataset)
    model_to_train: DiabeticRetinopathyNet = setup_model(attributes)
    optimizer: Adam = Adam(model_to_train.parameters(), lr=LEARNING_RATE)
    scheduler: StepLR = StepLR(optimizer, step_size=10, gamma=0.1)
    progress_tracker: dict[str, list] = get_progress_tracker()
    for epoch in tqdm(range(START_EPOCH, NUM_EPOCHS + 1)):
        model_to_train.train(True)
        progress_tracker, model_to_train, optimizer = run_epoch(
            progress_tracker, train_loader, model_to_train, optimizer, epoch
        )
        progress_tracker, _, val_loss = validate(
            progress_tracker, model_to_train, val_loader, epoch
        )
        scheduler.step()
        if epoch % SAVE_PROGRESS_INTERVAL == 0:
            save_progress_graph(progress_tracker, epoch)
            save_checkpoint(model_to_train, epoch)


def main() -> None:
    logger.info(f"\n{ITERATION}")
    torch.manual_seed(42)
    start: float = time.time()
    log_hyperparameters()
    train_model()
    logger.info(
        f"Took {round(time.time() - start, 3)} seconds to train {NUM_EPOCHS} epochs, which is {round(((time.time() - start) / 60), 3)} minutes"
    )


if __name__ == "__main__":
    main()
