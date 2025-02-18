import warnings
from pathlib import Path
from typing import Any

import torch
import torchvision.transforms as transforms
from model_architecture import DiabeticRetinopathyNet
from constants import DATA_PATH, ITERATION
from image_dataset import mean, std
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader
from utils import get_device, setup_logger

logger = setup_logger(
    "validation", Path(f"{DATA_PATH}/{ITERATION.replace(' ', '_')}/logs/training.log")
)


def model_validation_data(
    dataloader: DataLoader,
    model: DiabeticRetinopathyNet,
    validation_loss: float,
    validation_accuracy: float,
) -> tuple[float, float]:
    for batch_data in dataloader:
        image_to_model = batch_data["img"]
        target_scores = batch_data["levels"].to(get_device())
        model_output = model(image_to_model.to(get_device()))
        val_train = model.get_loss(model_output, target_scores)
        validation_loss += val_train.item()
        validation_accuracy += calculate_metrics(model_output, target_scores)
    average_loss = round(validation_loss / len(dataloader), 3)
    average_accuracy = round(100 * (validation_accuracy / len(dataloader)), 3)
    return average_loss, average_accuracy


def validate(
    progress_tracker: dict[str, list],
    model: DiabeticRetinopathyNet,
    dataloader: DataLoader,
    epoch: int,
) -> tuple[dict[str, list], float, float]:
    model.eval()
    with torch.no_grad():
        validation_loss: float = 0.0
        validation_accuracy: float = 0.0
        validation_loss, validation_accuracy = model_validation_data(
            dataloader, model, validation_loss, validation_accuracy
        )
    progress_tracker["Validation Loss"].append(validation_loss)
    progress_tracker["Validation Accuracy"].append(validation_accuracy)
    logger.info(
        f"epoch {epoch}: Validation  loss: {validation_loss}, accuracy: {validation_accuracy}%"
    )
    return progress_tracker, validation_accuracy, validation_loss


def calculate_metrics(output: Any, target: Any) -> float:
    _, predicted_score = output["level"].cpu().max(1)
    target_score = target.cpu()
    with (
        warnings.catch_warnings()
    ):
        warnings.simplefilter("ignore")
        accuracy = balanced_accuracy_score(
            y_true=target_score.numpy(), y_pred=predicted_score.numpy()
        )
    return accuracy


def validation_transforms() -> transforms.Compose:
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
