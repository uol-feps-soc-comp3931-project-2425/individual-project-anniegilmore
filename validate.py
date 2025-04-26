import warnings
from pathlib import Path
from typing import Any
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd

import torch
from focal_loss import FocalLoss
from model_architecture import DiabeticRetinopathyNet
from constants import DATA_PATH, ITERATION
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader
from utils import make_path, setup_logger

logger = setup_logger(
    "validation", Path(f"{DATA_PATH}/{ITERATION.replace(' ', '_')}/logs/training.log")
)


def get_confusion_matrix(y_true, y_pred, epoch, val) -> None:
    classes = ("0", "1", "2", "3")

    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(
        cf_matrix / np.sum(cf_matrix, axis=1)[:, None],
        index=[i for i in classes],
        columns=[i for i in classes],
    )
    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True)
    if val:
        path_to_matrix = Path(
            f"{DATA_PATH}/{ITERATION.replace(' ', '_')}/confusion_matrix/validation"
        )
    else:
        path_to_matrix = Path(
            f"{DATA_PATH}/{ITERATION.replace(' ', '_')}/confusion_matrix/train"
        )
    make_path(path_to_matrix)
    plt.savefig(f"{path_to_matrix}/{epoch}.png")
    plt.clf()
    plt.close()


def model_validation_data(
    dataloader: DataLoader,
    model: DiabeticRetinopathyNet,
    validation_loss: float,
    validation_accuracy: float,
    epoch: int,
    device,
) -> tuple[float, float]:
    y_pred = []
    y_true = []
    class_weights: torch.FloatTensor = torch.FloatTensor(
        dataloader.dataset.class_weights
    )
    for batch_data in dataloader:
        image_to_model = batch_data["img"]
        target_scores = batch_data["levels"].to(device)
        model_output = model(image_to_model.to(device))
        y_pred_labels = model_output["level"].argmax(dim=1).detach().cpu().numpy()

        y_pred.extend(y_pred_labels)
        labels = target_scores.data.cpu().numpy()
        y_true.extend(labels)
        criterion = FocalLoss(alpha=class_weights, gamma=3)
        val_loss = criterion(model_output["level"], target_scores)
        validation_loss += val_loss.item()
        validation_accuracy += calculate_metrics(y_pred_labels, batch_data["levels"])
    average_loss = round(validation_loss / len(dataloader), 3)
    average_accuracy = round(100 * (validation_accuracy / len(dataloader)), 3)
    get_confusion_matrix(y_true, y_pred, epoch, True)
    return average_loss, average_accuracy


def validate(
    progress_tracker: dict[str, list],
    model: DiabeticRetinopathyNet,
    dataloader: DataLoader,
    epoch: int,
    device,
) -> tuple[dict[str, list], float, float]:
    model.eval()
    with torch.no_grad():
        validation_loss: float = 0.0
        validation_accuracy: float = 0.0
        validation_loss, validation_accuracy = model_validation_data(
            dataloader, model, validation_loss, validation_accuracy, epoch, device
        )
    progress_tracker["Validation Loss"].append(validation_loss)
    progress_tracker["Validation Accuracy"].append(validation_accuracy)
    logger.info(
        f"epoch {epoch}: Validation  loss: {validation_loss}, accuracy: {validation_accuracy}%"
    )
    return progress_tracker, validation_accuracy, validation_loss


def calculate_metrics(output: Any, target: Any) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        accuracy = balanced_accuracy_score(y_true=target, y_pred=output)
    return accuracy
