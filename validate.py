from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd

import torch
from model_architecture import DiabeticRetinopathyNet
from constants import DATA_PATH, ITERATION
from torch.utils.data import DataLoader
from utils import make_path, setup_logger, calculate_accuracy


logger = setup_logger(
    "validation", Path(f"{DATA_PATH}/{ITERATION.replace(' ', '_')}/logs/training.log")
)

def create_confusion_matrix(y_true: list, y_pred: list, epoch: bool, validation_cm: bool) -> None:
    """ Creates and saves confusion matrix to visualise model predictions vs ground truth """
    classes = ("0", "1", "2", "3")
    cf_matrix: np.ndarray = confusion_matrix(y_true, y_pred)
    cf_matrix_dataframe: pd.DataFrame = pd.DataFrame(
        cf_matrix / np.sum(cf_matrix, axis=1)[:, None],
        index=[class_label for class_label in classes],
        columns=[class_label for class_label in classes],
    )
    plt.figure(figsize=(12, 7))
    sn.heatmap(cf_matrix_dataframe, annot=True)
    if validation_cm:
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
    return

def model_validation_data(
    dataloader: DataLoader,
    model: DiabeticRetinopathyNet,
    validation_loss: float,
    validation_accuracy: float,
    epoch: int,
    device,
) -> tuple[float, float]:
    """ Splits validation subset into batches and predicts DR levels of images """
    y_pred: list = []
    y_true: list = []
    for batch_data in dataloader:
        image_to_model = batch_data["img"]
        target_scores = batch_data["levels"].to(device)
        model_output = model(image_to_model.to(device))
        y_pred_labels = model_output["level"].argmax(dim=1).detach().cpu().numpy()

        y_pred.extend(y_pred_labels)
        labels = target_scores.data.cpu().numpy()
        y_true.extend(labels)
        val_loss: torch.Tensor = model.get_loss(model_output, labels)
        validation_loss += val_loss.item()
        validation_accuracy += calculate_accuracy(y_pred_labels, batch_data["levels"])
    average_loss: float = round(validation_loss / len(dataloader), 3)
    average_accuracy: float = round(100 * (validation_accuracy / len(dataloader)), 3)
    create_confusion_matrix(y_true, y_pred, epoch, True)
    return average_loss, average_accuracy


def validate(
    progress_tracker: dict[str, list],
    model: DiabeticRetinopathyNet,
    dataloader: DataLoader,
    epoch: int,
    device,
) -> tuple[dict[str, list], float, float]:
    """ Runs validation subset through model to assess model performance """
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

