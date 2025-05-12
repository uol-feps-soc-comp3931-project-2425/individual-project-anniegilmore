import csv
import logging
from pathlib import Path
import torch
from sklearn.metrics import balanced_accuracy_score
import warnings
from typing import Any


def make_path(path_to_create: Path) -> None:
    """Creates new directory"""
    Path.mkdir(path_to_create, parents=True, exist_ok=True)
    return


def setup_logger(
    name: str, log_file: Path, level: int = logging.INFO
) -> logging.Logger:
    """Initialises logger functionality for version control and tracking model progress"""
    make_path(log_file.parent)
    handler: logging.FileHandler = logging.FileHandler(str(log_file))
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger


def get_device() -> torch.device:
    """Returns the hardware available to train the model on"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_level_distribution_map(path_to_dataset: Path) -> dict[str, list[str]]:
    """Returns information about dataset distribution"""
    level_image_map: dict[str, list[str]] = {
        "0": [],
        "1": [],
        "2": [],
        "3": [],
    }
    with open(path_to_dataset, "r", newline="") as csvfile:
        dataset_reader = csv.reader(csvfile, delimiter=",")
        for rows in dataset_reader:
            for level in level_image_map.keys():
                if rows[1] == level:
                    level_image_map[level].append(rows[0])
    return level_image_map


def calculate_accuracy(output: Any, target: Any) -> float:
    """Returns accuracy of model predictions compared to ground truth values"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        accuracy: float = balanced_accuracy_score(y_true=target, y_pred=output)
    return accuracy
