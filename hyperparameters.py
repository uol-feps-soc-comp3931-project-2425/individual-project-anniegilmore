from pathlib import Path

from constants import ITERATION, DATA_PATH
from utils import setup_logger

BATCH_SIZE = 32
NUM_WORKERS = 4
LEARNING_RATE = 0.1
DROPOUT = 0.2
CLASSIFIER_STRUCT = """
nn.BatchNorm1d(last_channel),
nn.Dropout(p=DROPOUT),
nn.Linear(in_features=last_channel, out_features=n_quality_scores)"""


def log_hyperparameters() -> None:
    hyperparameter_logger = setup_logger(
        "hyperparameters",
        Path(
            f"{DATA_PATH}/{ITERATION.replace(' ', '_')}/logs/hyperparameters.log"
        ),
    )
    hyperparameter_logger.info("Hyperparameter details")
    hyperparameter_logger.info(f"Batch size: {BATCH_SIZE}")
    hyperparameter_logger.info(f"Number of workers: {NUM_WORKERS}")
    hyperparameter_logger.info(f"Learning rate: {LEARNING_RATE}")
    hyperparameter_logger.info(f"Dropout: {DROPOUT}")
    hyperparameter_logger.info("Model used in Transfer Learning: mobilenet v2")
    hyperparameter_logger.info(f"\nClassifier structure: {CLASSIFIER_STRUCT}\n\n")
