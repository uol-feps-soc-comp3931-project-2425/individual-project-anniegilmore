from pathlib import Path

from constants import ITERATION, DATA_PATH
from utils import setup_logger

BATCH_SIZE = 64
NUM_WORKERS = 4
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
DROPOUT = 0.5
CLASSIFIER_STRUCT = """
            last_channel = resnet50.fc.in_features
            resnet50.fc = nn.Linear(num_features, n_diabetic_retinopathy_levels)
            self.transfer_model = resnet50
            """
            
EXTRA_INFO = """
attempt to go back to it, 51 -> only using resnet layer 4 and fc
"""


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
    hyperparameter_logger.info(f"\nExtra information:\n{EXTRA_INFO}")
