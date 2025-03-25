from pathlib import Path

from constants import ITERATION, DATA_PATH
from utils import setup_logger

BATCH_SIZE = 32
NUM_WORKERS = 4
LEARNING_RATE = 0.01
DROPOUT = 0.2
CLASSIFIER_STRUCT = """
            nn.Linear(in_features=last_channel, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=4096),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=4096, out_features=n_diabetic_retinopathy_levels),
            nn.Softmax(dim=1),"""
            
EXTRA_INFO = """
Changes here:
Dataset still has 3 classes
7000 images in classes 0 and 1
class 3 has around 1300 images in
aiming to implement class weights through use of focal loss criterion when calculating loss
using resnet50 for transfer learning instead of mobilenet"""


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
