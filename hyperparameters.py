from pathlib import Path

from constants import ITERATION, DATA_PATH
from utils import setup_logger

BATCH_SIZE = 64
NUM_WORKERS = 4
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.0001
DROPOUT = 0.5
CLASSIFIER_STRUCT = """
            Resnet pretrained = True
            self.conv_head = nn.Sequential(
                nn.Conv2d(in_channels=2048, out_channels=32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout(0.2),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.AdaptiveAvgPool2d((56, 56)),
            )

            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features=128 *56 *56, out_features=128),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=128),
                nn.Dropout(0.5),
                nn.Linear(in_features=128, out_features=n_diabetic_retinopathy_levels),
            )
            """

EXTRA_INFO = """
inter level dropout increased
"""


def log_hyperparameters() -> None:
    hyperparameter_logger = setup_logger(
        "hyperparameters",
        Path(f"{DATA_PATH}/{ITERATION.replace(' ', '_')}/logs/hyperparameters.log"),
    )
    hyperparameter_logger.info("Hyperparameter details")
    hyperparameter_logger.info(f"Batch size: {BATCH_SIZE}")
    hyperparameter_logger.info(f"Number of workers: {NUM_WORKERS}")
    hyperparameter_logger.info(f"Learning rate: {LEARNING_RATE}")
    hyperparameter_logger.info(f"Dropout: {DROPOUT}")
    hyperparameter_logger.info("Model used in Transfer Learning: mobilenet v2")
    hyperparameter_logger.info(f"\nClassifier structure: {CLASSIFIER_STRUCT}\n\n")
    hyperparameter_logger.info(f"\nExtra information:\n{EXTRA_INFO}")
