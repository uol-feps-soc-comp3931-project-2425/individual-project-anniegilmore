from pathlib import Path

from constants import ITERATION, DATA_PATH
from utils import setup_logger

# defining hyperparameters for this iteration of training
BATCH_SIZE = 64
NUM_WORKERS = 4
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.001
DROPOUT = 0.5
CLASSIFIER_STRUCT = """
        resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        for name, param in resnet50.named_parameters():
            if "layer_4" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
                
        last_channel = resnet50.fc.in_features
        
        self.base_model = nn.Sequential(
            resnet50.conv1,
            resnet50.bn1,
            resnet50.relu,
            resnet50.maxpool,
            resnet50.layer1,
            resnet50.layer2,
            resnet50.layer3,
            resnet50.layer4,
        )
        
        self.custom_layer = nn.Sequential(
            nn.Conv2d(in_channels=last_channel, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.5),
        )
        

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=32 * 3 * 3),
            nn.Linear(in_features=32 * 3 * 3, out_features=n_diabetic_retinopathy_levels),
        )
            """

EXTRA_INFO = """
Potential final iteration
"""


def log_hyperparameters() -> None:
    """Logging hyperparameters for version control"""
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
