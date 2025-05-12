from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class DiabeticRetinopathyNet(nn.Module):
    """
    Defines CNN Model Structure for training
    """

    def __init__(self, n_diabetic_retinopathy_levels: int) -> None:
        """
        Initialises CNN with number of possible DR levels
        Defines model structure, namely architecture of learning and non-learning layers

        Args:
            n_diabetic_retinopathy_levels (int): number of classes in the multi-classification problem for DR diagnosis
        """
        super().__init__()
        resnet50: models.resnet50 = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT
        )
        for name, param in resnet50.named_parameters():
            if "layer_4" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        last_channel: int = resnet50.fc.in_features

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
            nn.Conv2d(
                in_channels=last_channel, out_channels=32, kernel_size=3, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.5),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=32 * 3 * 3, out_features=n_diabetic_retinopathy_levels
            ),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=3),
        )

    def activations_hook(self, grad):
        """For heatmap generation, stores the gradient values"""
        self.gradients = grad

    def forward(self, x: Any) -> dict[str, torch.Tensor]:
        """
        Forward pass of the model,
        Data passed through model architecture to get raw outputs of the final dense layer
        """
        x = self.base_model(x)
        x = self.custom_layer(x)
        self.feature_map = x
        x = self.classifier(x)
        return {"level": x}

    def get_loss(
        self, net_output: torch.Tensor, ground_truth: torch.Tensor
    ) -> torch.Tensor:
        """Calculates cross entropy loss for the model predictions"""
        loss: torch.Tensor = F.cross_entropy(net_output, ground_truth)
        return loss
