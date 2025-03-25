from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class DiabeticRetinopathyNet(nn.Module):
    def __init__(self, n_diabetic_retinopathy_levels: int) -> None:
        super().__init__()
        resnet50 = models.resnet50()
        for param in resnet50.parameters():
            param.requires_grad = False
        self.base_model = resnet50
        last_channel = resnet50.fc.in_features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
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
            nn.Softmax(dim=1),
        )

    def forward(self, x: Any) -> dict[str, torch.Tensor]:
        x = self.base_model(x)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        return {"level": self.classifier(x)}

    def get_loss(self, net_output: Any, ground_truth: torch.Tensor) -> torch.Tensor:
        loss: torch.Tensor = F.cross_entropy(net_output["level"], ground_truth)
        return loss
