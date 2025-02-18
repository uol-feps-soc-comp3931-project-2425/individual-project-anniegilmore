from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class DiabeticRetinopathyNet(nn.Module):
    def __init__(self, n_diabetic_retinopathy_levels: int) -> None:
        super().__init__()
        mobilenet = models.mobilenet_v2()
        self.base_model = mobilenet.features
        last_channel = mobilenet.last_channel
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(last_channel),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=last_channel, out_features=n_diabetic_retinopathy_levels),
        )

    def forward(self, x: Any) -> dict[str, torch.Tensor]:
        x = self.base_model(x)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        return {"level": self.classifier(x)}

    def get_loss(self, net_output: Any, ground_truth: torch.Tensor) -> torch.Tensor:
        loss: torch.Tensor = F.cross_entropy(net_output["level"], ground_truth)
        return loss
