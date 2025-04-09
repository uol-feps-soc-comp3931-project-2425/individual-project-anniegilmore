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

        super(DiabeticRetinopathyNet, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d((56, 56))
        self.flattened_tensor = nn.Flatten()

        self.block3 = nn.Sequential(
            nn.Linear(in_features=128 * 56 * 56, out_features=128),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=128),
            nn.Dropout(0.2),
            nn.Linear(in_features=128, out_features=n_diabetic_retinopathy_levels),
            nn.Softmax(dim=1),
        )
        self.gradients = None

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x: Any) -> dict[str, torch.Tensor]:
        conv_output = self.block2(self.block1(x))
        flat_output = self.flattened_tensor(conv_output)
        linear_output = self.block3(flat_output)
        return {"level": linear_output}

    def get_loss(
        self, net_output: torch.Tensor, ground_truth: torch.Tensor
    ) -> torch.Tensor:
        loss: torch.Tensor = F.cross_entropy(net_output, ground_truth)
        return loss
