from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class DiabeticRetinopathyNet(nn.Module):
    def __init__(self, n_diabetic_retinopathy_levels: int) -> None:
        super().__init__()
        resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        for name, param in resnet50.named_parameters():
            if "layer_4" in name or "fc" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        last_channel = resnet50.fc.in_features
        resnet50.fc = nn.Linear(last_channel, n_diabetic_retinopathy_levels)
        self.transfer_model = resnet50
        
        self.base_model = nn.Sequential(
            resnet50.conv1,
            resnet50.bn1,
            resnet50.relu,
            resnet50.maxpool,
            resnet50.layer1,
            resnet50.layer2,
            resnet50.layer3,
            resnet50.layer4  # Output: [B, 2048, 7, 7]
        )
        
        self.conv_head = nn.Sequential(
                nn.Conv2d(in_channels=2048, out_channels=32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout(0.2),
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
            

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x: Any) -> dict[str, torch.Tensor]:
        # x = self.base_model(x)          # [B, 2048, 7, 7]
        # x = self.conv_head(x)# [B, 128, 1, 1]
        # self.feature_map = x
        # x = self.classifier(x)  
        x = self.transfer_model(x)
        self.feature_map = x
        return {"level": x}


    def get_loss(self, net_output: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
        loss: torch.Tensor = F.cross_entropy(net_output, ground_truth)
        return loss
