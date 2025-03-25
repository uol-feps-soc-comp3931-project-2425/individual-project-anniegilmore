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
        
        super(DiabeticRetinopathyNet, self).__init__()
        
        # Convolutional Layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        
        # Max Pooling Layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(512 * 7 * 7, 4096)  # Assuming input images are 224x224
        self.fc2 = nn.Linear(4096, n_diabetic_retinopathy_levels)
        
        # Dropout Layer
        self.dropout = nn.Dropout(p=0.1)
        
        # self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.classifier = nn.Sequential(
        #     nn.Conv2d(in_features=last_channel, out_features=64),
        #     nn.ReLU(),
        #     nn.Linear(in_features=64, out_features=128),
        #     nn.ReLU(),
        #     nn.Linear(in_features=128, out_features=256),
        #     nn.ReLU(),
        #     nn.Linear(in_features=256, out_features=512),
        #     nn.ReLU(),
        #     nn.Linear(in_features=512, out_features=512),
        #     nn.ReLU(),
        #     nn.Linear(in_features=512, out_features=4096),
        #     nn.ReLU(),
        #     # nn.Dropout(0.2),
        #     nn.Linear(in_features=4096, out_features=n_diabetic_retinopathy_levels),
        #     nn.Softmax(dim=1),
        # )

    def forward(self, x: Any) -> dict[str, torch.Tensor]:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        
        x = F.relu(self.conv5(x))
        x = self.pool(x)
        
        # Flatten the output for the Fully Connected Layer
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        # Softmax Activation for Multi-Class Classification
        return {"level": F.softmax(x, dim=1)}
        
        
        # x = self.base_model(x)
        # x = self.pool(x)
        # x = torch.flatten(x, start_dim=1)
        # return {"level": self.classifier(x)}

    def get_loss(self, net_output: Any, ground_truth: torch.Tensor) -> torch.Tensor:
        loss: torch.Tensor = F.cross_entropy(net_output["level"], ground_truth)
        return loss
