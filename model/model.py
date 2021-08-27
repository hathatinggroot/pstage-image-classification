import torch
import torch.nn as nn
import torchvision
import numpy as np


class MyModel(nn.Module):
    def __init__(self, num_classes: int = 18):
        super(MyModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x

class ModelProvider:
    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
    
    def __call__(self) -> nn.Module:
        raise NotImplementedError
        
class MyResnet34(ModelProvider):
    def __init__(self, num_classes: int) -> None:
        super().__init__(num_classes)

    def __call__(self) -> nn.Module:
        model = torchvision.models.resnet34(pretrained=True)
        model.fc = nn.Linear(in_features=512, out_features=self.num_classes, bias=True)
        nn.init.xavier_uniform_(model.fc.weight)
        stdv = 1/np.sqrt(512)
        model.fc.bias.data.uniform_(-stdv, stdv)

        return model


class MyResnet50(ModelProvider):
    def __init__(self, num_classes: int) -> None:
        super().__init__(num_classes)

    def __call__(self) -> nn.Module:
        model = torchvision.models.resnet50(pretrained=True)
        model.fc = nn.Linear(in_features=2048, out_features=self.num_classes, bias=True)
        nn.init.xavier_uniform_(model.fc.weight)
        stdv = 1/np.sqrt(2048)
        model.fc.bias.data.uniform_(-stdv, stdv)

        return model