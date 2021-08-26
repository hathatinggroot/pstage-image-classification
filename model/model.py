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

MyResnet18 = torchvision.models.resnet18(pretrained=True)
MyResnet18.fc = nn.Linear(in_features=512, out_features=18, bias=True)

MyResnet101 = torchvision.models.resnet101(pretrained=True)
MyResnet101.fc = nn.Linear(in_features=512, out_features=18, bias=True)


def get_model(cat=None):
    of = 18
    if cat == 'mask' or cat == 'age':
        of = 3
    elif cat == 'gender':
        of = 2
    print(f'of: {of}')
    MyResnet34 = torchvision.models.resnet34(pretrained=True)
    MyResnet34.fc = nn.Linear(in_features=512, out_features=of, bias=True)
    nn.init.xavier_uniform_(MyResnet34.fc.weight)
    stdv = 1/np.sqrt(512)
    MyResnet34.fc.bias.data.uniform_(-stdv, stdv)
    
    return MyResnet34

