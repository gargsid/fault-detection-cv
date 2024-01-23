import torch
import torch.nn as nn 
from torchvision.models import resnet50, ResNet50_Weights

import os, sys, random
import numpy as np

class FaultDetectionModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.resnet_weights = ResNet50_Weights.DEFAULT
        self.resnet = resnet50(weights=self.resnet_weights)
        self.resnet.fc = nn.Linear(in_features=2048, out_features=1024, bias=True)
        self.bn_1 = nn.BatchNorm1d(1024)
        self.linear_1 = nn.Linear(in_features=1024, out_features=256, bias=True)
        self.bn_2 = nn.BatchNorm1d(256)
        self.linear_2 = nn.Linear(in_features=256, out_features=32, bias=True)
        self.bn_3 = nn.BatchNorm1d(32)
        self.logits = nn.Linear(in_features=32, out_features=num_classes, bias=True)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.gelu(self.bn_1(self.resnet(x)))
        x = self.gelu(self.bn_2(self.linear_1(x)))
        x = self.gelu(self.bn_3(self.linear_2(x)))
        x = self.logits(x)
        return x
