from collections import OrderedDict
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import numpy as np
from xml.etree import ElementTree
from PIL import Image
import matplotlib.pyplot as plt

def conv_layers(in_channels, out_channels):
    layer = nn.Sequential(OrderedDict([('conv',nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)),
            ('Bn', nn.BatchNorm2d(out_channels)), ('leaky_relu', nn.LeakyReLU(0.01))]))
    return layer

def pooling_layers():
    layer = nn.Sequential(OrderedDict([('max_pool', nn.AvgPool2d(kernel_size=2, stride=2))]))
    return layer

class TinyYOLO(nn.Module):
    def __init__(self, num_bboxes=2, num_classes=20):
        super().__init__()
        self.features = self.make_features()
        self.classifier = self.make_classifier(num_bboxes, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 7 * 7 * 256)
        x = self.classifier(x)
        return x

    def make_features(self):
        layers = []
        layers.append(conv_layers(3, 16))
        layers.append(pooling_layers())
        out_channels = 16
        
        for i in range(0, 5):
            layers.append(conv_layers(out_channels, out_channels * 2))
            layers.append(pooling_layers())
            out_channels = out_channels * 2
        layers.append(conv_layers(out_channels, out_channels * 2))
        out_channels = out_channels * 2
        layers.append(conv_layers(out_channels, 256))
        return nn.Sequential(*layers)

    def make_classifier(self, num_bboxes, num_classes):
        return nn.Sequential(nn.Sequential(nn.Linear(in_features = 256 * 7 * 7, out_features = 1470),
                nn.Sigmoid()))

