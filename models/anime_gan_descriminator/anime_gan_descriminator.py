from collections import OrderedDict
import torch
from torch import nn
import torch.nn.functional as F
from ..base import BaseModel
import numpy as np
import os
import sys
from xml.etree import ElementTree


class AnimeGanDescriminator(BaseModel):
    def __init__(self, pretrained=True):
        super(AnimeGanDescriminator, self).__init__()

        # Encoder layers
        self.features = nn.Sequential(OrderedDict([
            ('layer1', nn.Sequential(
                # in: 3 x 64 x 64

                nn.Conv2d(3, 64, kernel_size=4, stride=2,
                          padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2, inplace=True),
                # out: 64 x 32 x 32

                nn.Conv2d(64, 128, kernel_size=4,
                          stride=2, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),
                # out: 128 x 16 x 16

                nn.Conv2d(128, 256, kernel_size=4,
                          stride=2, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True),
                # out: 256 x 8 x 8

                nn.Conv2d(256, 512, kernel_size=4,
                          stride=2, padding=1, bias=False),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True),
                # out: 512 x 4 x 4

                nn.Conv2d(512, 1, kernel_size=4, stride=1,
                          padding=0, bias=False),
                # out: 1 x 1 x 1

                nn.Flatten(),
                nn.Sigmoid())
             # out: 3 x 64 x 64
             )]))

        self.classifier = nn.Sequential(OrderedDict())
        if pretrained:
            self.load_weight()
            print('Model is loaded')

    def forward(self, x):
        out = self.features(x)
        out = self.classifier(out)
        return out

    def load_weight(self):
        weight_file = './models/anime_gan_descriminator/pytorch_weights/D.pth'
        assert len(torch.load(weight_file, map_location=torch.device('cpu')).keys()) == len(
            self.state_dict().keys())
        dic = {}
        for now_keys, values in zip(self.state_dict().keys(), torch.load(weight_file, map_location=torch.device('cpu')).values()):
            dic[now_keys] = values
        self.load_state_dict(dic)
        print('Weights are loaded!')
