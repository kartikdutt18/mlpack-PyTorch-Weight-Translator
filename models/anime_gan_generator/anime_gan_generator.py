from collections import OrderedDict
import torch
from torch import nn
import torch.nn.functional as F
from ..base import BaseModel
import numpy as np
import os
import sys
from xml.etree import ElementTree


class AnimeGanGenerator(BaseModel):
    def __init__(self, pretrained=True):
        super(AnimeGanGenerator, self).__init__()

        # Encoder layers
        self.features = nn.Sequential(OrderedDict([
            ('layer1', nn.Sequential(
                # in: latent_size x 1 x 1

                nn.ConvTranspose2d(128, 512, kernel_size=4,
                                   stride=1, padding=0, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(True),
                # out: 512 x 4 x 4

                nn.ConvTranspose2d(512, 256, kernel_size=4,
                                   stride=2, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                # out: 256 x 8 x 8

                nn.ConvTranspose2d(256, 128, kernel_size=4,
                                   stride=2, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                # out: 128 x 16 x 16

                nn.ConvTranspose2d(128, 64, kernel_size=4,
                                   stride=2, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                # out: 64 x 32 x 32

                nn.ConvTranspose2d(64, 3, kernel_size=4,
                                   stride=2, padding=1, bias=False),
                nn.Tanh()
                # out: 3 x 64 x 64
            )
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
        weight_file = './models/anime_gan_generator/pytorch_weights/G.pth'
        assert len(torch.load(weight_file, map_location=torch.device('cpu')).keys()) == len(
            self.state_dict().keys())
        dic = {}
        for now_keys, values in zip(self.state_dict().keys(), torch.load(weight_file, map_location=torch.device('cpu')).values()):
            dic[now_keys] = values
        self.load_state_dict(dic)
        print('Weights are loaded!')
