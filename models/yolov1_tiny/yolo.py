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
        return nn.Sequential(nn.Sequential(nn.Linear(in_features=256 * 7 * 7, out_features=1470), nn.Sigmoid()))



# model output is like
# [x1, y1, w1, h1, c1, x2, y2, w2, h2, c2, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20]
class CustomyoloLoss(nn.Module):
    def __init__(self):
        super(CustomMSELoss, self).__init__()


    # predictions => [c1, c2]
    # targets => [c*]
    def objectiveness_loss(self, predictions, target):
        c1 = predictions[0]
        c2 = predictions[1]
        c = target[0]
        if c == 1:
            return torch.square(c1 - c) if c1 > c2 else torch.square(c2 - c)
        else:
            return torch.sum(torch.square(predictions))

    # predictions = [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20]
    # targets = [p1*, p2*, p3*, p4*, p5*, p6*, p7*, p8*, p9*, p10*, p11*, p12*, p13*, p14*, p15*, p16*, p17*, p18*, p19*, p20*]
    def classification_loss(self, predictions, targets):
        return torch.sum(torch.square(predictions - targets))

    # predictions = [x1, y1, w1, h1, c1, x2, y2, w2, h2, c2]
    # targets = [x*, y*, w*, h*]
    def box_regression_loss(self, predictions, targets):
        c1 = predictions[4]
        c2 = predictions[9]
        t_box_center = targets[0:2]
        t_h_w = targets[2:4]

        if c1 > c2:
            p_box_center = predictions[0:2]
            p_h_w = predictions[2:4]
        else:
            p_box_center = predictions[5:7]
            p_h_w = predictions[7:9]

        return torch.sum(torch.square(p_box_center - t_box_center)) + torch.sum(torch.square(torch.sqrt(p_h_w) - torch.sqrt(t_h_w)))


    # predictions => (50, 7*7*30) => (50, 1470)
    # target => (50, 7*7*25) => (50, 1225)
    def forward(self, predictions, targets):

        num_batches = predictions.shape[0]
        objectiveness_loss = 0
        class_loss = 0
        box_loss = 0

        predictions_ = predictions.reshape((num_batches, 7, 7, 30))
        targets_ = targets.reshape((num_batches, 7, 7, 25))

        for n_sample in range(num_batches):
            # data => (7*7*30)
            for row in range(7):
                for col in range(7):

                    c1 = predictions_[n_sample, row, col, 4]
                    c2 = predictions_[n_sample, row, col, 9]
                    c = targets_[n_sample, row, col, 4]

                    object_present = True if c == 1 else False


                    if object_present:
                        objectiveness_loss += self.objectiveness_loss(
                                torch.cat((predictions_[n_sample, row, col, 4:5], predictions_[n_sample, row, col, 9:10])),
                                targets_[n_sample, row, col, 4:5]
                        )

                        class_loss += self.classification_loss(
                                predictions_[n_sample, row, col, 10:],
                                targets_[n_sample, row, col, 5:]
                        )

                        box_loss += 5 * self.box_regression_loss(
                                predictions_[n_sample, row, col, :10],
                                targets_[n_sample, row, col, :5]
                        )

                    else:
                        objectiveness_loss += 0.5 * self.objectiveness_loss(
                                torch.cat((predictions_[n_sample, row, col, 4:5], predictions_[n_sample, row, col, 9:10])),
                                targets_[n_sample, row, col, 4:5]
                        )

        # overall loss will be the sum of all the loss
        loss = objectiveness_loss + class_loss + box_loss
        return loss