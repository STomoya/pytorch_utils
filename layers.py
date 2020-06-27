
import torch
import torch.nn as nn
import torch.nn.functional as F


class SingSqrt(nn.Module):
    '''
    singed square root
    '''
    def forward(self, x):
        x = torch.sqrt(F.relu(x)) - torch.sqrt(F.relu(-x))
        return x

class L2Norm(nn.Module):
    '''
    L2 normalization
    '''
    def forward(self, x):
        x = F.normalize(x)
        return x

class Flatten(nn.Module):
    '''
    flatten
    '''
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x

class ConvBatchnormRelu2d(nn.Module):
    def __init__(self,
        in_channels, out_channels, kernel_size,
        **kwargs
    ):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv(x)
        return x
