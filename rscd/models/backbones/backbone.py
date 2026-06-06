import torch 
import torch.nn as nn
from rscd.models.backbones.seaformer import *

class Base(nn.Module):
    def __init__(self, name, pretrained=False, weights=None):
        super().__init__()
        if name == 'Seaformer':
            if weights is None:
                self.backbone = SeaFormer_L(pretrained=pretrained)
            else:
                self.backbone = SeaFormer_L(pretrained=pretrained, weights=weights)
        else:
            raise ValueError(f"Unsupported backbone '{name}'. FSMamba release keeps SeaFormer only.")

    def forward(self, xA, xB):
        featuresA = self.backbone(xA)
        featuresB = self.backbone(xB)

        return [featuresA, featuresB]
