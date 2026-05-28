import torch
import torch.nn as nn
from torchvision import models

class CropYieldModel(nn.Module):
    def __init__(self):
        super(CropYieldModel, self).__init__()
        
        # load pretrained EfficientNet
        self.backbone = models.efficientnet_b0(weights='IMAGENET1K_V1')
        
        # get the number of features from the last layer
        in_features = self.backbone.classifier[1].in_features
        
        # replace classifier with regression head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # single output = yield value
        )
    
    def forward(self, x):
        return self.backbone(x).squeeze(1)


def get_model(device):
    model = CropYieldModel()
    model = model.to(device)
    return model