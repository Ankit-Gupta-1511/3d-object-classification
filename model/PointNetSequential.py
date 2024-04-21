import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNet(nn.Module):
    def __init__(self, output_classes=10):
        super(PointNet, self).__init__()

        # Sequential for convolutional and batch norm layers
        self.feature_extraction = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
        )

        # Sequential for fully connected layers
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, output_classes)
        )

    def forward(self, x):
        # Extract features
        x = self.feature_extraction(x)
        # Global max pooling
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)  # Flatten for fully connected layers
        # Classify
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)


