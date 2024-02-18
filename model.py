import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock3D(nn.Module):
    """Squeeze-and-Excitation Block for 3D Convolutional Networks."""
    def __init__(self, channel, reduction=16):
        super(SEBlock3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)

class Model(nn.Module):
    """Model with optional attention mechanism."""
    def __init__(self, use_attention=False):
        super(Model, self).__init__()
        self.use_attention = use_attention
        self.conv1 = nn.Conv3d(8, 32, 3, padding="same")
        self.conv2 = nn.Conv3d(32, 64, 3, padding="same")
        self.conv3 = nn.Conv3d(64, 80, 3, padding="same")
        if self.use_attention:
            self.se3 = SEBlock3D(80)
        self.conv4 = nn.Conv3d(80, 20, 3, padding="same")
        self.conv5 = nn.Conv3d(20, 20, 20, padding="same")
        if self.use_attention:
            self.se5 = SEBlock3D(20)
        self.conv6 = nn.Conv3d(20, 16, 3, padding="same")
        self.conv7 = nn.Conv3d(16, 1, 3, padding="same")
        self.dropout1 = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        if self.use_attention:
            x = self.se3(x)
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        if self.use_attention:
            x = self.se5(x)
        x = self.dropout1(x)
        x = F.relu(self.conv6(x))
        x = torch.sigmoid(self.conv7(x))
        return x
