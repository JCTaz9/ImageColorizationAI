import torch.nn as nn
import torchvision.models as models

class EnhancedNet(nn.Module):
    def __init__(self, input_size=128):
        super(EnhancedNet, self).__init__()
        # Initialize ResNet-34 with 1000 classes and adjust first convolutional layer to accept single-channel input
        resnet = models.resnet34(pretrained=True)
        # Modify the first convolutional layer to take a single channel input
        resnet.conv1.weight = nn.Parameter(resnet.conv1.weight.sum(dim=1).unsqueeze(1))
        # Use the initial part of ResNet-34 for mid-level feature extraction
        self.midlevel_resnet = nn.Sequential(*list(resnet.children())[0:6])
        
        # Define the upsampling path to enhance the resolution of the feature maps
        self.upsample = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 2, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2)
        )

    def forward(self, x):
        # Extract mid-level features from the input image
        midlevel_features = self.midlevel_resnet(x)
        # Upsample the extracted features to the target resolution
        output = self.upsample(midlevel_features)
        return output
