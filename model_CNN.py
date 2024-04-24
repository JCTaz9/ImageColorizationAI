import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet34_Weights

class EnhancedNet(nn.Module):
    def __init__(self, input_size=128):
        super(EnhancedNet, self).__init__()
        # Initialize ResNet-34 with the default pretrained weights
        weights = ResNet34_Weights.DEFAULT
        resnet = models.resnet34(weights=weights)
        
        # Modify the first convolutional layer to take a single channel input
        # Create a new Conv2d layer with modified parameters for 1 input channel
        original_first_layer = resnet.conv1
        resnet.conv1 = nn.Conv2d(1, original_first_layer.out_channels, 
                                 kernel_size=original_first_layer.kernel_size, 
                                 stride=original_first_layer.stride, 
                                 padding=original_first_layer.padding, 
                                 bias=original_first_layer.bias)

        # Adjust the new first layer weights by averaging the original weights across the input channels
        with torch.no_grad():
            resnet.conv1.weight = nn.Parameter(original_first_layer.weight.mean(dim=1, keepdim=True))
        
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
