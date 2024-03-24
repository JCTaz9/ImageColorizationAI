import torchvision.transforms as T
import torch
import numpy as np
from skimage.color import rgb2lab, rgb2gray
from torchvision import datasets

class EnhancedColorizeDataset(datasets.ImageFolder):
    def __getitem__(self, index):
        # Retrieve the image path and ignore its label using the provided index
        image_path, _ = self.imgs[index]
        # Load the image from the path
        loaded_image = self.loader(image_path)
        # Apply predefined transformations to the image
        transformed_image = self.transform(loaded_image)
        # Convert the transformed image into a numpy array
        transformed_image_array = np.asarray(transformed_image)
        # Change the color space of the image from RGB to LAB
        image_lab = rgb2lab(transformed_image_array)
        # Scale LAB values to a range between 0 and 1
        image_lab_normalized = (image_lab + 128) / 255
        # Extract the A and B color channels
        image_ab_channels = image_lab_normalized[:, :, 1:3]
        # Convert A and B channels into a torch tensor
        image_ab_channels_tensor = torch.from_numpy(image_ab_channels.transpose((2, 0, 1))).float()
        # Convert the original image to grayscale
        grayscale_image = rgb2gray(transformed_image_array)
        # Transform grayscale image to a torch tensor and add an extra dimension
        grayscale_image_tensor = torch.from_numpy(grayscale_image).unsqueeze(0).float()
        # Return the grayscale and AB channel images
        return grayscale_image_tensor, image_ab_channels_tensor
