from PIL import Image
import torchvision.transforms as T
import torch
import numpy as np
from skimage.color import rgb2lab, rgb2gray
from torchvision import datasets

class EnhancedColorizeDataset(datasets.ImageFolder):
    def __getitem__(self, index):
        image_path, _ = self.imgs[index]
        loaded_image = Image.open(image_path)

        # Convert palette images with transparency to RGBA and then to RGB
        if loaded_image.mode == 'P' or loaded_image.mode == 'RGBA':
            loaded_image = loaded_image.convert('RGBA').convert('RGB')

        # Convert PIL Image to NumPy array to check channels
        image_np = np.array(loaded_image)

        # Skip the image if it's grayscale
        if image_np.ndim == 2 or (image_np.ndim == 3 and image_np.shape[2] == 1):
            return None  # Return None or use another strategy such as loading a different image

        # Apply transformations
        transformed_image = self.transform(loaded_image)

        # Ensure transformed image is in the correct format for LAB conversion
        transformed_image_array = np.array(transformed_image)
        if transformed_image_array.ndim == 2:  # Check if the image is still grayscale after transformation
            transformed_image_array = np.stack([transformed_image_array] * 3, axis=-1)

        # Correctly transpose the array from (C, H, W) to (H, W, C) for rgb2lab
        transformed_image_array = transformed_image_array.transpose((1, 2, 0))

        # Convert image to LAB and normalize
        image_lab = rgb2lab(transformed_image_array)
        image_lab_normalized = (image_lab + 128) / 255
        image_ab_channels = image_lab_normalized[:, :, 1:3]
        image_ab_channels_tensor = torch.from_numpy(image_ab_channels.transpose((2, 0, 1))).float()

        # Convert image to grayscale for the L channel
        grayscale_image = rgb2gray(transformed_image_array)
        grayscale_image_tensor = torch.from_numpy(grayscale_image).unsqueeze(0).float()

        return grayscale_image_tensor, image_ab_channels_tensor
