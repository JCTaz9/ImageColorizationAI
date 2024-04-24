import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from data_colorize import EnhancedColorizeDataset # Custom dataset for colorization
from skimage.color import lab2rgb # LAB to RGB conversion for image colorization
import time
from model_CNN import EnhancedNet # Custom CNN model for image colorization
import torch.nn as nn
import torchvision.transforms as T
import cv2
import glob
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision import transforms
from torch.utils.data.dataloader import default_collate
from pathlib import Path

# Define device based on CUDA availability

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def custom_collate(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None  # Handle the case where the entire batch is None
    return default_collate(batch)

# Utility for measuring and tracking metrics
class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        # Initialize/reset all metrics
        self.val = self.avg = self.sum = self.count = 0
    def update(self, val, n=1):
        # Update metrics with new value
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# Trainer class for handling training and validation
class Trainer:
    def __init__(self):
        pass

    # Convert grayscale and ab channels to RGB and save if specified
    def to_rgb(self, grayscale_input, ab_input, save_path=None, save_name=None):
        plt.clf()  # Clear current figure to prevent overlap
        color_image = torch.cat((grayscale_input, ab_input), 0).numpy()  # Concatenate L and AB channels
        color_image = color_image.transpose((1, 2, 0))  # Adjust for matplotlib's image format
        color_image[:, :, 0:1] = color_image[:, :, 0:1] * 100  # Scale L channel
        color_image[:, :, 1:3] = color_image[:, :, 1:3] * 255 - 128  # Scale AB channels
        color_image = lab2rgb(color_image.astype(np.float64))  # Convert LAB to RGB

        grayscale_input = grayscale_input.squeeze().numpy()  # Process grayscale image for saving
        if save_path and save_name:
            os.makedirs(save_path['grayscale'], exist_ok=True)
            os.makedirs(save_path['colorized'], exist_ok=True)
            # Save grayscale and colorized images
            plt.imsave(arr=grayscale_input, fname=os.path.join(save_path['grayscale'], save_name), cmap='gray')
            plt.imsave(arr=color_image, fname=os.path.join(save_path['colorized'], save_name))

    # Train the model for one epoch
    def train(self, train_loader, epoch, model, criterion, optimizer, scheduler):
      print('Starting training for epoch {}'.format(epoch+1))

      model.train()
      for i, batch in enumerate(train_loader):
        if batch is None:
            continue  # Skip the iteration if the batch data is None
        
        input_gray, input_ab = batch


      batch_time, data_time, losses = AverageMeter(), AverageMeter(), AverageMeter()
      end = time.time()

      for i, (input_gray, input_ab) in enumerate(train_loader):
        # Load data to device
        input_gray, input_ab = input_gray.to(device), input_ab.to(device)
        data_time.update(time.time() - end) 

        # Forward pass
        output_ab = model(input_gray) 
        loss = criterion(output_ab, input_ab) 
        losses.update(loss.item(), input_gray.size(0))

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Record timings
        batch_time.update(time.time() - end)
        end = time.time()

        # Logging
        if i % 2 == 0:
          print('Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.6f} ({loss.avg:.6f})'.format(
                  epoch+1, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))

      print('Completed epoch {}'.format(epoch+1))

    def validate(self, val_loader, epoch, save_images, model, criterion):
      model.eval()
      # Initialize performance metrics and timers
      batch_time, data_time, losses = AverageMeter(), AverageMeter(), AverageMeter()
      end = time.time()
      already_saved_images = False

      for i, (input_gray, input_ab) in enumerate(val_loader):
        # Update data loading time
        data_time.update(time.time() - end)
        # Move inputs to the appropriate device
        input_gray, input_ab = input_gray.to(device), input_ab.to(device)

        # Perform forward pass through the model and compute loss
        output_ab = model(input_gray)
        loss = criterion(output_ab, input_ab)
        # Update loss metrics
        losses.update(loss.item(), input_gray.size(0))

        # Conditionally save output images once per epoch
        if save_images and not already_saved_images:
          already_saved_images = True
          # Limit the number of saved images to 10 per epoch
          for j in range(min(len(output_ab), 10)):
            # Define save paths for different image types
            save_path = {'grayscale': 'outputs/gray/', 'colorized': 'outputs/color/', 'ground_truth': 'outputs/ground_truth/'}
            save_name = 'img-{}-epoch-{}.jpg'.format(i * val_loader.batch_size + j, epoch+1)
            # Convert model output to RGB and save
            self.to_rgb(input_gray[j].cpu(), ab_input=output_ab[j].detach().cpu(), save_path=save_path, save_name=save_name)

        # Update batch processing time
        batch_time.update(time.time() - end)
        end = time.time()

        # Log validation stats periodically
        if i % 25 == 0:
          print('Validate: [{0}/{1}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Loss {loss.val:.6f} ({loss.avg:.6f})\t'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses))

      # Log completion of validation
      print('Finished validation. Model ready.')
      return losses.avg

if __name__ == "__main__":
    # Initialize default parameters for the training setup
    image_dir = 'dataset'
    n_val = 100
    epochs = 100
    lr = 1e-3
    weight_decay = 1e-4
    save_model = True
    loss_type = 'mse'
    batch_size = 32

    # Clear existing images from output folders
    for folder in ['outputs/color/*', 'outputs/gray/*']:
        files = glob.glob(folder)
        for f in files:
            os.remove(f)

    # Setup model, loss function, optimizer, and learning rate scheduler
    model = EnhancedNet().to(device)
    criterion = nn.MSELoss().to(device) if loss_type == 'mse' else nn.L1Loss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # Prepare datasets and data loaders
    all_transforms = T.Compose([
        T.Resize((256, 256)),  # Consider aspect ratio if necessary
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.ToTensor()  
    ])
    all_imagefolder = EnhancedColorizeDataset(image_dir, all_transforms)
    train_size = int(0.9 * len(all_imagefolder))
    val_size = len(all_imagefolder) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(all_imagefolder, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Execute training and validation cycles
    for epoch in range(epochs):
        Trainer().train(train_loader, epoch, model, criterion, optimizer, scheduler)
        scheduler.step()
        with torch.no_grad():
            Trainer().validate(val_loader, epoch, False, model, criterion)

    # Optionally save the trained model
    if save_model:
        models_directory = 'saved_models'
        os.makedirs(models_directory, exist_ok=True)
        model_save_path = os.path.join(models_directory, 'model_human.pth')
        torch.save(model, model_save_path)


