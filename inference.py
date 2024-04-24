import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from skimage.color import lab2rgb, rgb2gray
import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog
import os

class ImageColorizer:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.load(model_path)
        self.model.eval()

    def resize_image(self, img, max_size=256):
        h, w = img.shape[:2]
        scale = max_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized_img = cv2.resize(img, (new_w, new_h))
        return resized_img

    def pad_image(self, img, target_height=256, target_width=256):
        h, w = img.shape[:2]
        top = (target_height - h) // 2
        bottom = target_height - h - top
        left = (target_width - w) // 2
        right = target_width - w - left
        color = [0, 0, 0]  # Black padding for grayscale
        padded_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return padded_img

    def to_rgb(self, grayscale_input, ab_input, output_path='results/result_output.jpg'):
        plt.clf()
        color_image = torch.cat((grayscale_input, ab_input), 0).numpy()
        color_image = color_image.transpose((1, 2, 0))
        color_image[:, :, 0:1] = color_image[:, :, 0:1] * 100
        color_image[:, :, 1:3] = color_image[:, :, 1:3] * 255 - 128
        color_image = lab2rgb(color_image.astype(np.float64))
        plt.imsave(arr=color_image, fname=output_path)
        print(f"Colorized image saved at '{output_path}'")

    def process_image(self, image_path):
        input_gray = cv2.imread(image_path)
        input_gray = self.resize_image(input_gray)
        input_gray = self.pad_image(input_gray)
        input_gray = rgb2gray(input_gray)
        input_gray = torch.from_numpy(input_gray).unsqueeze(0).unsqueeze(0).float().to(self.device)

        output_ab = self.model(input_gray)
        self.to_rgb(input_gray[0].cpu(), output_ab[0].detach().cpu())

if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='saved_models/model_human_50.pth', type=str, help='Path to the saved model')
    args = parser.parse_args()

    colorizer = ImageColorizer(args.model_path)

    # Set up tkinter root window for file dialog
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title='Select grayscale test image',
                                           filetypes=[("JPEG files", "*.jpeg"), ("JPG files", "*.jpg"), ("PNG files", "*.png")])
    if not file_path:
        raise ValueError("No file selected")

    print('Beginning Inference')
    colorizer.process_image(file_path)
