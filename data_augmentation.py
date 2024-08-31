# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 20:01:14 2024

@author: Vladimir Zeljkovic
"""

import os
from PIL import Image, ImageEnhance
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

class Lighten:
    def __init__(self, factor=1.5):
        self.factor = factor

    def __call__(self, img):
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(self.factor)

class RandomHorizontalFlipAndLighten:
    def __init__(self, flip_prob=0.5, lighten_prob=0.5, lighten_factor=1.5):
        self.flip_prob = flip_prob
        self.lighten_prob = lighten_prob
        self.lighten = Lighten(lighten_factor)

    def __call__(self, img):
        # Apply horizontal flip
        if torch.rand(1).item() < self.flip_prob:
            img = TF.hflip(img)
        
        # Apply lightening
        if torch.rand(1).item() < self.lighten_prob:
            img = self.lighten(img)

        return img

def transform_images(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    files = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

    # Define transformations
    transformations = [
        RandomHorizontalFlipAndLighten(flip_prob=0.0, lighten_prob=0.0),  #Original
        RandomHorizontalFlipAndLighten(flip_prob=1.0, lighten_prob=0.0),  # Flipped
        RandomHorizontalFlipAndLighten(flip_prob=1.0, lighten_prob=1.0),  # Flipped and Lightened# Lightened
        RandomHorizontalFlipAndLighten(flip_prob=0.0, lighten_prob=1.0),   # Lightened
        RandomHorizontalFlipAndLighten(flip_prob=0.0, lighten_prob=1.0, lighten_factor=2)
    ]

    for file_name in files:
        image_path = os.path.join(input_folder, file_name)
        img = Image.open(image_path).convert('RGB')

        for i, transform in enumerate(transformations):
            transformed_img = transform(img)
            transformed_img.save(os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}_transformation_{i}.jpg"))

input_folder = 'slike'
output_folder = 'slike_trans'

transform_images(input_folder, output_folder)
