import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime as dt

import torch
from torch import optim, nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision.utils import make_grid
from torchvision import transforms as T
from torchvision import models, datasets

from collections import defaultdict
from PIL import Image
from torchvision.utils import save_image
import os

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

TRAIN_DIR = Path("/bsuhome/jonathanauyong/gradient/invertinggradients/inversefed/data/tinyimagenet/tiny-imagenet-200/train")
VALID_DIR = Path("/bsuhome/jonathanauyong/gradient/invertinggradients/inversefed/data/tinyimagenet/tiny-imagenet-200/val")

def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    
def show_batch(dataloader):
    dataiter = iter(dataloader)
    images, labels = dataiter.next()    
    imshow(make_grid(images)) # Using Torchvision.utils make_grid function
    
def show_image(dataloader):
    dataiter = iter(dataloader)
    images, labels = dataiter.next()
    random_num = randint(0, len(images)-1)
    imshow(images[random_num])
    label = labels[random_num]
    print(f'Label: {label}, Shape: {images[random_num].shape}')

def generate_dataloader(data, name, transform):
    if data is None: 
        return None
    
    # Read image files to pytorch dataset using ImageFolder, a generic data 
    # loader where images are in format root/label/filename
    # See https://pytorch.org/vision/stable/datasets.html
    if transform is None:
        dataset = datasets.ImageFolder(data, transform=T.ToTensor())
    else:
        dataset = datasets.ImageFolder(data, transform=transform)

    # Set options for device
    if use_cuda:
        kwargs = {"pin_memory": True, "num_workers": 1}
    else:
        kwargs = {}
    
    # Wrap image dataset (defined above) in dataloader 
    dataloader = DataLoader(dataset, batch_size=batch_size, 
                        shuffle=(name=="train"), 
                        **kwargs)
    
    return dataloader

preprocess_transform_pretrain = T.Compose([
                T.Resize(256), # Resize images to 256 x 256
                T.CenterCrop(224), # Center crop image
                T.RandomHorizontalFlip(),
                T.ToTensor(),  # Converting cropped images to tensors
                T.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
])

batch_size = 64

# Create DataLoaders for pre-trained models (normalized based on specific requirements)
train_loader_pretrain = generate_dataloader(TRAIN_DIR, "train",
                                  transform=preprocess_transform_pretrain)

val_loader_pretrain = generate_dataloader(VALID_DIR, "val",
                                 transform=preprocess_transform_pretrain)

def show_normalized_images(img_tensor):
    """De-normalizes and displays a tensor image or a grid of them."""
    # These are the mean and std values you used for normalization
    mean = torch.tensor([0.485, 0.456, 0.406], device=img_tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225], device=img_tensor.device)
    
    # De-normalize: rearrange dimensions for broadcasting and multiply
    denorm_img = img_tensor * std[:, None, None] + mean[:, None, None]
    
    # Convert to numpy and transpose for matplotlib
    npimg = denorm_img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# --- Display a batch from the training loader ---
print("Showing a batch of images from the train_loader...")
train_iter = iter(train_loader_pretrain)
images, labels = next(train_iter)

# Create a grid using torchvision's utility
img_grid = make_grid(images)

# Use our new function to show the de-normalized image grid
show_normalized_images(img_grid)


# --- Display a batch from the validation loader ---
print("\nShowing a batch of images from the val_loader...")
val_iter = iter(val_loader_pretrain)
images_val, labels_val = next(val_iter)

# Create and show the grid
img_grid_val = make_grid(images_val)
show_normalized_images(img_grid_val)

# Get class names from the dataset
class_names = train_loader_pretrain.dataset.classes

# Print the labels for the first 8 images in the training batch
print("Labels for the first 8 images in the training batch:")
print(' | '.join(f'{class_names[labels[j]]}' for j in range(8)))

