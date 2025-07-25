import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime as dt
import inversefed
import random

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

args = inversefed.options().parse_args()

arch = 'ConvNetTinyImage'
num_images = args.num_images
trained_model = False

import inversefed
setup = inversefed.utils.system_startup()
defs = inversefed.training_strategy('conservative')

dm = torch.as_tensor(inversefed.consts.tinyimagenet_mean, **setup)[:, None, None]
ds = torch.as_tensor(inversefed.consts.tinyimagenet_std, **setup)[:, None, None]

defs.seed = args.random_seed
np.random.seed(defs.seed)

loss_fn, trainloader, validloader =  inversefed.construct_dataloaders('TinyImageNet', defs)
args = inversefed.options().parse_args()



model, _ = inversefed.construct_model(arch, num_classes=200, num_channels=3)
model.to(**setup)
if trained_model:
    epochs = 120
    file = f'{arch}_{epochs}.pth'
    try:
        model.load_state_dict(torch.load(f'models/{file}'))
    except FileNotFoundError:
        inversefed.train(model, loss_fn, trainloader, validloader, defs, setup=setup)
        torch.save(model.state_dict(), f'models/{file}')
model.eval()

epoches = args.local_epoches
local_lr = args.local_lr
use_updates = True

config = dict(signed=True,
              boxed=True,
              cost_fn='sim',
              indices='def',
              weights='equal',
              lr=0.1,
              optim='adam',
              restarts=1,
              max_iterations=args.iterations,
              total_variation=args.tv,
              tv_min_value = args.tv_min_value,
              tv_scheduler_step= args.tv_scheduler_step,
              tv_scaling_factor = args.tv_scaling_factor,
              init='randn',
              filter='none',
              lr_decay=True,
              scoring_choice='loss',
              regularization_choice=args.regularization_choice)
              
output_image_dir = args.image_dir_name
output_plot_dir = args.plot_dir_name
output_image_dir = f'images/{output_image_dir}'
output_plot_dir = f'plots/{output_plot_dir}'

os.makedirs(output_image_dir, exist_ok=True) # Create directory for images
os.makedirs(output_plot_dir, exist_ok=True) # Create directory for plots

reconstructed_count = 0

rec_loss_sum = MSE_sum = PSNR_sum = FMSE_sum = 0



if(args.random_seed!=0):
    random.seed(args.random_seed)

for i in range(args.num_recoveries):
    inputs, targets = [], [] # choosen randomly ... just whatever you want
    while len(targets) < num_images:

        idx = random.randrange(100000)
        img, label = trainloader.dataset[idx]
        targets.append(torch.as_tensor((label,), device=setup['device']))
        inputs.append(img.to(**setup))
    inputs = torch.stack(inputs)
    targets = torch.cat(targets)

    model.eval()
    model.zero_grad()

    input_parameters = inversefed.reconstruction_algorithms.loss_steps(model, inputs, targets,
                                                            lr=local_lr, epoches=epoches,
                                                            use_updates=use_updates)
    input_parameters = [p.detach() for p in input_parameters]

    rec_machine = inversefed.FedAvgReconstructor(model, (dm, ds), epoches, local_lr, config,
                                                 num_images = inputs.shape[0], use_updates=use_updates)

    output, stats = rec_machine.reconstruct(input_parameters, targets, img_shape=(3, 64, 64))


    reconstructed_count += inputs.shape[0]


    test_mse = (output.detach() - inputs).pow(2).mean()
    feat_mse = (model(output.detach())- model(inputs)).pow(2).mean()  
    test_psnr = inversefed.metrics.psnr(output, inputs, factor=1/ds)

    rec_loss_sum += stats['opt']
    MSE_sum += test_mse
    FMSE_sum += feat_mse
    PSNR_sum += test_psnr

    print(f"Stats current batch {i+1}\n----------------------\nMean Rec Loss: {rec_loss_sum/(i+1)}\nMean MSE: {MSE_sum/(i+1)}\n"
        f"Mean PSNR: {PSNR_sum/(i+1)}\nMean FMSE:{FMSE_sum/(i+1)}\n", flush=True)

    num_images_in_batch = inputs.shape[0]

    fig, axes = plt.subplots(num_images_in_batch, 2, figsize=(10, 5 * num_images_in_batch))

    if num_images_in_batch == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(f"Rec. loss: {stats['opt']:2.4f} | MSE: {test_mse:2.4f} "
                 f"| PSNR: {test_psnr:4.2f} | FMSE: {feat_mse:2.4e} |", fontsize=16)

    # Denormalize the entire batch of images once before the loop
    inputs_denorm = inputs.clone().detach().mul_(ds).add_(dm).clamp_(0, 1)
    output_denorm_plot = output.clone().detach().mul_(ds).add_(dm).clamp_(0, 1)

    # Loop through each image in the batch to plot it
    for j in range(num_images_in_batch):
        # Plot original image in the left column
        axes[j, 0].imshow(inputs_denorm[j].permute(1, 2, 0).cpu())
        axes[j, 0].axis('off')

        # Plot reconstructed image in the right column
        axes[j, 1].imshow(output_denorm_plot[j].permute(1, 2, 0).cpu())
        axes[j, 1].axis('off')

        # Set titles for the columns, but only on the top row
        if j == 0:
            axes[j, 0].set_title('Original Inputs')
            axes[j, 1].set_title('Reconstructed Outputs')

    # Adjust layout to make room for the main title (suptitle)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    plt.show() # Display the plot for the current batch

    plot_filename = os.path.join(output_plot_dir, f'reconstruction_plot_{i:05d}.png')
    plt.savefig(plot_filename)
    plt.close(fig) 

    image_filename = os.path.join(output_image_dir, f'reconstructed_image_{i:05d}.png')
    save_image(output_denorm_plot, image_filename) # Saves the tensor as an image file
    print(f"Reconstructed batch {i+1}/{len(trainloader)}. Total images: {reconstructed_count}")

print(f"\nReconstruction complete. Total {reconstructed_count}.")