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


arch = 'ConvNetTinyImage'
num_images = 1
trained_model = False

import inversefed
setup = inversefed.utils.system_startup()
defs = inversefed.training_strategy('conservative_batch_size_1')


dm = torch.as_tensor(inversefed.consts.tinyimagenet_mean, **setup)[:, None, None]
ds = torch.as_tensor(inversefed.consts.tinyimagenet_std, **setup)[:, None, None]

args = inversefed.options().parse_args()

defs.seed = args.random_seed
np.random.seed(defs.seed)
print(defs.seed)

loss_fn, trainloader, validloader =  inversefed.construct_dataloaders('TinyImageNet', defs)



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

local_lr = 1e-4
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

for i, (inputs, targets) in enumerate(trainloader):
    inputs = inputs.to(**setup)
    targets = targets.to(device=setup['device'])
    # print(inputs.shape[0])
    # print(targets.shape[0])

    model.eval()
    model.zero_grad()

    input_parameters = inversefed.reconstruction_algorithms.loss_steps(model, inputs, targets,
                                                            lr=local_lr, epoches=epoches,
                                                            use_updates=use_updates)
    input_parameters = [p.detach() for p in input_parameters]

    rec_machine = inversefed.FedAvgReconstructor(model, (dm, ds), epoches, local_lr, config,
                                                 num_images = inputs.shape[0], use_updates=use_updates)
    
    output, stats = rec_machine.reconstruct(input_parameters, targets, img_shape=(3, 64, 64))

    reconstructed_count += 1

    test_mse = (output.detach() - inputs).pow(2).mean()
    feat_mse = (model(output.detach())- model(inputs)).pow(2).mean()  
    test_psnr = inversefed.metrics.psnr(output, inputs, factor=1/ds)
    
    rec_loss_sum += stats['opt']
    MSE_sum += test_mse
    FMSE_sum += feat_mse
    PSNR_sum += test_psnr

    print(f"Stats current image {reconstructed_count}\n----------------------\nMean Rec Loss: {rec_loss_sum/reconstructed_count}\nMean MSE: {MSE_sum/reconstructed_count}\n"
        f"Mean PSNR: {PSNR_sum/reconstructed_count}\nMean FMSE:{FMSE_sum/reconstructed_count}\n", flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5)) # Create a figure with 2 subplots

    # Denormalize and plot original image
    inputs_denorm = inputs.clone().detach()
    inputs_denorm.mul_(ds).add_(dm).clamp_(0, 1)
    axes[0].imshow(inputs_denorm[0].permute(1, 2, 0).cpu())
    axes[0].set_title('Original Input')
    axes[0].axis('off') # Hide axes ticks for cleaner look

    # Denormalize and plot reconstructed image
    output_denorm_plot = output.clone().detach()
    output_denorm_plot.mul_(ds).add_(dm).clamp_(0, 1)
    axes[1].imshow(output_denorm_plot[0].permute(1, 2, 0).cpu())
    stats_text = (f"Rec. loss: {stats['opt']:2.4f} | MSE: {test_mse:2.4f} | "
                      f"PSNR: {test_psnr:4.2f} | FMSE: {feat_mse:2.4e}")


    axes[1].set_title(f"Reconstructed Image")
    axes[1].axis('off') 

    fig.suptitle(f'{stats_text}', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])

    plt.tight_layout() # Adjust subplot parameters for a tight layout

    plot_filename = os.path.join(output_plot_dir, f'reconstruction_plot_{i:05d}.png')
    plt.savefig(plot_filename)
    plt.close(fig) # Close the figure to free up memory and prevent it from displaying if not desired
    # --- REMOVED plt.show() as plots are being saved ---


    # --- ADDED: Saving the reconstructed image data directly ---
    # Assuming output_denorm_plot is already denormalized
    image_filename = os.path.join(output_image_dir, f'reconstructed_image_{i:05d}.png')
    save_image(output_denorm_plot, image_filename) # Saves the tensor as an image file

    
    print(f"Reconstructed batch {i+1}/{len(trainloader)}. Total images: {reconstructed_count}", flush=True)

print(f"\nReconstruction complete. Total {reconstructed_count}.", flush=True)