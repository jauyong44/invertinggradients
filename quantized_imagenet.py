import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime as dt
from random import randrange
from inversefed.reconstruction_algorithms import quantize_gradient, dequantize_gradient
import numpy as np

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
image_count = 1

bits = [-1,8,4,2]

rec_loss_sum_list = [0] * len(bits)
MSE_sum_list = [0] * len(bits)
PSNR_sum_list = [0] * len(bits)
FMSE_sum_list = [0] * len(bits)

for i, (inputs, targets) in enumerate(trainloader):
    inputs = inputs.to(**setup)
    targets = targets.to(device=setup['device'])
    print(inputs.shape[0])
    print(targets.shape[0])

    model.eval()
    model.zero_grad()

    input_parameters = inversefed.reconstruction_algorithms.loss_steps(model, inputs, targets,
                                                            lr=local_lr, epoches=epoches,
                                                            use_updates=use_updates)
    input_parameters = [p.detach() for p in input_parameters]


    for b in range (len(bits)):
        curr_bits = bits[b]
        if (curr_bits>0):
            approximate_input_parameters = []
            for param in input_parameters:
                param_numpy = param.cpu().numpy()
                quantized_param, p_min, p_max = quantize_gradient(param_numpy, num_bits = curr_bits)
                # original_size = param_numpy.nbytes
                # transmitted_size = quantized_param.nbytes + np.dtype(np.float32).itemsize * 2
                # if(image_count==0):
                #     print(f"Original size: {original_size} bytes. Transmitted size: {transmitted_size} bytes.")
                #     print(f"Compression savings: {1 - transmitted_size / original_size:.2%}")

                approximated_param_numpy = dequantize_gradient(quantized_param, p_min, p_max, num_bits= curr_bits)
                approximated_param = torch.from_numpy(approximated_param_numpy)
                approximate_input_parameters.append(approximated_param)
            input_parameters = approximate_input_parameters

        rec_machine = inversefed.FedAvgReconstructor(model, (dm, ds), epoches, local_lr, config,
                                                    num_images = inputs.shape[0], use_updates=use_updates)

        output, stats = rec_machine.reconstruct(input_parameters, targets, img_shape=(3, 64, 64))


        test_mse = (output.detach() - inputs).pow(2).mean()
        feat_mse = (model(output.detach())- model(inputs)).pow(2).mean()  
        test_psnr = inversefed.metrics.psnr(output, inputs, factor=1/ds)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5)) 

        inputs_denorm = inputs.clone().detach()
        inputs_denorm.mul_(ds).add_(dm).clamp_(0, 1)
        axes[0].imshow(inputs_denorm[0].permute(1, 2, 0).cpu())
        axes[0].set_title('Original Input')
        axes[0].axis('off') 

        output_denorm_plot = output.clone().detach()
        output_denorm_plot.mul_(ds).add_(dm).clamp_(0, 1)
        axes[1].imshow(output_denorm_plot[0].permute(1, 2, 0).cpu())

        stats_text = (f"Rec. loss: {stats['opt']:2.4f} | MSE: {test_mse:2.4f} | "
                      f"PSNR: {test_psnr:4.2f} | FMSE: {feat_mse:2.4e}")


        axes[1].set_title(f"Reconstructed Image")
        axes[1].axis('off')
        
        if curr_bits > 0:
            fig.suptitle(f'Image Quantized to {curr_bits} bits', fontsize=16)
        else:
            fig.suptitle('Image Without Quantization', fontsize=16)

        fig.text(0.5, 0.00005, stats_text, ha='center')
        plt.tight_layout(rect=[0, 0.1, 1, 0.93])

        axes[1].axis('off') # Hide axes ticks
        rec_loss_sum_list[b] += stats['opt']
        MSE_sum_list[b] += test_mse
        PSNR_sum_list[b] += test_psnr
        FMSE_sum_list[b] += feat_mse

        print(f"Stats for image:{image_count+1}\n{bits[b]} bit quantization\n----------------------\nMean Rec Loss: {rec_loss_sum_list[b]/image_count}\nMean MSE: {MSE_sum_list[b]/image_count}\n"
        f"Mean PSNR: {PSNR_sum_list[b]/image_count}\nMean FMSE:{FMSE_sum_list[b]/image_count}\n", flush=True)

        plt.tight_layout() # Adjust subplot parameters for a tight layout

        plot_filename = os.path.join(output_plot_dir, f'reconstruction_plot_{image_count:05d}_{bits[b]:2d}bits.png')
        plt.savefig(plot_filename)
        plt.close(fig) # Close the figure to free up memory and prevent it from displaying if not desired
        # --- REMOVED plt.show() as plots are being saved ---


        # --- ADDED: Saving the reconstructed image data directly ---
        # Assuming output_denorm_plot is already denormalized
        image_filename = os.path.join(output_image_dir, f'reconstructed_image_{image_count:05d}_{bits[b]:2d}bits.png')
        save_image(output_denorm_plot, image_filename) # Saves the tensor as an image file

        
        reconstructed_count += 1
    print(f"Reconstructed batch {image_count+1}/{len(trainloader)}. Total images: {reconstructed_count}", flush=True)
    image_count +=1

print(f"\nReconstruction complete. Total quantized images:{image_count}. Total images in general:{reconstructed_count}.",flush=True)