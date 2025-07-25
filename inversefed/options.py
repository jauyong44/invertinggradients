"""Parser options."""

import argparse

def options():
    """Construct the central argument parser, filled with useful defaults."""
    parser = argparse.ArgumentParser(description='Reconstruct some image from a trained model.')

    # Central:
    parser.add_argument('--model', default='ConvNet', type=str, help='Vision model.')
    parser.add_argument('--dataset', default='CIFAR10', type=str)
    parser.add_argument('--dtype', default='float', type=str, help='Data type used during reconstruction [Not during training!].')


    parser.add_argument('--trained_model', action='store_true', help='Use a trained model.')
    parser.add_argument('--epochs', default=120, type=int, help='If using a trained model, how many epochs was it trained?')
    parser.add_argument('--local_epoches', default = 1, type=int, help='How many epoches will the client train on their data before sending gradients to server')

    parser.add_argument('--accumulation', default=0, type=int, help='Accumulation 0 is rec. from gradient, accumulation > 0 is reconstruction from fed. averaging.')
    parser.add_argument('--num_images', default=1, type=int, help='How many images should be recovered from the given gradient.')
    parser.add_argument('--num_recoveries', default = 2, type= int, help= 'How many batches of recoveries should we do?')
    parser.add_argument('--random_seed', default=0, type=int, help='Using a random seed or not (0 is no seed)')
    parser.add_argument('--target_id', default=None, type=int, help='Cifar validation image used for reconstruction.')
    parser.add_argument('--label_flip', action='store_true', help='Dishonest server permuting weights in classification layer.')
    parser.add_argument('--local_lr', default = 1e-4, help = 'Learning rate of the client model')
    # Rec. parameters
    parser.add_argument('--optim', default='ours', type=str, help='Use our reconstruction method or the DLG method.')

    parser.add_argument('--restarts', default=1, type=int, help='How many restarts to run.')
    parser.add_argument('--iterations', default=1000, type = int, help = 'How many iterations to run for recreating an image')
    parser.add_argument('--cost_fn', default='sim', type=str, help='Choice of cost function.')
    parser.add_argument('--indices', default='def', type=str, help='Choice of indices from the parameter list.')
    parser.add_argument('--weights', default='equal', type=str, help='Weigh the parameter list differently.')

    parser.add_argument('--optimizer', default='adam', type=str, help='Weigh the parameter list differently.')
    parser.add_argument('--signed', action='store_false', help='Do not used signed gradients.')
    parser.add_argument('--boxed', action='store_false', help='Do not used box constraints.')

    parser.add_argument('--scoring_choice', default='loss', type=str, help='How to find the best image between all restarts.')
    parser.add_argument('--init', default='randn', type=str, help='Choice of image initialization.')
    parser.add_argument('--tv', default=1e-4, type=float, help='Weight of TV penalty.')
    parser.add_argument('--tv_min_value', default = 0, type = float, help = 'The smallest possible value that TV can go to.')
    parser.add_argument('--tv_scheduler_step', default = 0, type= int, help ='After how many reconstruction iterations before we multiply by the reduction_ratio')
    parser.add_argument('--tv_scaling_factor', default = 0.1, type = float, help = "The ratio we multiply TV by after a scheduler step")
    parser.add_argument('--reconstruction_lr', default = 0.1, type = float, help = 'Learning rate of the reconstruction model')
    parser.add_argument('--regularization_choice', default='tv', type= str, help='Choice of regularization')


    # Files and folders:
    parser.add_argument('--save_image', action='store_true', help='Save the output to a file.')

    parser.add_argument('--image_path', default='images/', type=str)
    parser.add_argument('--model_path', default='models/', type=str)
    parser.add_argument('--table_path', default='tables/', type=str)
    parser.add_argument('--data_path', default='~/data', type=str)
    parser.add_argument('--image_dir_name', default = 'reconstructed_images', type = str)
    parser.add_argument('--plot_dir_name', default='reconstruction_plots', type=str)

    # Debugging:
    parser.add_argument('--name', default='iv', type=str, help='Name tag for the result table and model.')
    parser.add_argument('--deterministic', action='store_true', help='Disable CUDNN non-determinism.')
    parser.add_argument('--dryrun', action='store_true', help='Run everything for just one step to test functionality.')
    return parser
