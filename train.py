
import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

import atexit
import torch
import torch.nn as nn
from models.backbones import Backbones
from torchvision.utils import save_image
from torchvision.transforms import ToPILImage
import torch.profiler

import cv2
import numpy as np
import yaml
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, os.path.abspath('/home/pace/Poseidon/'))

from datasets.transforms.build import reverse_transforms
from models.best.PoseidonHeatMapVitPoseAttention12_vith_dropout_best_no_adaptive import Poseidon
from models.SimpleBaseline import SimpleBaseline
from datasets.zoo.posetrack.PoseTrack import PoseTrack 
from posetimation import get_cfg, update_config 
from engine.defaults import default_parse_args
from core.loss import get_loss_function
from core.optimizer import Optimizer
from utils.common import TRAIN_PHASE, VAL_PHASE, TEST_PHASE
from core.function import train, validate, train_batch_accumulation
from tqdm import tqdm
from utils.utils_save_results import *
from utils.utils_requests import *
import wandb  
import random
import sys
import torch
import numpy as np
from torch.optim.lr_scheduler import StepLR , CosineAnnealingLR

def load_config(config_path):
    """ Load the YAML configuration file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: Configuration dictionary.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def setup(args):
    cfg = get_cfg(args)
    update_config(cfg, args)
    return cfg

def main():
    # Load configuration from a YAML file.
    args = default_parse_args()
    cfg = setup(args)
    
    # set the seed
    set_seed(cfg)

    # wandb config Initialize a new run

    if cfg.SAVE_RESULTS:
        wandb.init(
            project="poseidon",

            # Set entity to specify your username or team name
            entity="krahim04",

            # Track hyperparameters and run metadata
            config=cfg,

            name=cfg.NAME_EXP
            )

    # set the phase
    phase = TRAIN_PHASE

    # set the device
    device = "cuda:" + str(cfg.GPUS[0]) if torch.cuda.is_available() else 'cpu'

    # print in red color the device
    print("\033[92m" + "Device: " + "\033[0m", device)

     # load the model
    if cfg.MODEL.METHOD == 'poseidon':
        model = Poseidon(cfg, phase=phase, device=device).to(device)
    elif cfg.MODEL.METHOD == 'simplebaseline':
        model = SimpleBaseline(cfg, phase=phase, device=device).to(device)

    # define loss function (criterion) and optimizer
    loss = get_loss_function(cfg, device)

    # define optimizer
    optimizer = Optimizer(model, cfg).get_optimizer()

     # if schedulare is used print in green color
    if cfg.TRAIN.LR_SCHEDULER:
        print("\033[92m" + "Scheduler: " + "\033[0m", cfg.TRAIN.LR_SCHEDULER)

    # Choose the scheduler based on cfg.TRAIN.LR_SCHEDULER
    if cfg.TRAIN.LR_SCHEDULER == 'StepLR':
        scheduler = StepLR(optimizer, step_size=cfg.TRAIN.LR_STEP, gamma=cfg.TRAIN.LR_FACTOR)
    elif cfg.TRAIN.LR_SCHEDULER == 'CosineAnnealingLR':
        # Parametrize T_max for CosineAnnealingLR (can adjust based on total epochs or other criteria)
        T_max = cfg.TRAIN.T_MAX if hasattr(cfg.TRAIN, 'T_MAX') else 10  # Default to 10 if not in cfg
        scheduler = CosineAnnealingLR(optimizer, T_max=T_max)
    

    # load the datasets train and val
    train_dataset = PoseTrack(cfg, phase=TRAIN_PHASE)

    val_dataset = PoseTrack(cfg, phase=VAL_PHASE)

    # load the dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.WORKERS,
        pin_memory=True,
    )

   # print in red color
    print("\033[92m" + "Train loader loaded successfully." + "\033[0m") 

    # print number of the elements in the train set
    print("Number of elements in the train set: ", len(train_dataset))

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.VAL.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=True,
    )

    # print in red color
    print("\033[92m" + "Val loader loaded successfully." + "\033[0m")

    # print number of the elements in the val set
    print("Number of elements in the val set: ", len(val_dataset))
    print("\n\n")

    best_perf = 0.0
    best_epoch = 0
    best_model = False
    results = {}

    # print start training
    print("\033[92m" + "Start training..." + "\033[0m")
    # send_start_training()
    
    # print number of epochs, learning rate, optimizer and loss
    print("Number of epochs: ", cfg.TRAIN.END_EPOCH)
    print("Optimizer: ", cfg.TRAIN.OPTIMIZER)
    print("Loss: ", cfg.LOSS.NAME)

    print()

    # Use the existing experiment directory if resuming
    if cfg.TRAIN.AUTO_RESUME:
        experiment_dir = cfg.TRAIN.EXPERIMENT_DIR  # Use the old experiment directory
        print(f"Resuming from experiment directory: {experiment_dir}")
    else:
        # Create a new output folder if not resuming
        experiment_dir = create_output_folder(cfg.OUTPUT_DIR)
        print(f"Starting new experiment, saving to: {experiment_dir}")

    # Check if resuming from a checkpoint
    if cfg.TRAIN.AUTO_RESUME:
        checkpoint_path = cfg.TRAIN.AUTO_RESUME_PATH
        model, optimizer, start_epoch = load_checkpoint(model, optimizer, checkpoint_path)
    else:
        start_epoch = cfg.TRAIN.BEGIN_EPOCH  # Start from the beginning if not resuming

    # Save the configuration file at the beginning
    if cfg.SAVE_RESULTS: 
        save_config(cfg, experiment_dir)

    # Initialize early stopping parameters
    early_stopping_patience = cfg.EARLY_STOPPING.PATIENCE if cfg.EARLY_STOPPING else 0
    early_stopping_enabled = cfg.EARLY_STOPPING and early_stopping_patience > 0
    epochs_without_improvement = 0

    if cfg.TRAIN.ACCUMULATION_STEPS > 1:
        print("\033[93m" + "Batch accumulation enabled." + "\033[0m")

    for epoch in range(start_epoch, cfg.TRAIN.END_EPOCH):
        
        print("\033[92m" + "Epoch: " + "\033[0m", epoch)
        
        if cfg.SAVE_RESULTS: 
            save_examples = (epoch == cfg.TRAIN.BEGIN_EPOCH)
        else:
            save_examples = False

        if cfg.TRAIN.ACCUMULATION_STEPS > 1:
            train_loss, train_acc = train_batch_accumulation(cfg, train_loader, model, loss, optimizer, epoch,
                output_dir=cfg.OUTPUT_DIR, device=device, experiment_dir=experiment_dir, save_examples=save_examples)
        else:
            train_loss, train_acc = train(cfg, train_loader, model, loss, optimizer, epoch,
                output_dir=cfg.OUTPUT_DIR, device=device, experiment_dir=experiment_dir, save_examples=save_examples)

        # Step the scheduler if applicable
        if cfg.TRAIN.LR_SCHEDULER == 'StepLR' or cfg.TRAIN.LR_SCHEDULER == 'CosineAnnealingLR':
            scheduler.step()

        # evaluate on validation set
        performance_values, perf_indicator, val_loss, val_acc = validate(cfg, val_loader, val_dataset, 
            model, loss, cfg.OUTPUT_DIR, epoch, device=device)

        # Save the results for this epoch
        results[epoch] = {
            'performance_values': performance_values,
            'perf_indicator': perf_indicator,
            'loss': val_loss,
        }

        # Save the results and configuration after each epoch
        if cfg.SAVE_RESULTS: 
            save_results(cfg, results, experiment_dir)

            # wandb log
            wandb.log({
                "train_loss": train_loss, 
                "train_acc": train_acc, 
                "val_loss": val_loss, 
                "val_acc": val_acc, 
                "mAP": perf_indicator,
                "learning_rate": optimizer.param_groups[0]['lr']  # Log the current learning rate
            })

            # Save the last model and the best model if applicable
            save_model(model, optimizer, epoch, experiment_dir, is_best=(perf_indicator > best_perf))

        # Save the model if it has the best performance
        if perf_indicator > best_perf:
            best_perf = perf_indicator
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= early_stopping_patience and early_stopping_enabled:
            print("\033[91m" + "Early stopping triggered." + "\033[0m")
            break

    if cfg.SAVE_RESULTS:
        send_training_complete( best_epoch, best_perf)

    print("Best performance: ", best_perf)
    print("Best epoch: ", best_epoch)

#send_training_complete()
print("\033[92m" + "Training complete." + "\033[0m")


def set_seed(config):

    # set the seed for reproducibility
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed(config.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(config.SEED)
    random.seed(config.SEED)

if __name__ == '__main__':

    # print env versioning in blue color
    print("\033[94m" + "Environment versioning:" + "\033[0m")

    # print python version
    print("Python version: ", sys.version)

    # print torch version
    print("PyTorch version: ", torch.__version__)

    # print numpy version
    print("Numpy version: ", np.__version__)

    print("\n\n")

    
    main()
