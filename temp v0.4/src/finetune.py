import argparse, os, time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import r2_score

# from .regression import get_fitting_data, fit_models
from .models import SaveLoadManager, ResNet18sw
from .data import DHSdataset, get_data_transformation_sequence

from .config import device, DEBUG_TOGGLE
from .training import get_directories


def finetune_entry(parser):
    parser.add_argument('--setup', default='DHS_RESNET18', type=str)
    args, unknown = parser.parse_known_args()

    if args.setup == 'DHS_RESNET18':
        finetune_dhs_resnet18(parser)
    else:
        raise NotImplementedError()

def finetune_dhs_resnet18(parser):
    from .config import default_configs
    dconf = default_configs['DHS_RESNET18']
    parser.add_argument('--PROJECT_NAME', default=dconf['PROJECT_NAME'], type=str)
    parser.add_argument('--CKPT_DIR', default=dconf['CKPT_DIR'], type=str)
    parser.add_argument('--MODEL_NAME', default=dconf['MODEL_NAME'], type=str)

    args, unknown = parser.parse_known_args()
    dargs = vars(args) # just converting the arguments to dictionary

    DIRS = get_directories(dargs)
    slm = SaveLoadManager()
    
    print('device:',device)