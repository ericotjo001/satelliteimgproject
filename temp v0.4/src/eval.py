import argparse
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

DEBUG_TOGGLE = False

def eval_entry(parser:argparse.ArgumentParser) -> None:
    print('@eval_entry')

    parser.add_argument('--setup', default='DHS_RESNET50', type=str)
    args, unknown = parser.parse_known_args()

    if args.setup == 'DHS_RESNET50':
        prep_for_revalidation_dhs_resnet50(parser)
    else:
        raise NotImplementedError()    

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def prep_for_revalidation_dhs_resnet50(parser:argparse.ArgumentParser) -> None:
    from .config import default_configs
    dconf = default_configs['DHS_RESNET50']
    parser.add_argument('--PROJECT_NAME', default=dconf['PROJECT_NAME'], type=str)
    parser.add_argument('--CKPT_DIR', default=dconf['CKPT_DIR'], type=str)
    parser.add_argument('--MODEL_NAME', default=dconf['MODEL_NAME'], type=str)

    args, unknown = parser.parse_known_args()
    dargs = vars(args) # just converting the arguments to dictionary

    from .models import SaveLoadManager, ResNet50sw
    from .data import DHSdataset, get_data_transformation_sequence
    from .training import get_directories

    DIRS = get_directories(dargs)
    slm = SaveLoadManager() 

    ##################################################
    # load model here or initialize new
    ##################################################
    ckpt = slm.saveload_ResNet50sw(init_new=False, MODEL_DIR=DIRS['MODEL_DIR'])
    net = ResNet50sw()
    net.load_state_dict(ckpt['model_state_dict'])
    net = net.to(device=device)
    net.eval()
    current_epoch = ckpt['epoch']
    losses = ckpt['losses']  
    # print('best_epoch:',ckpt['best_epoch'])

    ##################################################
    # Prepare data
    ##################################################
    transformseq = get_data_transformation_sequence(model='resnet50')
    dset = DHSdataset(dataname='dhs', split='train', transformseq=transformseq)
    loader = DataLoader(dset, batch_size=1, shuffle=False, num_workers=0)

    val_dset = DHSdataset(dataname='dhs', split='val', transformseq=transformseq)
    val_loader = DataLoader(val_dset, batch_size=1, shuffle=False, num_workers=0)


    train_y, train_y0, val_y, val_y0 = [], [], [], []
    with torch.no_grad():
        for i,(x,y0) in enumerate(loader):
            x,y0 = x.to(device=device), y0.to(device=device)
            y = net(x)     
            train_y.append(y[0].item())
            train_y0.append(y0[0].item())

            if DEBUG_TOGGLE:
                if i>=10: break

        for i,(x,y0) in enumerate(val_loader):
            x,y0 = x.to(device=device), y0.to(device=device)
            y = net(x)     
            val_y.append(y[0].item())
            val_y0.append(y0[0].item())

            if DEBUG_TOGGLE:
                if i>=10: break

    pd.DataFrame({'y':train_y, 'y0':train_y0}).to_csv( DIRS['REVALIDATE_TRAIN_DIR'], index=False)
    print(DIRS['REVALIDATE_TRAIN_DIR'], 'saved!')
    pd.DataFrame({'y':val_y, 'y0':val_y0}).to_csv( DIRS['REVALIDATE_DIR'], index=False)
    print(DIRS['REVALIDATE_DIR'], 'saved!')