import argparse, os, time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import r2_score

from .models import SaveLoadManager, ResNet50sw, NNHeadResNet50sw
from .data import DHSdataset, get_data_transformation_sequence, DHSFeatures

from .config import device, DEBUG_TOGGLE


def finetune_entry(parser:argparse.ArgumentParser)->None:
    print('finetuning...')
    parser.add_argument('--setup', default='DHS_RESNET50', type=str)
    args, unknown = parser.parse_known_args()

    if args.setup == 'DHS_RESNET50':
        finetune_dhs_resnet50(parser)
    else:
        raise NotImplementedError()        

def finetune_dhs_resnet50(parser):
    from .config import default_configs
    dconf = default_configs['DHS_RESNET50']
    parser.add_argument('--PROJECT_NAME', default=dconf['PROJECT_NAME'], type=str)
    parser.add_argument('--CKPT_DIR', default=dconf['CKPT_DIR'], type=str)
    parser.add_argument('--MODEL_NAME', default=dconf['MODEL_NAME'], type=str)   

    parser.add_argument('--b1', default=128, type=int)
    parser.add_argument('--b2', default=48, type=int)
    parser.add_argument('--n_epoch', default=1, type=int)
    parser.add_argument('--learning_rate', default=1e-3, type=float)    

    parser.add_argument('--print_every', default=100, type=int)
    parser.add_argument('--validate_every', default=100, type=int)


    parser.add_argument('--fmode', default=None, type=str)


    args, unknown = parser.parse_known_args()
    dargs = vars(args) # just converting the arguments to dictionary

    from .training import get_directories
    DIRS = get_directories(dargs)
    slm = SaveLoadManager()

    print('device:',device)

    ##################################################
    # extracted features
    ##################################################
    train_features, val_features = get_extracted_features(DIRS)

    datf_train = DHSFeatures(train_features)
    train_loader = DataLoader(datf_train, batch_size=len(datf_train), shuffle=True, num_workers=0)
    fit_loader = DataLoader(datf_train, batch_size=1, shuffle=False, num_workers=0)

    datf_val = DHSFeatures(val_features)
    val_loader = DataLoader(datf_val, batch_size=1, shuffle=False, num_workers=0)


    if args.fmode == 'val':
        validation2(val_loader, train_loader, dargs, DIRS)
        exit()

    ##################################################
    # load model here or initialize new
    ##################################################

    nnhead = NNHeadResNet50sw()
    ckptnnhead = slm.saveload_NNHeadResNet50sw(init_new=False, 
        FINETUNE_DIR=DIRS['FINETUNE_DIR'])
    if ckptnnhead['model_state_dict'] is not None:
        nnhead.load_state_dict(ckptnnhead['model_state_dict'])
    else:
        print('new!')

    nnhead.to(device=device)

    optimizer = optim.Adam(nnhead.parameters(),  lr=args.learning_rate, betas=(0.99,0.99), weight_decay=1e-3)
    if ckptnnhead ['optimizer_state_dict'] is not None:
        optimizer.load_state_dict(ckptnnhead['optimizer_state_dict'])

    current_epoch = ckptnnhead['epoch']
    losses = ckptnnhead['losses']

    scheduler = ReduceLROnPlateau(optimizer, 'min')
    if ckptnnhead['scheduler_state_dict'] is not None:
        scheduler.load_state_dict(ckptnnhead['scheduler_state_dict'])

    criterion = nn.MSELoss()

    start = time.time()
    total_iter = len(train_loader)
    batch_size = args.b1* args.b2
    print('training starts!\n')
    TARGET_REACHED = False
    for j in range(args.n_epoch):
        nnhead.train()
        for i,(x,y0) in enumerate(train_loader):
            x = x.to(device=device).to(torch.float)
            y0 = y0.to(device=device).to(torch.float)
            y = nnhead(x)

            loss = criterion(y.squeeze(1),y0)
            # loss = torch.sum((y-y0)**2)
            loss.backward()

            if DEBUG_TOGGLE: break

            if (i+1)%args.print_every==0 or (i+1)==total_iter:
                update_text = f'epoch:{current_epoch} iter:{i+1}/{total_iter}'
                print('%-64s'%(update_text), end='\r')

        losses.append(loss.item())
        max_norm = 1.
        torch.nn.utils.clip_grad_norm_(nnhead.parameters(), max_norm)    
        # gradients accumulated! optimizer will update here 
        optimizer.step()
        nnhead.zero_grad()
      

        current_epoch += 1

        if (j+1)%args.validate_every==0:
            # save after every epoch
            ckptnnhead['epoch'] = current_epoch
            ckptnnhead['model_state_dict'] = nnhead.state_dict()
            ckptnnhead['optimizer_state_dict'] = optimizer.state_dict()
            ckptnnhead['losses'] = losses

            torch.save(ckptnnhead, DIRS['FINETUNE_DIR'] )
            ckptnnhead, TARGET_REACHED = validation_nnhead_dhs_resnet50(j,loss.item(), val_loader, fit_loader, dargs, DIRS)

        if not scheduler is None:
            scheduler.step(loss)

        if TARGET_REACHED:
            print('target reached...!')
            break

    end = time.time()
    elapsed = end - start
    print('\n\ntime taken %s[s] = %s [min] '%(
        str(round(elapsed,1)), str(round(elapsed/60.,1)) ))



def validation2(val_loader, train_loader, dargs, DIRS):
    from .training import get_epochwise_dir
    from .models import SaveLoadManager
    slm = SaveLoadManager()

    nnhead = NNHeadResNet50sw()
    ckptnnhead = slm.saveload_NNHeadResNet50sw(init_new=False, 
        FINETUNE_DIR=DIRS['FINETUNE_DIR'])    

    nnhead.load_state_dict(ckptnnhead['model_state_dict'])
    nnhead = nnhead.to(device=device)
    nnhead.eval()
    total_iter = len(val_loader)
    y0_val, y_val = [], []
    with torch.no_grad():
        for i,(x,y0) in enumerate(val_loader):
            x,y0 = x.to(device=device).to(torch.float), y0.to(device=device)
            y = nnhead(x)
            y = 2.1*nnhead.act(y-0.001)

            y_val.append(y[0].item())
            y0_val.append(y0[0].item())
    r2 = r2_score(y_val, y0_val)  
    print(r2)

def validation_nnhead_dhs_resnet50(j, loss, val_loader, train_loader, dargs, DIRS):
    print('\nvalidation_nnhead_dhs_resnet50')

    TARGET_REACHED = False

    from .training import get_epochwise_dir
    from .models import SaveLoadManager
    slm = SaveLoadManager()

    nnhead = NNHeadResNet50sw()
    ckptnnhead = slm.saveload_NNHeadResNet50sw(init_new=False, 
        FINETUNE_DIR=DIRS['FINETUNE_DIR'])    

    nnhead.load_state_dict(ckptnnhead['model_state_dict'])
    nnhead = nnhead.to(device=device)
    nnhead.eval()

    ##################################################
    # validation starts!
    ##################################################    
    current_epoch = ckptnnhead['epoch']

    y0_val, y_val = [], []
    with torch.no_grad():
        for i,(x,y0) in enumerate(train_loader):
            x,y0 = x.to(device=device).to(torch.float), y0.to(device=device)
            y = nnhead(x)

            y_val.append(y[0].item())
            y0_val.append(y0[0].item())
            if DEBUG_TOGGLE:
                if i>=10: break
    r2_overfit = r2_score(y_val, y0_val)

    total_iter = len(val_loader)
    y0_val, y_val = [], []
    with torch.no_grad():
        for i,(x,y0) in enumerate(val_loader):
            x,y0 = x.to(device=device).to(torch.float), y0.to(device=device)
            y = nnhead(x)

            y_val.append(y[0].item())
            y0_val.append(y0[0].item())
            if DEBUG_TOGGLE:
                if i>=10: break
    r2 = r2_score(y_val, y0_val)  

    SAVE_EPOCH_CKPT = False
    if ckptnnhead ['best_epoch'] is None:
        SAVE_EPOCH_CKPT = True
        ckptnnhead['best_val_r2'] = r2

    if r2>=0.9:
        print('r2_overfit:',r2_overfit)
        ckptnnhead['best_val_r2'] = r2   
        SAVE_EPOCH_CKPT = True
        TARGET_REACHED = True

    if r2 >= ckptnnhead['best_val_r2']:
        ckptnnhead['best_val_r2'] = r2                
        SAVE_EPOCH_CKPT = True
    
    update_text = f'loss:{loss} r2:{r2_overfit} r2_val:{r2}' 
    print(update_text)

    if SAVE_EPOCH_CKPT:
        print('saving epochwise checkpoint at epoch %s'%(str(current_epoch)))
        ckptnnhead ['best_epoch'] = current_epoch 
        torch.save(ckptnnhead, DIRS['FINETUNE_DIR'] )
        torch.save(ckptnnhead, get_epochwise_dir(current_epoch, DIRS['FINETUNE_DIR']) )

    return ckptnnhead, TARGET_REACHED


def get_extracted_features(DIRS):
    if not (os.path.exists(DIRS['EXTRACTED_FEATURES_DIR.train'])
        and os.path.exists(DIRS['EXTRACTED_FEATURES_DIR.val'])):
        save_extracted_features_to_csv(DIRS)

    if not os.path.exists(DIRS['EXTRACTED_FEATURES_DIR.aug']):
        save_augmented_features_to_csv(DIRS)

    train_features = pd.read_csv(DIRS['EXTRACTED_FEATURES_DIR.train'])
    # aug_features = pd.read_csv(DIRS['EXTRACTED_FEATURES_DIR.aug'])
    val_features = pd.read_csv(DIRS['EXTRACTED_FEATURES_DIR.val'])
    return train_features, val_features

def save_extracted_features_to_csv(DIRS):
    slm = SaveLoadManager()
    ckpttemp = slm.saveload_ResNet50sw(init_new=False, MODEL_DIR=DIRS['MODEL_DIR'])

    temp = DIRS['MODEL_DIR'].find('.pth')
    bestepoch = ckpttemp ['best_epoch']
    BEST_MODEL_DIR = DIRS['MODEL_DIR'][:temp] + f'.{bestepoch}.pth'

    ckpt = slm.saveload_ResNet50sw(init_new=False, MODEL_DIR=BEST_MODEL_DIR)
    net = ResNet50sw()

    net.load_state_dict(ckpt['model_state_dict'])
    net.to(device=device)
    net.eval()    

    fitting_dset = DHSdataset(dataname='dhs', split='train', 
        transformseq=get_data_transformation_sequence(model='resnet50_val'))
    fitting_loader = DataLoader(fitting_dset, batch_size=1, shuffle=False, num_workers=0)

    val_dset = DHSdataset(dataname='dhs', split='val', 
        transformseq=get_data_transformation_sequence(model='resnet50_val'))
    val_loader = DataLoader(val_dset, batch_size=1, shuffle=False, num_workers=0)

    def get_df(loader):
        colname = ['y0'] + [f'x{i}' for i in range(40)]
        df = pd.DataFrame({},columns=colname)
        for i,(x,y0) in enumerate(loader):
            x,y0 = x.to(device=device).to(torch.float), y0.to(device=device)
            y_feature = net(x)

            newrow = [y0.item()] + y_feature[0].clone().detach().cpu().numpy().tolist()
            newrow = pd.DataFrame([newrow], columns=colname )
            df = pd.concat([df, newrow],join="inner" , ignore_index=True)
            if DEBUG_TOGGLE:    
                if i>=20: break
        return df

    print('preparing %s...'%(str(DIRS['EXTRACTED_FEATURES_DIR.train'])))
    get_df(fitting_loader).to_csv(DIRS['EXTRACTED_FEATURES_DIR.train'], index=False)
    print('preparing %s...'%(str(DIRS['EXTRACTED_FEATURES_DIR.val'])))
    get_df(val_loader).to_csv(DIRS['EXTRACTED_FEATURES_DIR.val'], index=False)

