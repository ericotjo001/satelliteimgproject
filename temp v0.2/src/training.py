import argparse, os, time

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

DEBUG_TOGGLE = False # just to be safe during debug
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def training_entry(parser:argparse.ArgumentParser) -> None:
    print('@training_entry')

    parser.add_argument('--setup', default='DHS_RESNET50', type=str)
    args, unknown = parser.parse_known_args()

    if args.setup == 'DHS_RESNET50':
        train_dhs_resnet50(parser)
    else:
        raise NotImplementedError()
    
def get_directories(dargs:dict )->dict:
    if not os.path.exists(dargs['CKPT_DIR']):
        print('creating dir:',dargs['CKPT_DIR'])
        os.makedirs(dargs['CKPT_DIR'],exist_ok=True)

    PROJECT_DIR = os.path.join(dargs['CKPT_DIR'],dargs['PROJECT_NAME'])
    if not os.path.exists(PROJECT_DIR):
        print('creating dir', PROJECT_DIR)
        os.makedirs(PROJECT_DIR, exist_ok=True)

    MODEL_DIR = os.path.join(PROJECT_DIR, dargs['MODEL_NAME'])

    DIRS = {
        'CKPT_DIR': dargs['CKPT_DIR'],
        'PROJECT_DIR': PROJECT_DIR,
        'MODEL_DIR': MODEL_DIR,        
    }
    return DIRS

def train_dhs_resnet50(parser:argparse.ArgumentParser) -> None:
    print('train_dhs_resnet50...')

    from .config import default_configs
    dconf = default_configs['DHS_RESNET50']
    parser.add_argument('--PROJECT_NAME', default=dconf['PROJECT_NAME'], type=str)
    parser.add_argument('--CKPT_DIR', default=dconf['CKPT_DIR'], type=str)
    parser.add_argument('--MODEL_NAME', default=dconf['MODEL_NAME'], type=str)

    # batch size is b1*b2. 
    # b1 is the max batch size per iteration. If your GPU memory is too small for large b1,
    #   you can reduce b1 and increase b2 to train with larger batch size.
    parser.add_argument('--b1', default=2, type=int)
    parser.add_argument('--b2', default=16, type=int)
    parser.add_argument('--n_epoch', default=1, type=int)

    parser.add_argument('--print_every', default=50, type=int)

    args, unknown = parser.parse_known_args()
    dargs = vars(args) # just converting the arguments to dictionary

    from .models import SaveLoadManager, ResNet50sw
    from .data import DHSdataset, get_data_transformation_sequence

    DIRS = get_directories(dargs)
    slm = SaveLoadManager()
    
    print('device:',device)

    ##################################################
    # load model here or initialize new
    ##################################################
    ckpt = slm.saveload_ResNet50sw(init_new=False, MODEL_DIR=DIRS['MODEL_DIR'])
    net = ResNet50sw()
    if ckpt['model_state_dict'] is not None:
        net.load_state_dict(ckpt['model_state_dict'])
    net = net.to(device=device)
    optimizer = optim.Adam(net.parameters(),  lr=0.001, betas=(0.5,0.999), weight_decay=1e-5)
    if ckpt['optimizer_state_dict'] is not None:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    current_epoch = ckpt['epoch']
    losses = ckpt['losses']

    ##################################################
    # Prepare data
    ##################################################
    transformseq = get_data_transformation_sequence(model='resnet50')
    dset = DHSdataset(dataname='dhs', split='train', transformseq=transformseq)
    loader = DataLoader(dset, batch_size=args.b1, shuffle=True, num_workers=0)

    val_dset = DHSdataset(dataname='dhs', split='val', transformseq=transformseq)
    val_loader = DataLoader(val_dset, batch_size=1, shuffle=False, num_workers=0)

    ##########################
    # training
    ##########################
    start = time.time()
    total_iter = len(loader)
    batch_size = args.b1* args.b2
    print('training starts!\n')
    for j in range(args.n_epoch):
        for i,(x,y0) in enumerate(loader):
            x,y0 = x.to(device=device), y0.to(device=device)
            y = net(x)

            loss = torch.sum((y-y0)**2)/batch_size
            loss.backward()
            losses.append(loss.item())

            if (i+1)%args.b2==0:
                # gradients accumulated! optimizer will update here 
                optimizer.step()
                net.zero_grad()

            if (i+1)%args.print_every==0 or (i+1)==total_iter:
                update_text = f'epoch:{current_epoch} iter:{i+1}/{total_iter}'
                print('%-64s'%(update_text), end='\r')

                if DEBUG_TOGGLE:
                    break

        # for the last bit of data
        if (i+1)%args.b2>0:
            optimizer.step()
            net.zero_grad()            

        current_epoch += 1
        
        # save after every epoch
        ckpt['epoch'] = current_epoch
        ckpt['model_state_dict'] = net.state_dict()
        ckpt['optimizer_state_dict'] = optimizer.state_dict()
        ckpt['losses'] = losses
        torch.save(ckpt, DIRS['MODEL_DIR'] )
        ckpt = validation(val_loader, dargs, DIRS)

    end = time.time()
    elapsed = end - start
    print('\n\ntime taken %s[s] = %s [min] '%(str(round(elapsed,1)), str(round(elapsed/60.,1)) ))

def validation(val_loader: DataLoader, dargs:dict, DIRS:dict) -> dict:
    print('\nvalidation()')
    from .models import SaveLoadManager
    slm = SaveLoadManager()
    assert(os.path.exists(DIRS['MODEL_DIR']))

    ##################################################
    # load model here or initialize new
    ##################################################
    from .models import SaveLoadManager, ResNet50sw
    ckpt = slm.saveload_ResNet50sw(init_new=False, MODEL_DIR=DIRS['MODEL_DIR'])
    net = ResNet50sw()
    net.load_state_dict(ckpt['model_state_dict'])
    net = net.to(device=device)
    net.eval()

    ##################################################
    # validation starts!
    ##################################################    
    current_epoch = ckpt['epoch']
    total_iter = len(val_loader)
    val_loss = 0
    with torch.no_grad():
        for i,(x,y0) in enumerate(val_loader):
            x,y0 = x.to(device=device), y0.to(device=device)
            y = net(x)
            loss = (y-y0)**2
            val_loss += loss.item()

            if DEBUG_TOGGLE:
                if i>=10: break

    loss = loss/total_iter        

    SAVE_EPOCH_CKPT = False
    if ckpt['best_epoch'] is None:
        SAVE_EPOCH_CKPT = True
    elif loss.item() <= ckpt['best_val_loss']:
        SAVE_EPOCH_CKPT = True

    if SAVE_EPOCH_CKPT:
        print('saving epochwise checkpoint at epoch %s\n  loss:%s'%(str(current_epoch),str(loss.item())))
        ckpt['best_epoch'] = current_epoch 
        ckpt['best_val_loss'] = loss.item()       
        torch.save(ckpt, DIRS['MODEL_DIR'] )
        torch.save(ckpt, get_epochwise_dir(current_epoch, DIRS['MODEL_DIR']) )

    print()
    return ckpt
        

def get_epochwise_dir(epoch:int, MODEL_DIR:str)->str:
    """
    for convention, MODEL_NAME extension is .pth
    """
    tempindex = MODEL_DIR.find('.pth')
    return MODEL_DIR[:tempindex] + '.' + str(epoch) + '.pth'
