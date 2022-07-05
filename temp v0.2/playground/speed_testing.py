"""
WARNING: for debugging only.
PLEASE ADD COMMENTS AND TYPESET PROPERLY LATER.
"""

import os, time
import pandas as pd
import numpy as np
from PIL import Image

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn

import torchvision.models as mod
from src.pytorch_pretrained_vit.model import ViT

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DHSdataset(Dataset):
    def __init__(self, dataname='dhs', split='train', transformseq=None):
        super(DHSdataset, self).__init__()
        self.IMG_FOLDER_DIR = os.path.join('data','imgs',f'{dataname}_{split}')
        self.CSV_FOLDER_DIR = os.path.join('data','extracted',f'{dataname}_wealth_index_{split}.csv')
        self.n = len(os.listdir(self.IMG_FOLDER_DIR))

        self.df = pd.read_csv(self.CSV_FOLDER_DIR, index_col=False)
        assert(self.n==len(self.df['id'].tolist()))

        self.transf = transforms.Compose([
            transforms.ToTensor(),transforms.RandomResizedCrop(384, scale=(0.8, 1.0), ratio=(0.75, 1.25)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]) if transformseq is None else transformseq

    def __getitem__(self,i):        
        pil_img = Image.open(os.path.join(self.IMG_FOLDER_DIR,
            self.df.loc[i]['id']+'.png')).convert('RGB')
        x = self.transf(np.array(pil_img)/255.).to(torch.float)

        y0 = self.df.loc[i]['wealth'] # ground-truth wealth index
        return x,y0

    def __len__(self):
        return self.n
        

def speedtest(parser):
    args, unknown = parser.parse_known_args()

    print('speedtest...')
    print(device)

    b1, b2 = 2,16
    if args.submode == 'dataonlyspeedtest':
        """ ===This is tested on RTX3060 GPU===
        With transformations it took only about 58 secs! Data loading doesn't seem to be any problem.
        
        for b1,b2=4,8
        dataonlyspeedtest
         time taken 64.9[s] = 1.1 [min] = 0.0 [hr]
        number of iteration: N_total=1084    

        for b1,b2=16,2
        dataonlyspeedtest
         time taken 78.0[s] = 1.3 [min] = 0.0 [hr]
        number of iteration: N_total=271  

        for b1,b2=2,16
        dataonlyspeedtest
         time taken 49.0[s] = 0.8 [min] = 0.0 [hr]
        number of iteration: N_total=2168
        """
        dataonlyspeedtest(b1, b2)
    elif args.submode == 'partialspeedtest':
        """ ===This is tested on RTX3060 GPU===

        for b1,b2=2,16 (so batch size is 32), we get:
        partialspeedtest...model:pretrained_resnet50
         time taken 5.3[s] = 0.1 [min]
        number of iteration: 16

        In other words, ONE batch update step takes 5.3 seconds. n_iter=16 
        >> 1 epoch will take N_total/n_iter * 5.3 seconds = 12 mins <<
        
        """
        partialspeedtest(b1, b2)
    elif args.submode == 'partialspeedtest_ViT':
        """ ===This is tested on RTX3060 GPU===
        b1,b2 = 4,8 already not enough memory!
        b1,b2 = 2,16

        Loaded pretrained weights.
         time taken 6.7[s] = 0.1 [min]
        number of iteration: 16

        In other words, ONE batch update step takes 6.7 seconds. n_iter=16 
        >> 1 epoch will take N_total/n_iter * 6.7 seconds = 15 mins <<
        """
        partialspeedtest(b1, b2, model='pretrained_ViT_B16')
    else:        
        raise NotImplementedError()

def dataonlyspeedtest(b1, b2):
    print('dataonlyspeedtest')
    dset = DHSdataset()
    loader = DataLoader(dset, batch_size=b1, shuffle=True, num_workers=0)

    start = time.time()
    for i,(x,y0) in enumerate(loader):
        # x,y0 = dset.__getitem__(i)
        # print(x.shape, torch.max(x), torch.min(x))

        if (i+1)%b2==0:
            # gradients accumulated! optimizer will update here 
            pass
    end = time.time()
    elapsed = end - start
    print(' time taken %s[s] = %s [min] = %s [hr]'%(str(round(elapsed,1)), str(round(elapsed/60.,1)), str(round(elapsed/3600.,1))))
    print('number of iteration:',i+1)


class ResNet50x(nn.Module):
    def __init__(self, ):
        super(ResNet50x, self).__init__()
        self.backbone = mod.resnet50(pretrained=True, progress=False)
        self.fc = nn.Linear(1000,1)
        
    def forward(self,x):
        return self.fc(self.backbone(x)).squeeze(1)

class ViTx(nn.Module):
    def __init__(self, ):
        super(ViTx, self).__init__()
        self.backbone = ViT('B_16_imagenet1k',pretrained=True)        
        self.fc = nn.Linear(1000,1)
        
    def forward(self,x):
        return self.fc(self.backbone(x)).squeeze(1)

def partialspeedtest(b1, b2, model='pretrained_resnet50'):
    print('partialspeedtest...model:%s'%(str(model)))
    dset = DHSdataset()
    if model=='pretrained_resnet50':
        net = ResNet50x()
        transformseq = transforms.Compose([
            transforms.ToTensor(), 
            transforms.RandomResizedCrop(384, scale=(0.8, 1.0), ratio=(0.75, 1.25)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif model =='pretrained_ViT_B16':
        net = ViTx()
        transformseq = transforms.Compose([
            transforms.ToTensor(), 
            transforms.RandomResizedCrop(384, scale=(0.8, 1.0), ratio=(0.75, 1.25)),
            transforms.Normalize(0.5,0.5),
        ])
    else:
        raise NotImplementedError()

    net = net.to(device=device)
    dset.transf = transformseq
    loader = DataLoader(dset, batch_size=b1, shuffle=True, num_workers=0)
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.5,0.999))

    start = time.time()
    for i,(x,y0) in enumerate(loader):
        # x,y0 = dset.__getitem__(i)
        # print(x.shape, torch.max(x), torch.min(x))
        x,y0 = x.to(device=device), y0.to(device=device)
        y = net(x)
        # print(x.shape, y.shape, y0.shape)

        loss = torch.sum((y-y0)**2)
        loss.backward()
        # print(loss)
        # raise Exception('g')

        if (i+1)%b2==0:
            # gradients accumulated! optimizer will update here 
            optimizer.step()
            net.zero_grad()
            break # ONE STEP DONE

    end = time.time()
    elapsed = end - start
    print(' time taken %s[s] = %s [min] '%(str(round(elapsed,1)), str(round(elapsed/60.,1)) ))
    print('number of iteration:',i+1)