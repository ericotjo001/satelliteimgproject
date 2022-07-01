import os
import pandas as pd
import numpy as np
from PIL import Image

import torch
from torchvision import transforms
from torch.utils.data import Dataset

def get_data_transformation_sequence(model):
    if model=='resnet50':
        transformseq = transforms.Compose([
            transforms.ToTensor(), 
            transforms.RandomResizedCrop(384, scale=(0.8, 1.0), ratio=(0.75, 1.25)),
            transforms.RandomRotation((0,360)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        raise NotImplementedError()

class DHSdataset(Dataset):
    def __init__(self, dataname='dhs', split='train', transformseq=None):
        super(DHSdataset, self).__init__()
        self.IMG_FOLDER_DIR = os.path.join('data','imgs',f'{dataname}_{split}')
        self.CSV_FOLDER_DIR = os.path.join('data','extracted',f'{dataname}_wealth_index_{split}.csv')
        self.n = len(os.listdir(self.IMG_FOLDER_DIR))

        self.df = pd.read_csv(self.CSV_FOLDER_DIR, index_col=False)
        assert(self.n==len(self.df['id'].tolist()))

        self.transf = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomResizedCrop(384, scale=(0.8, 1.0), ratio=(0.75, 1.25)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]) if transformseq is None else transformseq

    def __getitem__(self,i):        
        pil_img = Image.open(os.path.join(self.IMG_FOLDER_DIR,
            self.df.loc[i]['id']+'.png')).convert('RGB')
        x = self.transf(np.array(pil_img)/255.).to(torch.float)

        y0 = self.df.loc[i]['wealth'] # ground-truth wealth index
        return x,y0

    def __len__(self):
        return self.n
        