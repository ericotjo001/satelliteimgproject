import os, joblib
import pandas as pd
import numpy as np
from PIL import Image

import cv2
from matplotlib import pyplot as plt
from skimage.morphology import disk
from skimage.filters.rank import entropy


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
    elif model=='resnet50_val':
        transformseq = transforms.Compose([transforms.ToTensor(), 
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])        
    else:
        raise NotImplementedError()

class DHSdataset(Dataset):
    def __init__(self, dataname='dhs', split='train', transformseq=None, classify=False):
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

        """
        We use histogram to observe how to best divide the classes of wealth values evenly
        and arrived at the BINS below. This will create 7 classes: 
        x < -1.5 (class 0)
        -1.5 <= x < -0.75 (class 1)
        and so on, including x>= 1.5 (class 6)

        """
        self.BINS = [-1.5,-0.75,-0.25,0.25,0.75,1.5]
        self.classify = classify


        ENT_DIR = f'data/entropy_{dataname}_{split}.npy'
        if not os.path.exists(ENT_DIR):
            ent = []
            diskrad = 7
            from .config import DEBUG_TOGGLE
            for i in range(self.n):
                pil_img = Image.open(os.path.join(self.IMG_FOLDER_DIR,
                    self.df.loc[i]['id']+'.png')).convert('RGB')
                ent_img = entropy(cv2.cvtColor(np.array(pil_img),cv2.COLOR_RGB2GRAY), 
                    disk(diskrad))/diskrad
                ent.append(ent_img)

                if (i+1)%100==0:
                    print(i+1)
                if DEBUG_TOGGLE:
                    if i>=100: break
            joblib.dump(ent,ENT_DIR)
            print('done saving entropy at %s...'%(ENT_DIR))
        
        self.ent = joblib.load(ENT_DIR)
        print('done getting entropy at %s'%(str(ENT_DIR)))

    def __getitem__(self,i):        
        pil_img = Image.open(os.path.join(self.IMG_FOLDER_DIR,
            self.df.loc[i]['id']+'.png')).convert('RGB')
        x = np.array(pil_img)/255.
        x[:,:,2] = self.ent[i] 
        x = self.transf(x).to(torch.float)

        # ground-truth wealth index 
        y0 = self.df.loc[i]['wealth']
        if self.classify: # conver to classes
            y0 = self.conversion(y0) 
        return x,y0

    def __len__(self):
        return self.n

    def conversion(self, x):
        c = 0
        for partition in self.BINS:
            if x<partition: 
                return c
            c += 1
        return c
        