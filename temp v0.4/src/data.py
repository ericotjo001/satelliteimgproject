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
    if model=='resnet18':
        transformseq = transforms.Compose([
            transforms.ToTensor(), 
            transforms.RandomResizedCrop(256, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomRotation((-10,10)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif model=='resnet18_val':
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
            transforms.RandomResizedCrop(256, scale=(0.8, 1.0), ratio=(0.75, 1.25)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]) if transformseq is None else transformseq

        """
        We use histogram to observe how to best divide the classes of wealth values evenly
        and arrived at the BINS below. This will create 7 classes: 
        x < -1.5 (class 0)
        -1.5 <= x < -0.75 (class 1)
        and so on, including x>= 1.5 (class 6)

        """
        # self.BINS = np.linspace(-1.8,2.76,40-1) # 40 classes, see conversion()
        # self.classify = classify


    def __getitem__(self,i):        

        pil_img = Image.open(os.path.join(self.IMG_FOLDER_DIR,
            self.df.loc[i]['id']+'.png')).convert('RGB')

        # x = torch.cat((self.transf(pil_img),self.transf(infra_img))).to(torch.float)
        x = self.transf(pil_img).to(torch.float)
        # ground-truth wealth index 
        y0 = self.df.loc[i]['wealth']
        # if self.classify: # conver to classes
        #     y0 = self.conversion(y0) 
        return x,y0

    def __len__(self):
        return self.n

    # def conversion(self, x):
    #     c = 0
    #     for partition in self.BINS:
    #         if x<partition: 
    #             return c
    #         c += 1
    #     return c

# class DHSFeatures(Dataset):
#     def __init__(self, df_feature):
#         super(DHSFeatures, self).__init__()
#         self.df = df_feature
#         self.n = len(self.df)
        
#     def __getitem__(self,i):
#         y0 = self.df.loc[i][0]
#         x = self.df.loc[i][1:].to_numpy()
#         return x,y0

#     def __len__(self):
#         return self.n