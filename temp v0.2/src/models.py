
""" Note:
sw: satellite, wealth. This denotes our overarching objective, which is
    to use deep learning model to predict wealth index
"""

import os
import torch
import torch.nn as nn
import torchvision.models as mod


class ResNet50sw(nn.Module):
    def __init__(self) -> None :
        super(ResNet50sw, self).__init__()
        self.backbone = mod.resnet50(weights=mod.ResNet50_Weights.DEFAULT, progress=False) # the ImageNet pretraiend model!
        self.fc = nn.Linear(1000,1) # wealth index is a scalar, so output is 1 dim. 

    def forward(self,x):
        return self.fc(self.backbone(x)).squeeze(1)        


class SaveLoadManager():
    @staticmethod
    def saveload_ResNet50sw(init_new:bool = True, 
        MODEL_DIR:str='checkpoint/project1/ResNet50sw.pth')->dict:
        """
        initnew: bool. True: not using MODEL_DIR to load. If False, always
          try to load from MODEL_DIR. Init new if it is not found.
        MODEL_DIR: str. Where to save and load our model.
        """

        if init_new or not os.path.exists(MODEL_DIR):
            print('initializing new model...')
            ckpt = {
                'epoch':0,
                'losses': [],
                'model_state_dict':None,
                'optimizer_state_dict':None,
                
                'best_epoch': None,
                'best_val_loss': None,
            }
        else:
            print('loading model states...')
            ckpt = torch.load(MODEL_DIR)

        return ckpt
        

