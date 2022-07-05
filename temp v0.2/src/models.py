
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
        self.alpha = nn.Parameter(torch.tensor(5.))
        self.tanh = nn.Tanh()

    def forward(self,x):
        return self.tanh(self.backbone(x)) * self.alpha



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
                'best_val_r2': None,
                'best_fitting_model': None,
            }
        else:
            print('loading model states...')
            ckpt = torch.load(MODEL_DIR)

        return ckpt
        

