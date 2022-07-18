
""" Note:
sw: satellite, wealth. This denotes our overarching objective, which is
    to use deep learning model to predict wealth index
"""

import os
import torch
import torch.nn as nn
import torchvision.models as mod

""" Note:
net = mod.resnet18(weights=mod.ResNet18_Weights.IMAGENET1K_V1, progress=False) # old version
print(net.fc.weight.data.shape )
print(net.fc.bias.data.shape )

torch.Size([1000, 2048])
torch.Size([1000])
"""

class ResNet18sw(nn.Module):
    def __init__(self) -> None :
        super(ResNet18sw, self).__init__()
        self.backbone = mod.resnet18(weights=mod.ResNet18_Weights.DEFAULT, progress=False) # the ImageNet pretraiend model!
        self.fc = nn.Linear(1000,1,bias=True)

    def get_features(self,x):
        x = self.backbone(x)
        return x

    def forward(self,x):
        return self.fc(self.get_features(x)).squeeze(1)


class NNHeadResNet18sw(nn.Module):
    def __init__(self, ):
        super(NNHeadResNet18sw, self).__init__()
        self.fc = nn.Linear(1000,1,bias=True)         

    def forward(self,x):
        return self.out(self.act(self.fc(x)))



class SaveLoadManager():
    @staticmethod
    def saveload_ResNet18sw(init_new:bool = True, 
        MODEL_DIR:str='checkpoint/project1/ResNet18sw.pth')->dict:
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
                'scheduler_state_dict': None, # initiated during training
                
                'best_epoch': None,
                'best_val_r2': None,
                'best_fitting_model': None,
            }
        else:
            print('loading model states...')
            ckpt = torch.load(MODEL_DIR)

        return ckpt
        
    @staticmethod
    def saveload_NNHeadResNet18sw(init_new=True,
        FINETUNE_DIR:str='checkpoint/project1/nnhead.pth'):

        if init_new or not os.path.exists(FINETUNE_DIR):
            print('initializing new nn head...')
            ckpt = {
                'epoch': 0,
                'losses': [],
                'model_state_dict': None,
                'optimizer_state_dict': None,
                'scheduler_state_dict': None,

                'best_epoch': None,
                'best_val_r2': None,
            }
        else:
            print('loading nn head model states...')
            ckpt = torch.load(FINETUNE_DIR)

        return ckpt