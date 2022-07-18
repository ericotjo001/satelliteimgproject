import torch
DEBUG_TOGGLE = False # just to be safe during debug
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

default_configs = {
    'DHS_RESNET18': {
        'PROJECT_NAME': 'dhs_resnet18_0001',
        'CKPT_DIR': 'checkpoint',
        'MODEL_NAME': 'resnet18sw.pth',
    }
}