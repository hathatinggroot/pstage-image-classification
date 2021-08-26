import os
import datetime
import torch
import torch.nn as nn
from pytz import timezone

logs_dir = '/opt/ml/code/out/logs'
model_dir = '/opt/ml/code/out/models'



def log(title, msg):
    now = datetime.datetime.now(timezone('Asia/Seoul'))
    filePath = os.path.join(logs_dir, f'log_{title}.txt')
    
    if(os.path.exists(filePath)):
        f = open(filePath, mode='at', encoding='utf-8')
    else:
        f = open(filePath, mode='wt', encoding='utf-8')

    f.writelines(f'\n[{now}]\t{msg}')    
    f.close()
        
def save_model(model: nn.Module, cat=None):
    now = '_'.join(str(datetime.datetime.now(timezone('Asia/Seoul'))).split())
    if cat:
        modelPath = os.path.join(model_dir, cat, f'{model.__class__.__name__}_{now}.pt')
    else:
        modelPath = os.path.join(model_dir, f'{model.__class__.__name__}_{now}.pt')

    f = open(modelPath, mode='wt')
    f.close()

    torch.save(model.state_dict(), modelPath)

