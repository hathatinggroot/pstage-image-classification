import os
import datetime
import torch
import torch.nn as nn

logs_dir = '/opt/ml/code/out/logs'
model_dir = '/opt/ml/code/out/models'



def log(title, msg):
    now = datetime.datetime.now()
    filePath = os.path.join(logs_dir, f'log_{title}.txt')
    
    if(os.path.exists(filePath)):
        f = open(filePath, mode='at', encoding='utf-8')
    else:
        f = open(filePath, mode='wt', encoding='utf-8')

    f.writelines(f'\n[{now}]\t{msg}')    
    f.close()
        
def save_model(model: nn.Module):
    now = '_'.join(str(datetime.datetime.now()).split())
    modelPath = os.path.join(model_dir, f'{model.__class__.__name__}_{now}.pt')

    f = open(modelPath, mode='wt')
    f.close()

    torch.save(model.state_dict(), modelPath)

