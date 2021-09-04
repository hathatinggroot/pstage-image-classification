
import os
import torch
import torch.nn as nn
import datetime
from pytz import timezone
from typing import Union

import model.model as m
import config as cf

def getModel(name: str) -> Union[nn.Module, m.ModelProvider]:
    loadedModel = None
    try:
        loadedModel = m.__dict__[name]
        if not (issubclass(loadedModel, nn.Module) or issubclass(loadedModel, m.ModelProvider)):
            print(f'{name} is not model from nn.Module or ModelProvider')
            return None
    except(KeyError):
        print(f'Model not exists with name : {name}')

    return loadedModel

def loadWithCheckpoint(model, checkpointName: str) -> nn.Module:
    checkpoint = torch.load(os.path.join(cf.checkpointsDir, checkpointName))
    model.load_state_dict(checkpoint)

    return model

def saveCheckpoint(model: nn.Module, checkpointName: str) -> None:
    now = '_'.join(str(datetime.datetime.now(timezone('Asia/Seoul'))).split())
    checkpointName = '_'.join([checkpointName, now, '.pt'])
    torch.save(model.state_dict(), os.path.join(cf.outModelsDir, checkpointName))

