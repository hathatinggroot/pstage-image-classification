import os
import datetime
import torch
import torch.nn as nn
from pytz import timezone
from tqdm.auto import tqdm

import config as cf
import model.modeler as modeler

def log(title: str, msg: str):
    filePath = os.path.join(cf.outLogsDir, f'log_{title}.txt')
    with open(filePath, mode='a+', encoding='utf-8') as f:
        now = datetime.datetime.now(timezone('Asia/Seoul'))
        f.writelines(f'\n[{now}]\t{msg}')

# train hooks
def defaultAfterEpochHook(trainer, epoch, **kwargs):
    PrintEvery = 1
    if ((epoch%PrintEvery)==0) or (epoch==(trainer.epochs-1)):
        train_accr = func_eval(trainer.model,trainer.data_loader, trainer.device)
        
        # FIXME
        # test_accr = func_eval(model,data,device)
        test_accr = 0.01
        
        msg = "epoch:[%d] loss:[%.3f] train_accr:[%.3f] test_accr:[%.3f]."%(epoch, kwargs['loss_val_avg'], train_accr, test_accr)
        log(trainer.expr_name, msg)
        modeler.saveCheckpoint(trainer.model, trainer.expr_name)

def defaultOptimizerHook(optimizer, loss_out):
    optimizer.zero_grad()
    loss_out.backward()
    optimizer.step()

def func_eval(model,data_iter,device):
    print ("Eval Start.... ")
    with torch.no_grad():
        n_total,n_correct = 0,0
        model.eval() # evaluate (affects DropOut and BN)
        for i, (batch_in,batch_out) in enumerate(tqdm(data_iter)):
            y_trgt = batch_out.to(device)
            model_pred = model.forward(batch_in.to(device))
            _,y_pred = torch.max(model_pred,1)
            n_correct += (y_pred==y_trgt).sum().item()
            n_total += batch_in.size(0)

            if i ==2:
                break
            
        val_accr = (n_correct/n_total)
        model.train() # back to train mode 
    print ("Eval End.... ")
    return val_accr