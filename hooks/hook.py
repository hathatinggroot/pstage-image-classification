import os
import datetime
import torch
import torch.nn as nn
from pytz import timezone
from tqdm.auto import tqdm
from sklearn.metrics import f1_score
from torch.utils.tensorboard import SummaryWriter



import config as cf
import model.modeler as modeler
from data.data_loader import kfold_valid_iter


def log(title: str, msg: str):
    filePath = os.path.join(cf.outLogsDir, f'log_{title}.txt')
    with open(filePath, mode='a+', encoding='utf-8') as f:
        now = datetime.datetime.now(timezone('Asia/Seoul'))
        f.writelines(f'\n[{now}]\t{msg}')

# train hooks
def defaultAfterEpochHook(trainer, epoch, **kwargs):
    PrintEvery = 1
    # kwargs
    y_true = kwargs['y_true'].cpu()
    y_pred = kwargs['y_pred'].cpu()
    loss_val_avg = kwargs['loss_val_avg']
    tb_writer = kwargs['tb_writer']

    if ((epoch%PrintEvery)==0) or (epoch==(trainer.epochs-1)):
        train_accr = func_eval(trainer.model, trainer.data_loader, trainer.device)
        
        valid_accr = func_eval(trainer.model, kfold_valid_iter, trainer.device)
        f1 = f1_score(y_true, y_pred, average='macro')
        msg = "epoch:[%d] loss:[%.3f] train_accr:[%.3f] valid_accr:[%.3f] f1_score:[%.3f]"%(epoch, loss_val_avg, train_accr, valid_accr, f1)
        
        if tb_writer:
            tb_writer.add_scalar(f'{trainer.expr_name}/Loss/train', loss_val_avg, epoch)
            tb_writer.add_scalar(f'{trainer.expr_name}/Acc/train', train_accr, epoch)
            tb_writer.add_scalar(f'{trainer.expr_name}/Acc/valid', valid_accr, epoch)
            tb_writer.add_scalar(f'{trainer.expr_name}/F1Score/train', f1, epoch)
            tb_writer.flush()

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
            
        val_accr = (n_correct/n_total)
        model.train() # back to train mode 
    print ("Eval End.... ")
    return val_accr