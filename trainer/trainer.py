
import os
from typing import Counter
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from torchvision.transforms import Resize, ToTensor, Normalize

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

import model.model as M
import dataset.data_loader as D


from hooks.hook import log, defaultAfterEpochHook, defaultOptimizerHook

class BaseTrainer:
    def __init__(self, expr_name: str, data_loader, model: nn.Module, optim, loss_fn, lr, epochs=5, optimizerHook=defaultOptimizerHook, afterEpochHook=defaultAfterEpochHook) -> None:
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.expr_name = expr_name
        self.data_loader = data_loader
        self.model = model.to(self.device)
        self.optim = optim
        self.loss_fn = loss_fn
        self.lr = lr
        self.epochs = epochs
        # hooks
        self.optimizerHook = optimizerHook
        self.afterEpochHook = afterEpochHook

    def train(self):
        log(self.expr_name, f'[{self.expr_name}]Start training....')
        log(self.expr_name, self.info())

        self.model.train()
        
        for epoch in range(self.epochs):
            loss_val_sum = 0
            for i, (inputs, labels) in enumerate(tqdm(self.data_loader)):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(inputs)
                _, y_hat = torch.max(logits, 1)
                loss_out = self.loss_fn(logits, labels.type(torch.LongTensor).to(self.device))

                # hook
                self.optimizerHook(self.optim, loss_out)

                loss_val_sum += loss_out
                if i == 2:
                    break

            loss_val_avg = loss_val_sum / len(self.data_loader)
            
            # hook
            if self.afterEpochHook:
                self.afterEpochHook(self, epoch, loss_val_avg=loss_val_avg)
            
        log(self.expr_name, f"[{self.expr_name}]End training....")

    def info(self):
        info = f'\n[{self.expr_name} Info]'
        info += f'\n\t - DataLoader : {self.data_loader.__class__.__name__}'
        info += f'\n\t - Model : {self.model.__class__.__name__}'
        info += f'\n\t - Optimizer : {self.optim.__class__.__name__}'
        info += f'\n\t - Loss_fn : {self.loss_fn.__class__.__name__}'
        info += f'\n\t - LearningRate : {self.lr}'
        info += f'\n\t - Epochs : {self.epochs}'
        info += '\n' + '-'*10
        return info

    





device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# model = M.MyModel()
# model = M.get_model(3)

hparams = {
    'lr' : 1e-2,
    'EPOCHs' : 10,
    'EPOCH_EVERY' : 1
}

trainProps = {
    # 'model' : model.to(device),
    'loss_fn' : nn.CrossEntropyLoss(),
    'optimzier' : torch.optim.Adam,
    # 'data' : D.mask_train_img_iter_numworker_batch
}

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


def train(expr_name, data=D.train_img_iter_numworker_batch, cat=None, pretrained=None):
    print ("Start training.")
    lr, EPOCHs, EPOCH_EVERY = hparams.values()
    
    # _, loss_fn, optimizer, _ = trainProps.values()
    loss_fn, optimizer = trainProps.values()
    model = M.get_model(cat).to(device)
    if pretrained:
        model.load_state_dict(pretrained)

    model.train()
    optimizer = optimizer(model.parameters(), lr=lr)

    for epoch in range(EPOCHs):
        loss_val_sum = 0
        for i, (batch_in,batch_out) in enumerate(tqdm(data)):
            # Forward path
            batch_in = batch_in.to(device)
            batch_out = batch_out.to(device)

            logits = model(batch_in)
            _, y_pred = torch.max(logits, 1)
            loss_out = loss_fn(logits, batch_out.type(torch.LongTensor).to(device))
            
            # Update
            optimizer.zero_grad() # reset gradient 
            loss_out.backward() # backpropagate
            optimizer.step() # optimizer update
            loss_val_sum += loss_out
    
        loss_val_avg = loss_val_sum/len(data)
        # Print
        if ((epoch%EPOCH_EVERY)==0) or (epoch==(EPOCHs-1)):
            train_accr = func_eval(model,data,device)
            # test_accr = func_eval(model,data,device)
            test_accr = 0.01
            msg = "epoch:[%d] loss:[%.3f] train_accr:[%.3f] test_accr:[%.3f]."%(epoch,loss_val_avg,train_accr,test_accr)
            log(expr_name, msg)
            print (msg)
            save_model(model, cat)
    
    print ("Done")


def main():
    # train('expr_7_mask', D.mask_train_img_iter_numworker_batch, 'mask')
    model_dir = '/opt/ml/code/out/models'
    age_pretrained = torch.load(os.path.join(model_dir, 'age', 'ResNet_2021-08-25_14:26:58.313473.pt')) 
    train('expr_7_age', D.age_train_img_iter_numworker_batch, 'age')
    gen_pretrained = torch.load(os.path.join(model_dir, 'gender', 'ResNet_2021-08-25_15:13:27.042853.pt')) 
    train('expr_7_gender', D.gender_train_img_iter_numworker_batch, 'gender')
    
    # log('tester', 'this is test')
    # save_model()

    
    
    


if __name__ == '__main__':
    main()