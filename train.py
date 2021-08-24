
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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = M.MyModel()

hparams = {
    'lr' : 1e-3,
    'EPOCHs' : 1,
    'EPOCH_EVERY' : 1
}

trainProps = {
    'model' : model.to(device),
    'loss_fn' : nn.CrossEntropyLoss(),
    'optimzier' : torch.optim.Adam(model.parameters(), hparams['lr']),
    'data' : D.train_img_iter_basic
}

def func_eval(model,data_iter,device):
    with torch.no_grad():
        n_total,n_correct = 0,0
        model.eval() # evaluate (affects DropOut and BN)
        for batch_in,batch_out in tqdm(data_iter):
            y_trgt = batch_out.to(device)
            model_pred = model.forward(batch_in.to(device))
            _,y_pred = torch.max(model_pred,1)
            n_correct += (y_pred==y_trgt).sum().item()
            n_total += batch_in.size(0)
        val_accr = (n_correct/n_total)
        model.train() # back to train mode 
    return val_accr
print ("Done")

def tmp_test():
    train_accr = func_eval(trainProps['model'], trainProps['data'], device)
    print(f'train_accr : {train_accr}')

def train():
    print ("Start training.")
    _, EPOCHs, EPOCH_EVERY = hparams.values()
    
    model, loss_fn, optimizer, data = trainProps.values()

    for epoch in range(EPOCHs):
        loss_val_sum = 0
        for batch_in,batch_out in tqdm(data):
            # Forward path
            y_pred = model.forward(batch_in.to(device))
            
            loss_out = loss_fn(y_pred, batch_out.type(torch.LongTensor).to(device))
            # Update
            optimizer.zero_grad() # reset gradient 
            loss_out.backward() # backpropagate
            optimizer.step() # optimizer update
            loss_val_sum += loss_out
        loss_val_avg = loss_val_sum/len(data)
        # Print
        if ((epoch%EPOCH_EVERY)==0) or (epoch==(EPOCHs-1)):
            train_accr = func_eval(model,data,device)
            test_accr = func_eval(model,data,device)
            print ("epoch:[%d] loss:[%.3f] train_accr:[%.3f] test_accr:[%.3f]."%
                (epoch,loss_val_avg,train_accr,test_accr))
    print ("Done")


def main():
    train()
    
    


if __name__ == '__main__':
    main()