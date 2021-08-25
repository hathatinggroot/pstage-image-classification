
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

from metric.metric import log, save_model



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# model = M.MyModel()
model = M.get_model()

hparams = {
    'lr' : 1e-3,
    'EPOCHs' : 10,
    'EPOCH_EVERY' : 1
}

trainProps = {
    'model' : model.to(device),
    'loss_fn' : nn.CrossEntropyLoss(),
    'optimzier' : torch.optim.Adam(model.parameters(), hparams['lr']),
    'data' : D.train_img_iter_numworker_batch,
    'name' : '4_expr'
}

def func_eval(model,data_iter,device):
    print ("Eval Start.... ")
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
    print ("Eval End.... ")
    return val_accr


def train():
    print ("Start training.")
    _, EPOCHs, EPOCH_EVERY = hparams.values()
    
    model, loss_fn, optimizer, data, expr_name = trainProps.values()
    model.train()

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
            save_model(model)
    
    print ("Done")


def main():
    train()
    # log('tester', 'this is test')
    # save_model()

    
    
    


if __name__ == '__main__':
    main()