import torch
from torch.cuda.amp.grad_scaler import GradScaler
import torch.nn as nn
from tqdm.auto import tqdm
from torch.cuda.amp import autocast
from torch.utils.tensorboard import SummaryWriter

from hooks.hook import log, defaultAfterEpochHook, defaultOptimizerHook

class BaseTrainer:
    def __init__(self, expr_name: str, data_loader, model: nn.Module, optim, loss_fn, lr, epochs=5, optimizerHook=defaultOptimizerHook, afterEpochHook=defaultAfterEpochHook, tensorboard:bool=False) -> None:
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
        # monitoring
        self.tensorboard = tensorboard

    def train(self):
        log(self.expr_name, f'[{self.expr_name}]Start training....')
        log(self.expr_name, self.info())

        self.model.train()
        tb_writer = None
        if self.tensorboard:
            tb_writer = SummaryWriter()
        
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

            loss_val_avg = loss_val_sum / len(self.data_loader)
            
            # hook
            self.afterEpochHook(self, epoch, loss_val_avg=loss_val_avg, tb_writer=tb_writer)
            
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

class AMPTrainer(BaseTrainer):
    def __init__(self, expr_name: str, data_loader, model: nn.Module, optim, loss_fn, lr, epochs=5, optimizerHook=None, afterEpochHook=defaultAfterEpochHook, tensorboard:bool=False) -> None:
        super().__init__(expr_name, data_loader, model, optim, loss_fn, lr, epochs=epochs, optimizerHook=optimizerHook, afterEpochHook=afterEpochHook, tensorboard=tensorboard)
        self.scaler = GradScaler()

    def train(self):
        log(self.expr_name, f'[{self.expr_name}]Start training....')
        log(self.expr_name, self.info())

        self.model.train()
        # tensorboard
        tb_writer = None
        if self.tensorboard:
            tb_writer = SummaryWriter()

        for epoch in range(self.epochs):
            loss_val_sum = 0
            for i, (inputs, labels) in enumerate(tqdm(self.data_loader)):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                with autocast():
                    logits = self.model(inputs)
                    _, y_hat = torch.max(logits, 1)
                    loss_out = self.loss_fn(logits, labels.type(torch.LongTensor).to(self.device))                
                
                
                self.optim.zero_grad()
                self.scaler.scale(loss_out).backward()
                self.scaler.step(self.optim)
                self.scaler.update()

                loss_val_sum += loss_out

            loss_val_avg = loss_val_sum / len(self.data_loader)
            
            # hook
            if self.afterEpochHook:
                self.afterEpochHook(self, epoch, loss_val_avg=loss_val_avg, y_pred=y_hat, y_true=labels, tb_writer=tb_writer)
        
        if tb_writer:
            tb_writer.close()
        log(self.expr_name, f"[{self.expr_name}]End training....")