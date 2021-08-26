import torch
import torch.nn as nn

from trainer.trainer import BaseTrainer
from dataset.data_loader import train_img_iter_numworker_batch
from model.modeler import getModel



class BaseExpr:
    trainer: BaseTrainer    

    def execute(self):
        if not self.trainer:
            raise NotImplementedError
        self.trainer.train()
        
class Expr_A(BaseExpr):
    def __init__(self) -> None:
        super().__init__()

        model = getModel('MyResnet34')(num_classes=18)()
        lr = 1e-2

        self.trainer = BaseTrainer(
            expr_name='expr_A',
            data_loader=train_img_iter_numworker_batch,
            model=model,
            optim=torch.optim.Adam(model.parameters(), lr=lr),
            loss_fn=nn.CrossEntropyLoss(),
            lr=lr,
            epochs=1
        )

