from data.data_set import MergedTrainDataSet
import torch
import torch.nn as nn

from trainer.trainer import AMPTrainer, BaseTrainer
from data.data_loader import train_img_iter_numworker_batch, kfold_train_iter
from model.modeler import getModel, loadWithCheckpoint
from metric.metric import FocalLoss, F1_Loss



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

class Expr_AMP_Basic(BaseExpr):
    def __init__(self) -> None:
        super().__init__()

        model = getModel('MyResnet50')(num_classes=18)()
        lr = 1e-3

        self.trainer = AMPTrainer(
            expr_name='Expr_AMP_Basic',
            data_loader=train_img_iter_numworker_batch,
            model=model,
            optim=torch.optim.Adam(model.parameters(), lr=lr),
            loss_fn=nn.CrossEntropyLoss(),
            lr=lr,
            epochs=5
        )

class Expr_AMP_Basic_More(BaseExpr):
    def __init__(self) -> None:
        super().__init__()
        
        modelFrame = getModel('MyResnet50')(num_classes=18)()
        model = loadWithCheckpoint(modelFrame, 'Expr_AMP_Basic_2021-08-29_20:40:21.200216+09:00_.pt')
        lr = 1e-3

        self.trainer = AMPTrainer(
            expr_name='Expr_AMP_Basic_More',
            data_loader=train_img_iter_numworker_batch,
            model=model,
            optim=torch.optim.Adam(model.parameters(), lr=lr),
            loss_fn=nn.CrossEntropyLoss(),
            lr=lr,
            epochs=5
        )


class Expr_Focal_Loss(BaseExpr):
    def __init__(self) -> None:
        super().__init__()

        model = getModel('MyResnet50')(num_classes=18)()
        lr = 1e-3

        self.trainer = AMPTrainer(
            expr_name='Expr_Focal_Loss',
            data_loader=train_img_iter_numworker_batch,
            model=model,
            optim=torch.optim.Adam(model.parameters(), lr=lr),
            loss_fn=FocalLoss(),
            lr=lr,
            epochs=5
        )


class Expr_Focal_Loss_Relay(BaseExpr):
    def __init__(self) -> None:
        super().__init__()

        modelFrame = getModel('MyResnet50')(num_classes=18)()
        model = loadWithCheckpoint(modelFrame, 'Expr_Focal_Loss_2021-08-29_22:35:27.846395+09:00_.pt')
        
        lr = 1e-3

        self.trainer = AMPTrainer(
            expr_name='Expr_Focal_Loss_Relay',
            data_loader=train_img_iter_numworker_batch,
            model=model,
            optim=torch.optim.Adam(model.parameters(), lr=lr),
            loss_fn=FocalLoss(),
            lr=lr,
            epochs=15
        )


class Expr_F1_Loss(BaseExpr):
    def __init__(self) -> None:
        super().__init__()

        model = getModel('MyResnet50')(num_classes=18)()
        lr = 1e-3

        self.trainer = AMPTrainer(
            expr_name='Expr_F1_Loss',
            data_loader=train_img_iter_numworker_batch,
            model=model,
            optim=torch.optim.Adam(model.parameters(), lr=lr),
            loss_fn=F1_Loss().cuda(),
            lr=lr,
            epochs=5
        )


class Expr_Focal_Loss_Folded(BaseExpr):
    def __init__(self) -> None:
        super().__init__()

        model = getModel('MyResnet50')(num_classes=18)()
        lr = 1e-3
        
        self.trainer = AMPTrainer(
            expr_name='Expr_Focal_Loss_Folded',
            data_loader=kfold_train_iter,
            model=model,
            optim=torch.optim.Adam(model.parameters(), lr=lr),
            loss_fn=FocalLoss(),
            lr=lr,
            epochs=10
        )



class Expr_Focal_Loss_Folded_Relay(BaseExpr):
    def __init__(self) -> None:
        super().__init__()

        modelFrame = getModel('MyResnet50')(num_classes=18)()
        model = loadWithCheckpoint(modelFrame, 'Expr_Focal_Loss_Folded_2021-08-30_15:17:11.568602+09:00_.pt')
        
        lr = 1e-3
        
        self.trainer = AMPTrainer(
            expr_name='Expr_Focal_Loss_Folded_Relay',
            data_loader=kfold_train_iter,
            model=model,
            optim=torch.optim.Adam(model.parameters(), lr=lr),
            loss_fn=FocalLoss(),
            lr=lr,
            epochs=10,
            tensorboard=True
        )