import torch
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()



def writeTensorBoard(writer: SummaryWriter, tag: str, scalar_val, epoch):
    writer.add_scalar(tag, scalar_value=scalar_val, global_step=epoch)

