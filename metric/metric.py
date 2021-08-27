import torch
import torch.optim as optim

# loss functions

def f1_loss(inputs, targets):
    _, y_hat = torch.max(inputs, 1)
    