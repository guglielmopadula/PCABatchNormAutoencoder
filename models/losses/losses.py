
import torch.nn.functional as F
import numpy as np
import torch
def L2_loss(x_hat, x):
    loss=F.mse_loss(x.reshape(-1), x_hat.reshape(-1), reduction="none")
    loss=loss.mean()
    return loss


