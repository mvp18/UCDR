import os
import numpy as np

import torch
import torch.optim as optim
from torch.autograd import Function


class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        reversed_scaled_grad = torch.neg(ctx.lambda_*grad_output.clone())
        return reversed_scaled_grad, None

def grad_reverse(x, LAMBDA):
    return GradReverse.apply(x, LAMBDA)


def numeric_classes(tags_classes, dict_tags):
    num_classes = np.array([dict_tags.get(t) for t in tags_classes])
    return num_classes


def create_dict_texts(texts):
    d = {l: i for i, l in enumerate(texts)}
    return d


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def save_checkpoint(state, directory, save_name, last_chkpt):

    if not os.path.isdir(directory):
        os.makedirs(directory)
    
    checkpoint_file = os.path.join(directory, save_name+'.pth')
    torch.save(state, checkpoint_file)
    last_chkpt_file = os.path.join(directory, last_chkpt+'.pth')
    
    if os.path.isfile(last_chkpt_file):
        os.remove(last_chkpt_file)
    else:
        print("Error: {} file not found".format(last_chkpt_file))