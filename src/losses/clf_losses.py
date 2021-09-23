import torch
import torch.nn as nn
import torch.nn.functional as F


class EMSLoss(nn.Module):
    
    def __init__(self, m=4):
        super(EMSLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.m = m

    def forward(self, inputs, targets):
        
        mmatrix = torch.ones_like(inputs)
        for ii in range(inputs.size()[0]):
            mmatrix[ii, int(targets[ii])]=self.m
            
        inputs_m = torch.mul(inputs, mmatrix)
        
        return self.criterion(inputs_m, targets)


class SoftCrossEntropy(nn.Module):
    
    def __init__(self):
        super(SoftCrossEntropy, self).__init__()
        return

    def forward(self, input_logits, target_logits, mask_pos=None, mask=None):
        """
        :param input_logits: prediction logits
        :param target_logits: target logits
        :return: loss
        """
        log_likelihood = - F.log_softmax(input_logits, dim=1)
        
        if mask_pos is not None:
            target_logits = target_logits + mask_pos
        
        if mask is None:
            sample_num, class_num = target_logits.shape
            loss = torch.sum(torch.mul(log_likelihood, F.softmax(target_logits, dim=1)))/sample_num
        else:
            sample_num = torch.sum(mask)
            loss = torch.sum(torch.mul(torch.mul(log_likelihood, F.softmax(target_logits, dim=1)), mask))/sample_num

        return loss