import torch
import torch.nn as nn

from .senet import cse_resnet50_hashing


class EMSLayer(nn.Module):
	def __init__(self, num_classes, hashing_dim):
		super(EMSLayer, self).__init__()
		self.cpars = torch.nn.Parameter(torch.randn(num_classes, hashing_dim))
		self.relu = torch.nn.ReLU(inplace=True)
		
	def forward(self, x):
		out = pairwise_distances(x, self.cpars)
		out = - self.relu(out).sqrt()
		return out
		
	
def pairwise_distances(x, y=None):
	'''
	Input: x is a Nxd matrix
		   y is an optional Mxd matirx
	Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
			if y is not given then use 'y=x'.
	i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
	'''
	x_norm = (x**2).sum(1).view(-1, 1)
	if y is not None:
		y_norm = (y**2).sum(1).view(1, -1)
	else:
		y = x
		y_norm = x_norm.view(1, -1)

	dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
	return dist


class CSEResnetModel_KDHashing(nn.Module):
    
    def __init__(self, num_classes, ems, hashing_dim, pretrained=True):
        super(CSEResnetModel_KDHashing, self).__init__()
        
        if pretrained:
            self.original_model = cse_resnet50_hashing(hashing_dim)
        else:
            self.original_model = cse_resnet50_hashing(hashing_dim, pretrained=None)
        
        if ems:
            self.linear = EMSLayer(num_classes, hashing_dim)
        else:
            self.linear = nn.Linear(in_features=hashing_dim, out_features=num_classes)

                    
    def forward(self, x, y):
        out_o = self.original_model.features(x, y)
        out_o = self.original_model.hashing(out_o)
        
        out = self.linear(out_o)
        out_kd = self.original_model.logits(out_o)

        return out_o, out, out_kd