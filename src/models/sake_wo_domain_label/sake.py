import torch
import torch.nn as nn

from ..senet import se_resnet50


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


class Student(nn.Module):

	def __init__(self, num_classes=93, ems=True, hashing_dim=512, pretrained='imagenet'):
	
		super(Student, self).__init__()
		
		self.base_model = se_resnet50(pretrained=pretrained)
		
		feat_dim = self.base_model.last_linear.in_features
		num_imagenet_classes = self.base_model.last_linear.out_features
		self.hashing_layer = nn.Linear(feat_dim, hashing_dim)
		self.base_model.last_linear = nn.Linear(hashing_dim, num_imagenet_classes)
		
		if ems:
			self.linear_o = EMSLayer(num_classes, hashing_dim)
		else:
			self.linear_o = nn.Linear(hashing_dim, num_classes)

	def forward(self, x):

		features = self.base_model.features(x)
		
		out = self.base_model.avg_pool(features)
		if self.base_model.dropout is not None:
			out = self.base_model.dropout(out)
		out = out.view(out.size(0), -1)
		out = self.hashing_layer(out)

		return out, self.linear_o(out), self.base_model.last_linear(out)


class Teacher(nn.Module):

	def __init__(self, pretrained='imagenet'):
	
		super(Teacher, self).__init__()
		
		self.base_model = se_resnet50(pretrained=pretrained)

	def forward(self, x):
		return self.base_model(x)