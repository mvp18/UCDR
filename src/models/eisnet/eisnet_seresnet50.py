import torch
import torch.nn as nn

from .senet import se_resnet50
from ..soft_attention import SoftAttention


class Normalize(nn.Module):

	def __init__(self, power=2):
		super(Normalize, self).__init__()
		self.power = power

	def forward(self, x):
		norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
		out = x.div(norm)
		return out


class EISNet_SEResNet50(nn.Module):

	def __init__(self, semantic_dim=300, jigsaw_classes=31, pretrained='imagenet'):
	
		super(EISNet_SEResNet50, self).__init__()
		
		self.base_model = se_resnet50(pretrained=pretrained)
		feat_dim = self.base_model.last_linear.in_features
		self.base_model.last_linear = nn.Linear(feat_dim, semantic_dim)

		self.jigsaw_classifier = nn.Linear(feat_dim, jigsaw_classes)
		self.embedding = nn.Sequential(nn.Linear(feat_dim, 128))
		self.l2norm = Normalize(2)

	def forward(self, x):

		features = self.base_model.features(x)
		
		out = self.base_model.avg_pool(features)
		if self.base_model.dropout is not None:
			out = self.base_model.dropout(out)
		
		out = out.view(out.size(0), -1)
		
		return self.base_model.last_linear(out), self.jigsaw_classifier(out), self.l2norm(self.embedding(out))