import torch
import torch.nn as nn

from ..senet import se_resnet50


class CuMix_SEResNet50(nn.Module):

	def __init__(self, semantic_dim=300, pretrained='imagenet', num_tr_classes=93,):
	
		super(CuMix_SEResNet50, self).__init__()
		
		self.base_model = se_resnet50(pretrained=pretrained)
		feat_dim = self.base_model.last_linear.in_features
		self.base_model.last_linear = nn.Linear(feat_dim, semantic_dim)

	def forward(self, x):

		out = self.base_model.features(x)		
		out = self.base_model.avg_pool(out)
		if self.base_model.dropout is not None:
			out = self.base_model.dropout(out)		
		out = out.view(out.size(0), -1)
		
		return out