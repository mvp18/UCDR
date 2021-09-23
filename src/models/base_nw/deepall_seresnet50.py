import torch
import torch.nn as nn

from ..senet import se_resnet50
from ..soft_attention import SoftAttention


class DeepAll_SEResNet50(nn.Module):

	def __init__(self, semantic_dim=300, pretrained='imagenet'):
	
		super(DeepAll_SEResNet50, self).__init__()
		
		self.base_model = se_resnet50(pretrained=pretrained)
		feat_dim = self.base_model.last_linear.in_features
		self.base_model.last_linear = nn.Linear(feat_dim, semantic_dim)

		self.attention_layer = SoftAttention(input_dim=feat_dim)

	def forward(self, x):

		features = self.base_model.features(x)
		feat_attn = self.attention_layer(features)
		
		out = self.base_model.avg_pool(feat_attn)
		if self.base_model.dropout is not None:
			out = self.base_model.dropout(out)
		
		out = out.view(out.size(0), -1)
		out = self.base_model.last_linear(out)
		
		return out