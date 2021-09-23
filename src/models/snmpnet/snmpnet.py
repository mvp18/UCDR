import torch
import torch.nn as nn

from ..senet import se_resnet50
from ..soft_attention import SoftAttention


class SnMpNet(nn.Module):

	def __init__(self, semantic_dim=300, pretrained='imagenet', num_tr_classes=93):
	
		super(SnMpNet, self).__init__()
		
		self.base_model = se_resnet50(pretrained=pretrained)
		feat_dim = self.base_model.last_linear.in_features
		self.base_model.last_linear = nn.Linear(feat_dim, semantic_dim)

		self.attention_layer = SoftAttention(input_dim=feat_dim)
		self.ratio_predictor = nn.Linear(feat_dim, num_tr_classes)

	def forward(self, x):

		features = self.base_model.features(x)
		feat_attn = self.attention_layer(features)
		
		out = self.base_model.avg_pool(feat_attn)
		if self.base_model.dropout is not None:
			out = self.base_model.dropout(out)
		
		feat_final = out.view(out.size(0), -1)
		mixup_ratio = self.ratio_predictor(feat_final)
		
		return mixup_ratio, feat_final