import torch
import torch.nn as nn
import torchvision

from ..soft_attention import SoftAttention


class DeepAll_VGG16(nn.Module):
    
    def __init__(self, semantic_dim=300, pretrained=True):
        super(DeepAll_VGG16, self).__init__()
        
        self.base_model = torchvision.models.vgg16_bn(pretrained)
        self.attention_layer = SoftAttention(input_dim=512)        
        self.base_model.classifier._modules['6'] = nn.Linear(4096, semantic_dim)

    def forward(self, x):
        
        features = self.base_model.features(x)
        out = self.attention_layer(features)
        
        out = out.view(out.size(0), -1)
        out = self.base_model.classifier(out)
        
        return out