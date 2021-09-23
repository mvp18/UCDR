import torch.nn as nn


class SoftAttention(nn.Module):
    
    def __init__(self, input_dim=2048, hidden_layer=380):
        super(SoftAttention, self).__init__()        
        
        self.net = nn.Sequential(
            nn.Conv2d(input_dim, hidden_layer, kernel_size=1), 
            nn.Conv2d(hidden_layer, 1, kernel_size=1))

    def forward(self, x):
        
        attn_mask = self.net(x)
        attn_mask = attn_mask.view(attn_mask.size(0), -1)
        attn_mask = nn.Softmax(dim=1)(attn_mask)
        attn_mask = attn_mask.view(attn_mask.size(0), 1, x.size(2), x.size(3)) 
        x_attn = x * attn_mask
        x = x + x_attn
        
        return x