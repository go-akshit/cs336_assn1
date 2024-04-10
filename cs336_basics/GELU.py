import torch
import torch.nn as nn

class gelu(nn.Module):
    def __init__(self, ):
        super(gelu, self).__init__()
    
    def forward(self, x):
        temp = 1 + torch.erf(x/torch.sqrt(torch.tensor(2.0)))
        return 0.5 * x * temp
    
