import torch
import torch.nn as nn

import sys
sys.path.insert(0, '../cs336_assn1/cs336_basics')
from GELU import gelu

class poswise_ffn(nn.Module):
    def __init__(self, w1, w2):
        super(poswise_ffn, self).__init__()
        self.w1 = w1
        self.w2 = w2
        self.gelu = gelu()

    def forward(self, x):
        #import pdb; pdb.set_trace()
        linear1 = torch.matmul(x, self.w1.T)
        gelu = self.gelu(linear1)
        linear2 = torch.matmul(gelu, self.w2.T)
        return linear2


    
    
