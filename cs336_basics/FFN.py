import torch
import torch.nn as nn

import sys
sys.path.insert(0, '../cs336_assn1/cs336_basics')
from Others import gelu

class poswise_ffn(nn.Module):
    def __init__(self, d_model, dff):
        super(poswise_ffn, self).__init__()
        self.gelu = gelu()
        self.d_model = d_model
        self.dff = dff
        self.linear1 = nn.Linear(in_features = self.d_model, out_features = self.dff, bias = False) 
        self.linear2 = nn.Linear(in_features = self.dff, out_features = self.d_model, bias = False)
    
    def forward(self, x):
        # import pdb; pdb.set_trace()
        linear1 = self.linear1(x)
        gelu = self.gelu(linear1)
        linear2 = self.linear2(gelu)
        return linear2


    
    
