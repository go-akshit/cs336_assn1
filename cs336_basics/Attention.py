import torch
import torch.nn as nn
import torch.nn.functional as F
from Others import softmax

def scaled_attention(q, k, v, mask, pdrop):
    #import pdb; pdb.set_trace()
    d_k = q.shape[-1]
    q_kt = torch.matmul(q, k.transpose(-2, -1))
    scores = q_kt / torch.sqrt(torch.tensor(d_k).float())

    if mask is not None:
        mask = mask* (-1e9)
        scores = scores + mask
    
    attn_weights = softmax(scores, dim = -1)
    attn_weights = F.dropout(attn_weights, p=pdrop)
    output = torch.matmul(attn_weights, v)
    return output

class multi_head_attention(nn.Module):
    def __init__(self, d_model, num_heads, attn_pdrop):
        super(multi_head_attention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.attn_pdrop = attn_pdrop
        self.q_proj = nn.Linear(self.d_model, self.d_model, bias = False)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias = False)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias = False)

        self.output_linear = nn.Linear(self.d_model, self.d_model, bias = False)
        
    def forward(self, x):
        #import pdb; pdb.set_trace()
        qw = self.q_proj(x) #dim = batch x seq_length x num_heads*d_model
        kw = self.k_proj(x) #dim = batch x seq_length x num_heads*d_model
        vw = self.v_proj(x) #dim = batch x seq_length x num_heads*d_model
        seq_length = qw.shape[1]
        batches = qw.shape[0]
        qw = qw.view(batches, seq_length, self.num_heads, self.d_model//self.num_heads)
        kw = kw.view(batches, seq_length, self.num_heads, self.d_model//self.num_heads)
        vw = vw.view(batches, seq_length, self.num_heads, self.d_model//self.num_heads)

        qw = qw.transpose(1,2)
        kw = kw.transpose(1,2)
        vw = vw.transpose(1,2)
        
        mask = torch.triu(torch.ones((seq_length, seq_length)), diagonal = 1).unsqueeze(0).unsqueeze(0)
        outputs = scaled_attention(qw, kw, vw, mask, self.attn_pdrop)
        outputs = outputs.transpose(1,2)
        outputs = outputs.contiguous()
        outputs = outputs.view(batches, seq_length, self.d_model)

        return self.output_linear(outputs)