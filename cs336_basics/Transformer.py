import torch
import torch.nn as nn
import torch.nn.functional as F

from Others import rms_norm, softmax
from Attention import multi_head_attention 
from FFN import poswise_ffn

class transformer_block(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, attn_pdrop, residual_pdrop):
        super(transformer_block, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.attn_pdrop = attn_pdrop
        self.residual_pdrop = residual_pdrop

        self.rms_norm1 = rms_norm(d_model)
        self.rms_norm2 = rms_norm(d_model)

        self.mha = multi_head_attention(d_model, num_heads, attn_pdrop)
        self.ffn = poswise_ffn(d_model, d_ff)
    
    def forward(self, x):
        #import pdb; pdb.set_trace()
        rms_out1 = self.rms_norm1(x)
        mha_out = self.mha(rms_out1)
        dropout_out1 = F.dropout(mha_out, self.residual_pdrop)
        sub_block1_out =  x + dropout_out1
        rms_out2 = self.rms_norm2(sub_block1_out)
        ffn_out = self.ffn(rms_out2)
        dropout_out2 = F.dropout(ffn_out, self.residual_pdrop)
        sub_block2_out = sub_block1_out + dropout_out2
        return sub_block2_out

class transformer_lm(nn.Module):
    def __init__(self, vocab_size, context_length, num_layers, d_model, num_heads, d_ff, attn_pdrop, residual_pdrop):
        super(transformer_lm, self).__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.attn_pdrop = attn_pdrop
        self.residual_pdrop = residual_pdrop

        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(context_length, d_model)
        self.layers = nn.ModuleList([transformer_block(d_model, num_heads, d_ff, attn_pdrop, residual_pdrop) for _ in range(num_layers)])
        self.norm_final = rms_norm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias = False)

    def forward(self, x):
        #import pdb; pdb.set_trace()
        token_embed = self.token_embed(x)
        seq_length = x.shape[1]
        pos = torch.arange(0, seq_length, device=x.device)
        pos = pos.unsqueeze(0).expand_as(x) #add the batch dimension. Will get broadcasted when added to token_embed
        pos_embed = self.pos_embed(pos)

        combined_embed = token_embed + pos_embed
        dropout1 = F.dropout(combined_embed, self.residual_pdrop)
        transformer_block_out = self.layers[0](dropout1)
        for i in range(1, self.num_layers):
            transformer_block_out = self.layers[i](transformer_block_out)
        
        norm_final = self.norm_final(transformer_block_out)
        head = self.head(norm_final)
        return head
