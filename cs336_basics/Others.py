import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class gelu(nn.Module):
    def __init__(self, ):
        super(gelu, self).__init__()
    
    def forward(self, x):
        temp = 1 + torch.erf(x/torch.sqrt(torch.tensor(2.0)))
        return 0.5 * x * temp
    
class rms_norm(nn.Module):
    def __init__(self, input_size, epsilon = 1e-5):
        super(rms_norm, self).__init__()
        self.input_size = input_size
        self.epsilon = epsilon
        self.gains = nn.Parameter(torch.ones(input_size))

    def rms(self, x):
        temp = torch.mean(x**2, keepdim = True, dim = -1) + self.epsilon
        return torch.sqrt(temp)

    def forward(self, x):
        rms = self.rms(x)
        return (1/rms * x * self.gains)
    
def softmax(input, dim):
    #import pdb; pdb.set_trace()
    max = torch.max(input, dim = dim, keepdim = True)[0]
    temp = input - max
    num = torch.exp(temp)
    den = torch.sum(num, dim = dim, keepdim = True)
    return num/den

def log_softmax(input, dim):
    max = torch.max(input, dim = dim, keepdim = True)[0]
    temp = input - max
    num = torch.exp(temp)
    den = torch.sum(num, dim = dim, keepdim = True)
    return temp - torch.log(den)
    

def cross_entropy(logits, targets):
    log_probs = log_softmax(logits, dim=-1)
    target_log_probs = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    loss = -target_log_probs
    return loss.mean()

def learning_rate_schedule(t, alpha_max, alpha_min, t_w, t_c):
    
    alpha_t = None
    if t < t_w:
        alpha_t = (t/t_w)*alpha_max
    elif (t >= t_w and t <= t_c):
        #import pdb; pdb.set_trace()
        temp = math.cos(math.pi*(t-t_w)/(t_c - t_w))
        alpha_t = alpha_min + 0.5*(1 + temp)*(alpha_max - alpha_min)
    else: 
        alpha_t = alpha_min
    return alpha_t

def gradient_clipping(parameters, max_l2_norm, eps=1e-6):
    for p in parameters:
        l2_norm = torch.linalg.norm(p.grad.data)
        if l2_norm < max_l2_norm:
            continue
        else: 
            p.grad.data = p.grad.data - max_l2_norm/(l2_norm + eps)

def data_loading(x, batch_size, context_length, device):
    num_samples = len(x) - context_length
    
    # Randomly select starting indices for the sequences
    start_indices = np.random.choice(num_samples, batch_size, replace=False)
    
    # Initialize tensors for inputs and targets
    input_sequences = torch.zeros((batch_size, context_length), dtype=torch.long)
    next_token_targets = torch.zeros((batch_size, context_length), dtype=torch.long)
    
    # Construct input and target tensors
    for i, start_idx in enumerate(start_indices):
        end_idx = start_idx + context_length
        input_sequences[i] = torch.tensor(x[start_idx:end_idx], dtype=torch.long)
        next_token_targets[i] = torch.tensor(x[start_idx+1:end_idx+1], dtype=torch.long)
    
    # Move tensors to the specified device
    input_sequences = input_sequences.to(device)
    next_token_targets = next_token_targets.to(device)
    
    return input_sequences, next_token_targets

def save_checkpoint(model, optimizer, iteration, out):
    checkpoint = {'model_params': model.state_dict(), 
                  'optimizer_state': optimizer.state_dict(),
                  'iteration': iteration}
    torch.save(checkpoint, out)

def load_checkpoint(src, model, optimizer):
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model_params'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    return checkpoint['iteration']

