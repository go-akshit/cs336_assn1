import torch
import torch.nn as nn

class rms_norm(nn.Module):
    def __init__(self, input_size, weights, epsilon = 1e-5):
        super(rms_norm, self).__init__()
        self.input_size = input_size
        self.epsilon = epsilon
        self.gains = weights

    def rms(self, x):
        temp = torch.mean(x**2, keepdim = True, dim = -1) + self.epsilon
        return torch.sqrt(temp)

    def forward(self, x):
        rms = self.rms(x)
        return (1/rms * x * self.gains)



