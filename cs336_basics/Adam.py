import torch
import torch.nn as nn

class adam(torch.optim.Optimizer):
    def __init__(self, params, lr, weight_decay, betas, eps):
        defaults = {"lr": lr, "weight_decay": weight_decay, "betas": betas, "eps": eps}
        super(adam, self).__init__(params, defaults)
    
    def step(self, closure = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                alpha = group['lr']
                weight_decay = group['weight_decay']
                beta1, beta2 = group['betas']
                eps = group['eps']

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)
                state['step'] += 1
                m = state['m']
                v = state['v']
                m = beta1*m + (1-beta1)*grad
                v = beta2*v + (1-beta2)*grad**2
                state['m'] = m
                state['v'] = v
                alpha_t = alpha * torch.sqrt(torch.tensor(1-beta2**state['step']))/(1-beta1**state['step'])
                p.data = p.data - alpha_t*m/(torch.sqrt(v) + group['eps'])
                p.data = p.data - alpha*group['weight_decay']*p.data
        return loss