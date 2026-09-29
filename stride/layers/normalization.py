import torch
import torch.nn as nn

class Normalize(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=False, subtract_last=False, use_norm=True):
        super(Normalize, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine                
        self.subtract_last = subtract_last  
        self.use_norm = use_norm            

        if self.affine:  
            self._init_params()

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))    
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))     

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))    

        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)    
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()    
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if not self.use_norm:  
            return x

        center = self.last if self.subtract_last else self.mean
        x = (x - center) / self.stdev

        if self.affine:
            x = x * self.affine_weight + self.affine_bias

        return x

    def _denormalize(self, x):
        if not self.use_norm:
            return x

        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps * self.eps)

        center = self.last if self.subtract_last else self.mean
        return x * self.stdev + center

    def forward(self, x, mode: str):

        if mode == 'norm':
            self._get_statistics(x)  
            x = self._normalize(x)  
        elif mode == 'denorm':
            x = self._denormalize(x)  
        else:
            raise NotImplementedError  
        return x
