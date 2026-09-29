import math
import torch
import torch.nn as nn

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad = False  
        position = torch.arange(0, max_len).float().unsqueeze(1)            
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-(math.log(10000.0) / d_model)))            
        pe[:, 0::2] = torch.sin(position * div_term)                        
        pe[:, 1::2] = torch.cos(position * div_term)                        
        self.register_buffer('pe', pe.unsqueeze(0))                         

    def forward(self, x):               
        return self.pe[:, :x.size(1)]   

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super().__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(c_in, d_model, kernel_size=3,
                                   padding=padding, padding_mode='circular', bias=False)
        nn.init.kaiming_normal_(self.tokenConv.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):                           
        x = x.permute(0, 2, 1).contiguous()         
        x = self.tokenConv(x).transpose(1, 2)       
        return x

class DataEmbedding(nn.Module):

    def __init__(self, c_in, d_model, dropout=0.1):
        super().__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        v = self.value_embedding(x)                  
        p = self.position_embedding(v)               
        out = v + p

        return self.dropout(out)
