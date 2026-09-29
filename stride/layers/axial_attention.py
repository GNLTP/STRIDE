import torch
import torch.nn as nn

class AxialAttention(nn.Module):
    def __init__(self, dim=16, d_ff=32, heads=4, axis='W', dropout=0.1):

        super().__init__()
        self.axis = axis
        self.heads = heads
        self.d_ff = d_ff
        self.d_model = dim
        assert self.d_ff % self.heads == 0,  f"d_ff ({self.d_ff}) must be divisible by num_heads ({self.heads})"
        self.dim_head = self.d_ff // self.heads

        self.scale = self.dim_head ** -0.5
        self.norm = nn.LayerNorm(self.d_model)

        self.pre_proj = nn.Linear(self.d_model, self.d_ff)
        self.qkv_proj = nn.Linear(self.d_ff, self.d_ff * 3, bias=False)
        self.out_proj = nn.Linear(self.d_ff, self.d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        B,C,H,W = x.shape
        x = x.permute(0, 2, 3, 1)        

        if self.axis == 'H':    
            x = x.transpose(1, 2)   
            length = H
        elif self.axis == 'W':  
            length = W                         
        else:
            raise ValueError("Invalid axis: self.axis must be 'H' or 'W'.")

        x = self.norm(x)    
        x_shape = x.shape
        x = x.reshape(-1, x.shape[2], C)    

        x = self.pre_proj(x)                
        qkv = self.qkv_proj(x).chunk(3, dim=-1)  
        
        q, k, v = map(lambda t: t.reshape(t.shape[0], t.shape[1], self.heads, -1).transpose(1, 2), qkv)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).reshape(x.shape[0], length, self.d_ff)
        out = self.out_proj(out)  
        out = out.reshape(*x_shape)  

        if self.axis == 'H':
            out = out.transpose(1, 2)  
        out = out.permute(0, 3, 1, 2)  

        return out
