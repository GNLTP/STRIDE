import torch
import torch.nn as nn
import torch.nn.functional as F

from stride.layers.image_backbone import ImageResNet18Light
from stride.layers.embedding import DataEmbedding
from stride.layers.axial_attention import AxialAttention
from stride.layers.period_estimation import Period_init
from stride.layers.normalization import Normalize

class AxisStripGate(nn.Module):
    def __init__(self, C: int, k: int = 3, dilation: int = 1, tau: float = 1.0):
        super().__init__()
        pad = dilation * (k // 2)
        self.tau = tau
        self.dw_h = nn.Conv1d(C, C, kernel_size=k, padding=pad, dilation=dilation, groups=C, bias=False)
        self.bn_h = nn.BatchNorm1d(C, eps=1e-3)
        self.dw_w = nn.Conv1d(C, C, kernel_size=k, padding=pad, dilation=dilation, groups=C, bias=False)
        self.bn_w = nn.BatchNorm1d(C, eps=1e-3)

    def forward(self, x_bt):

        h_strip = x_bt.mean(dim=3)                      
        w_strip = x_bt.mean(dim=2)                      
        w_h = self.bn_h(self.dw_h(h_strip))
        w_w = self.bn_w(self.dw_w(w_strip))
        w_h = F.softmax(w_h / self.tau, dim=-1)   
        w_w = F.softmax(w_w / self.tau, dim=-1)   
        return w_h.unsqueeze(-1), w_w.unsqueeze(-2)

class ChannelWeightsS(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv_s = nn.Conv1d(3, 1, kernel_size=1)

    def _branch_map(self, x: torch.Tensor) -> torch.Tensor:
        avg_s = x.mean(dim=1)                         
        std_s = x.var(dim=1, unbiased=False).sqrt()   
        max_s = x.amax(dim=1)                         

        fs = torch.stack([avg_s, std_s, max_s], dim=1)   
        ws = self.conv_s(fs)                             
        return ws  

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        B, C, T, S = x1.shape
        x1 = x1.reshape(B*C, T, S)
        x2 = x2.reshape(B*C, T, S)  

        ws1 = self._branch_map(x1)   
        ws2 = self._branch_map(x2)   

        w = torch.stack([ws1, ws2], dim=0)      
        w = torch.softmax(w, dim=0)             

        w = w.reshape(2, B, C, 1, S)               
        w = w.expand(-1, -1, -1, T, -1)         
        return w

class ImageSTRBlock(nn.Module):
    def __init__(self, configs, freq_k: int = 3):
        super().__init__()
        self.configs = configs
        self.freq_k = freq_k
        self.seq_len = configs.seq_len
        self.d_model = configs.d_model       

        self.encoder_out = 128
        self.c_mid = self.encoder_out // 2   
        self.encoder = ImageResNet18Light(
            input_channels=configs.enc_img_in,
            pretrain=getattr(configs, "pretrained_backbone", True),
            encoder_list=[True, True, False],
        )

        self.axis_gate = AxisStripGate(C=self.c_mid)
        self.chan_ts = ChannelWeightsS()
        self.temporal_conv = nn.Sequential(
            nn.Conv3d(self.encoder_out, self.c_mid, kernel_size=3, dilation=(3,1,1), padding=(3,1,1)),
            nn.BatchNorm3d(self.c_mid, eps=1e-3),
            nn.GELU()
        )

        self.data_embedding = DataEmbedding(
            c_in=self.c_mid,
            d_model=self.c_mid,
            dropout=configs.dropout_modal
        )

        self.attn_W = AxialAttention(dim=self.c_mid, heads=configs.n_heads, axis='W')
        self.attn_H = AxialAttention(dim=self.c_mid, heads=configs.n_heads, axis='H')

        self.trend_proj2d = nn.Conv2d(self.c_mid, self.d_model, kernel_size=1, bias=False)
        self.detail_proj2d = nn.Conv2d(self.c_mid, self.d_model, kernel_size=1, bias=False)

    def _extract_feat(self, x: torch.Tensor) -> torch.Tensor:
        
        B, T, C, H, W = x.shape
        x2d = x.flatten(0, 1)              
        if H != W: raise ValueError(f"[ImageSTRBlock] Expected square images, got H={H}, W={W}.")
        x2d = self.encoder(x2d)                             
        x2d = x2d.unflatten(0, (B, T)).movedim(1, 2)        
        x3d = self.temporal_conv(x2d)                       

        B,Cmid,T,H_last,W_last = x3d.shape
        
        x_bt = x3d.permute(0, 2, 1, 3, 4).reshape(B*T, Cmid, H_last, W_last)
        w_h, w_w = self.axis_gate(x_bt)     

        seqH_bt = (x_bt * w_w).sum(dim=3)   
        seqW_bt = (x_bt * w_h).sum(dim=2)   

        seqH = seqH_bt.reshape(B, T, Cmid, H_last).permute(0, 2, 1, 3).contiguous()
        seqW = seqW_bt.reshape(B, T, Cmid, W_last).permute(0, 2, 1, 3).contiguous()    

        w_ts = self.chan_ts(seqH, seqW)     
        wH_ts, wW_ts = w_ts[0], w_ts[1]     
        xs = wH_ts * seqH + wW_ts * seqW    
        return xs

    def _image_similarity(self, xs: torch.Tensor) -> torch.Tensor:
        B, C, T, S = xs.shape
        scale = S ** 0.5
        out = xs.new_zeros(B, T, C)     

        for k in range(1, T):           
            diag_up   = xs[:,:,:-k,:]   
            diag_down = xs[:,:,k: ,:]   
            
            v = ((diag_up * diag_down).sum(dim=-1) / max(scale, 1.0)).permute(0,2,1)

            out[:,:-k,:] += v
            out[:,k: ,:] += v   

        if T > 1: out = out / (T - 1)
        return out

    def _fold_by_period(self, series_bt: torch.Tensor, period_init):
        B,T,C = series_bt.shape
        outs = []
        for (L,H,W) in period_init:
            if L != H * W:
                raise RuntimeError(f"period_shape mismatch: L={L}, H*W={H * W} (H={H}, W={W})")
            if L < T:
                raise RuntimeError(f"period_init should satisfy L>=T (e.g., ceil(T/p)*p).")
            if L > T:
                pad_len = L - T
                x = torch.cat([series_bt, series_bt.new_zeros(B, pad_len, C)], dim=1)   
            else:
                x = series_bt
            outs.append(x.reshape(B,H,W,C).permute(0,3,1,2).contiguous())    
        return outs

    def forward(self, image: torch.Tensor, period_data = None):

        feat = self._extract_feat(image)      
        agg = self._image_similarity(feat)    
        if period_data is None:
            with torch.no_grad():
                period = Period_init(agg, self.freq_k)
        else:
            period = period_data
        agg_emb = self.data_embedding(agg)  
        x_list = self._fold_by_period(agg_emb, period)

        trend_list, detail_list, next_x = [], [], []
        for (LHW, x) in zip(period, x_list):
            last_t = self.attn_W(x)
            last_s = self.attn_H(x)
            trend_d = self.trend_proj2d(last_t)     
            detail_d = self.detail_proj2d(last_s)   
            trend_list.append(trend_d)
            detail_list.append(detail_d)

        return trend_list, detail_list
