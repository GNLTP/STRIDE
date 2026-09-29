import torch
import torch.nn as nn
import torch.nn.functional as F

from stride.layers.axial_attention import AxialAttention
from stride.layers.trend_detail import MultiScaleTrendMixing2D, MultiScaleDetailMixing2D

class DataSTRBlock(nn.Module):
    def __init__(self, configs):
        super(DataSTRBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.k = configs.top_k
        self.down_sampling_layers = configs.down_sampling_layers  

        self.downsamplers = nn.ModuleList([
            nn.AvgPool2d(kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
            for _ in range(self.down_sampling_layers)
        ])

        self.attn_W = AxialAttention(dim=configs.d_model, heads=configs.n_heads, axis='W')  
        self.attn_H = AxialAttention(dim=configs.d_model, heads=configs.n_heads, axis='H')

        self.mixing_multi_scale_detail = MultiScaleDetailMixing2D(configs)
        self.mixing_multi_scale_trend  = MultiScaleTrendMixing2D(configs)

        self.layer_norm = nn.LayerNorm(configs.d_model)

    @staticmethod
    def _fold2d_pad(series_bt: torch.Tensor, period_shape):
        B, L, C = series_bt.shape
        L_period, H, W = period_shape

        assert L_period == H * W, f"Invalid period grid: L_period={L_period}, H*W={H*W}"

        if L < L_period:
            pad_len = L_period - L
            pad = series_bt.new_zeros(B, pad_len, C)
            series_bt = torch.cat([series_bt, pad], dim=1)
        elif L > L_period:
            raise RuntimeError(
                f"Unexpected: L({L}) > L_period({L_period}). "
                f"Period grid must be computed from the same sequence length or use a shared period strategy."
            )

        series_2d = series_bt.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return series_2d

    def forward(self, x_list, period_init):

        next_x, detail_list, trend_list = [], [], []
        for i, series_bt in enumerate(x_list):
            B, L, N = series_bt.shape
            series_2d = self._fold2d_pad(series_bt, period_init[i])  
            out = series_2d
            detail_list_tmp, trend_list_tmp = [], []

            for ds in self.downsamplers:
                # Phase (W) is Trend; Cycle (H) is Detail.
                trend = self.attn_W(out)
                detail = self.attn_H(out)
                detail_list_tmp.append(detail)
                trend_list_tmp.append(trend)
                out = ds(out)

            # Preserve original module keys for checkpoint compatibility.
            trend_outputs_res = self.mixing_multi_scale_detail(trend_list_tmp)
            detail_outputs_res = self.mixing_multi_scale_trend(detail_list_tmp)
            fin_trend = trend_outputs_res[-1]
            fin_detail = detail_outputs_res[0]

            trend = F.interpolate(fin_trend, size=fin_detail.shape[2:], mode='bilinear', align_corners=False)
            detail = fin_detail

            detail_list.append(detail)
            trend_list.append(trend)

            fused = detail + trend      
            fused = self.layer_norm(fused.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)      
            fused = series_2d + fused

            fused = fused.permute(0, 2, 3, 1).reshape(B, -1, N)
            fused = fused[:, :self.seq_len, :]
            next_x.append(fused)

        return next_x, detail_list, trend_list
