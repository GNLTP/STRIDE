import torch
import torch.nn as nn

class CrossAxialAttention(nn.Module):
    def __init__(self, dim=16, d_ff=64, heads=4, dropout=0.1, axis='W'):

        super().__init__()
        self.axis = axis
        self.d_model = dim
        self.d_ff = d_ff
        self.heads = heads
        assert self.d_ff % self.heads == 0, f"d_ff ({d_ff}) must be divisible by num_heads ({self.heads})"

        self.dim_head = self.d_ff // heads
        self.scale = self.dim_head ** -0.5

        self.norm_image = nn.LayerNorm(self.d_model)
        self.norm_data = nn.LayerNorm(self.d_model)

        self.pre_image = nn.Linear(self.d_model, self.d_ff)  
        self.pre_data = nn.Linear(self.d_model, self.d_ff)

        self.q_image = nn.Linear(self.d_ff, self.d_ff, bias=False)
        self.kv_image = nn.Linear(self.d_ff, self.d_ff * 2, bias=False)
        self.q_data = nn.Linear(self.d_ff, self.d_ff, bias=False)
        self.kv_data = nn.Linear(self.d_ff, self.d_ff * 2, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.out_image = nn.Linear(self.d_ff, self.d_model)
        self.out_data = nn.Linear(self.d_ff, self.d_model)

    def _attend(self, Q, K, V):
        B, H, L, D = Q.shape  
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)  
        return out.transpose(1, 2).reshape(B, L, H * D)

    def forward(self, image, data):
        B, C, H, W = image.shape
        image = image.permute(0, 2, 3, 1)  
        data = data.permute(0, 2, 3, 1)  

        if self.axis == 'H':  
            image = image.transpose(1, 2)  
            data = data.transpose(1, 2)
            length = H
        elif self.axis == 'W':  
            length = W  
        else:
            raise ValueError("Invalid axis: self.axis must be 'H' or 'W'.")

        _image = image.reshape(-1, C)
        _data = data.reshape(-1, C)
        _image = self.norm_image(_image).reshape(*image.shape).contiguous()
        _data = self.norm_data(_data).reshape(*data.shape).contiguous()

        _image = _image.reshape(-1, length, C)
        _data = _data.reshape(-1, length, C)
        _image = self.pre_image(_image)
        _data = self.pre_data(_data)

        Q_image = self.q_image(_image).reshape(-1, length, self.heads, self.dim_head).transpose(1, 2)
        K_data, V_data = self.kv_data(_data).chunk(2, dim=-1)
        K_data = K_data.reshape(-1, length, self.heads, self.dim_head).transpose(1, 2)
        V_data = V_data.reshape(-1, length, self.heads, self.dim_head).transpose(1, 2)
        Att_image2data = self._attend(Q_image, K_data, V_data)  

        Q_data = self.q_data(_data).reshape(-1, length, self.heads, self.dim_head).transpose(1, 2)
        K_image, V_image = self.kv_image(_image).chunk(2, dim=-1)
        K_image = K_image.reshape(-1, length, self.heads, self.dim_head).transpose(1, 2)
        V_image = V_image.reshape(-1, length, self.heads, self.dim_head).transpose(1, 2)
        Att_data2image = self._attend(Q_data, K_image, V_image)  

        att_image = self.out_image(Att_image2data).reshape(*image.shape)  
        att_data = self.out_data(Att_data2image).reshape(*data.shape)  

        if self.axis == 'H':  
            att_image = att_image.transpose(1, 2)  
            att_data = att_data.transpose(1, 2)

        att_image = att_image.permute(0, 3, 1, 2)       
        att_data = att_data.permute(0, 3, 1, 2)

        return att_image, att_data

class TrendDetailFusion(nn.Module):
    def __init__(self, dim=32, d_ff=64, num_heads=4, dropout=0.1, output_len=96):
        super().__init__()

        self.out_seq = output_len
        self.cross_trend = CrossAxialAttention(dim=dim, d_ff=d_ff, heads=num_heads, dropout=dropout, axis='W')
        self.cross_detail = CrossAxialAttention(dim=dim, d_ff=d_ff, heads=num_heads, dropout=dropout, axis='H')
        self.norm_trend  = nn.LayerNorm(dim)
        self.norm_detail = nn.LayerNorm(dim)

    @staticmethod
    def _space2time(x, T):
        B, C, H, W = x.shape
        y = x.permute(0, 2, 3, 1).reshape(B, H * W, C)  
        if H * W >= T:
            return y[:, :T, :].contiguous()
        pad = y.new_zeros(B, T - H * W, C)
        return torch.cat([y, pad], dim=1).contiguous()

    def _cross_block(self, image, data, cross_mod):
        att_image, att_data = cross_mod(image, data)  

        image_out  = image + att_image
        data_out = data + att_data      
        return image_out, data_out

    def forward(self, trend_image_list, trend_data_list, detail_image_list, detail_data_list):
        if not (len(trend_image_list) == len(trend_data_list) == len(detail_image_list) == len(detail_data_list)):
            raise ValueError("The input lists must have the same length")

        K = len(trend_image_list)
        trend_acc, detail_acc = None, None
        for trend_image, trend_data, detail_image, detail_data in zip(trend_image_list, trend_data_list, detail_image_list, detail_data_list):

            trend_image_out, trend_data_out = self._cross_block(trend_image, trend_data, self.cross_trend)
            detail_image_out, detail_data_out = self._cross_block(detail_image, detail_data, self.cross_detail)
            trend_out = trend_image_out + trend_data_out      
            detail_out = detail_image_out + detail_data_out   

            trend_seq  = self._space2time(trend_out,  self.out_seq)  
            detail_seq = self._space2time(detail_out, self.out_seq)  
            trend_seq  = self.norm_trend(trend_seq)
            detail_seq = self.norm_detail(detail_seq)

            if trend_acc is None:
                trend_acc, detail_acc = trend_seq, detail_seq
            else:
                trend_acc  = trend_acc + trend_seq
                detail_acc = detail_acc + detail_seq

        scale = K ** -0.5
        trend_acc  = trend_acc * scale
        detail_acc = detail_acc * scale

        return trend_acc, detail_acc
