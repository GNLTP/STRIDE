import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleDetailMixing2D(nn.Module):
    def __init__(self, configs):
        super(MultiScaleDetailMixing2D, self).__init__() 

        self.downsamplers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels=configs.d_model,                
                    out_channels=configs.d_ff,                  
                    kernel_size=(3, 3),                         
                    stride=(configs.down_sampling_window, 1),   
                    padding=(1, 1)                              
                ),
                
                nn.GELU(),
                
                nn.Conv2d(
                    in_channels=configs.d_ff,                   
                    out_channels=configs.d_model,               
                    kernel_size=(1, 1),                         
                    stride=1,                                   
                    padding=0                                   
                )
            )
            for _ in range(configs.down_sampling_layers - 1)    
        ])

    def forward(self, detail_list):
        
        detail_list_copy = detail_list.copy()
        origin_high = detail_list_copy[0]
        out_high= detail_list_copy[0]    
        out_low = detail_list_copy[1]    
        out_detail_list = [out_high]

        for i in range(len(detail_list_copy) - 1):
            out_low_res = self.downsamplers[i](out_high) 
            out_low = out_low + out_low_res              
            out_high = out_low                           

            origin_scaled = F.interpolate(origin_high, size=out_high.shape[2:], mode='bilinear', align_corners=False)
            out_high = out_high + origin_scaled

            if i + 2 <= len(detail_list) - 1:
                out_low = detail_list[i + 2]             
            out_detail_list.append(out_high)
        return out_detail_list

class MultiScaleTrendMixing2D(nn.Module):
    def __init__(self, configs):
        super(MultiScaleTrendMixing2D, self).__init__()

        self.upsamplers = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=configs.d_model, 
                    out_channels=configs.d_ff, 
                    kernel_size=(3, 3), 
                    stride=(configs.down_sampling_window, 1), 
                    padding=(1, 1)
                ),
                nn.GELU(),
                nn.Conv2d(
                    in_channels=configs.d_ff,       
                    out_channels=configs.d_model,   
                    kernel_size=(1, 1),             
                    stride=1,                       
                    padding=0                       
                )
            )
            for _ in range(configs.down_sampling_layers - 1)  
        ])
    
    def forward(self, trend_list):
        
        trend_list_copy = trend_list.copy()
        origin_low = trend_list_copy[-1]
        trend_list_copy.reverse()
        out_low = trend_list_copy[0]    
        out_high = trend_list_copy[1]   
        out_trend_list = [out_low]

        for i in range(len(trend_list_copy) - 1):
            out_high_res = self.upsamplers[i](out_low)  

            if out_high_res.shape[2:] != out_high.shape[2:]:
                target_size = out_high.shape[2:]
                out_high_res = F.interpolate(out_high_res, size=target_size, mode='bilinear', align_corners=False)

            out_high = out_high + out_high_res          

            origin_scaled = F.interpolate(origin_low, size=out_high.shape[2:], mode='bilinear', align_corners=False)
            out_high = out_high + origin_scaled

            out_low = out_high                          
            if i + 2 <= len(trend_list_copy) - 1:
                out_high = trend_list_copy[i + 2]
            out_trend_list.append(out_low)

        out_trend_list.reverse()
        
        return out_trend_list
