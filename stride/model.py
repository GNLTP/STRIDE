import torch
import torch.nn as nn
import torch.nn.functional as F

from stride.layers.embedding import DataEmbedding
from stride.layers.normalization import Normalize
from stride.layers.period_estimation import Period_init
from stride.layers.cross_modal_fusion import TrendDetailFusion
from stride.layers.data_encoder import DataSTRBlock
from stride.layers.image_encoder import ImageSTRBlock
from stride.layers.semantic_head import Stage1SemanticHead
from stride.utils.checkpoint import CheckpointWrapper

class STRIDEStage1(nn.Module):
    def __init__(self, configs):
        super().__init__()

        self.mode = configs.model_mode
        if self.mode not in {"data", "img", "data & img"}:
            raise ValueError(f"Unsupported model_mode: {self.mode}")
        self.layer = configs.e_layers
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len - configs.excluded_future_steps    
        self.excluded_future_steps = configs.excluded_future_steps
        self.k = configs.top_k              
        self.use_ckpt = getattr(configs, "use_checkpoint", True)

        if self.mode in {"data", "data & img"}:
            self.normalize_layers = Normalize(
                configs.enc_data_in,
                affine=False,               
                subtract_last=False,        
                use_norm=configs.use_norm   
            )

            self.data_embedding = DataEmbedding(
                c_in=configs.enc_data_in,
                d_model=configs.d_model,
                dropout=configs.dropout_modal
            )

            self.data_encoder = nn.ModuleList([
                CheckpointWrapper(DataSTRBlock(configs), use_checkpoint=self.use_ckpt)
                for _ in range(self.layer)
            ])

        if self.mode in {"img", "data & img"}:
            self.image_encoder = CheckpointWrapper(
                ImageSTRBlock(configs, freq_k=configs.top_k),
                use_checkpoint=self.use_ckpt,
            )

        self.fusion = CheckpointWrapper(
            TrendDetailFusion(
                dim=configs.d_model,
                d_ff=configs.d_ff,
                num_heads=configs.n_heads,
                output_len=self.seq_len
            ),
            use_checkpoint=self.use_ckpt,
        )

        self.semantic_head = CheckpointWrapper(
            Stage1SemanticHead(
                d_model=configs.d_model,
                num_classes=configs.num_classes,
                num_experts=configs.num_experts,
                dropout=configs.dropout_modal
            ),
            use_checkpoint=self.use_ckpt,
        )

    def forward(self, imgs, data=None, return_features=False):
        if self.mode == "img":
            if imgs is None:
                raise ValueError("imgs is required for img mode")
            img_trend, img_detail = self.image_encoder(imgs, None)
            trend_seq, detail_seq = self.fusion(img_trend, img_trend, img_detail, img_detail)
        else:
            if data is None:
                raise ValueError(f"data is required for {self.mode} mode")
            data = self.normalize_layers(data, 'norm')          
            with torch.no_grad():
                period_init = Period_init(data, self.k)         
            data = self.data_embedding(data)
            data_list = [data.clone() for _ in period_init]     

            if self.mode == "data & img":
                if imgs is None:
                    raise ValueError("imgs is required for data & img mode")
                for i in range(self.layer):
                    data_list, data_detail, data_trend = self.data_encoder[i](data_list, period_init)
                img_trend, img_detail = self.image_encoder(imgs, period_init)
                trend_seq, detail_seq = self.fusion(img_trend, data_trend, img_detail, data_detail)

            else:
                for i in range(self.layer):
                    data_list, data_detail, data_trend = self.data_encoder[i](data_list, period_init)
                trend_seq, detail_seq = self.fusion(data_trend, data_trend, data_detail, data_detail)

        out = self.semantic_head(trend_seq, detail_seq)

        if return_features:
            out = {**out, "trend_tokens": trend_seq, "detail_tokens": detail_seq}

        return out
