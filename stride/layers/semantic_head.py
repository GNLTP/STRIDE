import torch
import torch.nn as nn
import torch.nn.functional as F

class Stage1SemanticHead(nn.Module):
    """Aggregate trend/detail tokens into routing and operation-label logits."""

    def __init__(self, d_model: int = 16, num_classes: int = 8,
                 num_experts: int = 32, dropout: float = 0.1,
                 fuse_mode="cat"):

        super().__init__()
        assert fuse_mode in ["add", "cat"]
        self.fuse_mode = fuse_mode
        self.d_model     = d_model
        self.d_ff        = d_model * 4
        self.joint_dim   = d_model * 2
        self.dropout     = dropout
        self.num_classes = num_classes
        self.num_experts = num_experts

        self.trend_block = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_ff),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_ff, self.d_model),
        )

        self.detail_block = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_ff),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_ff, self.d_model),
        )

        if fuse_mode == "cat":
            self.routing_head = nn.Sequential(
                nn.LayerNorm(self.joint_dim),
                nn.Linear(self.joint_dim, self.d_ff),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.d_ff, self.num_experts),
            )
        else:
            self.routing_head = nn.Sequential(
                nn.LayerNorm(self.d_model),
                nn.Linear(self.d_model, self.d_ff),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.d_ff, self.num_experts),
            )

        self.operation_readout = nn.Sequential(
            nn.LayerNorm(self.num_experts),
            nn.Linear(self.num_experts, self.d_ff),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_ff, self.num_classes),
        )

    def forward(self, trend_seq, detail_seq):
        assert trend_seq.shape == detail_seq.shape
        B, T, C = trend_seq.shape

        trend_feat  = trend_seq  + self.trend_block(trend_seq)
        detail_feat = detail_seq + self.detail_block(detail_seq)

        if self.fuse_mode == "add":
            joint = 0.5 * (trend_feat + detail_feat)    
        else:
            joint = torch.cat([trend_feat, detail_feat], dim=-1)  

        routing_logits_per_step = self.routing_head(joint)        
        routing_logits = routing_logits_per_step.mean(dim=1)      

        # Keep the original stop-gradient between the two readouts.
        operation_logits = self.operation_readout(routing_logits.detach())

        return {
            "routing_logits": routing_logits,
            "operation_logits":  operation_logits,
        }
