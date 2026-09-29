from dataclasses import dataclass


@dataclass
class STRIDEConfig:
    """Configuration for the STRIDE representation model."""

    model_mode: str = "data & img"
    task_name: str = "stage1"
    seq_len: int = 32
    excluded_future_steps: int = 0
    top_k: int = 3
    e_layers: int = 3
    enc_img_in: int = 6
    enc_data_in: int = 15
    pretrained_backbone: bool = True
    d_model: int = 24
    d_ff: int = 48
    n_heads: int = 4
    dropout_modal: float = 0.1
    num_classes: int = 8
    num_experts: int = 16
    down_sampling_layers: int = 3
    down_sampling_window: int = 2
    use_norm: bool = False
    use_checkpoint: bool = False
