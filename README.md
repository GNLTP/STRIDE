# STRIDE: Spectral Trend-Detail Representation

This repository provides the multimodal representation model developed for **STRIDE: Spectral Trend-Detail Representation for Interpretable Dual-Head Evaluation**. It processes time-aligned multi-view images and numerical state sequences to capture complementary temporal patterns.

The model estimates salient periods with spectral analysis, organizes temporal features into period-aware representations, and learns Trend and Detail features through axial attention and cross-modal fusion. The resulting representations can be used for operation-label evaluation or as conditioning features for a downstream Flow Matching model.

## Interface

- Images: `(B, T, C_img, H, W)`; the example uses two RGB views (`C_img=6`). Images must be square.
- States: `(B, T, C_state)`; the example uses `C_state=15`.
- Outputs: `routing_logits` and `operation_logits`. Set `return_features=True` to also obtain `trend_tokens` and `detail_tokens`, each shaped `(B, T, d_model)`.

Supported modes are `data`, `img`, and `data & img`. The `img` mode estimates periods from images. On the period grid, the horizontal phase axis (W) yields Trend and the vertical cycle axis (H) yields Detail.

Modify the parameters in `stride/config.py`. The routing scores are learned representation outputs that may be used for downstream conditioning.

## Start

```bash
pip install -r requirements.txt
python -m main.forward_dummy
```

The example performs a forward pass with random inputs and does not download model weights. To load pretrained ResNetV2 weights when available, set `pretrained_backbone=True` in the configuration; this may require network access.

## License

Released under the [MIT License](LICENSE).
