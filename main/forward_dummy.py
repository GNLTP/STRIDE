import torch

from stride import STRIDEConfig, STRIDEStage1


def main():
    config = STRIDEConfig(pretrained_backbone=False)
    model = STRIDEStage1(config).eval()
    images = torch.randn(1, config.seq_len, config.enc_img_in, 96, 96)
    states = torch.randn(1, config.seq_len, config.enc_data_in)

    with torch.no_grad():
        output = model(images, states, return_features=True)

    for name, tensor in output.items():
        print(name, tuple(tensor.shape))


if __name__ == "__main__":
    main()
