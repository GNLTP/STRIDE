import torch
import torch.nn as nn
import timm

def ImageResNet18Light(input_channels: int = 3, pretrain=True, encoder_list=[True,True,False]) -> nn.Module:

    encoder = timm.create_model('resnetv2_18', pretrained=pretrain)

    old_conv: nn.Conv2d = encoder.stem.conv
    C_pre = old_conv.in_channels  
    C_new = input_channels

    assert C_new % C_pre == 0, \
        f"input_channels={C_new} must be a multiple of backbone in_channels={C_pre}"

    groups = C_new // C_pre

    new_conv = nn.Conv2d(
        in_channels=C_new,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        dilation=old_conv.dilation,
        bias=old_conv.bias is not None,
    )

    with torch.no_grad():
        w = old_conv.weight  
        new_conv.weight.zero_()

        for g in range(groups):
            c_start = g * C_pre
            c_end = c_start + C_pre
            new_conv.weight[:, c_start:c_end, :, :] = w

        if old_conv.bias is not None:
            new_conv.bias.data.copy_(old_conv.bias.data)

    encoder.stem.conv = new_conv

    del encoder.stages[3]
    del encoder.stages[2]

    encoder.norm = nn.Identity()
    encoder.head.global_pool = nn.Identity()
    encoder.head.fc = nn.Identity()
    encoder.head.flatten = nn.Identity()

    for p in encoder.stem.parameters():
        p.requires_grad = encoder_list[0]
    for p in encoder.stages[0].parameters():
        p.requires_grad = encoder_list[1]
    for p in encoder.stages[1].parameters():
        p.requires_grad = encoder_list[2]

    def forward(x):
        x = encoder.stem(x)        
        x = encoder.stages[0](x)   
        x = encoder.stages[1](x)   

        return x

    encoder.forward = forward

    return encoder
