import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import timm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import types
import math
from archs.WKDB import *
from archs.kan import KANLinear
from torch.nn import init
from archs.wtconv2d import WTConv2d
from archs.CCKFT import CCKFT



class KANLayer(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., no_kan=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features

        grid_size = 5
        spline_order = 3
        scale_noise = 0.1
        scale_base = 1.0
        scale_spline = 1.0
        base_activation = torch.nn.SiLU
        grid_eps = 0.02
        grid_range = [-1, 1]


        if not no_kan:
            self.fc1 = KANLinear(
                in_features,
                hidden_features,
                grid_size=grid_size,
                spline_order=spline_order,
                scale_noise=scale_noise,
                scale_base=scale_base,
                scale_spline=scale_spline,
                base_activation=base_activation,
                grid_eps=grid_eps,
                grid_range=grid_range,
            )
            self.fc2 = KANLinear(
                hidden_features,
                out_features,
                grid_size=grid_size,
                spline_order=spline_order,
                scale_noise=scale_noise,
                scale_base=scale_base,
                scale_spline=scale_spline,
                base_activation=base_activation,
                grid_eps=grid_eps,
                grid_range=grid_range,
            )
            self.fc3 = KANLinear(
                hidden_features,
                out_features,
                grid_size=grid_size,
                spline_order=spline_order,
                scale_noise=scale_noise,
                scale_base=scale_base,
                scale_spline=scale_spline,
                base_activation=base_activation,
                grid_eps=grid_eps,
                grid_range=grid_range,
            )
        else:
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.fc2 = nn.Linear(hidden_features, out_features)
            self.fc3 = nn.Linear(hidden_features, out_features)

        self.dwconv_1 = DW_bn_relu(hidden_features)
        self.dwconv_2 = DW_bn_relu(hidden_features)
        self.dwconv_3 = DW_bn_relu(hidden_features)

        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape

        x = self.fc1(x.reshape(B * N, C))
        x = x.reshape(B, N, C).contiguous()
        x = self.dwconv_1(x, H, W)
        x = self.fc2(x.reshape(B * N, C))
        x = x.reshape(B, N, C).contiguous()
        x = self.dwconv_2(x, H, W)
        x = self.fc3(x.reshape(B * N, C))
        x = x.reshape(B, N, C).contiguous()
        x = self.dwconv_3(x, H, W)

        return x



class KANBlock(nn.Module):
    def __init__(self, dim, drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, no_kan=False):
        super().__init__()
        # drop_path防止过拟合
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim)

        self.layer = KANLayer(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
                              no_kan=no_kan)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):

        x = x + self.drop_path(self.layer(self.norm2(x), H, W))

        return x



class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = WTConv2d(dim, dim, kernel_size=3, stride=1)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x



class DW_bn_relu(nn.Module):
    def __init__(self, dim=768):
        super(DW_bn_relu, self).__init__()
        self.dwconv = WTConv2d(dim, dim, kernel_size=3, stride=1)
        self.bn = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU()

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = x.flatten(2).transpose(1, 2)

        return x



class FPNMultiScalePatchEmbed(nn.Module):


    def __init__(self, img_size=224, kernel_sizes=[3, 5, 7], strides=[1, 1, 1], in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.in_chans = in_chans
        self.embed_dim = embed_dim


        self.projs = nn.ModuleList([
            nn.Conv2d(in_chans, embed_dim // len(kernel_sizes), kernel_size=ks, stride=s, padding=ks // 2)
            for ks, s in zip(kernel_sizes, strides)
        ])


        self.att_fusion = ChannelAttention(embed_dim)
        self.fusion_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.norm = nn.LayerNorm(embed_dim)


        H_out = (img_size[0] + 2 * (kernel_sizes[0] // 2) - kernel_sizes[0]) // strides[0] + 1
        W_out = (img_size[1] + 2 * (kernel_sizes[0] // 2) - kernel_sizes[0]) // strides[0] + 1
        self.base_H, self.base_W = H_out, W_out

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        B, C, H, W = x.shape
        multi_scale_features = []

        # 提取多尺度特征
        for proj in self.projs:
            feat = proj(x)
            multi_scale_features.append(feat)


        target_H, target_W = max(f.shape[2] for f in multi_scale_features), max(
            f.shape[3] for f in multi_scale_features)
        aligned_features = []
        for feat in multi_scale_features:
            _, _, h, w = feat.shape
            if (h, w) != (target_H, target_W):
                feat = F.interpolate(feat, size=(target_H, target_W), mode='bilinear', align_corners=False)
            aligned_features.append(feat)


        fused_feat = torch.cat(aligned_features, dim=1)
        fused_feat = self.att_fusion(fused_feat)
        fused_feat = self.fusion_conv(fused_feat)


        _, _, H_new, W_new = fused_feat.shape
        fused_feat = fused_feat.flatten(2).transpose(1, 2)
        fused_feat = self.norm(fused_feat)

        return fused_feat, H_new, W_new


class WaveKANet(nn.Module):
    def __init__(self, num_classes, input_channels=3, img_size=256, patch_size=16, in_chans=3,
                 embed_dims=[32, 64, 256, 300, 600], no_kan=False,
                 drop_rate=0.1, drop_path_rate=0.1, norm_layer=nn.LayerNorm, depths=[1, 1, 1], **kwargs):
        super().__init__()
        self.channel_num = embed_dims


        self.encoder1 = DownSample(3, embed_dims[0])
        self.encoder2 = DownSample(embed_dims[0], embed_dims[1])
        self.encoder3 = DownSample(embed_dims[1], embed_dims[2])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Stage 4
        self.kan_down3 = DownSample(embed_dims[2], embed_dims[2] // 2)
        self.patch_embed3 = FPNMultiScalePatchEmbed(
            img_size=img_size // 8,
            kernel_sizes=[3, 5, 7],
            strides=[1, 1, 1],
            in_chans=embed_dims[2] // 2,
            embed_dim=embed_dims[3]
        )
        self.block1 = nn.ModuleList([KANBlock(
            dim=embed_dims[3],
            drop=drop_rate, drop_path=dpr[0], norm_layer=norm_layer
        )])
        self.norm3 = norm_layer(embed_dims[3])

        # Stage 5
        self.kan_down4 = DownSample(embed_dims[3], embed_dims[3])
        self.patch_embed4 = FPNMultiScalePatchEmbed(
            img_size=img_size // 16,
            kernel_sizes=[3, 5, 7],
            strides=[1, 1, 1],
            in_chans=embed_dims[3],
            embed_dim=embed_dims[4]
        )
        self.block2 = nn.ModuleList([KANBlock(
            dim=embed_dims[4],
            drop=drop_rate, drop_path=dpr[1], norm_layer=norm_layer
        )])
        self.norm4 = norm_layer(embed_dims[4])


        self.dnorm3 = norm_layer(embed_dims[3])
        self.dnorm4 = norm_layer(embed_dims[2])

        self.dblock1 = nn.ModuleList([KANBlock(
            dim=embed_dims[3],
            drop=drop_rate, drop_path=dpr[0], norm_layer=norm_layer
        )])
        self.dblock2 = nn.ModuleList([KANBlock(
            dim=embed_dims[2],
            drop=drop_rate, drop_path=dpr[1], norm_layer=norm_layer
        )])

        self.decoder1 = Up_wt_KAN(embed_dims[4], embed_dims[3])
        self.decoder2 = Up_wt_KAN(embed_dims[3], embed_dims[2])
        self.decoder3 = Up_wt_KAN(embed_dims[2], embed_dims[2] // 4)
        self.decoder4 = Up_wt_KAN(embed_dims[2] // 4, embed_dims[2] // 8)
        self.decoder5 = Up_wt_KAN(embed_dims[2] // 8, embed_dims[2] // 8)
        self.final = nn.Conv2d(embed_dims[2] // 8, num_classes, kernel_size=1)

        class Config:
            transformer = {
                "num_heads": 4,
                "num_layers": 1,
                "embeddings_dropout_rate": 0.1,
                "attention_dropout_rate": 0.1,
                "dropout_rate": 0.1
            }
            KV_size = 1252
            expand_ratio = 4

        self.channel_transformer = CCKFT(
            config=Config,
            vis=False,
            img_size=img_size,
            channel_num=embed_dims,
            patchSize=[16, 8, 4, 2, 1]
        )

        self.encoder1_output = nn.Identity()
        self.encoder2_output = nn.Identity()
        self.encoder3_output = nn.Identity()
        self.encoder4_output = nn.Identity()
        self.encoder5_output = nn.Identity()
        self.decoder1_output = nn.Identity()
        self.decoder2_output = nn.Identity()
        self.decoder3_output = nn.Identity()
        self.decoder4_output = nn.Identity()
        self.decoder5_output = nn.Identity()

    def forward(self, x, return_features=False):
        B = x.shape[0]

        ### Encoder
        out = self.encoder1(x)
        t1 = out
        H1, W1 = out.shape[2], out.shape[3]

        out = self.encoder2(out)
        t2 = out
        H2, W2 = out.shape[2], out.shape[3]

        out = self.encoder3(out)
        t3 = out
        H3, W3 = out.shape[2], out.shape[3]

        out = self.kan_down3(out)
        t4 = out
        H4, W4 = out.shape[2], out.shape[3]

        out, H, W = self.patch_embed3(t4)
        for i, blk in enumerate(self.block1):
            out = blk(out, H, W)
        out = self.norm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = out
        H4, W4 = H, W

        out = self.kan_down4(out)  # [B, 300, 11, 11]
        t5 = out
        H5, W5 = out.shape[2], out.shape[3]

        out, H, W = self.patch_embed4(t5)
        for i, blk in enumerate(self.block2):
            out = blk(out, H, W)
        out = self.norm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t5 = out
        H5, W5 = H, W

        en1 = t1
        en2 = t2
        en3 = t3
        en4 = t4
        en5 = t5

        x1, x2, x3, x4, x5, attn_weights = self.channel_transformer(en1, en2, en3, en4, en5)

        t1 = t1 + x1 if x1 is not None else t1
        t2 = t2 + x2 if x2 is not None else t2
        t3 = t3 + x3 if x3 is not None else t3
        t4 = t4 + x4 if x4 is not None else t4
        t5 = t5 + x5 if x5 is not None else t5


        t1 = self.encoder1_output(t1)
        t2 = self.encoder2_output(t2)
        t3 = self.encoder3_output(t3)
        t4 = self.encoder4_output(t4)
        t5 = self.encoder5_output(t5)

        ### Decoder
        out = self.decoder1(t5, t4)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.dblock1):
            out = blk(out, H, W)
        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = self.decoder1_output(out)

        out = self.decoder2(out, t3)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.dblock2):
            out = blk(out, H, W)
        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = self.decoder2_output(out)

        out = self.decoder3(out, t2)
        out = self.decoder3_output(out)

        out = self.decoder4(out, t1)
        out = self.decoder4_output(out)

        out = self.decoder5(out, None)
        out = self.decoder5_output(out)


        features = out
        final_out = self.final(out)


        if final_out.shape[1] == 1 and len(final_out.shape) == 3:
            final_out = final_out.unsqueeze(1)

        if return_features:
            return final_out, features
        return final_out

