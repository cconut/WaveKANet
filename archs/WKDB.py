import torch
import torch.nn as nn
from einops import rearrange
from pytorch_wavelets import DWTForward, IDWT
from archs.fastkanconv import FastKANConvLayer


class ChannelAttention(nn.Module):
    def __init__(self, in_ch, reduction=2):
        super().__init__()
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_ch, in_ch // reduction, 1, bias=False)
        self.relu = nn.GELU()
        self.fc2 = nn.Conv2d(in_ch // reduction, in_ch, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.pool(x)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        return x * self.sigmoid(y)


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_pool, max_pool], dim=1)
        spatial_att = self.conv(concat)
        spatial_att = self.sigmoid(spatial_att)
        return x * spatial_att


class ECAM(nn.Module):
    def __init__(self, in_channels, reduction=2):
        super().__init__()
        self.in_channels = in_channels
        self.channel_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_max_pool = nn.AdaptiveMaxPool2d(1)
        self.channel_var_pool = lambda x: torch.var(x, dim=(2, 3), keepdim=True)
        self.channel_fc = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels // reduction, 1),
            nn.GELU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=3, padding=1)
        self.spatial_sigmoid = nn.Sigmoid()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()

        avg_out = self.channel_avg_pool(x)
        max_out = self.channel_max_pool(x)
        var_out = self.channel_var_pool(x)
        channel_cat = torch.cat([avg_out, max_out, var_out], dim=1)
        channel_weight = self.channel_fc(channel_cat).view(b, c, 1, 1)
        x = x * channel_weight

        avg_spatial = torch.mean(x, dim=1, keepdim=True)
        max_spatial, _ = torch.max(x, dim=1, keepdim=True)
        spatial_cat = torch.cat([avg_spatial, max_spatial], dim=1)
        spatial_weight = self.spatial_sigmoid(self.spatial_conv(spatial_cat))
        x = x * spatial_weight

        gate_weight = self.gate(x)
        x = x * gate_weight
        return x

class KANWaveDownLayer(nn.Module):
    def __init__(self, in_ch, out_ch, use_attention=True):
        super().__init__()
        self.out_ch = out_ch

        self.base_conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True)
        )
        self.base_adapter = nn.Conv2d(in_ch, out_ch, 1)
        self.downsample = nn.AvgPool2d(2)


        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.wavelet_encoder = nn.Sequential(
            nn.Conv2d(in_ch * 4, out_ch * 2, 3, padding=1),
            nn.BatchNorm2d(out_ch * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch * 2, out_ch, 1)
        )


        self.low_kan = FastKANConvLayer(in_ch, out_ch, 3, padding=1)
        self.high_kans = nn.ModuleList([
            FastKANConvLayer(in_ch, out_ch, 3, padding=1) for _ in range(3)
        ])


        self.attn = ECAM(out_ch * 3, reduction=2) if use_attention else nn.Identity()
        self.fusion = nn.Sequential(
            nn.Conv2d(out_ch * 3, out_ch * 2, 3, padding=1),
            nn.BatchNorm2d(out_ch * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch * 2, out_ch, 1)
        )

    def forward(self, x):

        base = self.base_conv(x)
        base_down = self.downsample(base)
        base_out = self.base_adapter(base_down)


        yL, yH = self.wt(x)
        HL, LH, HH = yH[0].unbind(2)
        wavelet_cat = torch.cat([yL, HL, LH, HH], dim=1)
        wavelet_out = self.wavelet_encoder(wavelet_cat)


        low_feat = self.low_kan(yL)
        high_feats = [kan(h) for kan, h in zip(self.high_kans, [HL, LH, HH])]
        kan_feats = torch.stack([low_feat] + high_feats, dim=1)
        kan_feats = rearrange(kan_feats, 'b n c h w -> b (n c) h w')[:, :self.out_ch]

        combined = torch.cat([base_out, wavelet_out, kan_feats], dim=1)
        attn_feats = self.attn(combined)
        output = self.fusion(attn_feats) + base_out

        return output

class DownSample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.down_kan = KANWaveDownLayer(in_ch, out_ch)

    def forward(self, input):
        return self.down_kan(input)


class FKSA(nn.Module):
    def __init__(self, F_g, F_x, num_grids=4, kernel_size=3):
        super().__init__()

        self.kan_g = FastKANConvLayer(
            in_channels=F_g,
            out_channels=1,
            kernel_size=kernel_size,
            padding=kernel_size//2,
            num_grids=num_grids,
            use_base_update=True,
            kan_type="RBF"
        )
        self.kan_x = FastKANConvLayer(
            in_channels=F_x,
            out_channels=1,
            kernel_size=kernel_size,
            padding=kernel_size//2,
            num_grids=num_grids,
            use_base_update=True,
            kan_type="RBF"
        )
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = nn.Parameter(torch.ones(1))


    def forward(self, g, x):
        channel_att_g = self.kan_g(g)
        channel_att_x = self.kan_x(x)
        channel_att_sum = (channel_att_g + channel_att_x) / 2.0
        scale = self.sigmoid(channel_att_sum)
        x_after_channel = x * scale
        out = x + self.scale_factor * x_after_channel
        out = self.relu(out)
        return out


class Up_wt_KAN(nn.Module):
    def __init__(self, in_ch, out_ch, num_grids=4, kernel_size=3):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch),
            nn.Conv2d(in_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch)
        )

        self.fksa = FKSA(F_g=out_ch, F_x=out_ch, num_grids=num_grids, kernel_size=kernel_size)
        self.spatial_att = SpatialAttention()
        self.post_process = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch),
            nn.GELU()
        )

    def forward(self, kan_feat, skip_feat):

        upsampled_feat = self.upsample(kan_feat)

        if skip_feat is not None:

            skip_feat_att = self.fksa(g=upsampled_feat, x=skip_feat)
            fused_feature = upsampled_feat + skip_feat_att
        else:
            fused_feature = upsampled_feat

        fused_feature = self.spatial_att(fused_feature)

        return self.post_process(fused_feature)


