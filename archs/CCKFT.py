
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import logging
import math
import torch
import torch.nn as nn
import numpy as np
from torch.nn import Dropout, Softmax, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from archs.kan import KANLinear

logger = logging.getLogger(__name__)

class Channel_Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self,config, patchsize, img_size, in_channels):
        super().__init__()
        img_size = _pair(img_size)
        patch_size = _pair(patchsize)
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])

        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=in_channels,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, in_channels))
        self.dropout = Dropout(config.transformer["embeddings_dropout_rate"])

    def forward(self, x):
        if x is None:
            return None
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class Reconstruct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(Reconstruct, self).__init__()
        if kernel_size == 3:
            padding = 1
        else:
            padding = 0
        self.conv = nn.Conv2d(in_channels, out_channels,kernel_size=kernel_size, padding=padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        if x is None:
            return None

        B, n_patch, hidden = x.size()
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = x.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = nn.Upsample(scale_factor=self.scale_factor)(x)

        out = self.conv(x)
        out = self.norm(out)
        out = self.activation(out)
        return out

class Attention_org(nn.Module):
    def __init__(self, config, vis, channel_num):
        super(Attention_org, self).__init__()
        self.vis = vis
        self.KV_size = config.KV_size
        self.channel_num = channel_num
        self.num_attention_heads = config.transformer["num_heads"]

        self.query1 = nn.ModuleList()
        self.query2 = nn.ModuleList()
        self.query3 = nn.ModuleList()
        self.query4 = nn.ModuleList()
        self.query5 = nn.ModuleList()
        self.key = nn.ModuleList()
        self.value = nn.ModuleList()

        for _ in range(config.transformer["num_heads"]):
            query1 = nn.Linear(channel_num[0], channel_num[0], bias=False)
            query2 = nn.Linear(channel_num[1], channel_num[1], bias=False)
            query3 = nn.Linear(channel_num[2], channel_num[2], bias=False)
            query4 = nn.Linear(channel_num[3], channel_num[3], bias=False)
            query5 = nn.Linear(channel_num[4], channel_num[4], bias=False)
            key = nn.Linear(self.KV_size, self.KV_size, bias=False)
            value = nn.Linear(self.KV_size, self.KV_size, bias=False)
            self.query1.append(copy.deepcopy(query1))
            self.query2.append(copy.deepcopy(query2))
            self.query3.append(copy.deepcopy(query3))
            self.query4.append(copy.deepcopy(query4))
            self.query5.append(copy.deepcopy(query5))
            self.key.append(copy.deepcopy(key))
            self.value.append(copy.deepcopy(value))
        self.psi = nn.InstanceNorm2d(self.num_attention_heads)
        self.softmax = Softmax(dim=3)
        self.out1 = nn.Linear(channel_num[0], channel_num[0], bias=False)
        self.out2 = nn.Linear(channel_num[1], channel_num[1], bias=False)
        self.out3 = nn.Linear(channel_num[2], channel_num[2], bias=False)
        self.out4 = nn.Linear(channel_num[3], channel_num[3], bias=False)
        self.out5 = nn.Linear(channel_num[4], channel_num[4], bias=False)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

    def forward(self, emb1, emb2, emb3, emb4, emb5, emb_all):
        multi_head_Q1_list, multi_head_Q2_list, multi_head_Q3_list, multi_head_Q4_list, multi_head_Q5_list = [], [], [], [], []
        multi_head_K_list, multi_head_V_list = [], []

        if emb1 is not None:
            for query1 in self.query1:
                Q1 = query1(emb1)
                multi_head_Q1_list.append(Q1)
        if emb2 is not None:
            for query2 in self.query2:
                Q2 = query2(emb2)
                multi_head_Q2_list.append(Q2)
        if emb3 is not None:
            for query3 in self.query3:
                Q3 = query3(emb3)
                multi_head_Q3_list.append(Q3)
        if emb4 is not None:
            for query4 in self.query4:
                Q4 = query4(emb4)
                multi_head_Q4_list.append(Q4)
        if emb5 is not None:
            for query5 in self.query5:
                Q5 = query5(emb5)
                multi_head_Q5_list.append(Q5)

        for key in self.key:
            K = key(emb_all)
            multi_head_K_list.append(K)
        for value in self.value:
            V = value(emb_all)
            multi_head_V_list.append(V)

        multi_head_Q1 = torch.stack(multi_head_Q1_list, dim=1) if emb1 is not None else None
        multi_head_Q2 = torch.stack(multi_head_Q2_list, dim=1) if emb2 is not None else None
        multi_head_Q3 = torch.stack(multi_head_Q3_list, dim=1) if emb3 is not None else None
        multi_head_Q4 = torch.stack(multi_head_Q4_list, dim=1) if emb4 is not None else None
        multi_head_Q5 = torch.stack(multi_head_Q5_list, dim=1) if emb5 is not None else None
        multi_head_K = torch.stack(multi_head_K_list, dim=1)
        multi_head_V = torch.stack(multi_head_V_list, dim=1)

        multi_head_Q1 = multi_head_Q1.transpose(-1, -2) if emb1 is not None else None
        multi_head_Q2 = multi_head_Q2.transpose(-1, -2) if emb2 is not None else None
        multi_head_Q3 = multi_head_Q3.transpose(-1, -2) if emb3 is not None else None
        multi_head_Q4 = multi_head_Q4.transpose(-1, -2) if emb4 is not None else None
        multi_head_Q5 = multi_head_Q5.transpose(-1, -2) if emb5 is not None else None

        attention_scores1 = torch.matmul(multi_head_Q1, multi_head_K) if emb1 is not None else None
        attention_scores2 = torch.matmul(multi_head_Q2, multi_head_K) if emb2 is not None else None
        attention_scores3 = torch.matmul(multi_head_Q3, multi_head_K) if emb3 is not None else None
        attention_scores4 = torch.matmul(multi_head_Q4, multi_head_K) if emb4 is not None else None
        attention_scores5 = torch.matmul(multi_head_Q5, multi_head_K) if emb5 is not None else None

        attention_scores1 = attention_scores1 / math.sqrt(self.KV_size) if emb1 is not None else None
        attention_scores2 = attention_scores2 / math.sqrt(self.KV_size) if emb2 is not None else None
        attention_scores3 = attention_scores3 / math.sqrt(self.KV_size) if emb3 is not None else None
        attention_scores4 = attention_scores4 / math.sqrt(self.KV_size) if emb4 is not None else None
        attention_scores5 = attention_scores5 / math.sqrt(self.KV_size) if emb5 is not None else None

        attention_probs1 = self.softmax(self.psi(attention_scores1)) if emb1 is not None else None
        attention_probs2 = self.softmax(self.psi(attention_scores2)) if emb2 is not None else None
        attention_probs3 = self.softmax(self.psi(attention_scores3)) if emb3 is not None else None
        attention_probs4 = self.softmax(self.psi(attention_scores4)) if emb4 is not None else None
        attention_probs5 = self.softmax(self.psi(attention_scores5)) if emb5 is not None else None

        if self.vis:
            weights = []
            weights.append(attention_probs1.mean(1) if emb1 is not None else None)
            weights.append(attention_probs2.mean(1) if emb2 is not None else None)
            weights.append(attention_probs3.mean(1) if emb3 is not None else None)
            weights.append(attention_probs4.mean(1) if emb4 is not None else None)
            weights.append(attention_probs5.mean(1) if emb5 is not None else None)
        else:
            weights = None

        attention_probs1 = self.attn_dropout(attention_probs1) if emb1 is not None else None
        attention_probs2 = self.attn_dropout(attention_probs2) if emb2 is not None else None
        attention_probs3 = self.attn_dropout(attention_probs3) if emb3 is not None else None
        attention_probs4 = self.attn_dropout(attention_probs4) if emb4 is not None else None
        attention_probs5 = self.attn_dropout(attention_probs5) if emb5 is not None else None

        multi_head_V = multi_head_V.transpose(-1, -2)
        context_layer1 = torch.matmul(attention_probs1, multi_head_V) if emb1 is not None else None
        context_layer2 = torch.matmul(attention_probs2, multi_head_V) if emb2 is not None else None
        context_layer3 = torch.matmul(attention_probs3, multi_head_V) if emb3 is not None else None
        context_layer4 = torch.matmul(attention_probs4, multi_head_V) if emb4 is not None else None
        context_layer5 = torch.matmul(attention_probs5, multi_head_V) if emb5 is not None else None

        context_layer1 = context_layer1.permute(0, 3, 2, 1).contiguous() if emb1 is not None else None
        context_layer2 = context_layer2.permute(0, 3, 2, 1).contiguous() if emb2 is not None else None
        context_layer3 = context_layer3.permute(0, 3, 2, 1).contiguous() if emb3 is not None else None
        context_layer4 = context_layer4.permute(0, 3, 2, 1).contiguous() if emb4 is not None else None
        context_layer5 = context_layer5.permute(0, 3, 2, 1).contiguous() if emb5 is not None else None

        context_layer1 = context_layer1.mean(dim=3) if emb1 is not None else None
        context_layer2 = context_layer2.mean(dim=3) if emb2 is not None else None
        context_layer3 = context_layer3.mean(dim=3) if emb3 is not None else None
        context_layer4 = context_layer4.mean(dim=3) if emb4 is not None else None
        context_layer5 = context_layer5.mean(dim=3) if emb5 is not None else None

        O1 = self.out1(context_layer1) if emb1 is not None else None
        O2 = self.out2(context_layer2) if emb2 is not None else None
        O3 = self.out3(context_layer3) if emb3 is not None else None
        O4 = self.out4(context_layer4) if emb4 is not None else None
        O5 = self.out5(context_layer5) if emb5 is not None else None

        O1 = self.proj_dropout(O1) if emb1 is not None else None
        O2 = self.proj_dropout(O2) if emb2 is not None else None
        O3 = self.proj_dropout(O3) if emb3 is not None else None
        O4 = self.proj_dropout(O4) if emb4 is not None else None
        O5 = self.proj_dropout(O5) if emb5 is not None else None

        return O1, O2, O3, O4, O5, weights




class KANLayer(nn.Module):
    def __init__(self, in_features, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features

        grid_size = 5
        spline_order = 3
        scale_noise = 0.1
        scale_base = 1.0
        scale_spline = 1.0
        base_activation = torch.nn.SiLU
        grid_eps = 0.02
        grid_range = [-1, 1]

        self.fc = KANLinear(
            in_features,
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

        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # x: (B, N, C)
        B, N, C = x.shape
        x = self.fc(x.reshape(B * N, C))
        x = x.reshape(B, N, -1)
        x = self.drop(x)
        return x



class Block_ViT(nn.Module):
    def __init__(self, config, vis, channel_num):
        super(Block_ViT, self).__init__()
        expand_ratio = config.expand_ratio
        self.attn_norm1 = LayerNorm(channel_num[0], eps=1e-6)
        self.attn_norm2 = LayerNorm(channel_num[1], eps=1e-6)
        self.attn_norm3 = LayerNorm(channel_num[2], eps=1e-6)
        self.attn_norm4 = LayerNorm(channel_num[3], eps=1e-6)
        self.attn_norm5 = LayerNorm(channel_num[4], eps=1e-6)
        self.attn_norm = LayerNorm(config.KV_size, eps=1e-6)
        self.channel_attn = Attention_org(config, vis, channel_num)

        self.ffn_norm1 = LayerNorm(channel_num[0], eps=1e-6)
        self.ffn_norm2 = LayerNorm(channel_num[1], eps=1e-6)
        self.ffn_norm3 = LayerNorm(channel_num[2], eps=1e-6)
        self.ffn_norm4 = LayerNorm(channel_num[3], eps=1e-6)
        self.ffn_norm5 = LayerNorm(channel_num[4], eps=1e-6)
        self.ffn1 = KANLayer(in_features=channel_num[0], out_features=channel_num[0],
                             drop=config.transformer["dropout_rate"])
        self.ffn2 = KANLayer(in_features=channel_num[1], out_features=channel_num[1],
                             drop=config.transformer["dropout_rate"])
        self.ffn3 = KANLayer(in_features=channel_num[2], out_features=channel_num[2],
                             drop=config.transformer["dropout_rate"])
        self.ffn4 = KANLayer(in_features=channel_num[3], out_features=channel_num[3],
                             drop=config.transformer["dropout_rate"])
        self.ffn5 = KANLayer(in_features=channel_num[4], out_features=channel_num[4],
                             drop=config.transformer["dropout_rate"])

    def forward(self, emb1, emb2, emb3, emb4, emb5):
        embcat = []
        org1, org2, org3, org4, org5 = emb1, emb2, emb3, emb4, emb5
        for i in range(5):
            var_name = "emb" + str(i + 1)
            tmp_var = locals()[var_name]
            if tmp_var is not None:
                embcat.append(tmp_var)

        emb_all = torch.cat(embcat, dim=2)

        cx1 = self.attn_norm1(emb1) if emb1 is not None else None
        cx2 = self.attn_norm2(emb2) if emb2 is not None else None
        cx3 = self.attn_norm3(emb3) if emb3 is not None else None
        cx4 = self.attn_norm4(emb4) if emb4 is not None else None
        cx5 = self.attn_norm5(emb5) if emb5 is not None else None
        emb_all = self.attn_norm(emb_all)
        cx1, cx2, cx3, cx4, cx5, weights = self.channel_attn(cx1, cx2, cx3, cx4, cx5, emb_all)

        cx1 = org1 + cx1 if emb1 is not None else None
        cx2 = org2 + cx2 if emb2 is not None else None
        cx3 = org3 + cx3 if emb3 is not None else None
        cx4 = org4 + cx4 if emb4 is not None else None
        cx5 = org5 + cx5 if emb5 is not None else None

        org1, org2, org3, org4, org5 = cx1, cx2, cx3, cx4, cx5
        x1 = self.ffn_norm1(cx1) if emb1 is not None else None
        x2 = self.ffn_norm2(cx2) if emb2 is not None else None
        x3 = self.ffn_norm3(cx3) if emb3 is not None else None
        x4 = self.ffn_norm4(cx4) if emb4 is not None else None
        x5 = self.ffn_norm5(cx5) if emb5 is not None else None
        x1 = self.ffn1(x1) if emb1 is not None else None
        x2 = self.ffn2(x2) if emb2 is not None else None
        x3 = self.ffn3(x3) if emb3 is not None else None
        x4 = self.ffn4(x4) if emb4 is not None else None
        x5 = self.ffn5(x5) if emb5 is not None else None

        x1 = x1 + org1 if emb1 is not None else None
        x2 = x2 + org2 if emb2 is not None else None
        x3 = x3 + org3 if emb3 is not None else None
        x4 = x4 + org4 if emb4 is not None else None
        x5 = x5 + org5 if emb5 is not None else None

        return x1, x2, x3, x4, x5, weights


class Encoder(nn.Module):
    def __init__(self, config, vis, channel_num):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm1 = LayerNorm(channel_num[0], eps=1e-6)
        self.encoder_norm2 = LayerNorm(channel_num[1], eps=1e-6)
        self.encoder_norm3 = LayerNorm(channel_num[2], eps=1e-6)
        self.encoder_norm4 = LayerNorm(channel_num[3], eps=1e-6)
        self.encoder_norm5 = LayerNorm(channel_num[4], eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block_ViT(config, vis, channel_num)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, emb1, emb2, emb3, emb4, emb5):
        attn_weights = []
        for layer_block in self.layer:
            emb1, emb2, emb3, emb4, emb5, weights = layer_block(emb1, emb2, emb3, emb4, emb5)
            if self.vis:
                attn_weights.append(weights)
        emb1 = self.encoder_norm1(emb1) if emb1 is not None else None
        emb2 = self.encoder_norm2(emb2) if emb2 is not None else None
        emb3 = self.encoder_norm3(emb3) if emb3 is not None else None
        emb4 = self.encoder_norm4(emb4) if emb4 is not None else None
        emb5 = self.encoder_norm5(emb5) if emb5 is not None else None
        return emb1, emb2, emb3, emb4, emb5, attn_weights


class CCKFT(nn.Module):
    def __init__(self, config, vis, img_size, channel_num=[32, 64, 256, 300, 600], patchSize=[16, 8, 4, 2, 1]):
        super().__init__()

        stage_sizes = [
            img_size // 2,
            img_size // 4,
            img_size // 8,
            img_size // 16,
            img_size // 32
        ]


        assert all(stage_sizes[i] % patchSize[i] == 0 for i in range(len(stage_sizes))), \
            "Patch sizes must evenly divide stage resolutions"

        self.patchSize_1, self.patchSize_2, self.patchSize_3, self.patchSize_4, self.patchSize_5 = patchSize


        self.embeddings_1 = Channel_Embeddings(config, self.patchSize_1, img_size=stage_sizes[0], in_channels=channel_num[0])
        self.embeddings_2 = Channel_Embeddings(config, self.patchSize_2, img_size=stage_sizes[1], in_channels=channel_num[1])
        self.embeddings_3 = Channel_Embeddings(config, self.patchSize_3, img_size=stage_sizes[2], in_channels=channel_num[2])
        self.embeddings_4 = Channel_Embeddings(config, self.patchSize_4, img_size=stage_sizes[3], in_channels=channel_num[3])
        self.embeddings_5 = Channel_Embeddings(config, self.patchSize_5, img_size=stage_sizes[4], in_channels=channel_num[4])

        self.encoder = Encoder(config, vis, channel_num)


        self.reconstruct_1 = Reconstruct(channel_num[0], channel_num[0], kernel_size=1, scale_factor=(self.patchSize_1, self.patchSize_1))
        self.reconstruct_2 = Reconstruct(channel_num[1], channel_num[1], kernel_size=1, scale_factor=(self.patchSize_2, self.patchSize_2))
        self.reconstruct_3 = Reconstruct(channel_num[2], channel_num[2], kernel_size=1, scale_factor=(self.patchSize_3, self.patchSize_3))
        self.reconstruct_4 = Reconstruct(channel_num[3], channel_num[3], kernel_size=1, scale_factor=(self.patchSize_4, self.patchSize_4))
        self.reconstruct_5 = Reconstruct(channel_num[4], channel_num[4], kernel_size=1, scale_factor=(self.patchSize_5, self.patchSize_5))

    def forward(self, en1, en2, en3, en4, en5):

        emb1 = self.embeddings_1(en1)
        emb2 = self.embeddings_2(en2)
        emb3 = self.embeddings_3(en3)
        emb4 = self.embeddings_4(en4)
        emb5 = self.embeddings_5(en5)


        encoded1, encoded2, encoded3, encoded4, encoded5, attn_weights = self.encoder(emb1, emb2, emb3, emb4, emb5)


        x1 = self.reconstruct_1(encoded1) if en1 is not None else None
        x2 = self.reconstruct_2(encoded2) if en2 is not None else None
        x3 = self.reconstruct_3(encoded3) if en3 is not None else None
        x4 = self.reconstruct_4(encoded4) if en4 is not None else None
        x5 = self.reconstruct_5(encoded5) if en5 is not None else None

        x1 = x1 + en1 if en1 is not None else None
        x2 = x2 + en2 if en2 is not None else None
        x3 = x3 + en3 if en3 is not None else None
        x4 = x4 + en4 if en4 is not None else None
        x5 = x5 + en5 if en5 is not None else None


        return x1, x2, x3, x4, x5, attn_weights
