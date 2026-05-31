import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
# from ..base_model import BaseModel
from opengait.modeling.base_model import BaseModel
from ..modules import HorizontalPoolingPyramid, PackSequenceWrapper, SeparateFCs, SeparateBNNecks, SetBlockWrapper, \
    conv3x3, conv1x1, BasicBlock2D, BasicBlockP3D

from einops import rearrange

import copy


class SHIFFusion(nn.Module):
    def __init__(self, in_channels=64):
        super(SHIFFusion, self).__init__()
        self.in_channels = in_channels

        # Local feature fusion
        self.local_fusion = nn.Sequential(
            nn.Conv3d(in_channels * 2, in_channels, 1),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True)
        )

        # Global feature fusion
        self.global_fusion = nn.Sequential(
            nn.Conv3d(in_channels * 2, in_channels, 1),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True)
        )

        # Upper body attention
        self.upper_attention = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // 4, 1),
            nn.BatchNorm3d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // 4, 1, 1),
            nn.Sigmoid()
        )

        # Lower body attention
        self.lower_attention = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // 4, 1),
            nn.BatchNorm3d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // 4, 1, 1),
            nn.Sigmoid()
        )

        # Global attention
        self.global_attention = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // 4, 1),
            nn.BatchNorm3d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // 4, 1, 1),
            nn.Sigmoid()
        )

        # Difference feature extraction (for upper and lower body) - reduced channels
        self.upper_diff = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // 4, 3, padding=1),  # reduced from //2 to //4
            nn.BatchNorm3d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // 4, in_channels, 1)
        )

        self.lower_diff = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // 4, 3, padding=1),  # reduced from //2 to //4
            nn.BatchNorm3d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // 4, in_channels, 1)
        )

        # Common pattern extraction - reduced channels
        self.common_pattern = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // 4, 3, padding=1),  # reduced from //2 to //4
            nn.BatchNorm3d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // 4, in_channels, 1)
        )

        # Auxiliary feature fusion - simplified
        self.aux_fusion = nn.Sequential(
            nn.Conv3d(in_channels * 2, in_channels, 1),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True)
        )

        # Final fusion
        self.final_fusion = nn.Sequential(
            nn.Conv3d(in_channels * 2, in_channels, 1),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True)
        )

        # Weight for auxiliary features - reduced initial weight
        self.aux_weight = nn.Parameter(torch.tensor(0.05))  # reduced from 0.1 to 0.05

    def forward(self, sil_feat, map_feat):
        # Initial fusion for local and global features
        local_feat = self.local_fusion(torch.cat([sil_feat, map_feat], dim=1))
        global_feat = self.global_fusion(torch.cat([sil_feat, map_feat], dim=1))

        # Split features for upper and lower body
        b, c, s, h, w = sil_feat.shape
        upper_h = h // 2

        # Upper body features
        sil_upper = sil_feat[:, :, :, :upper_h, :]
        map_upper = map_feat[:, :, :, :upper_h, :]
        local_upper = local_feat[:, :, :, :upper_h, :]

        # Lower body features
        sil_lower = sil_feat[:, :, :, upper_h:, :]
        map_lower = map_feat[:, :, :, upper_h:, :]
        local_lower = local_feat[:, :, :, upper_h:, :]

        # Upper body fusion and difference feature extraction
        upper_att = self.upper_attention(local_upper)
        upper_fused = upper_att * sil_upper + (1 - upper_att) * map_upper
        upper_diff = self.upper_diff(upper_fused)

        # Lower body fusion and difference feature extraction
        lower_att = self.lower_attention(local_lower)
        lower_fused = lower_att * sil_lower + (1 - lower_att) * map_lower
        lower_diff = self.lower_diff(lower_fused)

        # Combine upper and lower body features
        local_fused = torch.cat([upper_fused, lower_fused], dim=3)
        local_diff = torch.cat([upper_diff, lower_diff], dim=3)

        # Global feature fusion with attention and common pattern extraction
        global_att = self.global_attention(global_feat)
        global_fused = global_att * sil_feat + (1 - global_att) * map_feat
        global_common = self.common_pattern(global_fused)

        # Original model's main fusion
        main_feat = self.final_fusion(torch.cat([local_fused, global_fused], dim=1))

        # Auxiliary features fusion
        aux_feat = self.aux_fusion(torch.cat([local_diff, global_common], dim=1))

        # Combine main and auxiliary features with learnable weight
        final_feat = main_feat + self.aux_weight * aux_feat

        return final_feat


class SkeletonGaitPP(BaseModel):

    def build_network(self, model_cfg):
        # B, C = [1, 4, 4, 1], 2
        in_C, B, C = model_cfg['Backbone']['in_channels'], model_cfg['Backbone']['blocks'], model_cfg['Backbone']['C']
        self.inference_use_emb = model_cfg['use_emb2'] if 'use_emb2' in model_cfg else False

        self.inplanes = 32 * C
        self.sil_layer0 = SetBlockWrapper(nn.Sequential(
            conv3x3(1, self.inplanes, 1),
            nn.BatchNorm2d(self.inplanes),
            nn.ReLU(inplace=True)
        ))

        self.map_layer0 = SetBlockWrapper(nn.Sequential(
            conv3x3(1, self.inplanes, 1),
            nn.BatchNorm2d(self.inplanes),
            nn.ReLU(inplace=True)
        ))

        self.sil_layer1 = SetBlockWrapper(
            self.make_layer(BasicBlock2D, 32 * C, stride=[1, 1], blocks_num=B[0], mode='2d'))
        self.map_layer1 = copy.deepcopy(self.sil_layer1)
        self.fusion = SHIFFusion(32 * C)

        self.layer2 = self.make_layer(BasicBlockP3D, 64 * C, stride=[2, 2], blocks_num=B[1], mode='p3d')
        self.layer3 = self.make_layer(BasicBlockP3D, 128 * C, stride=[2, 2], blocks_num=B[2], mode='p3d')
        self.layer4 = self.make_layer(BasicBlockP3D, 256 * C, stride=[1, 1], blocks_num=B[3], mode='p3d')

        self.FCs = SeparateFCs(16, 256 * C, 128 * C)
        self.BNNecks = SeparateBNNecks(16, 128 * C, class_num=model_cfg['SeparateBNNecks']['class_num'])

        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=[16])

    def make_layer(self, block, planes, stride, blocks_num, mode='2d'):

        if max(stride) > 1 or self.inplanes != planes * block.expansion:
            if mode == '3d':
                downsample = nn.Sequential(
                    nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=[1, 1, 1], stride=stride,
                              padding=[0, 0, 0], bias=False), nn.BatchNorm3d(planes * block.expansion))
            elif mode == '2d':
                downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride=stride),
                                           nn.BatchNorm2d(planes * block.expansion))
            elif mode == 'p3d':
                downsample = nn.Sequential(
                    nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=[1, 1, 1], stride=[1, *stride],
                              padding=[0, 0, 0], bias=False), nn.BatchNorm3d(planes * block.expansion))
            else:
                raise TypeError('xxx')
        else:
            downsample = lambda x: x

        layers = [block(self.inplanes, planes, stride=stride, downsample=downsample)]
        self.inplanes = planes * block.expansion
        s = [1, 1] if mode in ['2d', 'p3d'] else [1, 1, 1]
        for i in range(1, blocks_num):
            layers.append(
                block(self.inplanes, planes, stride=s)
            )
        return nn.Sequential(*layers)

    def inputs_pretreament(self, inputs):
        ### Ensure the same data augmentation for heatmap and silhouette
        pose_sils = inputs[0]
        new_data_list = []
        for pose, sil in zip(pose_sils[0], pose_sils[1]):
            sil = sil[:, np.newaxis, ...]  # [T, 1, H, W]
            pose_h, pose_w = pose.shape[-2], pose.shape[-1]
            sil_h, sil_w = sil.shape[-2], sil.shape[-1]
            if sil_h != sil_w and pose_h == pose_w:
                cutting = (sil_h - sil_w) // 2
                pose = pose[..., cutting:-cutting]
            # Only use joint heatmap and silhouette
            pose = pose[:, :1, ...]  # Only keep joint heatmap
            cat_data = np.concatenate([pose, sil], axis=1)  # [T, 2, H, W]
            new_data_list.append(cat_data)
        new_inputs = [[new_data_list], inputs[1], inputs[2], inputs[3], inputs[4]]
        return super().inputs_pretreament(new_inputs)

    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs

        pose = ipts[0]
        pose = pose.transpose(1, 2).contiguous()
        assert pose.size(-1) in [44, 48, 88, 96]
        maps = pose[:, :1, ...]  # joint heatmap
        sils = pose[:, 1:, ...]  # silhouette

        del ipts
        map0 = self.map_layer0(maps)
        map1 = self.map_layer1(map0)

        sil0 = self.sil_layer0(sils)
        sil1 = self.sil_layer1(sil0)

        out1 = self.fusion(sil1, map1)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)  # [n, c, s, h, w]

        # Temporal Pooling, TP
        outs = self.TP(out4, seqL, options={"dim": 2})[0]  # [n, c, h, w]
        n, c, h, w = outs.size()

        # Horizontal Pooling Matching, HPM
        feat = self.HPP(outs)  # [n, c, p]

        embed_1 = self.FCs(feat)  # [n, c, p]
        embed_2, logits = self.BNNecks(embed_1)  # [n, c, p]

        if self.inference_use_emb:
            embed = embed_2
        else:
            embed = embed_1

        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed_1, 'labels': labs},
                'softmax': {'logits': logits, 'labels': labs}
            },
            'visual_summary': {
                'image/sils': rearrange(pose * 255., 'n c s h w -> (n s) c h w'),
            },
            'inference_feat': {
                'embeddings': embed
            }
        }
        return retval


class AttentionFusion(nn.Module):
    def __init__(self, in_channels=64, squeeze_ratio=16):
        super(AttentionFusion, self).__init__()
        hidden_dim = int(in_channels / squeeze_ratio)
        self.conv = SetBlockWrapper(
            nn.Sequential(
                conv1x1(in_channels * 2, hidden_dim),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                conv3x3(hidden_dim, hidden_dim),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                conv1x1(hidden_dim, in_channels * 2),
            )
        )

    def forward(self, sil_feat, map_feat):
        '''
            sil_feat: [n, c, s, h, w]
            map_feat: [n, c, s, h, w]
        '''
        c = sil_feat.size(1)
        feats = torch.cat([sil_feat, map_feat], dim=1)
        score = self.conv(feats)  # [n, 2 * c, s, h, w]
        score = rearrange(score, 'n (d c) s h w -> n d c s h w', d=2)
        score = F.softmax(score, dim=1)
        retun = sil_feat * score[:, 0] + map_feat * score[:, 1]
        return retun


class CatFusion(nn.Module):
    def __init__(self, in_channels=64):
        super(CatFusion, self).__init__()
        self.conv = SetBlockWrapper(
            nn.Sequential(
                conv1x1(in_channels * 2, in_channels),
            )
        )

    def forward(self, sil_feat, map_feat):
        '''
            sil_feat: [n, c, s, h, w]
            map_feat: [n, c, s, h, w]
        '''
        feats = torch.cat([sil_feat, map_feat])
        retun = self.conv(feats)
        return retun


class PlusFusion(nn.Module):
    def __init__(self):
        super(PlusFusion, self).__init__()

    def forward(self, sil_feat, map_feat):
        '''
            sil_feat: [n, c, s, h, w]
            map_feat: [n, c, s, h, w]
        '''
        return sil_feat + map_feat
