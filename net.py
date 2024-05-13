import os
import torch
import torch.nn as nn
from collections import OrderedDict

import torch.nn.functional as F
from functools import reduce
from embedding import get_logits_with_matching
from correlation import Correlation

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class STB(nn.Module):
    def __init__(self, device, inplane, outplane, ref_len, correlation_window_size, save_cost, embedding_dim):
        super(STB, self).__init__()
        self.device = device
        inplane = reduce(lambda x, y: x + y, inplane)
        self.feat_conv = SingleConv(inplane, outplane, 1, 1, 0)
        self.ref_len = ref_len
        self.correlation_window_size = correlation_window_size
        self.save_cost = save_cost
        self.embedding_dim = embedding_dim

        self.embedding_conv2d = DoubleConv(in_channels_1=outplane, out_channels_1=embedding_dim,
                                           kernel_size_1=[3, 3], stride_1=(1, 1), padding_1=[1, 1],
                                           in_channels_2=embedding_dim, out_channels_2=embedding_dim,
                                           kernel_size_2=[3, 3], stride_2=(1, 1), padding_2=[1, 1])

        self.embedding_seg_conv2d = DoubleConv(in_channels_1=outplane+1+1, out_channels_1=embedding_dim,
                                           kernel_size_1=[3, 3], stride_1=(1, 1), padding_1=[1, 1],
                                           in_channels_2=embedding_dim, out_channels_2=embedding_dim,
                                           kernel_size_2=[3, 3], stride_2=(1, 1), padding_2=[1, 1])
        if save_cost:
            self.corr = Correlation(pad_size=correlation_window_size, kernel_size=1,
                                    max_displacement=correlation_window_size, stride1=1,
                                    stride2=1, corr_multiply=embedding_dim)     # wait to check

        self.se = SELayer(inplane + embedding_dim)
        self.merge_conv2d = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(inplane + embedding_dim, embedding_dim, [3, 3], (1, 1), [1, 1])),
                # ('dropout', nn.Dropout3d(p=0.5, inplace=True)),
                ('lrule', nn.LeakyReLU(negative_slope=0.01, inplace=True)),
                ('instnorm',
                 nn.InstanceNorm2d(embedding_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)),
                ('conv_out', nn.Conv2d(embedding_dim, 2, [3, 3], (1, 1), [1, 1])),
            ]))


    def forward(self, feature, out):
        s = out.size()[0]  # batch size
        h = out.size()[2]  # 128
        w = out.size()[3]  # 128

        high_h = feature[0].size(-2)  # 16
        high_w = feature[0].size(-1)  # 16
        feature = reduce(lambda x, y: torch.cat(    # concat low_features
            (x, F.interpolate(y, size=(high_h, high_w), mode='bilinear', align_corners=True)), dim=1), feature)
        matching_input = self.feat_conv(feature.detach())
        # shape: (batch_size, outplane(256), 128, 128)

        out = torch.argmax(out, dim=1).unsqueeze(1).float()  # original prediction
        # shape: (batch_size, 1, 128, 128)

        stb_embedding, coor_info = get_logits_with_matching(self,
                                                 matching_input,
                                                 reference_labels=out.detach(),
                                                 ref_len=self.ref_len,
                                                 correlation_window_size=self.correlation_window_size,
                                                 save_cost=self.save_cost)

        m_input = torch.cat((stb_embedding, feature), dim=1)    # 496 + 128 = 624
        m_input = self.se(m_input)
        m_output = self.merge_conv2d(m_input)
        return m_output, coor_info


class SingleConv(nn.Module):
    def __init__(self, in_channels_1, out_channels_1, kernel_size_1, stride_1, padding_1):
        super(SingleConv, self).__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(in_channels_1, out_channels_1, kernel_size_1, stride_1, padding_1)),
                # ('dropout', nn.Dropout3d(p=0.5, inplace=True)),
                ('lrule', nn.LeakyReLU(negative_slope=0.01, inplace=True)),
                ('instnorm',
                 nn.InstanceNorm2d(out_channels_1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False))
            ]))
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

class DoubleConv(nn.Module):
    def __init__(self, in_channels_1, out_channels_1, kernel_size_1, stride_1, padding_1,
                 in_channels_2, out_channels_2, kernel_size_2, stride_2, padding_2):
        super(DoubleConv, self).__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(in_channels_1, out_channels_1, kernel_size_1, stride_1, padding_1)),
                # ('dropout', nn.Dropout3d(p=0.5, inplace=True)),
                ('lrule', nn.LeakyReLU(negative_slope=0.01, inplace=True)),
                ('instnorm',
                 nn.InstanceNorm2d(out_channels_1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False))
            ])),
            nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(in_channels_2, out_channels_2, kernel_size_2, stride_2, padding_2)),
                # ('dropout', nn.Dropout3d(p=0.5, inplace=True)),
                ('lrule', nn.LeakyReLU(negative_slope=0.01, inplace=True)),
                ('instnorm',
                 nn.InstanceNorm2d(out_channels_1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False))
            ]))
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


'''Vanilla Deep Sup Unet for size [128, 128]'''
class Unet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Unet, self).__init__()
        self.conv_blocks_context = nn.ModuleList([
            DoubleConv(in_channels_1=in_channels, out_channels_1=16, kernel_size_1=[3, 3], stride_1=(1, 1),
                       padding_1=[1, 1],
                       in_channels_2=16, out_channels_2=16, kernel_size_2=[3, 3], stride_2=(1, 1),
                       padding_2=[1, 1]),    # [bs, 16, 128, 128]
            DoubleConv(in_channels_1=16, out_channels_1=32, kernel_size_1=[3, 3], stride_1=[2, 2],
                       padding_1=[1, 1],
                       in_channels_2=32, out_channels_2=32, kernel_size_2=[3, 3], stride_2=(1, 1),
                       padding_2=[1, 1]),    # [bs, 32, 64, 64]
            DoubleConv(in_channels_1=32, out_channels_1=64, kernel_size_1=[3, 3], stride_1=[2, 2],
                       padding_1=[1, 1],
                       in_channels_2=64, out_channels_2=64, kernel_size_2=[3, 3], stride_2=(1, 1),
                       padding_2=[1, 1]),    # [bs, 64, 32, 32]
            DoubleConv(in_channels_1=64, out_channels_1=128, kernel_size_1=[3, 3], stride_1=[2, 2],
                       padding_1=[1, 1],
                       in_channels_2=128, out_channels_2=128, kernel_size_2=[3, 3], stride_2=(1, 1),
                       padding_2=[1, 1]),    # [bs, 128, 16, 16]
            DoubleConv(in_channels_1=128, out_channels_1=256, kernel_size_1=[3, 3], stride_1=[2, 2],
                       padding_1=[1, 1],
                       in_channels_2=256, out_channels_2=256, kernel_size_2=[3, 3], stride_2=(1, 1),
                       padding_2=[1, 1]),    # [bs, 256, 8, 8]
        ])

        self.conv_blocks_localization = nn.ModuleList([
            DoubleConv(in_channels_1=256, out_channels_1=128, kernel_size_1=[3, 3], stride_1=(1, 1),
                       padding_1=[1, 1],
                       in_channels_2=128, out_channels_2=128, kernel_size_2=[3, 3], stride_2=(1, 1),
                       padding_2=[1, 1]),    # [bs, 128, 16, 16]
            DoubleConv(in_channels_1=128, out_channels_1=64, kernel_size_1=[3, 3], stride_1=(1, 1),
                       padding_1=[1, 1],
                       in_channels_2=64, out_channels_2=64, kernel_size_2=[3, 3], stride_2=(1, 1),
                       padding_2=[1, 1]),    # [bs, 64, 32, 32]
            DoubleConv(in_channels_1=64, out_channels_1=32, kernel_size_1=[3, 3], stride_1=(1, 1),
                       padding_1=[1, 1],
                       in_channels_2=32, out_channels_2=32, kernel_size_2=[3, 3], stride_2=(1, 1),
                       padding_2=[1, 1]),    # [bs, 32, 64, 64]
            DoubleConv(in_channels_1=32, out_channels_1=16, kernel_size_1=[3, 3], stride_1=(1, 1),
                       padding_1=[1, 1],
                       in_channels_2=16, out_channels_2=16, kernel_size_2=[3, 3], stride_2=(1, 1),
                       padding_2=[1, 1]),    # [bs, 16, 128, 128]
        ])

        self.tu = nn.ModuleList([
            nn.ConvTranspose2d(256, 128, kernel_size=(2, 2), stride=(2, 2), bias=False),
            nn.ConvTranspose2d(128, 64, kernel_size=(2, 2), stride=(2, 2), bias=False),
            nn.ConvTranspose2d(64, 32, kernel_size=(2, 2), stride=(2, 2), bias=False),
            nn.ConvTranspose2d(32, 16, kernel_size=(2, 2), stride=(2, 2), bias=False)
            ])

        self.seg = nn.Conv2d(16, out_channels, kernel_size=(1, 1), stride=(1, 1), bias=False)

        self.stb = STB(device="cuda",
                       inplane=[16, 32, 64, 128, 256],  # feature channel
                       outplane=256,
                       ref_len=1,
                       correlation_window_size=3,
                       save_cost=True,
                       embedding_dim=128)

    def forward(self, x):
        x1 = self.conv_blocks_context[0](x)
        x2 = self.conv_blocks_context[1](x1)
        x3 = self.conv_blocks_context[2](x2)
        x4 = self.conv_blocks_context[3](x3)
        x5 = self.conv_blocks_context[4](x4)

        x = self.tu[0](x5)
        x = torch.cat((x, x4), dim=1)
        x = self.conv_blocks_localization[0](x)

        x = self.tu[1](x)
        x = torch.cat((x, x3), dim=1)
        x = self.conv_blocks_localization[1](x)

        x = self.tu[2](x)
        x = torch.cat((x, x2), dim=1)
        x = self.conv_blocks_localization[2](x)

        x = self.tu[3](x)
        x = torch.cat((x, x1), dim=1)
        x = self.conv_blocks_localization[3](x)

        x = self.seg(x)

        output, coor_info = self.stb([x1, x2, x3, x4, x5], x)
        return output, x, coor_info



if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    net = Unet(1, 2).cuda()
    out, x, dist = net(torch.randn(24, 1, 128, 128).cuda())
    print(out.shape)
