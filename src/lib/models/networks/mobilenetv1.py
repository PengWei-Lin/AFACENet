from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch.nn.functional as F
from torch import nn
from torch.nn import init
#from .DCNv2.dcn_v2 import DCN


class DeformConv(nn.Module):
    def __init__(self, chi, cho):
        super(DeformConv, self).__init__()
        self.actf = nn.Sequential(
            nn.BatchNorm2d(cho, momentum=0.1),
            nn.ReLU(inplace=True)
        )
        #self.conv = DCN(chi, cho, kernel_size=(3, 3), stride=1, padding=1, dilation=1, deformable_groups=1)
        self.conv = nn.Conv2d(chi, cho, kernel_size=3, stride=1, padding=1, dilation=1, groups=1)
    def forward(self, x):
        x = self.conv(x)
        x = self.actf(x)
        return x


class IDAUp(nn.Module):

    def __init__(self, o, channels, up_f):
        super(IDAUp, self).__init__()

        layers = []
        for i in range(1, len(channels)):
            c = channels[i]
            f = int(up_f[i])
            #print(f)
            proj = DeformConv(c, o)
            node = DeformConv(o, o)
            #up = nn.ConvTranspose2d(o, o, f * 2, stride=f,
                                    #padding=f // 2, output_padding=0,
                                    #groups=o, bias=False)
            up = nn.Sequential(
                nn.Upsample(size=None, scale_factor=f, mode='bilinear'),
                #nn.UpsamplingNearest2d(size=None, scale_factor=f),
                nn.Conv2d(o, o, kernel_size=1, stride=1, padding=0, groups=o, bias=False)
            )
            #fill_up_weights(up)
            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)
            setattr(self, 'node_' + str(i), node)

    def forward(self, layers, startp, endp):
        for i in range(startp + 1, endp):
            upsample = getattr(self, 'up_' + str(i - startp))
            project = getattr(self, 'proj_' + str(i - startp))
            layers[i] = upsample(project(layers[i]))
            node = getattr(self, 'node_' + str(i - startp))
            layers[i] = node(layers[i] + layers[i - 1])

'MobileNetv3 Large'
class MobileNetV1(nn.Module):
    def __init__(self, heads, final_kernel, head_conv):
        super(MobileNetV1, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
                )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                # pw
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
                )

        self.first_conv = nn.Sequential(
            conv_bn(3, 16, 2)
        )
        
        self.block0 = nn.Sequential(
            conv_dw(16, 32, 1),
            conv_dw(32, 64, 2),
        )
        
        self.block1 = nn.Sequential(
            conv_dw(64, 64, 1),
            conv_dw(64, 128, 2),
        )
        
        self.block2 = nn.Sequential(
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
        )
        
        self.block3 = nn.Sequential(
            conv_dw(256, 256, 1),
            conv_dw(256, 256, 1),
            conv_dw(256, 256, 1),
            conv_dw(256, 256, 1),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
        )

        self.ida_up = IDAUp(64, [64, 128, 256, 512], [2 ** i for i in range(4)])

        self.heads = heads
        for head in self.heads:
            classes = self.heads[head]
            fc = nn.Sequential(
                nn.Conv2d(64, head_conv,
                          kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_conv, classes,
                          kernel_size=final_kernel, stride=1,
                          padding=final_kernel // 2, bias=True))
            fc[-1].bias.data.fill_(-2.19)
            self.__setattr__(head, fc)
            
    def forward(self, x):
        out = self.first_conv(x)
        out0 = self.block0(out)
        out1 = self.block1(out0)
        out2 = self.block2(out1)
        out3 = self.block3(out2)
        out = [out0, out1, out2, out3]
        y = []
        for i in range(4):
            y.append(out[i].clone())
        self.ida_up(y, 0, len(y))
        z = {}
        for head in self.heads:
            z[head] = self.__getattr__(head)(y[-1])
        return [z]
