from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch.nn.functional as F
import torch
from torch import nn
from torch.nn import init
#from .DCNv2.dcn_v2 import DCN


class CSAModule(nn.Module):
    def __init__(self, in_size):
        super(CSAModule, self).__init__()
        self.ch_at = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, in_size, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Sigmoid(),
        )
        
        self.sp_at = nn.Sequential(
            nn.Conv2d(in_size, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid(),
        )
        
    def forward(self, x):
        chat = self.ch_at(x)
        spat = self.sp_at(x)
        
        ch_out = x * chat
        sp_out = x * spat
        
        return ch_out + sp_out
    

class CSSAModel(nn.Module):
    def __init__(self, in_size):
        super(CSSAModel, self).__init__()
        self.ch_at = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, in_size, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Sigmoid(),
        )
        
        self.ss_at = nn.Sequential(
            #nn.Conv2d(in_size, in_size, kernel_size=1, stride=1, padding=0, groups=in_size, bias=True),
            #nn.BatchNorm2d(in_size),
            #nn.ReLU(inplace=True),
            nn.Conv2d(in_size, in_size, kernel_size=1, stride=1, padding=0, groups=in_size, bias=True),
            #nn.BatchNorm2d(in_size),
            nn.ReLU(inplace=True),
            nn.Sigmoid(),
        )
        
    def forward(self, x):
        chat = self.ch_at(x)
        spat = self.ss_at(x)
        
        ch_out = x * chat
        sp_out = x * spat
        
        return ch_out + sp_out


class RDFE(nn.Module):
    def __init__(self, chi, cho):
        super(RDFE, self).__init__()
        self.actf = nn.Sequential(
            nn.BatchNorm2d(cho//5, momentum=0.1),
            nn.ReLU(inplace=True)
        )
        self.conv_b1 = nn.Sequential(
            nn.Conv2d(chi, cho//5, kernel_size=1, stride=1, padding=0, dilation=1, groups=1),
            nn.BatchNorm2d(cho//5, momentum=0.1),
			nn.ReLU(inplace=True),
            )
        self.conv_b2 = nn.Sequential(
            nn.Conv2d(chi, cho//5, kernel_size=3, stride=1, padding=6, dilation=6, groups=1),
            nn.BatchNorm2d(cho//5, momentum=0.1),
			nn.ReLU(inplace=True),
            )
        '''
        self.conv_d2 = nn.Sequential(
            nn.Conv2d(chi, 256, kernel_size=3, stride=1, padding=int((3-1)/2)*2, dilation=2, groups=1),
            nn.BatchNorm2d(256, momentum=0.1),
			nn.ReLU(inplace=True),
            )
        '''
        self.conv_b3 = nn.Sequential(
            nn.Conv2d(chi, cho//5, kernel_size=3, stride=1, padding=12, dilation=12, groups=1),
            nn.BatchNorm2d(cho//5, momentum=0.1),
			nn.ReLU(inplace=True),
            )
        self.conv_b4 = nn.Sequential(
            nn.Conv2d(chi, cho//5, kernel_size=3, stride=1, padding=18, dilation=18, groups=1),
            nn.BatchNorm2d(cho//5, momentum=0.1),
			nn.ReLU(inplace=True),
            )
        self.pool_b5 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(chi, cho//5, 1, stride=1, bias=False),
            nn.BatchNorm2d(cho//5, momentum=0.1), 
            nn.ReLU(inplace=True)
            )
        self.conv_cat = nn.Sequential(
			nn.Conv2d(cho, cho, 1, 1, padding=0, bias=True),
			nn.BatchNorm2d(cho, momentum=0.1),
			nn.ReLU(inplace=True),		
		)
        
    def forward(self, x):
        [b, c, row, col] = x.size()
        
        
        
        b1 = self.conv_b1(x)
        b2 = self.conv_b2(x)
        b3 = self.conv_b3(x)
        b4 = self.conv_b4(x)
        
        b5 = self.pool_b5(x)
        b5 = F.interpolate(b5, (row, col), None, 'bilinear', True)
        
        final_cat = torch.cat([b1, b2, b3, b4, b5], dim=1)
        feature_cat = self.conv_cat(final_cat)
        
        residual = x + feature_cat

        return residual


class Att_Proj_Node_o(nn.Module):
    def __init__(self, chi, cho):
        super(Att_Proj_Node_o, self).__init__()
        #self.conv = DCN(chi, cho, kernel_size=(3, 3), stride=1, padding=1, dilation=1, deformable_groups=1)
        self.conv = nn.Conv2d(chi, cho, kernel_size=3, stride=1, padding=1, dilation=1, groups=1) #bias=True)
        self.actf = nn.Sequential(
            nn.BatchNorm2d(cho, momentum=0.1),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        x = self.conv(x)
        out = self.actf(x)
        
        return out

class Att_Proj_Node_n(nn.Module):
    def __init__(self, chi, cho):
        super(Att_Proj_Node_n, self).__init__()
        #self.conv = DCN(chi, cho, kernel_size=(3, 3), stride=1, padding=1, dilation=1, deformable_groups=1)
        self.CSAM = CSAModule(chi)
        self.conv = nn.Conv2d(chi, cho, kernel_size=3, stride=1, padding=1, dilation=1, groups=1) #bias=True)
        self.actf = nn.Sequential(
            nn.BatchNorm2d(cho, momentum=0.1),
            nn.ReLU(inplace=True)
        )
        #self.conv_L1 = nn.Conv2d(chi, cho, kernel_size=(15, 1), stride=1, padding=4, dilation=1, groups=1)
        #self.conv_L2 = nn.Conv2d(cho, cho, kernel_size=(1, 15), stride=1, padding=3, dilation=1, groups=1)
        #self.conv_R1 = nn.Conv2d(chi, cho, kernel_size=(1, 15), stride=1, padding=4, dilation=1, groups=1)
        #self.conv_R2 = nn.Conv2d(cho, cho, kernel_size=(15, 1), stride=1, padding=3, dilation=1, groups=1)
    def forward(self, x):
        
        x = self.CSAM(x)
        x = self.conv(x)
        out = self.actf(x)
        #L = self.conv_L1(x)
        #L = self.conv_L2(L)
        
        #R = self.conv_R1(x)
        #R = self.conv_R2(R)
        
        return out#L+R

class Att_Proj_Node(nn.Module):
    def __init__(self, chi, cho):
        super(Att_Proj_Node, self).__init__()
        #self.conv = DCN(chi, cho, kernel_size=(3, 3), stride=1, padding=1, dilation=1, deformable_groups=1)
        self.conv = nn.Conv2d(chi, cho, kernel_size=3, stride=1, padding=1, dilation=1, groups=1) #bias=True)
        self.CSAM = CSAModule(cho)
        self.actf = nn.Sequential(
            nn.BatchNorm2d(cho, momentum=0.1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        
        x = self.conv(x)
        x = self.CSAM(x)
        out = self.actf(x)
        
        return out#L+R

class IDAUp(nn.Module):

    def __init__(self, o, channels, up_f):
        super(IDAUp, self).__init__()

        for i in range(1, len(channels)):
            c = channels[i]
            f = int(up_f[i])
            #print(f)
            proj = Att_Proj_Node_o(c, o)
            node = Att_Proj_Node_o(o, o)
            #up = nn.ConvTranspose2d(o, o, f * 2, stride=f,
                                    #padding=f // 2, output_padding=0,
                                    #groups=o, bias=False)
            up = nn.Sequential(
                nn.Upsample(size=None, scale_factor=f, mode='bilinear'),
                nn.Conv2d(o, o, kernel_size=1, stride=1, padding=0, groups=o, bias=False),
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


class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out


class eSeModule(nn.Module):
    def __init__(self, in_size):
        super(eSeModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, in_size, kernel_size=1, stride=1, padding=0, bias=True),#bias=False),
            nn.BatchNorm2d(in_size),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return x * self.se(x)


class Block(nn.Module):
    '''expand + depthwise + pointwise'''

    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(Block, self).__init__()
        self.stride = stride
        self.se = semodule
        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride,
                               padding=kernel_size // 2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = nolinear
        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_size != out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_size),
            )

    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.nolinear2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.se != None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out


class CSPBlock(nn.Module):
    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride, part_ratio=0.2):
        super(CSPBlock, self).__init__()
        self.part1_chnls = int(in_size * part_ratio)
        self.part2_chnls = in_size - self.part1_chnls
        self.block = Block(kernel_size, self.part2_chnls, expand_size, out_size-self.part1_chnls, nolinear, semodule, stride)
        #self.dwsp = nn.Conv2d(self.part1_chnls, self.part1_chnls, kernel_size=kernel_size, stride=2, padding=kernel_size//2, groups= self.part1_chnls, bias=False)
        
        #self.dw_flag = stride
        
    def forward(self, x):
        part1 = x[:, :self.part1_chnls, :, :]
        part2 = x[:, self.part1_chnls:, :, :]
        
        part2 = self.block(part2)
        #if self.dw_flag == 2:
            #part1 = self.dwsp(part1)
        out = torch.cat((part1, part2), dim=1)
        return out


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


'MobileNetv3 Small'
class MobileNetV3S(nn.Module):
    def __init__(self, heads, final_kernel, head_conv):
        super(MobileNetV3S, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = hswish()

        self.bneck0 = nn.Sequential(
            Block(3, 16, 16, 16, nn.ReLU(inplace=True), eSeModule(16), 2),
        )
        self.bneck1 = nn.Sequential(
            Block(3, 16, 72, 24, nn.ReLU(inplace=True), None, 2),
            Block(3, 24, 88, 24, nn.ReLU(inplace=True), eSeModule(24), 1)
        )
        self.bneck2 = nn.Sequential(
            Block(5, 24, 96, 40, hswish(), eSeModule(40), 2),
            Block(5, 40, 240, 40, hswish(), eSeModule(40), 1),
            Block(5, 40, 240, 40, hswish(), eSeModule(40), 1),
            Block(5, 40, 120, 48, hswish(), eSeModule(48), 1),
            Block(5, 48, 144, 48, hswish(), eSeModule(48), 1),
        )
        self.bneck3 = nn.Sequential(
            Block(5, 48, 288, 96, hswish(), eSeModule(96), 2),
            Block(5, 96, 576, 96, hswish(), eSeModule(96), 1),
            Block(5, 96, 576, 96, hswish(), eSeModule(96), 1),
        )
        self.conv2 = nn.Conv2d(96, 576, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(576)
        self.hs2 = hswish()
        self.init_params()

        self.ida_up = IDAUp(16, [24, 24, 48, 576], [2 ** i for i in range(4)])

        self.heads = heads
        for head_id, head in enumerate(self.heads):
            classes = self.heads[head]
            if head_id == 0:
                fc = nn.Sequential(
                    #nn.Conv2d(64, head_conv, kernel_size=3, padding=1, bias=True),
                    nn.Conv2d(16, head_conv, kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_conv, classes, kernel_size=final_kernel, stride=1, padding=final_kernel // 2, bias=True),
                    CSAModule(in_size=classes),
                    )
                if 'hm' in head:
                    fc[-2].bias.data.fill_(-2.19)
                self.__setattr__(head, fc)
            else:
                fc = nn.Sequential(
                    #nn.Conv2d(64, head_conv,
                    nn.Conv2d(16, head_conv,
                    kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    #CSAModule(in_size=head_conv),
                    nn.Conv2d(head_conv, classes,
                    kernel_size=final_kernel, stride=1,
                    padding=final_kernel // 2, bias=True))
                #fc[-1].bias.data.fill_(-2.19)
                self.__setattr__(head, fc)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.hs1(self.bn1(self.conv1(x)))
        out0 = self.bneck0(out)
        out1 = self.bneck1(out0)
        out2 = self.bneck2(out1)
        out3 = self.bneck3(out2)
        out3 = self.hs2(self.bn2(self.conv2(out3)))
        out = [out0, out1, out2, out3]
        y = []
        for i in range(4):
            y.append(out[i].clone())
        self.ida_up(y, 0, len(y))
        z = {}
        for head in self.heads:
            z[head] = self.__getattr__(head)(y[-1])
        return [z]


'Cross Stage Partial MobileNetV3 Small'
class MobileNetV3SCSP(nn.Module):
    def __init__(self, heads, final_kernel, head_conv):
        super(MobileNetV3SCSP, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = hswish()

        self.bneck0 = nn.Sequential(
            CSPBlock(3, 16, 16, 16, nn.ReLU(inplace=True), eSeModule(8), 2),
        )
        self.bneck1 = nn.Sequential(
            Block(3, 16, 72, 24, nn.ReLU(inplace=True), None, 2),
            CSPBlock(3, 24, 88, 24, nn.ReLU(inplace=True), eSeModule(12), 1)
        )
        self.bneck2 = nn.Sequential(
            Block(5, 24, 96, 40, hswish(), eSeModule(28), 2),
            CSPBlock(5, 40, 240, 40, hswish(), eSeModule(20), 1),
            CSPBlock(5, 40, 240, 40, hswish(), eSeModule(20), 1),
            CSPBlock(5, 40, 120, 48, hswish(), eSeModule(28), 1),
            CSPBlock(5, 48, 144, 48, hswish(), eSeModule(24), 1),
        )
        self.bneck3 = nn.Sequential(
            Block(5, 48, 288, 96, hswish(), eSeModule(72), 2),
            CSPBlock(5, 96, 576, 96, hswish(), eSeModule(48), 1),
            CSPBlock(5, 96, 576, 96, hswish(), eSeModule(48), 1),
        )
        self.conv2 = nn.Conv2d(96, 576, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(576)
        self.hs2 = hswish()
        self.init_params()

        self.ida_up = IDAUp(16, [24, 24, 48, 576], [2 ** i for i in range(4)])

        self.heads = heads
        for head_id, head in enumerate(self.heads):
            classes = self.heads[head]
            if head_id == 0:
                fc = nn.Sequential(
                    #nn.Conv2d(64, head_conv, kernel_size=3, padding=1, bias=True),
                    nn.Conv2d(24, head_conv, kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_conv, classes, kernel_size=final_kernel, stride=1, padding=final_kernel // 2, bias=True),
                    CSAModule(in_size=classes),
                    )
                if 'hm' in head:
                    fc[-1].bias.data.fill_(-2.19)
                self.__setattr__(head, fc)
            else:
                fc = nn.Sequential(
                    #nn.Conv2d(64, head_conv,
                    nn.Conv2d(24, head_conv,
                    kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    #CSAModule(in_size=head_conv),
                    nn.Conv2d(head_conv, classes,
                    kernel_size=final_kernel, stride=1,
                    padding=final_kernel // 2, bias=True))
                #fc[-1].bias.data.fill_(-2.19)
                self.__setattr__(head, fc)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.hs1(self.bn1(self.conv1(x)))
        out0 = self.bneck0(out)
        out1 = self.bneck1(out0)
        out2 = self.bneck2(out1)
        out3 = self.bneck3(out2)
        out3 = self.hs2(self.bn2(self.conv2(out3)))
        out = [out0, out1, out2, out3]
        y = []
        for i in range(4):
            y.append(out[i].clone())
        self.ida_up(y, 0, len(y))
        z = {}
        for head in self.heads:
            z[head] = self.__getattr__(head)(y[-1])
        return [z]


'MobileNetV3 Large'
class MobileNetV3(nn.Module):
    def __init__(self, heads, final_kernel, head_conv):
        super(MobileNetV3, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = hswish()

        self.bneck0 = nn.Sequential(
            Block(3, 16, 16, 16, nn.ReLU(inplace=True), None, 1),
            Block(3, 16, 64, 24, nn.ReLU(inplace=True), None, 2),
            Block(3, 24, 72, 24, nn.ReLU(inplace=True), None, 1),
        )
        self.bneck1 = nn.Sequential(
            Block(5, 24, 72, 40, nn.ReLU(inplace=True), eSeModule(40), 2),
            Block(5, 40, 120, 40, nn.ReLU(inplace=True), eSeModule(40), 1),
            Block(5, 40, 120, 40, nn.ReLU(inplace=True), eSeModule(40), 1),
        )
        self.bneck2 = nn.Sequential(
            Block(3, 40, 240, 80, hswish(), None, 2),
            Block(3, 80, 200, 80, hswish(), None, 1),
            Block(3, 80, 184, 80, hswish(), None, 1),
            Block(3, 80, 184, 80, hswish(), None, 1),
            Block(3, 80, 480, 112, hswish(), eSeModule(112), 1),
            Block(3, 112, 672, 112, hswish(), eSeModule(112), 1),
            Block(5, 112, 672, 160, hswish(), eSeModule(160), 1),
        )
        self.bneck3 = nn.Sequential(
            Block(5, 160, 672, 160, hswish(), eSeModule(160), 2),
            Block(5, 160, 960, 160, hswish(), eSeModule(160), 1),
        )
        self.conv2 = nn.Conv2d(160, 960, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(960)
        self.hs2 = hswish()
        self.init_params()

        self.ida_up = IDAUp(24, [24, 40, 160, 960], [2 ** i for i in range(4)])

        self.heads = heads
        for head_id, head in enumerate(self.heads):
            classes = self.heads[head]
            if head_id == 0:
                fc = nn.Sequential(
                    nn.Conv2d(24, head_conv, kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_conv, classes, kernel_size=final_kernel, stride=1, padding=final_kernel // 2, bias=True),
                    #CSAModule(in_size=classes),
                    )
                if 'hm' in head:
                    #fc[-3].bias.data.fill_(-2.19)
                    fc[-1].bias.data.fill_(-2.19)
                self.__setattr__(head, fc)
            else:
                fc = nn.Sequential(
                    nn.Conv2d(24, head_conv,
                    kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    #CSAModule(in_size=head_conv),
                    nn.Conv2d(head_conv, classes,
                    kernel_size=final_kernel, stride=1,
                    padding=final_kernel // 2, bias=True))
                fill_fc_weights(fc)
                self.__setattr__(head, fc)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.hs1(self.bn1(self.conv1(x)))
        out0 = self.bneck0(out)
        out1 = self.bneck1(out0)
        out2 = self.bneck2(out1)
        out3 = self.bneck3(out2)
        out3 = self.hs2(self.bn2(self.conv2(out3)))
        out = [out0, out1, out2, out3]
        y = []
        for i in range(4):
            y.append(out[i].clone())
        self.ida_up(y, 0, len(y))
        z = {}
        for head in self.heads:
            z[head] = self.__getattr__(head)(y[-1])
        return [z]


'Cross Stage Partial MobileNetV3 Large'
class MobileNetV3CSP(nn.Module):
    def __init__(self, heads, final_kernel, head_conv):
        super(MobileNetV3CSP, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = hswish()

        self.bneck0 = nn.Sequential(
            CSPBlock(3, 16, 16, 16, nn.ReLU(inplace=True), None, 1),
            Block(3, 16, 64, 24, nn.ReLU(inplace=True), None, 2),
            CSPBlock(3, 24, 72, 24, nn.ReLU(inplace=True), None, 1),
        )
        self.bneck1 = nn.Sequential(
            Block(5, 24, 72, 40, nn.ReLU(inplace=True), eSeModule(40), 2),
            CSPBlock(5, 40, 120, 40, nn.ReLU(inplace=True), eSeModule(32), 1),
            CSPBlock(5, 40, 120, 40, nn.ReLU(inplace=True), eSeModule(32), 1),
        )
        self.bneck2 = nn.Sequential(
            Block(3, 40, 240, 80, hswish(), None, 2),
            CSPBlock(3, 80, 200, 80, hswish(), None, 1),
            CSPBlock(3, 80, 184, 80, hswish(), None, 1),
            CSPBlock(3, 80, 184, 80, hswish(), None, 1),
            CSPBlock(3, 80, 480, 112, hswish(), eSeModule(96), 1),
            CSPBlock(3, 112, 672, 112, hswish(), eSeModule(90), 1),
            CSPBlock(5, 112, 672, 160, hswish(), eSeModule(138), 1),
        )
        self.bneck3 = nn.Sequential(
            Block(5, 160, 672, 160, hswish(), eSeModule(160), 2),
            CSPBlock(5, 160, 960, 160, hswish(), eSeModule(128), 1),
        )
        self.conv2 = nn.Conv2d(160, 960, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(960)
        self.hs2 = hswish()
        self.init_params()

        self.ida_up = IDAUp(24, [24, 40, 160, 960], [2 ** i for i in range(4)])
        
        #self.RDFE=RDFE(24, 24)
        #self.RDFE=RDFE(960, 960)
        
        #self.conv_up_level1 = nn.Conv2d(960, 256, kernel_size=1, stride=1, padding=0)
        #self.conv_up_level2 = nn.Conv2d(416, 128, kernel_size=1, stride=1, padding=0)
        #self.conv_up_level3 = nn.Conv2d(168, 64, kernel_size=1, stride=1, padding=0)
        #self.conv_up_level4 = nn.Conv2d(88, 64, kernel_size=1, stride=1, padding=0)
        
        #self.CSAM_level2_p = CSAModule(in_size=416)
        #self.CSAM_level3_p = CSAModule(in_size=64)
        #self.CSAM_level4_p = CSAModule(in_size=64)
        
        #self.CSAM_level2_n = CSAModule(in_size=416)
        #self.CSAM_level3_n = CSAModule(in_size=168)
        #self.CSAM_level4_n = CSAModule(in_size=64)
        
        '''
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
        '''
        
        self.heads = heads
        for head_id, head in enumerate(self.heads):
            classes = self.heads[head]
            if head_id == 0:
                fc = nn.Sequential(
                    #nn.Conv2d(64, head_conv, kernel_size=3, padding=1, bias=True),
                    nn.Conv2d(24, head_conv, kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_conv, classes, kernel_size=final_kernel, stride=1, padding=final_kernel // 2, bias=True),
                    CSAModule(in_size=classes),
                    )
                if 'hm' in head:
                    fc[-2].bias.data.fill_(-2.19)
                self.__setattr__(head, fc)
            else:
                fc = nn.Sequential(
                    #nn.Conv2d(64, head_conv,
                    nn.Conv2d(24, head_conv,
                    kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    #CSAModule(in_size=head_conv),
                    nn.Conv2d(head_conv, classes,
                    kernel_size=final_kernel, stride=1,
                    padding=final_kernel // 2, bias=True))
                #fc[-1].bias.data.fill_(-2.19)
                fill_fc_weights(fc)
                self.__setattr__(head, fc)
        
    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        #_, _, input_h, input_w = x.size()
        #hm_h, hm_w = input_h // 4, input_w // 4
        
        out = self.hs1(self.bn1(self.conv1(x)))
        out0 = self.bneck0(out)
        out1 = self.bneck1(out0)
        out2 = self.bneck2(out1)
        out3 = self.bneck3(out2)
        out3 = self.hs2(self.bn2(self.conv2(out3)))
        #out3 = self.RDFE(out3)
        out = [out0, out1, out2, out3]
        #direct_up = F.interpolate(out3, scale_factor=8)#size=(hm_h, hm_w))
        
        
        #up_level1 = F.interpolate(self.conv_up_level1(out3), scale_factor=2, mode='bilinear', align_corners=True)
        
        #cat_level2 = torch.cat([up_level1, out2], dim=1)
        #up_level2 = F.interpolate(self.conv_up_level2(cat_level2), scale_factor=2, mode='bilinear', align_corners=True)
        
        #cat_level3 = torch.cat([up_level2, out1], dim=1)
        #up_level3 = F.interpolate(self.conv_up_level3(cat_level3), scale_factor=2, mode='bilinear', align_corners=True)
        
        #out_final = self.conv_up_level4(torch.cat([out0, up_level3], dim=1))
        
        
        y = []
        for i in range(4):
            y.append(out[i].clone())
        self.ida_up(y, 0, len(y))
        
        #out_final = self.DFE(y[-1])
        #out_final = self.DFE(direct_up)
        #out_final = self.DFE(out_final)
        
        z = {}
        
        for head in self.heads:
            z[head] = self.__getattr__(head)(y[-1])
        return [z]
        
        
        #for head in self.heads:
            #z[head] = self.__getattr__(head)(out_final)
        #return [z]
        