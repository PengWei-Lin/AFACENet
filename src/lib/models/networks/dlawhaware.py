from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import logging
import numpy as np
from os.path import join

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

#from .DCNv2.dcn_v2 import DCN
#from mmcv.ops import DeformConv2dPack as DCN
from mmcv.ops import ModulatedDeformConv2dPack as DCN

from mmcv.ops import CARAFEPack

#from .dcn.modules.deform_conv import ModulatedDeformConvPack as DCN

#from .dcnv1 import DeformableConv2d
#from .dcnv2 import DeformableConv2d


BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

focal=707.0493

def get_model_url(data='imagenet', name='dla34', hash='ba72cf86'):
    return join('http://dl.yf.io/dla/models', data, '{}-{}.pth'.format(name, hash))


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class mSEModule(nn.Module):
    def __init__(self, out_channel):
        super(mSEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(out_channel, out_channel, kernel_size=1, padding=0)
        #self.Sigmoid = nn.Sigmoid()

    def forward(self, x):
        input = x
        x = self.avg_pool(x)
        x = self.fc(x)
        #x = self.Sigmoid(x)
        x = F.softmax(x, dim=1)
        return input * x


class CSAModuleV2(nn.Module):
    def __init__(self, in_size):
        super(CSAModuleV2, self).__init__()
        self.ch_at = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, in_size, kernel_size=1,
                      stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),    # important!!!
            nn.Sigmoid(),
        )

        self.sp_at = nn.Sequential(
            nn.Conv2d(in_size, in_size, kernel_size=1, stride=1,
                      padding=0, bias=True, groups=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        chat = self.ch_at(x)
        spat = self.sp_at(x)

        ch_out = x * chat
        sp_out = x * spat

        return ch_out + sp_out



class Learnable_Parameter(nn.Module):
    def __init__(self, in_size, out_size):
        super(Learnable_Parameter, self).__init__()
        self.parameter = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=True, groups=1),
                nn.Sigmoid(),
                )
    
    def forward(self, x):
        x = self.parameter(x)
        return x




class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out
    
class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(Bottleneck, self).__init__()
        expansion = Bottleneck.expansion
        bottle_planes = planes // expansion
        self.conv1 = nn.Conv2d(inplanes, bottle_planes,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(bottle_planes, planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class BottleneckX(nn.Module):
    expansion = 2
    cardinality = 32

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BottleneckX, self).__init__()
        cardinality = BottleneckX.cardinality
        dim = int(math.floor(planes * (BottleneckX.expansion / 64.0)))
        bottle_planes = dim * cardinality
        bottle_planes = planes * cardinality // 32
        self.conv1 = nn.Conv2d(inplanes, bottle_planes,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3,
                               stride=stride, padding=dilation, bias=False,
                               dilation=dilation, groups=cardinality)
        self.bn2 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(bottle_planes, planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, residual):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 1,
            stride=1, bias=False, padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)

        return x


class Tree(nn.Module):
    def __init__(self, levels, block, in_channels, out_channels, stride=1,
                 level_root=False, root_dim=0, root_kernel_size=1,
                 dilation=1, root_residual=False):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride,
                               dilation=dilation)
            self.tree2 = block(out_channels, out_channels, 1,
                               dilation=dilation)
        else:
            self.tree1 = Tree(levels - 1, block, in_channels, out_channels,
                              stride, root_dim=0,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
            self.tree2 = Tree(levels - 1, block, out_channels, out_channels,
                              root_dim=root_dim + out_channels,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
        if levels == 1:
            self.root = Root(root_dim, out_channels, root_kernel_size,
                             root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)  # Max better!!!!!
            
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
            )

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x  #double SE 2.7058   # Head RWSE (New CSAMV2) 2.7167  # Single RWSE(New CSAMV2) 2.7429   # double eSE 2.7497   #Single eca 2.7506   #Single SE 2.7576   #Single CSAMV2+CSAMV2 Head 2.7721   #SE CSAMV2 2.7769    #SE_CSMA 2.7805     #dobule CSAM 2.7879  # Non 2.7889  # double eca 2.7894   #double RWSE (New CSAMV2) 2.7944    #Single CSAM 2.8173    # double SE with RWSE head 2.8528  loose!   #eca + RWSE head 2.8508  loose!  #eSE 2.86 loose!    #eca_CSMAV2 loose! 2.9xx  #double eca + CSAMV2 head 2.9906 fail..... 
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


class DLA(nn.Module):
    def __init__(self, levels, channels, num_classes=1000,
                 block=BasicBlock, residual_root=False, linear_root=False):
        super(DLA, self).__init__()
        self.channels = channels
        self.num_classes = num_classes
        self.base_layer = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=7, stride=1,
                      padding=3, bias=False),
            nn.BatchNorm2d(channels[0], momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True))
        self.level0 = self._make_conv_level(
            channels[0], channels[0], levels[0])
        self.level1 = self._make_conv_level(
            channels[0], channels[1], levels[1], stride=2)
        self.level2 = Tree(levels[2], block, channels[1], channels[2], 2,
                           level_root=False,
                           root_residual=residual_root)
        self.level3 = Tree(levels[3], block, channels[2], channels[3], 2,
                           level_root=True, root_residual=residual_root)
        self.level4 = Tree(levels[4], block, channels[3], channels[4], 2,
                           level_root=True, root_residual=residual_root)
        self.level5 = Tree(levels[5], block, channels[4], channels[5], 2,
                           level_root=True, root_residual=residual_root)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def _make_level(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.MaxPool2d(stride, stride=stride),
                nn.Conv2d(inplanes, planes,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample=downsample))
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2d(inplanes, planes, kernel_size=3,
                          stride=stride if i == 0 else 1,
                          padding=dilation, bias=False, dilation=dilation),
                nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)])
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(self, x):
        y = []
        x = self.base_layer(x)
        for i in range(6):
            x = getattr(self, 'level{}'.format(i))(x)
            y.append(x)
        return y

    def load_pretrained_model(self, data='imagenet', name='dla34', hash='ba72cf86'):
        # fc = self.fc
        if name.endswith('.pth'):
            model_weights = torch.load(data + name)
        else:
            model_url = get_model_url(data, name, hash)
            model_weights = model_zoo.load_url(model_url)
        num_classes = len(model_weights[list(model_weights.keys())[-1]])
        self.fc = nn.Conv2d(
            self.channels[-1], num_classes,
            kernel_size=1, stride=1, padding=0, bias=True)
        self.load_state_dict(model_weights, strict=False)
        # self.fc = fc


def dlawhaware34(pretrained=True, **kwargs):  # DLA-34
    model = DLA([1, 1, 1, 2, 2, 1],
                [16, 32, 64, 128, 256, 512],
                block=BasicBlock, **kwargs)
    # block=BBlock, **kwargs)
    # block=BottleneckX, **kwargs)
    # block=Bottleneck, **kwargs)
    if pretrained:
        model.load_pretrained_model(
            data='imagenet', name='dla34', hash='ba72cf86')
    return model


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


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


class DeformConv(nn.Module):
    def __init__(self, chi, cho):
        super(DeformConv, self).__init__()
        self.actf = nn.Sequential(
            nn.BatchNorm2d(cho, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        #self.conv = DCN(chi, cho, kernel_size=(3, 3), stride=1, padding=1, dilation=1, deformable_groups=1)
        self.conv = DCN(chi, cho, kernel_size=3, stride=1, padding=1, dilation=1, deform_groups=1)
        #self.conv = DeformableConv2d(chi, cho, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.actf(x)
        return x


class DLASeg(nn.Module):
    def __init__(self, base_name, heads, pretrained, down_ratio, final_kernel,
                 last_level, head_conv, out_channel=0):
        super(DLASeg, self).__init__()
        assert down_ratio in [2, 4, 8, 16]
        self.first_level = int(np.log2(down_ratio))
        self.last_level = last_level
        self.base = globals()[base_name](pretrained=pretrained)
        channels = self.base.channels

        if out_channel == 0:
            out_channel = channels[self.first_level]
        
        
        self.conv_up_level1 = DeformConv(512, 256)
        #self.carafe1 = CARAFEPack(channels = 256, scale_factor=2, up_kernel = 5, up_group = 1, encoder_kernel = 3, encoder_dilation = 1, compressed_channels = 64)
        self.conv_cat1 = DeformConv(512, 256)
        
        self.conv_up_level2 = DeformConv(256, 128)
        #self.carafe2 = CARAFEPack(channels = 128, scale_factor=2, up_kernel = 5, up_group = 1, encoder_kernel = 3, encoder_dilation = 1, compressed_channels = 64)
        self.conv_cat2 = DeformConv(256, 128)
        
        self.conv_up_level3 = DeformConv(128, 64)
        #self.carafe3 = CARAFEPack(channels = 64, scale_factor=2, up_kernel = 5, up_group = 1, encoder_kernel = 3, encoder_dilation = 1, compressed_channels = 64)
        self.conv_cat3 = DeformConv(128, 64)
        
        
        
        self.carafe_hm1_3 = CARAFEPack(channels = 3, scale_factor=4, up_kernel = 5, up_group = 1, encoder_kernel = 3, encoder_dilation = 1, compressed_channels = 64)
        self.carafe_hm2_3 = CARAFEPack(channels = 3, scale_factor=2, up_kernel = 5, up_group = 1, encoder_kernel = 3, encoder_dilation = 1, compressed_channels = 64)
        
        self.carafe_dim1_3 = CARAFEPack(channels = 3, scale_factor=4, up_kernel = 5, up_group = 1, encoder_kernel = 3, encoder_dilation = 1, compressed_channels = 64)
        self.carafe_dim2_3 = CARAFEPack(channels = 3, scale_factor=2, up_kernel = 5, up_group = 1, encoder_kernel = 3, encoder_dilation = 1, compressed_channels = 64)
        
        self.carafe_dep1_1 = CARAFEPack(channels = 1, scale_factor=4, up_kernel = 5, up_group = 1, encoder_kernel = 3, encoder_dilation = 1, compressed_channels = 64)
        self.carafe_dep2_1 = CARAFEPack(channels = 1, scale_factor=2, up_kernel = 5, up_group = 1, encoder_kernel = 3, encoder_dilation = 1, compressed_channels = 64)
        
        self.carafe_rot1_8 = CARAFEPack(channels = 8, scale_factor=4, up_kernel = 5, up_group = 1, encoder_kernel = 3, encoder_dilation = 1, compressed_channels = 64)
        self.carafe_rot2_8 = CARAFEPack(channels = 8, scale_factor=2, up_kernel = 5, up_group = 1, encoder_kernel = 3, encoder_dilation = 1, compressed_channels = 64)
        
        self.carafe_reg1_2 = CARAFEPack(channels = 2, scale_factor=4, up_kernel = 5, up_group = 1, encoder_kernel = 3, encoder_dilation = 1, compressed_channels = 64)
        self.carafe_reg2_2 = CARAFEPack(channels = 2, scale_factor=2, up_kernel = 5, up_group = 1, encoder_kernel = 3, encoder_dilation = 1, compressed_channels = 64)
        
        self.carafe_wh1_2 = CARAFEPack(channels = 2, scale_factor=4, up_kernel = 5, up_group = 1, encoder_kernel = 3, encoder_dilation = 1, compressed_channels = 64)
        self.carafe_wh2_2 = CARAFEPack(channels = 2, scale_factor=2, up_kernel = 5, up_group = 1, encoder_kernel = 3, encoder_dilation = 1, compressed_channels = 64)
        
        
        # Add amodel center
        self.carafe_act1_2 = CARAFEPack(channels = 2, scale_factor=4, up_kernel = 5, up_group = 1, encoder_kernel = 3, encoder_dilation = 1, compressed_channels = 64)
        self.carafe_act2_2 = CARAFEPack(channels = 2, scale_factor=2, up_kernel = 5, up_group = 1, encoder_kernel = 3, encoder_dilation = 1, compressed_channels = 64)
        # Add amodel center
        
        
        self.attention_head = CSAModuleV2(in_size=3)
        self.lpr = Learnable_Parameter(2, 1)
        
        self.heads = heads
        fpn_channels = [256, 128, 64]
        for fpn_idx, fpn_c in enumerate(fpn_channels):
            for head in sorted(self.heads):
                classes = self.heads[head]
                if head_conv > 0:
                    if 'hm' in head:
                        fc = nn.Sequential(
                            nn.Conv2d(fpn_c, head_conv, kernel_size=3, padding=1, bias=True),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(head_conv, classes,
                                      kernel_size=final_kernel, stride=1,
                                      padding=final_kernel // 2, bias=True),
                        )
                        fc[-1].bias.data.fill_(-2.19)
                    else:
                        fc = nn.Sequential(
                            nn.Conv2d(fpn_c, head_conv, kernel_size=3, padding=1, bias=True),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(head_conv, classes,
                                      kernel_size=final_kernel, stride=1,
                                      padding=final_kernel // 2, bias=True))
                        fill_fc_weights(fc)
                else:
                    fc = nn.Conv2d(fpn_c, classes,
                                   kernel_size=final_kernel, stride=1,
                                   padding=final_kernel // 2, bias=True)
                    if 'hm' in head:
                        fc.bias.data.fill_(-2.19)
                    else:
                        fill_fc_weights(fc)
                self.__setattr__('fpn{}_{}'.format(fpn_idx, head), fc)
        
                

    def forward(self, x):
        _, _, input_h, input_w = x.size()
        hm_h, hm_w = input_h // 4, input_w // 4
        x = self.base(x)

        # New
        # up_level1: torch.Size([b, 512, 14, 14])
        up_level1 = F.interpolate(self.conv_up_level1(x[5]), scale_factor=2, mode='bilinear', align_corners=True)
        #up_level1 = self.carafe1(self.conv_up_level1(x[5]))
        #print("Level1:{}".format(up_level1.shape))

        concat_level1 = self.conv_cat1(torch.cat((up_level1, x[4]), dim=1))
        # up_level2: torch.Size([b, 256, 28, 28])
        up_level2 = F.interpolate(self.conv_up_level2(concat_level1), scale_factor=2, mode='bilinear', align_corners=True)
        #up_level2 = self.carafe2(self.conv_up_level2(concat_level1))
        #print("Level2:{}".format(up_level2.shape))

        concat_level2 = self.conv_cat2(torch.cat((up_level2, x[3]), dim=1))
        # up_level3: torch.Size([b, 128, 56, 56]),
        up_level3 = F.interpolate(self.conv_up_level3(concat_level2), scale_factor=2, mode='bilinear', align_corners=True)
        #up_level3 = self.carafe3(self.conv_up_level3(concat_level2))
        #print("Level3:{}".format(up_level3.shape))
        # up_level4: torch.Size([b, 64, 56, 56])
        concat_level3 = self.conv_cat3(torch.cat((up_level3, x[2]), dim=1))
        #print("Level4:{}".format(concat_level3.shape))
        
        
        ret = {}
        #head_count = 0
        for head in self.heads:
            #head_count += 1
            #print(head_count)
            temp_outs = []
            # original     up_level2, up_level3, up_level4
            for fpn_idx, fdn_input in enumerate([concat_level1, concat_level2, concat_level3]):
                fpn_out = self.__getattr__(
                    'fpn{}_{}'.format(fpn_idx, head))(fdn_input)
                _, _, fpn_out_h, fpn_out_w = fpn_out.size()
                #_, fpn_out_c, fpn_out_h, fpn_out_w = fpn_out.size()
                # Make sure the added features having same size of heatmap output
                #if (fpn_out_w != hm_w) or (fpn_out_h != hm_h):
                    #fpn_out = F.interpolate(fpn_out, size=(hm_h, hm_w))
                
                
                if hm_w // fpn_out_w == 4 and head == 'hm':
                    fpn_out = self.carafe_hm1_3(fpn_out)
                elif hm_w // fpn_out_w == 2 and head == 'hm':
                    fpn_out =self.carafe_hm2_3(fpn_out)
                # Add amodel center
                elif hm_w // fpn_out_w == 4 and head == 'act':
                    fpn_out = self.carafe_act1_2(fpn_out)
                elif hm_w // fpn_out_w == 2 and head == 'act':
                    fpn_out = self.carafe_act2_2(fpn_out)
                # Add amodel center
                elif hm_w // fpn_out_w == 4 and head == 'dep':
                    fpn_out = self.carafe_dep1_1(fpn_out)
                elif hm_w // fpn_out_w == 2 and head == 'dep':
                    fpn_out =self.carafe_dep2_1(fpn_out)
                elif hm_w // fpn_out_w == 4 and head == 'rot':
                    fpn_out = self.carafe_rot1_8(fpn_out)
                elif hm_w // fpn_out_w == 2 and head == 'rot':
                    fpn_out = self.carafe_rot2_8(fpn_out)
                elif hm_w // fpn_out_w == 4 and head == 'dim':
                    fpn_out = self.carafe_dim1_3(fpn_out)
                elif hm_w // fpn_out_w == 2 and head == 'dim':
                    fpn_out = self.carafe_dim2_3(fpn_out)
                elif hm_w // fpn_out_w == 4 and head == 'wh':
                    fpn_out = self.carafe_wh1_2(fpn_out)
                elif hm_w // fpn_out_w == 2 and head == 'wh':
                    fpn_out = self.carafe_wh2_2(fpn_out)
                elif hm_w // fpn_out_w == 4 and head == 'reg':
                    fpn_out = self.carafe_reg1_2(fpn_out)
                elif hm_w // fpn_out_w == 2 and head == 'reg':
                    fpn_out = self.carafe_reg2_2(fpn_out)
                #print("FPN out:{}".format(fpn_out.shape))
                temp_outs.append(fpn_out)
            # Take the softmax in the keypoint feature pyramid network
            #print("temp:{}".format(temp_outs))
            final_out = self.apply_kfpn(temp_outs)
            
            if head == 'hm':
                final_out = self.attention_head(final_out)
                
            elif head == 'dim':
                rH = final_out[:,0:1,:,:]
                #rW = final_out[:,1:2,:,:]
                
            elif head == 'wh':
                #pw = final_out[:,0:1,:,:]
                ph = final_out[:,1:2,:,:]
                
            elif head == 'dep':
                zh = focal*rH/ph
                #zw = focal*rW/pw
                z = final_out
                
                #zc = torch.cat((zh, zw), dim=1)
                #zc = torch.cat((zc, z), dim=1)
                zc = torch.cat((zh, z), dim=1)
                
                #sft_out = F.softmax(zc, dim=1)
                #final_out = (zc * sft_out).sum(dim=1)
                alpha = self.lpr(zc)
                final_out = alpha * zh + (1-alpha) * z
                #print("final out{}".format(final_out.shape))
            
            
            
                
            ret[head] = final_out

        return [ret]

    # Previous Modifield

    def apply_kfpn(self, outs):
        outs = torch.cat([out.unsqueeze(-1) for out in outs], dim=-1)
        softmax_outs = F.softmax(outs, dim=-1)
        ret_outs = (outs * softmax_outs).sum(dim=-1)
        return ret_outs
    

def get_pose_net(num_layers, heads, head_conv=256, down_ratio=4):
    model = DLASeg('dlawhaware{}'.format(num_layers), heads,
                   pretrained=True,
                   down_ratio=down_ratio,
                   final_kernel=1,
                   last_level=5,
                   head_conv=head_conv)
    return model
