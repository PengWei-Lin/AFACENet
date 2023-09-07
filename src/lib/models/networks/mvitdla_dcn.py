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

from typing import Tuple, Optional, Callable
from timm.models.layers import GroupNorm1
from timm.models.layers import DropPath
from timm.models.layers import ConvNormAct
from timm.models.layers import BatchNormAct2d
from timm.models.layers import make_divisible


#from .DCNv2.dcn_v2 import DCN
from mmcv.ops import DeformConv2dPack as DCN
#from mmcv.ops import ModulatedDeformConv2dPack as DCN
#from .dcn.modules.deform_conv import ModulatedDeformConvPack as DCN

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)




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
        #self.ch_at[-3].bias.data.fill_(-2.19)
        #self.sp_at[-2].bias.data.fill_(-2.19)
    def forward(self, x):
        chat = self.ch_at(x)
        spat = self.sp_at(x)
        
        ch_out = x * chat
        sp_out = x * spat
        
        return ch_out + sp_out
    
    
class CSAModuleV2(nn.Module):
    def __init__(self, in_size):
        super(CSAModuleV2, self).__init__()
        self.ch_at = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, in_size, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Sigmoid(),
        )
        
        self.sp_at = nn.Sequential(
            nn.Conv2d(in_size, in_size, kernel_size=1, stride=1, padding=0, bias=True, groups=in_size),
            nn.Sigmoid(),
        )
    def forward(self, x):
        chat = self.ch_at(x)
        spat = self.sp_at(x)
        
        ch_out = x * chat
        sp_out = x * spat
        
        return ch_out + sp_out


class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y



class eSEModule(nn.Module):
    def __init__(self, out_channel):
        super(eSEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(out_channel, out_channel, kernel_size=1, padding=0)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x):
        input = x
        x = self.avg_pool(x)
        x = self.fc(x)
        x = self.Sigmoid(x)
        return input * x


def get_model_url(data='imagenet', name='dla34', hash='ba72cf86'):
    return join('http://dl.yf.io/dla/models', data, '{}-{}.pth'.format(name, hash))


#def get_model_url():
    #return join('https://drive.google.com/file/d/1G2tCOeQVBTPkV9nncFDNrG2E12s81oQw/view?usp=share_link')


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


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
        #out = out + residual
        out = self.relu(out)

        return out



class FBasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(FBasicBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3,
                                stride=stride, padding=dilation,
                                bias=False, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        
        
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=(1, 3),
                               stride=(1, stride), padding=(0, dilation),
                               bias=False, dilation=dilation, groups=inplanes)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv1_1 = nn.Conv2d(planes, planes, kernel_size=(3, 1),
                                 stride=(stride, 1), padding=(dilation, 0),
                                 bias=False, dilation=dilation, groups=planes)
        self.bn1_1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        #self.conv2 = nn.Conv2d(planes, planes, kernel_size=(1, 3),  # (3, 1) better
                               #stride=1, padding=(0, dilation),
                               #bias=False, dilation=dilation, groups=planes)
        #self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        #self.conv2_1 = nn.Conv2d(planes, planes, kernel_size=(3, 1), #(1, 3) better
                                 #stride=1, padding=(dilation, 0),
                                 #bias=False, dilation=dilation, groups=planes)
        #self.bn2_1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        
        
        '''
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=(3, 1),
                               stride=(stride, 1), padding=(dilation, 0),
                                   bias=False, dilation=dilation, groups=inplanes)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv1_1 = nn.Conv2d(planes, planes, kernel_size=(1, 3),
                                 stride=(1, stride), padding=(0, dilation),
                                 bias=False, dilation=dilation, groups=planes)
        self.bn1_1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=(3, 1),  # (3, 1) better
                               stride=1, padding=(dilation, 0),
                               bias=False, dilation=dilation, groups=planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2_1 = nn.Conv2d(planes, planes, kernel_size=(1, 3), #(1, 3) better
                                 stride=1, padding=(0, dilation),
                                 bias=False, dilation=dilation, groups=planes)
        self.bn2_1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        '''
        
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.conv1_1(out)
        out = self.bn1_1(out)
        #out = self.relu(out)
        #out = self.conv2(out)
        #out = self.bn2(out)
        #out = self.conv2_1(out)
        #out = self.bn2_1(out)
            
        out += residual
        out = self.relu(out)
        
        return out




'''For RMNet'''
def rm_r_BasicBlock(block):
    block.eval()
    in_planes = block.conv1.in_channels
    mid_planes = in_planes + block.conv1.out_channels
    out_planes = block.conv2.out_channels

    #merge conv1 and bn1
    block.conv1=nn.utils.fuse_conv_bn_eval(block.conv1,block.bn1)
    #new conv1
    idconv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=3, stride=block.stride, padding=1).eval()
    #origional channels
    idconv1.weight.data[in_planes:]=block.conv1.weight.data
    idconv1.bias.data[in_planes:]=block.conv1.bias.data
    #reserve input featuremaps with dirac initialized channels
    nn.init.dirac_(idconv1.weight.data[:in_planes])
    nn.init.zeros_(idconv1.bias.data[:in_planes])

    #merge conv2 and bn2
    block.conv2=nn.utils.fuse_conv_bn_eval(block.conv2,block.bn2)
    #new conv
    idconv2 = nn.Conv2d(mid_planes, out_planes, kernel_size=3, stride=1, padding=1).eval()
    #origional channels
    idconv2.weight.data[:,in_planes:]=block.conv2.weight.data
    idconv2.bias.data=block.conv2.bias.data
    #merge input featuremaps to output featuremaps
    if in_planes==out_planes:
        nn.init.dirac_(idconv2.weight.data[:,:in_planes])
    else:
        #if there are a downsample layer
        downsample=nn.utils.fuse_conv_bn_eval(block.downsample[0],block.downsample[1])
        #conv1*1 -> conv3*3
        idconv2.weight.data[:,:in_planes]=F.pad(downsample.weight.data, [1, 1, 1, 1])
        idconv2.bias.data+=downsample.bias.data
    return nn.Sequential(*[idconv1,block.relu,idconv2,block.relu])

# ResBlock
class ResBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(ResBlock, self).__init__()

        self.in_planes = inplanes
        self.mid_planes = inplanes + planes
        self.out_planes = planes
        self.stride = stride

        self.conv1 = nn.Conv2d(inplanes, self.mid_planes, kernel_size=3, stride=stride, padding=dilation, bias=False, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(self.mid_planes)
        
        self.conv2 = nn.Conv2d(self.mid_planes, planes, kernel_size=3, stride=1, padding=dilation, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.relu = nn.ReLU(inplace=True)
        
        self.downsample=nn.Sequential()
        if self.in_planes != self.out_planes or self.stride != 1:
            self.downsample=nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes))
        self.running1 = nn.BatchNorm2d(inplanes,affine=False)
        self.running2 = nn.BatchNorm2d(planes,affine=False)
        
    def forward(self, x):
        if self.in_planes == self.out_planes and self.stride == 1:
            self.running1(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.downsample(x)
        self.running2(out)
        return self.relu(out)
    
    def deploy(self, merge_bn=False):
        idconv1 = nn.Conv2d(self.in_planes, self.mid_planes, kernel_size=3, stride=self.stride, padding=1, bias=False).eval()
        idbn1=nn.BatchNorm2d(self.mid_planes).eval()
        
        nn.init.dirac_(idconv1.weight.data[:self.in_planes])
        bn_var_sqrt=torch.sqrt(self.running1.running_var + self.running1.eps)
        idbn1.weight.data[:self.in_planes]=bn_var_sqrt
        idbn1.bias.data[:self.in_planes]=self.running1.running_mean
        idbn1.running_mean.data[:self.in_planes]=self.running1.running_mean
        idbn1.running_var.data[:self.in_planes]=self.running1.running_var
        
        idconv1.weight.data[self.in_planes:]=self.conv1.weight.data
        idbn1.weight.data[self.in_planes:]=self.bn1.weight.data
        idbn1.bias.data[self.in_planes:]=self.bn1.bias.data
        idbn1.running_mean.data[self.in_planes:]=self.bn1.running_mean
        idbn1.running_var.data[self.in_planes:]=self.bn1.running_var
        
        idconv2 = nn.Conv2d(self.mid_planes, self.out_planes, kernel_size=3, stride=1, padding=1, bias=False).eval()
        idbn2=nn.BatchNorm2d(self.out_planes).eval()
        downsample_bias=0
        if self.in_planes==self.out_planes:
            nn.init.dirac_(idconv2.weight.data[:,:self.in_planes])
        else:
            idconv2.weight.data[:,:self.in_planes],downsample_bias=self.fuse(F.pad(self.downsample[0].weight.data, [1, 1, 1, 1]),self.downsample[1].running_mean,self.downsample[1].running_var,self.downsample[1].weight,self.downsample[1].bias,self.downsample[1].eps)

        idconv2.weight.data[:,self.in_planes:],bias=self.fuse(self.conv2.weight,self.bn2.running_mean,self.bn2.running_var,self.bn2.weight,self.bn2.bias,self.bn2.eps)
        
        bn_var_sqrt=torch.sqrt(self.running2.running_var + self.running2.eps)
        idbn2.weight.data=bn_var_sqrt
        idbn2.bias.data=self.running2.running_mean
        idbn2.running_mean.data=self.running2.running_mean+bias+downsample_bias
        idbn2.running_var.data=self.running2.running_var
        
        if merge_bn:
            return [torch.nn.utils.fuse_conv_bn_eval(idconv1,idbn1),self.relu,torch.nn.utils.fuse_conv_bn_eval(idconv2,idbn2),self.relu]
        else:
            return [idconv1,idbn1,self.relu,idconv2,idbn2,self.relu]


    def fuse(self,conv_w, bn_rm, bn_rv,bn_w,bn_b, eps):
        bn_var_rsqrt = torch.rsqrt(bn_rv + eps)
        conv_w = conv_w * (bn_w * bn_var_rsqrt).reshape([-1] + [1] * (len(conv_w.shape) - 1))
        conv_b = bn_rm * bn_var_rsqrt * bn_w-bn_b
        return conv_w,conv_b
##########################################################################################################################
##########################################################################################################################


class FResBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(FResBlock, self).__init__()

        self.in_planes = inplanes
        self.mid_planes = inplanes + planes
        self.out_planes = planes
        self.stride = stride

        self.conv1 = nn.Conv2d(inplanes, self.mid_planes, kernel_size=(1, 3), stride=(1, stride), padding=(0, dilation), bias=False, dilation=dilation, groups=inplanes)
        self.bn1 = nn.BatchNorm2d(self.mid_planes)
        
        self.conv2 = nn.Conv2d(self.mid_planes, planes, kernel_size=(3, 1), stride=(stride, 1), padding=(dilation, 0), bias=False, dilation=dilation, groups=inplanes)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.relu = nn.ReLU(inplace=True)
        
        self.downsample=nn.Sequential()
        if self.in_planes != self.out_planes or self.stride != 1:
            self.downsample=nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes))
        self.running1 = nn.BatchNorm2d(inplanes,affine=False)
        self.running2 = nn.BatchNorm2d(planes,affine=False)
        
    def forward(self, x):
        if self.in_planes == self.out_planes and self.stride == 1:
            self.running1(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.downsample(x)
        self.running2(out)
        return self.relu(out)
    
    def deploy(self, merge_bn=False):
        idconv1 = nn.Conv2d(self.in_planes, self.mid_planes, kernel_size=3, stride=self.stride, padding=1, bias=False).eval()
        idbn1=nn.BatchNorm2d(self.mid_planes).eval()
        
        nn.init.dirac_(idconv1.weight.data[:self.in_planes])
        bn_var_sqrt=torch.sqrt(self.running1.running_var + self.running1.eps)
        idbn1.weight.data[:self.in_planes]=bn_var_sqrt
        idbn1.bias.data[:self.in_planes]=self.running1.running_mean
        idbn1.running_mean.data[:self.in_planes]=self.running1.running_mean
        idbn1.running_var.data[:self.in_planes]=self.running1.running_var
        
        idconv1.weight.data[self.in_planes:]=self.conv1.weight.data
        idbn1.weight.data[self.in_planes:]=self.bn1.weight.data
        idbn1.bias.data[self.in_planes:]=self.bn1.bias.data
        idbn1.running_mean.data[self.in_planes:]=self.bn1.running_mean
        idbn1.running_var.data[self.in_planes:]=self.bn1.running_var
        
        idconv2 = nn.Conv2d(self.mid_planes, self.out_planes, kernel_size=3, stride=1, padding=1, bias=False).eval()
        idbn2=nn.BatchNorm2d(self.out_planes).eval()
        downsample_bias=0
        if self.in_planes==self.out_planes:
            nn.init.dirac_(idconv2.weight.data[:,:self.in_planes])
        else:
            idconv2.weight.data[:,:self.in_planes],downsample_bias=self.fuse(F.pad(self.downsample[0].weight.data, [1, 1, 1, 1]),self.downsample[1].running_mean,self.downsample[1].running_var,self.downsample[1].weight,self.downsample[1].bias,self.downsample[1].eps)

        idconv2.weight.data[:,self.in_planes:],bias=self.fuse(self.conv2.weight,self.bn2.running_mean,self.bn2.running_var,self.bn2.weight,self.bn2.bias,self.bn2.eps)
        
        bn_var_sqrt=torch.sqrt(self.running2.running_var + self.running2.eps)
        idbn2.weight.data=bn_var_sqrt
        idbn2.bias.data=self.running2.running_mean
        idbn2.running_mean.data=self.running2.running_mean+bias+downsample_bias
        idbn2.running_var.data=self.running2.running_var
        
        if merge_bn:
            return [torch.nn.utils.fuse_conv_bn_eval(idconv1,idbn1),self.relu,torch.nn.utils.fuse_conv_bn_eval(idconv2,idbn2),self.relu]
        else:
            return [idconv1,idbn1,self.relu,idconv2,idbn2,self.relu]


    def fuse(self,conv_w, bn_rm, bn_rv,bn_w,bn_b, eps):
        bn_var_rsqrt = torch.rsqrt(bn_rv + eps)
        conv_w = conv_w * (bn_w * bn_var_rsqrt).reshape([-1] + [1] * (len(conv_w.shape) - 1))
        conv_b = bn_rm * bn_var_rsqrt * bn_w-bn_b
        return conv_w,conv_b






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
        # dim = int(math.floor(planes * (BottleneckV5.expansion / 64.0)))
        # bottle_planes = dim * cardinality
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


class ConvMlp(nn.Module):
    def __init__(
            self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, norm_layer=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=True)
        self.norm = norm_layer(hidden_features) if norm_layer else nn.Identity()
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=True)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


def num_groups(group_size, channels):
    if not group_size:  # 0 or None
        return 1  # normal conv with 1 group
    else:
        # NOTE group_size == 1 -> depthwise conv
        assert channels % group_size == 0
        return channels // group_size



class LayerFn:
    conv_norm_act: Callable = ConvNormAct
    norm_act: Callable = BatchNormAct2d
    act: Callable = nn.ReLU
    attn: Optional[Callable] = None
    self_attn: Optional[Callable] = None



class LinearSelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        bias: bool = True,
    ): #-> None:
        super().__init__()
        self.embed_dim = embed_dim

        self.qkv_proj = nn.Conv2d(
            in_channels=embed_dim,
            out_channels=1 + (2 * embed_dim),
            bias=bias,
            kernel_size=1,
        )
        self.attn_drop = nn.Dropout(attn_drop)
        self.out_proj = nn.Conv2d(
            in_channels=embed_dim,
            out_channels=embed_dim,
            bias=bias,
            kernel_size=1,
        )
        self.out_drop = nn.Dropout(proj_drop)

    def _forward_self_attn(self, x: torch.Tensor) -> torch.Tensor:
        # [B, C, P, N] --> [B, h + 2d, P, N]
        qkv = self.qkv_proj(x)

        # Project x into query, key and value
        # Query --> [B, 1, P, N]
        # value, key --> [B, d, P, N]
        query, key, value = qkv.split([1, self.embed_dim, self.embed_dim], dim=1)

        # apply softmax along N dimension
        context_scores = F.softmax(query, dim=-1)
        context_scores = self.attn_drop(context_scores)

        # Compute context vector
        # [B, d, P, N] x [B, 1, P, N] -> [B, d, P, N] --> [B, d, P, 1]
        context_vector = (key * context_scores).sum(dim=-1, keepdim=True)

        # combine context vector with values
        # [B, d, P, N] * [B, d, P, 1] --> [B, d, P, N]
        out = F.relu(value) * context_vector.expand_as(value)
        out = self.out_proj(out)
        out = self.out_drop(out)
        return out

    @torch.jit.ignore()
    def _forward_cross_attn(self, x: torch.Tensor, x_prev: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x --> [B, C, P, N]
        # x_prev = [B, C, P, M]
        batch_size, in_dim, kv_patch_area, kv_num_patches = x.shape
        q_patch_area, q_num_patches = x.shape[-2:]

        assert (
            kv_patch_area == q_patch_area
        ), "The number of pixels in a patch for query and key_value should be the same"

        # compute query, key, and value
        # [B, C, P, M] --> [B, 1 + d, P, M]
        qk = F.conv2d(
            x_prev,
            weight=self.qkv_proj.weight[:self.embed_dim + 1],
            bias=self.qkv_proj.bias[:self.embed_dim + 1],
        )

        # [B, 1 + d, P, M] --> [B, 1, P, M], [B, d, P, M]
        query, key = qk.split([1, self.embed_dim], dim=1)
        # [B, C, P, N] --> [B, d, P, N]
        value = F.conv2d(
            x,
            weight=self.qkv_proj.weight[self.embed_dim + 1],
            bias=self.qkv_proj.bias[self.embed_dim + 1] if self.qkv_proj.bias is not None else None,
        )

        # apply softmax along M dimension
        context_scores = F.softmax(query, dim=-1)
        context_scores = self.attn_drop(context_scores)

        # compute context vector
        # [B, d, P, M] * [B, 1, P, M] -> [B, d, P, M] --> [B, d, P, 1]
        context_vector = (key * context_scores).sum(dim=-1, keepdim=True)

        # combine context vector with values
        # [B, d, P, N] * [B, d, P, 1] --> [B, d, P, N]
        out = F.relu(value) * context_vector.expand_as(value)
        out = self.out_proj(out)
        out = self.out_drop(out)
        return out

    def forward(self, x: torch.Tensor, x_prev: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x_prev is None:
            return self._forward_self_attn(x)
        else:
            return self._forward_cross_attn(x, x_prev=x_prev)




class LinearTransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        mlp_ratio: float = 2.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer=None,
        norm_layer=None,
    ):
        super().__init__()
        act_layer = act_layer or nn.SiLU
        norm_layer = norm_layer or GroupNorm1

        self.norm1 = norm_layer(embed_dim)
        self.attn = LinearSelfAttention(embed_dim=embed_dim, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path1 = DropPath(drop_path)

        self.norm2 = norm_layer(embed_dim)
        self.mlp = ConvMlp(
            in_features=embed_dim,
            hidden_features=int(embed_dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop)
        #self.mlp = 
        self.drop_path2 = DropPath(drop_path)

    def forward(self, x: torch.Tensor, x_prev: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x_prev is None:
            # self-attention
            x = x + self.drop_path1(self.attn(self.norm1(x)))
        else:
            # cross-attention
            res = x
            x = self.norm1(x)  # norm
            x = self.attn(x, x_prev)  # attn
            x = self.drop_path1(x) + res  # residual

        # Feed forward network
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x



class MobileVitV3Block(nn.Module):
    def __init__(
        self,
        in_chs: int,
        out_chs: Optional[int] = None,
        kernel_size: int = 3,
        bottle_ratio: float = 1.0,
        group_size: Optional[int] = 1,
        dilation: Tuple[int, int] = (1, 1),
        mlp_ratio: float = 2.0,
        transformer_dim: Optional[int] = None,
        transformer_depth: int = 2,
        patch_size: int = 2,#4,
        attn_drop: float = 0.,
        drop: int = 0.,
        drop_path_rate: float = 0.,
        layers: LayerFn = None,
        transformer_norm_layer: Callable = GroupNorm1,
        **kwargs,  # eat unused args
    ):
        super(MobileVitV3Block, self).__init__()
        layers = layers or LayerFn()
        groups = num_groups(group_size, in_chs)
        out_chs = out_chs or in_chs
        transformer_dim = transformer_dim or make_divisible(bottle_ratio * in_chs)

        self.conv_kxk = layers.conv_norm_act(
            in_chs, in_chs, kernel_size=kernel_size,
            stride=1, groups=groups, dilation=dilation[0])
        self.conv_1x1 = nn.Conv2d(in_chs, transformer_dim, kernel_size=1, bias=False)

        self.transformer = nn.Sequential(*[
            LinearTransformerBlock(
                transformer_dim,
                mlp_ratio=mlp_ratio,
                attn_drop=attn_drop,
                drop=drop,
                drop_path=drop_path_rate,
                act_layer=nn.ReLU,
                norm_layer=transformer_norm_layer
            )
            for _ in range(transformer_depth)
        ])
        self.norm = transformer_norm_layer(transformer_dim)

        #self.conv_proj = layers.conv_norm_act(transformer_dim, out_chs, kernel_size=1, stride=1, apply_act=False)
        
        #self.conv_proj = layers.conv_norm_act(transformer_dim*2, out_chs, kernel_size=1, stride=1, apply_act=False)
        self.conv_proj = nn.Conv2d(transformer_dim*2, out_chs, kernel_size=1)
        
        self.patch_size = (patch_size, patch_size)#to_2tuple(patch_size)
        self.patch_area = self.patch_size[0] * self.patch_size[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        patch_h, patch_w = self.patch_size
        new_h, new_w = math.ceil(H / patch_h) * patch_h, math.ceil(W / patch_w) * patch_w
        num_patch_h, num_patch_w = new_h // patch_h, new_w // patch_w  # n_h, n_w
        num_patches = num_patch_h * num_patch_w  # N
        if new_h != H or new_w != W:
            x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=True)
            
        y = x.clone()
        # Local representation
        x = self.conv_kxk(x)
        x = self.conv_1x1(x)
        b,c,h,w = x.size()
        z = x.clone()
        #print("inner {}, {}, {}, {}".format(b,c,h,w))
        # Unfold (feature map -> patches), [B, C, H, W] -> [B, C, P, N]
        C = x.shape[1]
        x = x.reshape(B, C, num_patch_h, patch_h, num_patch_w, patch_w).permute(0, 1, 3, 5, 2, 4)
        x = x.reshape(B, C, -1, num_patches)

        # Global representations
        x = self.transformer(x)
        x = self.norm(x)

        # Fold (patches -> feature map), [B, C, P, N] --> [B, C, H, W]
        x = x.reshape(B, C, patch_h, patch_w, num_patch_h, num_patch_w).permute(0, 1, 4, 2, 5, 3)
        x = x.reshape(B, C, num_patch_h * patch_h, num_patch_w * patch_w)
        x = torch.cat((x, z), 1)
        x = self.conv_proj(x)
        return x+y


class MobileVitV2Block(nn.Module):
    """
    This class defines the `MobileViTv2 block <>`_
    """

    def __init__(
        self,
        in_chs: int,
        out_chs: Optional[int] = None,
        kernel_size: int = 3,
        bottle_ratio: float = 1.0,
        group_size: Optional[int] = 1,
        dilation: Tuple[int, int] = (1, 1),
        mlp_ratio: float = 2.0,
        transformer_dim: Optional[int] = None,
        transformer_depth: int = 2,
        patch_size: int = 4,
        attn_drop: float = 0.,
        drop: int = 0.,
        drop_path_rate: float = 0.,
        layers: LayerFn = None,
        transformer_norm_layer: Callable = GroupNorm1,
        **kwargs,  # eat unused args
    ):
        super(MobileVitV2Block, self).__init__()
        layers = layers or LayerFn()
        groups = num_groups(group_size, in_chs)
        out_chs = out_chs or in_chs
        transformer_dim = transformer_dim or make_divisible(bottle_ratio * in_chs)

        self.conv_kxk = layers.conv_norm_act(
            in_chs, in_chs, kernel_size=kernel_size,
            stride=1, groups=groups, dilation=dilation[0])
        self.conv_1x1 = nn.Conv2d(in_chs, transformer_dim, kernel_size=1, bias=False)

        self.transformer = nn.Sequential(*[
            LinearTransformerBlock(
                transformer_dim,
                mlp_ratio=mlp_ratio,
                attn_drop=attn_drop,
                drop=drop,
                drop_path=drop_path_rate,
                act_layer=nn.ReLU,
                norm_layer=transformer_norm_layer
            )
            for _ in range(transformer_depth)
        ])
        self.norm = transformer_norm_layer(transformer_dim)

        self.conv_proj = layers.conv_norm_act(transformer_dim, out_chs, kernel_size=1, stride=1, apply_act=False)

        self.patch_size = (patch_size, patch_size)#to_2tuple(patch_size)
        self.patch_area = self.patch_size[0] * self.patch_size[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        patch_h, patch_w = self.patch_size
        new_h, new_w = math.ceil(H / patch_h) * patch_h, math.ceil(W / patch_w) * patch_w
        num_patch_h, num_patch_w = new_h // patch_h, new_w // patch_w  # n_h, n_w
        num_patches = num_patch_h * num_patch_w  # N
        if new_h != H or new_w != W:
            x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=True)

        # Local representation
        x = self.conv_kxk(x)
        x = self.conv_1x1(x)
        b,c,h,w = x.size()
        #print("inner {}, {}, {}, {}".format(b,c,h,w))
        # Unfold (feature map -> patches), [B, C, H, W] -> [B, C, P, N]
        C = x.shape[1]
        x = x.reshape(B, C, num_patch_h, patch_h, num_patch_w, patch_w).permute(0, 1, 3, 5, 2, 4)
        x = x.reshape(B, C, -1, num_patches)

        # Global representations
        x = self.transformer(x)
        x = self.norm(x)

        # Fold (patches -> feature map), [B, C, P, N] --> [B, C, H, W]
        x = x.reshape(B, C, patch_h, patch_w, num_patch_h, num_patch_w).permute(0, 1, 4, 2, 5, 3)
        x = x.reshape(B, C, num_patch_h * patch_h, num_patch_w * patch_w)

        x = self.conv_proj(x)
        return x


'''
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
'''

class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, residual):
        super(Root, self).__init__()
        #self.conv = nn.Conv2d(
            #in_channels, out_channels, 1,
            #stride=1, bias=False, padding=(kernel_size - 1) // 2)
        self.Mvitv2block = MobileVitV2Block(in_chs=in_channels, out_chs=out_channels, bottle_ratio=0.5, group_size=1, transformer_depth=1)
        self.bn = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.Mvitv2block(torch.cat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
            #x = x + children[0]
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
            #root_dim = root_dim + in_channels
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
            #self.downsample = nn.MaxPool2d(stride, stride=stride)
            
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(in_channels, momentum=BN_MOMENTUM),
                SEModule(in_channels)
                )
            #self.downsample = nn.Sequential(
                #nn.Conv2d(in_channels, in_channels, kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0), groups=in_channels),
                #nn.BatchNorm2d(in_channels, momentum=BN_MOMENTUM),
                #nn.Conv2d(in_channels, in_channels, kernel_size=(1, 3), stride=(1, stride), padding=(0, 1), groups=in_channels),
                #nn.BatchNorm2d(in_channels, momentum=BN_MOMENTUM),
                #SEModule(in_channels)
                #)
            #self.downsample = nn.Sequential(
                #nn.Conv2d(in_channels, in_channels, kernel_size=(1, 3), stride=(1, stride), padding=(0, 1), groups=in_channels),
                #nn.BatchNorm2d(in_channels, momentum=BN_MOMENTUM),
                #nn.Conv2d(in_channels, in_channels, kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0), groups=in_channels),
                #nn.BatchNorm2d(in_channels, momentum=BN_MOMENTUM),
                #SEModule(in_channels)
                #)
            
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
                SEModule(out_channels)
            )

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
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


'''
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
            #self.tree1 = Tree(levels - 1, block, in_channels, out_channels,
                              #stride, root_dim=0,
                              #root_kernel_size=root_kernel_size,
                              #dilation=dilation, root_residual=root_residual)
            self.tree1 = Tree(levels - 1, block, in_channels, out_channels, stride, root_dim=0, root_kernel_size=root_kernel_size, dilation=dilation)
            #self.tree2 = Tree(levels - 1, block, out_channels, out_channels,
                              #root_dim=root_dim + out_channels,
                              #root_kernel_size=root_kernel_size,
                              #dilation=dilation, root_residual=root_residual)
            self.tree2 = Tree(levels - 1, block, out_channels, out_channels, root_dim=root_dim + out_channels, root_kernel_size=root_kernel_size, dilation=dilation)
        if levels == 1:
            self.root = Root(root_dim, out_channels, root_kernel_size,
                             root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
            
            #self.downsample = nn.Sequential(
                #nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1),
                ########nn.Conv2d(in_channels, in_channels, kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),#, groups=in_channels),
                #nn.BatchNorm2d(in_channels, momentum=BN_MOMENTUM),
                ########nn.Conv2d(in_channels, in_channels, kernel_size=(1, 3), stride=(1, stride), padding=(0, 1)),#, groups=in_channels),
                ########nn.BatchNorm2d(in_channels, momentum=BN_MOMENTUM),
                #SEModule(in_channels)
                #)
            
        
        #if in_channels != out_channels:
            #self.project = nn.Sequential(
                #nn.Conv2d(in_channels, out_channels,
                          #kernel_size=1, stride=1, bias=False),
                #nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
                ##########SEModule(out_channels)
            #)
        

    #def forward(self, x, residual=None, children=None):
    def forward(self, x, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        #residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x)
        #x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x
'''



class MvitDLA(nn.Module):
    def __init__(self, levels, channels, num_classes=1000,
                 block=BasicBlock, residual_root=False, linear_root=False):
        super(MvitDLA, self).__init__()
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
            #model_weights = torch.load(data + name)
            model_weights = torch.load(name)
        else:
            model_url = get_model_url(data, name, hash)
            model_weights = model_zoo.load_url(model_url)
        num_classes = len(model_weights[list(model_weights.keys())[-1]])
        #num_classes = 200
        self.fc = nn.Conv2d(
            self.channels[-1], num_classes,
            kernel_size=1, stride=1, padding=0, bias=True)
        self.load_state_dict(model_weights, strict=False)
        # self.fc = fc
        


def mvitdla34(pretrained=False, **kwargs):  # DLA-34
    model = MvitDLA([1, 1, 1, 2, 2, 1],
                [16, 32, 64, 128, 256, 512],
                block=BasicBlock, **kwargs)
                #block=FResBlock, **kwargs)
    
    if pretrained:
        model.load_pretrained_model(data='imagenet', name='dla34', hash='ba72cf86')
        #model.load_pretrained_model(data='imagenet', name='mvitdla.pth', hash='ba72cf86')
    #if pretrained:
        #model.load_pretrained_model()
        #model.load_state_dict(torch.load('mvitdla.pth'))
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
        #self.conv = DCN(chi, cho, kernel_size=(3,3), stride=1, padding=1, dilation=1, deformable_groups=1)
        self.conv = DCN(chi, cho, kernel_size=3, stride=1, padding=1, dilation=1, deform_groups=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.actf(x)
        return x


class IDAUp(nn.Module):

    def __init__(self, o, channels, up_f):
        super(IDAUp, self).__init__()
        for i in range(1, len(channels)):
            c = channels[i]
            f = int(up_f[i])
            proj = DeformConv(c, o)
            node = DeformConv(o, o)
            
            up = nn.ConvTranspose2d(o, o, f * 2, stride=f, 
                                    padding=f // 2, output_padding=0,
                                    groups=o, bias=False)
            fill_up_weights(up)
            
            #up = nn.Sequential(
                    #nn.Upsample(size=None, scale_factor=f, mode='bilinear'),
                    #nn.UpsamplingNearest2d(size=None, scale_factor=f),
                    #nn.Conv2d(o, o, kernel_size=1, stride=1, padding=0, groups=o, bias=True)
            #)
            #up[-1].bias.data.fill_(-2.19)
            
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



class DLAUp(nn.Module):
    def __init__(self, startp, channels, scales, in_channels=None):
        super(DLAUp, self).__init__()
        self.startp = startp
        if in_channels is None:
            in_channels = channels
        self.channels = channels
        channels = list(channels)
        scales = np.array(scales, dtype=int)
        for i in range(len(channels) - 1):
            j = -i - 2
            setattr(self, 'ida_{}'.format(i),
                    IDAUp(channels[j], in_channels[j:],
                          scales[j:] // scales[j]))
            scales[j + 1:] = scales[j]
            in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]

    def forward(self, layers):
        out = [layers[-1]] # start with 32
        for i in range(len(layers) - self.startp - 1):
            ida = getattr(self, 'ida_{}'.format(i))
            ida(layers, len(layers) -i - 2, len(layers))
            out.insert(0, layers[-1])
        return out


class Interpolate(nn.Module):
    def __init__(self, scale, mode):
        super(Interpolate, self).__init__()
        self.scale = scale
        self.mode = mode
        
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale, mode=self.mode, align_corners=False)
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
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        self.dla_up = DLAUp(self.first_level, channels[self.first_level:], scales)

        if out_channel == 0:
            out_channel = channels[self.first_level]

        self.ida_up = IDAUp(out_channel, channels[self.first_level:self.last_level], 
                            [2 ** i for i in range(self.last_level - self.first_level)])        
        self.heads = heads
        for head in self.heads:
            classes = self.heads[head]
            if head_conv > 0:
                if 'hm' in head:
                    fc = nn.Sequential(
                        nn.Conv2d(channels[self.first_level], head_conv,
                          kernel_size=3, padding=1, bias=True),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(head_conv, classes, 
                          kernel_size=final_kernel, stride=1, 
                          padding=final_kernel // 2, bias=True),
                        #CSAModuleV2(in_size=classes),
                        )
                    fc[-1].bias.data.fill_(-2.19)
                else:
                    fc = nn.Sequential(
                        nn.Conv2d(channels[self.first_level], head_conv,
                                  kernel_size=3, padding=1, bias=True),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(head_conv, classes,
                                  kernel_size=final_kernel, stride=1,
                                  padding=final_kernel // 2, bias=True))
                    fill_fc_weights(fc)
            else:
              fc = nn.Conv2d(channels[self.first_level], classes, 
                  kernel_size=final_kernel, stride=1, 
                  padding=final_kernel // 2, bias=True)
              if 'hm' in head:
                fc.bias.data.fill_(-2.19)
              else:
                fill_fc_weights(fc)
            self.__setattr__(head, fc)

    def forward(self, x):
        x = self.base(x)
        x = self.dla_up(x)

        y = []
        for i in range(self.last_level - self.first_level):
            y.append(x[i].clone())
        self.ida_up(y, 0, len(y))

        z = {}
        for head in self.heads:
            z[head] = self.__getattr__(head)(y[-1])
        return [z]
    

def get_pose_net(num_layers, heads, head_conv=256, down_ratio=4):
  model = DLASeg('mvitdla{}'.format(num_layers), heads,
                 pretrained=True,
                 down_ratio=down_ratio,
                 final_kernel=1,
                 last_level=5,
                 head_conv=head_conv)
  return model

