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

from .DCNv2.dcn_v2 import DCN
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


def get_model_url(data='imagenet', name='dla34', hash='ba72cf86'):
    return join('http://dl.yf.io/dla/models', data, '{}-{}.pth'.format(name, hash))


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)



'Local to Global    Globle to Local'
# inputs: x(b c h w) z(b m d)
# output: z(b m d)  to_q, to_out   Linear
class Mobile2Former(nn.Module):
    def __init__(self, dim, heads, channel, dropout=0.):
        super(Mobile2Former, self).__init__()
        inner_dim = heads * channel
        self.heads = heads
        self.to_q = nn.Linear(dim, inner_dim)
        self.attend = nn.Softmax(dim=-1)
        self.scale = channel ** -0.5
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, z):
        b, m, d = z.shape
        b, c, h, w = x.shape
        x =  x.reshape(b, c, h*w).transpose(1,2).unsqueeze(1)
        q = self.to_q(z).view(b, self.heads, m, c)
        dots = q @ x.transpose(2, 3) * self.scale
        attn = self.attend(dots)
        out = attn @ x
        #out = rearrange(out, 'b h m c -> b m (h c)')
        out = torch.reshape(out, (out.shape[0], out.shape[2], out.shape[1] * out.shape[3]))
        return z + self.to_out(out)


# inputs: x(b c h w) z(b m d)
# output: x(b c h w)    to_k   to_v   to_out    Linear
class Former2Mobile(nn.Module):
    def __init__(self, dim, heads, channel, dropout=0.):
        super(Former2Mobile, self).__init__()
        inner_dim = heads * channel
        self.heads = heads
        self.to_k = nn.Linear(dim, inner_dim)
        self.to_v = nn.Linear(dim, inner_dim)
        self.attend = nn.Softmax(dim=-1)
        self.scale = channel ** -0.5

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, channel),
            nn.Dropout(dropout)
        )

    def forward(self, x, z):
        b, m, d = z.shape
        b, c, h, w = x.shape
        q =  x.reshape(b, c, h*w).transpose(1,2).unsqueeze(1)
        k = self.to_k(z).view(b, self.heads, m, c)
        v = self.to_v(z).view(b, self.heads, m, c)
        dots = q @ k.transpose(2, 3) * self.scale
        attn = self.attend(dots)
        out = attn @ v
        #out = rearrange(out, 'b h l c -> b l (h c)')
        out = out.permute(0, 2, 1, 3).reshape(out.shape[0], out.shape[2], -1)
        out = self.to_out(out)
        out = out.view(b, c, h, w)
        return x + out



'MobileNet(CNN)'
class MyDyRelu(nn.Module):
    def __init__(self, k):
        super(MyDyRelu, self).__init__()
        self.k = k

    def forward(self, inputs):
        x, relu_coefs = inputs
        # BxCxHxW -> HxWxBxCx1
        x_perm = x.permute(2, 3, 0, 1).unsqueeze(-1)
        # h w b c 1 -> h w b c k
        output = x_perm * relu_coefs[:, :, :self.k] + relu_coefs[:, :, self.k:]
        # HxWxBxCxk -> BxCxHxW
        result = torch.max(output, dim=-1)[0].permute(2, 3, 0, 1)
        return result



class Mobile(nn.Module):
    def __init__(self, ks, inp, hid, out, se, stride, dim, reduction=4, k=2):
        super(Mobile, self).__init__()
        self.hid = hid
        self.k = k
        self.fc1 = nn.Linear(dim, dim // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(dim // reduction, 2 * k * hid)
        self.sigmoid = nn.Sigmoid()

        self.register_buffer('lambdas', torch.Tensor([1.] * k + [0.5] * k).float())
        self.register_buffer('init_v', torch.Tensor([1.] + [0.] * (2 * k - 1)).float())
        self.stride = stride
        # self.se = DyReLUB(channels=out, k=1) if dyrelu else se
        self.se = se

        self.conv1 = nn.Conv2d(inp, hid, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(hid)
        self.act1 = MyDyRelu(2)

        self.conv2 = nn.Conv2d(hid, hid, kernel_size=ks, stride=stride,
                               padding=ks // 2, groups=hid, bias=False)
        self.bn2 = nn.BatchNorm2d(hid)
        self.act2 = MyDyRelu(2)

        self.conv3 = nn.Conv2d(hid, out, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out)

        self.shortcut = nn.Identity()
        if stride == 1 and inp != out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inp, out, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out),
            )

    def get_relu_coefs(self, z):
        theta = z[:, 0, :]
        # b d -> b d//4
        theta = self.fc1(theta)
        theta = self.relu(theta)
        # b d//4 -> b 2*k
        theta = self.fc2(theta)
        theta = 2 * self.sigmoid(theta) - 1
        # b 2*k
        return theta

    def forward(self, x, z):
        theta = self.get_relu_coefs(z)
        # b 2*k*c -> b c 2*k                                     2*k            2*k
        relu_coefs = theta.view(-1, self.hid, 2 * self.k) * self.lambdas + self.init_v

        out = self.bn1(self.conv1(x))
        out_ = [out, relu_coefs]
        out = self.act1(out_)

        out = self.bn2(self.conv2(out))
        out_ = [out, relu_coefs]
        out = self.act2(out_)

        out = self.bn3(self.conv3(out))
        if self.se is not None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out



class MobileDown(nn.Module):
    def __init__(self, ks, inp, hid, out, se, stride, dim, reduction=4, k=2):
        super(MobileDown, self).__init__()
        self.dim = dim
        self.hid, self.out = hid, out
        self.k = k
        self.fc1 = nn.Linear(dim, dim // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(dim // reduction, 2 * k * hid)
        self.sigmoid = nn.Sigmoid()
        self.register_buffer('lambdas', torch.Tensor([1.] * k + [0.5] * k).float())
        self.register_buffer('init_v', torch.Tensor([1.] + [0.] * (2 * k - 1)).float())
        self.stride = stride
        # self.se = DyReLUB(channels=out, k=1) if dyrelu else se
        self.se = se

        self.dw_conv1 = nn.Conv2d(inp, hid, kernel_size=ks, stride=stride,
                                  padding=ks // 2, groups=inp, bias=False)
        self.dw_bn1 = nn.BatchNorm2d(hid)
        self.dw_act1 = MyDyRelu(2)

        self.pw_conv1 = nn.Conv2d(hid, inp, kernel_size=1, stride=1, padding=0, bias=False)
        self.pw_bn1 = nn.BatchNorm2d(inp)
        self.pw_act1 = nn.ReLU()

        self.dw_conv2 = nn.Conv2d(inp, hid, kernel_size=ks, stride=1,
                                  padding=ks // 2, groups=inp, bias=False)
        self.dw_bn2 = nn.BatchNorm2d(hid)
        self.dw_act2 = MyDyRelu(2)

        self.pw_conv2 = nn.Conv2d(hid, out, kernel_size=1, stride=1, padding=0, bias=False)
        self.pw_bn2 = nn.BatchNorm2d(out)

        self.shortcut = nn.Identity()
        if stride == 1 and inp != out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inp, out, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out),
            )

    def get_relu_coefs(self, z):
        theta = z[:, 0, :]
        # b d -> b d//4
        theta = self.fc1(theta)
        theta = self.relu(theta)
        # b d//4 -> b 2*k
        theta = self.fc2(theta)
        theta = 2 * self.sigmoid(theta) - 1
        # b 2*k
        return theta

    def forward(self, x, z):
        theta = self.get_relu_coefs(z)
        # b 2*k*c -> b c 2*k                                     2*k            2*k
        relu_coefs = theta.view(-1, self.hid, 2 * self.k) * self.lambdas + self.init_v

        out = self.dw_bn1(self.dw_conv1(x))
        out_ = [out, relu_coefs]
        out = self.dw_act1(out_)
        out = self.pw_act1(self.pw_bn1(self.pw_conv1(out)))

        out = self.dw_bn2(self.dw_conv2(out))
        out_ = [out, relu_coefs]
        out = self.dw_act2(out_)
        out = self.pw_bn2(self.pw_conv2(out))

        if self.se is not None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super(PreNorm, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super(Attention, self).__init__()
        inner_dim = heads * dim_head  # head数量和每个head的维度
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):  # 2,65,1024 batch,patch+cls_token,dim (每个patch相当于一个token)
        # 输入x每个token的维度为1024，在注意力中token被映射16个64维的特征（head*dim_head），
        # 最后再把所有head的特征合并为一个（16*1024）的特征，作为每个token的输出
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # 2,65,1024 -> 2,65,1024*3
        q, k, v = map(lambda t: t.view(t.shape[0], t.shape[1], self.heads, -1).permute(0, 2, 1, 3), qkv) # 2,65,(16*64) -> 2,16,65,64 ,16个head，每个head维度64
        dots = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # b,16,65,64 @ b,16,64*65 -> b,16,65,65 : q@k.T
        attn = self.attend(dots)  # 注意力 2,16,65,65  16个head，注意力map尺寸65*65，对应token（patch）[i,j]之间的注意力
        # 每个token经过每个head的attention后的输出 
        out = torch.matmul(attn, v)# atten@v 2,16,65,65 @ 2,16,65,64 -> 2,16,65,64
        out = out.permute(0, 2, 1, 3).reshape(out.shape[0], out.shape[2], -1)# 合并所有head的输出(16*64) -> 1024 得到每个token当前的特征
        return self.to_out(out)



# inputs: n L C
# output: n L C
class Former(nn.Module):
    def __init__(self, dim, depth=1, heads=2, dim_head=32, dropout=0.3):
        super(Former, self).__init__()
        mlp_dim = dim * 2
        self.layers = nn.ModuleList([])
        # dim_head = dim // heads
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


'Mobileformer'
class mfBaseBlock(nn.Module):
    def __init__(self, inp, exp, out, se, stride, heads, dim):
        super(mfBaseBlock, self).__init__()
        if stride == 2:
            self.mobile = MobileDown(3, inp, exp, out, se, stride, dim)
        else:
            self.mobile = Mobile(3, inp, exp, out, se, stride, dim)
        self.mobile2former = Mobile2Former(dim=dim, heads=heads, channel=inp)
        self.former = Former(dim=dim)
        self.former2mobile = Former2Mobile(dim=dim, heads=heads, channel=out)

    def forward(self, inputs):
        x, z = inputs
        z_hid = self.mobile2former(x, z)
        z_out = self.former(z_hid)
        x_hid = self.mobile(x, z_out)
        x_out = self.former2mobile(x_hid, z_out)
        return [x_out, z_out]



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
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
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


class DLA(nn.Module):
    def __init__(self, levels, channels, num_classes=1000,
                 block=mfBaseBlock, residual_root=False, linear_root=False):
        super(DLA, self).__init__()
        self.channels = channels
        self.num_classes = num_classes
        self.base_layer = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=7, stride=1,
                      padding=3, bias=False),
            nn.BatchNorm2d(channels[0], momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True))
        #self.level0 = self._make_conv_level(channels[0], channels[0], levels[0])
        self.level0 = mfBaseBlock(inp=16, exp=64, out=16, se=None, stride=1, heads=2, dim=128)
        #self.level1 = self._make_conv_level(channels[0], channels[1], levels[1], stride=2)
        self.level1 = mfBaseBlock(inp=16, exp=64, out=32, se=None, stride=1, heads=2, dim=128)
        self.level2 = Tree(levels[2], block, channels[1], channels[2], 2,
                           level_root=False,
                           root_residual=residual_root)
        #self.level3 = Tree(levels[3], block, channels[2], channels[3], 2,level_root=True, root_residual=residual_root)
        self.level3 = Tree(2, block, channels[2], channels[3], 2,level_root=True, root_residual=residual_root)
        #self.level4 = Tree(levels[4], block, channels[3], channels[4], 2, level_root=True, root_residual=residual_root)
        self.level4 = Tree(2, block, channels[3], channels[4], 2, level_root=True, root_residual=residual_root)
        #self.level5 = Tree(levels[5], block, channels[4], channels[5], 2, level_root=True, root_residual=residual_root)
        self.level5 = Tree(1, block, channels[4], channels[5], 2, level_root=True, root_residual=residual_root)

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
        self.load_state_dict(model_weights)
        # self.fc = fc


def dla34(pretrained=False, **kwargs):  # DLA-34
    model = DLA([1, 1, 1, 2, 2, 1],
                [16, 32, 64, 128, 256, 512],
                #block=BasicBlock, **kwargs)
                block=mfBaseBlock, **kwargs)
    #if pretrained:
        #model.load_pretrained_model(data='imagenet', name='dla34', hash='ba72cf86')
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


'Good for Basic Block'
class Att_Proj_Node_n(nn.Module):
    def __init__(self, chi, cho):
        super(Att_Proj_Node_n, self).__init__()
        self.CSAM = CSAModule(chi)
        self.conv = nn.Conv2d(chi, cho, kernel_size=3, stride=1, padding=1, dilation=1, groups=1) #bias=True)
        self.actf = nn.Sequential(
            nn.BatchNorm2d(cho, momentum=0.1),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        
        x = self.CSAM(x)
        x = self.conv(x)
        out = self.actf(x)
        
        return out


class Att_Proj_Node(nn.Module):
    def __init__(self, chi, cho):
        super(Att_Proj_Node, self).__init__()
        self.conv = nn.Conv2d(chi, cho, kernel_size=3, stride=1, padding=1, dilation=1, groups=1)
        self.CSAM = CSAModule(cho)
        self.actf = nn.Sequential(
            nn.BatchNorm2d(cho, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.CSAM(x)
        x = self.actf(x)
        return x



class DeformConv(nn.Module):
    def __init__(self, chi, cho):
        super(DeformConv, self).__init__()
        self.actf = nn.Sequential(
            nn.BatchNorm2d(cho, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        self.conv = DCN(chi, cho, kernel_size=(3,3), stride=1, padding=1, dilation=1, deformable_groups=1)

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
            #proj = Att_Proj_Node_n(c, o)
            #node = Att_Proj_Node_n(o, o)
            proj = DeformConv(c, o)
            node = DeformConv(o, o)
            
            up = nn.ConvTranspose2d(o, o, f * 2, stride=f, 
                                    padding=f // 2, output_padding=0,
                                    groups=o, bias=False)
            fill_up_weights(up)
            
            #up = nn.Sequential(
                    #nn.Upsample(size=None, scale_factor=f, mode='bilinear'),
                    #nn.UpsamplingNearest2d(size=None, scale_factor=f),
                    #nn.Conv2d(o, o, kernel_size=1, stride=1, padding=0, groups=o, bias=False)
            #)
            
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
                        #CSAModule(in_size=classes),
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
  model = DLASeg('dla{}'.format(num_layers), heads,
                 pretrained=True,
                 down_ratio=down_ratio,
                 final_kernel=1,
                 last_level=5,
                 head_conv=head_conv)
  return model

