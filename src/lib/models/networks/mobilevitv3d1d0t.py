import math
from typing import Tuple, Optional, Callable


import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init
from timm.models.layers import GroupNorm1
#from timm.models.layers import ConvMlp
from timm.models.layers import DropPath
from timm.models.layers import ConvNormAct
from timm.models.layers import BatchNormAct2d
from timm.models.layers import make_divisible
#from timm.models.layers import to_2tuple
from timm.models.layers import AvgPool2dSame

#from .DCNv2.dcn_v2 import DCN
from mmcv.ops import DeformConv2dPack as DCN
#from mmcv.ops import ModulatedDeformConv2dPack as DCN

class ConvMlp(nn.Module):
    """ MLP using 1x1 convs that keeps spatial dims
    """
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


class LayerFn:
    conv_norm_act: Callable = ConvNormAct
    norm_act: Callable = BatchNormAct2d
    act: Callable = nn.ReLU
    attn: Optional[Callable] = None
    self_attn: Optional[Callable] = None



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


class DeformConv(nn.Module):
    def __init__(self, chi, cho):
        super(DeformConv, self).__init__()
        self.actf = nn.Sequential(
            nn.BatchNorm2d(cho, momentum=0.1),
            nn.ReLU(inplace=True)
        )
        #self.conv = DCN(chi, cho, kernel_size=(3,3), stride=1, padding=1, dilation=1, deformable_groups=1)
        self.conv = DCN(chi, cho, kernel_size=3, stride=1, padding=1, dilation=1, deform_groups=1)
        #self.conv = DeformableConv2d(chi, cho, kernel_size=3, padding=1, bias=False)

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
            #print(f)
            proj = DeformConv(c, o)
            node = Att_Proj_Node_o(o, o)
            up = nn.ConvTranspose2d(o, o, f * 2, stride=f,
                                    padding=f // 2, output_padding=0,
                                    groups=o, bias=False)
            #up = nn.Sequential(
                #nn.Upsample(size=None, scale_factor=f, mode='bilinear'),
                #nn.Conv2d(o, o, kernel_size=1, stride=1, padding=0, groups=o, bias=False),
            #)
            fill_up_weights(up)
            
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



def conv_nxn_bn(inp, oup, kernal_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernal_size, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )


class DownsampleAvg(nn.Module):
    def __init__(self, in_chs, out_chs, stride=1, dilation=1, apply_act=False, layers: LayerFn = None):
        """ AvgPool Downsampling as in 'D' ResNet variants."""
        super(DownsampleAvg, self).__init__()
        layers = layers or LayerFn()
        avg_stride = stride if dilation == 1 else 1
        if stride > 1 or dilation > 1:
            avg_pool_fn = AvgPool2dSame if avg_stride == 1 and dilation > 1 else nn.AvgPool2d
            self.pool = avg_pool_fn(2, avg_stride, ceil_mode=True, count_include_pad=False)
        else:
            self.pool = nn.Identity()
        self.conv = layers.conv_norm_act(in_chs, out_chs, 1, apply_act=apply_act)

    def forward(self, x):
        return self.conv(self.pool(x))



def create_shortcut(downsample_type, layers: LayerFn, in_chs, out_chs, stride, dilation, **kwargs):
    assert downsample_type in ('avg', 'conv1x1', '')
    if in_chs != out_chs or stride != 1 or dilation[0] != dilation[1]:
        if not downsample_type:
            return None  # no shortcut
        elif downsample_type == 'avg':
            return DownsampleAvg(in_chs, out_chs, stride=stride, dilation=dilation[0], **kwargs)
        else:
            return layers.conv_norm_act(in_chs, out_chs, kernel_size=1, stride=stride, dilation=dilation[0], **kwargs)
    else:
        return nn.Identity()  # identity shortcut


    
class BottleneckBlock(nn.Module):
    """ ResNet-like Bottleneck Block - 1x1 - kxk - 1x1
    """

    def __init__(
            self,
            in_chs,
            out_chs,
            kernel_size=3,
            stride=1,
            dilation=(1, 1),
            bottle_ratio=1.,
            group_size=None,
            downsample='avg',
            attn_last=False,
            linear_out=False,
            extra_conv=False,
            bottle_in=False,
            layers: LayerFn = None,
            drop_block=None,
            drop_path_rate=0.,
    ):
        super(BottleneckBlock, self).__init__()
        layers = layers or LayerFn()
        mid_chs = make_divisible((in_chs if bottle_in else out_chs) * bottle_ratio)
        groups = num_groups(group_size, mid_chs)

        self.shortcut = create_shortcut(
            downsample, in_chs=in_chs, out_chs=out_chs, stride=stride, dilation=dilation,
            apply_act=False, layers=layers)

        self.conv1_1x1 = layers.conv_norm_act(in_chs, mid_chs, 1)
        self.conv2_kxk = layers.conv_norm_act(
            mid_chs, mid_chs, kernel_size, stride=stride, dilation=dilation[0], groups=groups, drop_layer=drop_block)
        if extra_conv:
            self.conv2b_kxk = layers.conv_norm_act(mid_chs, mid_chs, kernel_size, dilation=dilation[1], groups=groups)
        else:
            self.conv2b_kxk = nn.Identity()
        self.attn = nn.Identity() if attn_last or layers.attn is None else layers.attn(mid_chs)
        self.conv3_1x1 = layers.conv_norm_act(mid_chs, out_chs, 1, apply_act=False)
        self.attn_last = nn.Identity() if not attn_last or layers.attn is None else layers.attn(out_chs)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.act = nn.Identity() if linear_out else nn.ReLU(inplace=True)

    def init_weights(self, zero_init_last: bool = False):
        if zero_init_last and self.shortcut is not None and getattr(self.conv3_1x1.bn, 'weight', None) is not None:
            nn.init.zeros_(self.conv3_1x1.bn.weight)
        for attn in (self.attn, self.attn_last):
            if hasattr(attn, 'reset_parameters'):
                attn.reset_parameters()

    def forward(self, x):
        shortcut = x
        x = self.conv1_1x1(x)
        x = self.conv2_kxk(x)
        x = self.conv2b_kxk(x)
        x = self.attn(x)
        x = self.conv3_1x1(x)
        x = self.attn_last(x)
        x = self.drop_path(x)
        if self.shortcut is not None:
            x = x + self.shortcut(shortcut)
        return self.act(x)


def num_groups(group_size, channels):
    if not group_size:  # 0 or None
        return 1  # normal conv with 1 group
    else:
        # NOTE group_size == 1 -> depthwise conv
        assert channels % group_size == 0
        return channels // group_size


class LinearSelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        bias: bool = True,
        ):
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

    def forward(self, x: torch.Tensor, x_prev: Optional[torch.Tensor] = None) -> torch.Tensor:
        #print("x: {}".format(x.shape))
        # [B, C, P, N] --> [B, h + 2d, P, N]
        qkv = self.qkv_proj(x)
        #print("qkv: {}".format(qkv.shape))
        # Project x into query, key and value
        # Query --> [B, 1, P, N]
        # value, key --> [B, d, P, N]
        query, key, value = qkv.split([1, self.embed_dim, self.embed_dim], dim=1)
        #print("query: {}".format(query.shape))
        #print("key: {}".format(key.shape))
        #print("value: {}".format(value.shape))
        # apply softmax along N dimension
        context_scores = F.softmax(query, dim=-1)
        #print("query softmax: {}".format(context_scores.shape))
        context_scores = self.attn_drop(context_scores)
        #print("query softmax(drop): {}".format(context_scores.shape))
        # Compute context vector
        # [B, d, P, N] x [B, 1, P, N] -> [B, d, P, N] --> [B, d, P, 1]
        context_vector = (key * context_scores).sum(dim=-1, keepdim=True)
        #print("q,k,v: {}".format(context_vector.shape))
        # combine context vector with values
        # [B, d, P, N] * [B, d, P, 1] --> [B, d, P, N]
        out = F.relu(value) * context_vector.expand_as(value)
        #print("out(ReLU): {}".format(out.shape))
        out = self.out_proj(out)
        #print("out(proj): {}".format(out.shape))
        out = self.out_drop(out)
        #print("out(drop): {}".format(out.shape))
        return out



class LinearTransformerBlock(nn.Module):
    """
    This class defines the pre-norm transformer encoder with linear self-attention in `MobileViTv2 paper <>`_
    Args:
        embed_dim (int): :math:`C_{in}` from an expected input of size :math:`(B, C_{in}, P, N)`
        mlp_ratio (float): Inner dimension ratio of the FFN relative to embed_dim
        drop (float): Dropout rate. Default: 0.0
        attn_drop (float): Dropout rate for attention in multi-head attention. Default: 0.0
        drop_path (float): Stochastic depth rate Default: 0.0
        norm_layer (Callable): Normalization layer. Default: layer_norm_2d
    Shape:
        - Input: :math:`(B, C_{in}, P, N)` where :math:`B` is batch size, :math:`C_{in}` is input embedding dim,
            :math:`P` is number of pixels in a patch, and :math:`N` is number of patches,
        - Output: same shape as the input
    """

    def __init__(
        self,
        embed_dim: int,
        mlp_ratio: float = 2.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer=None,
        norm_layer=None,
    ): #-> None:
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

'''
def _mobilevitv2_cfg(multiplier=1.0):
    chs = (64, 128, 256, 384, 512)
    if multiplier != 1.0:
        chs = tuple([int(c * multiplier) for c in chs])
    cfg = ByoModelCfg(
        blocks=(
            _inverted_residual_block(d=1, c=chs[0], s=1, br=2.0),
            _inverted_residual_block(d=2, c=chs[1], s=2, br=2.0),
            _mobilevitv2_block(d=1, c=chs[2], s=2, transformer_depth=2),
            _mobilevitv2_block(d=1, c=chs[3], s=2, transformer_depth=4),
            _mobilevitv2_block(d=1, c=chs[4], s=2, transformer_depth=3),
        ),
        stem_chs=int(32 * multiplier),
        stem_type='3x3',
        stem_pool='',
        downsample='',
        act_layer='silu',
    )
    return
'''

def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class MobileViTv3d1d0t(nn.Module):
    def __init__(self, heads, final_kernel, head_conv):
        super(MobileViTv3d1d0t, self).__init__()
        self.conv1 = conv_nxn_bn(3, 64, stride=2)
        self.first = BottleneckBlock(in_chs=64, out_chs=64, bottle_ratio=2, group_size=1)
        
        self.block0 = nn.Sequential(
            BottleneckBlock(in_chs=64, out_chs=128, stride=2, bottle_ratio=2, group_size=1),
            BottleneckBlock(in_chs=128, out_chs=128, bottle_ratio=2, group_size=1),
        )
        
        self.block1 = nn.Sequential(
            BottleneckBlock(stride=2, in_chs=128, out_chs=256, bottle_ratio=2, group_size=1),
            MobileVitV2Block(in_chs=256, out_chs=256, bottle_ratio=0.5, group_size=1, transformer_depth=2),
        )
        
        self.block2 = nn.Sequential(
            BottleneckBlock(stride=2, in_chs=256, out_chs=384, bottle_ratio=2, group_size=1),
            MobileVitV2Block(in_chs=384, out_chs=384, bottle_ratio=0.5, group_size=1, transformer_depth=4),
        )
        
        self.block3 = nn.Sequential(
            BottleneckBlock(stride=2, in_chs=384, out_chs=512, bottle_ratio=2, group_size=1),
            MobileVitV2Block(in_chs=512, out_chs=512, bottle_ratio=0.5, group_size=1, transformer_depth=3),
        )
        
        self.init_params()
        
        self.ida_up = IDAUp(128, [128, 256, 384, 512], [2 ** i for i in range(4)])
        
        self.heads = heads
        for head_id, head in enumerate(self.heads):
            classes = self.heads[head]
            if head_id == 0:
                fc = nn.Sequential(
                    nn.Conv2d(128, head_conv, kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_conv, classes, kernel_size=final_kernel, stride=1, padding=final_kernel // 2, bias=True),
                    )
                if 'hm' in head:
                    fc[-1].bias.data.fill_(-2.19)
                self.__setattr__(head, fc)
            else:
                fc = nn.Sequential(
                    nn.Conv2d(128, head_conv,
                    kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
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
        x = self.conv1(x)
        x = self.first(x)

        out0 = self.block0(x)
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
        
