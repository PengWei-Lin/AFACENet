import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F


class Att_Proj_Node_o(nn.Module):
    def __init__(self, chi, cho):
        super(Att_Proj_Node_o, self).__init__()
        self.conv = nn.Conv2d(chi, cho, kernel_size=3, stride=1, padding=1, dilation=1, groups=1)
        self.actf = nn.Sequential(
            nn.BatchNorm2d(cho, momentum=0.1),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        x = self.conv(x)
        out = self.actf(x)
        
        return out


class IDAUp(nn.Module):
    def __init__(self, o, channels, up_f):
        super(IDAUp, self).__init__()

        for i in range(1, len(channels)):
            c = channels[i]
            f = int(up_f[i])
            proj = Att_Proj_Node_o(c, o)
            node = Att_Proj_Node_o(o, o)

            up = nn.Sequential(
                nn.Upsample(size=None, scale_factor=f, mode='bilinear'),
                nn.Conv2d(o, o, kernel_size=1, stride=1, padding=0, groups=o, bias=False),
            )
            
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


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )


def DWConv_nxn_bn(chs, kernal_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(chs, chs, kernal_size, stride, padding=1, groups=chs, bias=False),
        nn.BatchNorm2d(chs),
        nn.SiLU()
    )


def conv_nxn_bn(inp, oup, kernal_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernal_size, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)


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
        #print(self.embed_dim)
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

    def forward(self, x):#: torch.Tensor):
        # [B, C, P, N] --> [B, h + 2d, P, N]
        #b,c,h,w = x.size()
        #print("{}, {}, {}, {}".format(b,c,h,w))
        x = x.permute(0, 3, 2, 1)
        #b,c,h,w = x.size()
        #print("{}, {}, {}, {}".format(b,c,h,w))
        qkv = self.qkv_proj(x)
        #b,c,h,w = qkv.size()
        #print("{}, {}, {}, {}".format(b,c,h,w))
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
        b,c,h,w = out.size()
        print("{}, {}, {}, {}".format(b,c,h,w))
        return out


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        
        self.d_head = dim_head
        
        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, p, n, d = x.shape
        h = self.heads
        d_head = self.d_head
        qkv = self.to_qkv(x).view(b, p, n, h, 3 * d_head).transpose(2, 3)

        q, k, v = qkv.chunk(3, dim=-1)
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()


        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        b, p ,h, n, d = out.shape
        new_shape = (b, p, n, h*d)
        out = out.permute(0, 1, 3, 2, 4).reshape(*new_shape)
        return self.to_out(out)


def conv_1x1(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
    )

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                #PreNorm(dim, LinearSelfAttention(dim, 0, 0)),
                PreNorm(dim, Attention(dim, heads, dim_head, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))
    
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class MV2Block(nn.Module):
    def __init__(self, inp, oup, stride=1, expansion=4):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expansion)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileViTBlock(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size

        self.conv1 = DWConv_nxn_bn(channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, dim)

        self.transformer = Transformer(dim, depth, 4, 8, mlp_dim, dropout)

        self.conv3 = conv_1x1_bn(dim, channel)
        #self.conv4 = conv_1x1_bn(2 * channel, channel)#conv_nxn_bn(2 * channel, channel, kernel_size)
        self.conv4 = conv_1x1_bn(dim + channel, channel)
    
    def forward(self, x):
        y = x.clone()

        # Local representations
        x = self.conv1(x)
        x = self.conv2(x)
        z = x.clone()
        
        # Global representations
        b, c, h, w = x.shape
        x = x.view(b, c, h//self.ph, self.ph, w//self.pw, self.pw)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.view(b, -1, h//self.ph * w//self.pw, c)
        x = self.transformer(x)
        b, _, h_w, d = x.shape
        h, w, ph, pw = h // self.ph, w // self.pw, self.ph, self.pw
        new_shape = (b, d, h, ph, w, pw)
        x = x.reshape(*new_shape).permute(0, 1, 3, 2, 4, 5).reshape(b, d, h*ph, w*pw)

        # Fusion
        x = self.conv3(x)
        #x = torch.cat((x, y), 1)
        #b,c,w,h=x.size()
        #print("{}".format(c))
        x = torch.cat((x, z), 1)
        x = self.conv4(x)
        return x+y


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class MobileViTv3xxstt(nn.Module):
    def __init__(self, heads, final_kernel, head_conv):
        super(MobileViTv3xxstt, self).__init__()
        ih = 384
        iw = 1280
        ph = 2
        pw = 2
        assert ih % ph == 0 and iw % pw == 0

        #self.conv1 = conv_nxn_bn(3, channels[0], stride=2)
        self.conv1 = conv_nxn_bn(3, 16, stride=2)
        self.first_mv2 = MV2Block(16, 16, 1, 4)
        self.block0 = nn.Sequential(
            MV2Block(16, 24, 2, 4),
            MV2Block(24, 24, 1, 4),
            MV2Block(24, 24, 1, 4)
        )
        
        self.block1 = nn.Sequential(
            MV2Block(24, 48, 2, 4),
            MobileViTBlock(64, 2, 48, 3, (2, 2), 128),
        )
        
        self.block2 = nn.Sequential(
            MV2Block(48, 64, 2, 4),
            MobileViTBlock(80, 4, 64, 3, (2, 2), 320)
        )
        
        self.block3 = nn.Sequential(
            MV2Block(64, 80, 2, 4),
            MobileViTBlock(96, 3, 80, 3, (2, 2), 384),
        )
        
        self.conv2 = conv_1x1_bn(80, 320)
        self.init_params()

        self.ida_up = IDAUp(24, [24, 48, 64, 320], [2 ** i for i in range(4)])
        
        self.heads = heads
        for head_id, head in enumerate(self.heads):
            classes = self.heads[head]
            if head_id == 0:
                fc = nn.Sequential(
                    nn.Conv2d(24, head_conv, kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_conv, classes, kernel_size=final_kernel, stride=1, padding=final_kernel // 2, bias=True),
                    )
                if 'hm' in head:
                    fc[-1].bias.data.fill_(-2.19)
                self.__setattr__(head, fc)
            else:
                fc = nn.Sequential(
                    nn.Conv2d(24, head_conv,
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
        x = self.first_mv2(x)

        out0 = self.block0(x)
        out1 = self.block1(out0)
        out2 = self.block2(out1)      # Repeat
        out3 = self.block3(out2)
        out3 = self.conv2(out3)
        
        out = [out0, out1, out2, out3]
        
        y = []
        for i in range(4):
            y.append(out[i].clone())
        self.ida_up(y, 0, len(y))
        z = {}
        
        for head in self.heads:
            z[head] = self.__getattr__(head)(y[-1])
        return [z]