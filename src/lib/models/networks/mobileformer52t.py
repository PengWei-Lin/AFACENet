import torch
from torch import nn, einsum
from einops import rearrange
import torch.nn.functional as F
from torch.nn import init


'Local to Global    Globle to Local'
# inputs: x(b c h w) z(b m d)
# output: z(b m d)
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
# output: x(b c h w)
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

class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out


class SeModule(nn.Module):
    def __init__(self, inp, reduction=4):
        super(SeModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Linear(inp, inp // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(inp // reduction, inp, bias=False),
            hsigmoid()
        )

    def forward(self, x):
        se = self.avg_pool(x)
        b, c, _, _ = se.size()
        se = se.view(b, c)
        se = self.se(se).view(b, c, 1, 1)
        return x * se.expand_as(x)


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


'Former(Transformer)'
def pair(t):
    return t if isinstance(t, tuple) else (t, t)


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

    def forward(self, x):# 2,65,1024 batch,patch+cls_token,dim (每个patch相当于一个token)
        _, _, _, h = *x.shape, self.heads
        #print("x: {}".format(x.shape))
        # 输入x每个token的维度为1024，在注意力中token被映射16个64维的特征（head*dim_head），
        # 最后再把所有head的特征合并为一个（16*1024）的特征，作为每个token的输出
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # 2,65,1024 -> 2,65,1024*3
        #print("qkv: {}".format(qkv.shape))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)  # 2,65,(16*64) -> 2,16,65,64 ,16个head，每个head维度64
        #print("q: {}".format(q.shape))
        #print("k: {}".format(k.shape))
        #print("v: {}".format(v.shape))
        #q, k, v = map(lambda t: t.view(t.shape[0], t.shape[1], self.heads, -1).permute(0, 2, 1, 3), qkv)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale  # b,16,65,64 @ b,16,64*65 -> b,16,65,65 : q@k.T
        #print("dots: {}".format(dots.shape))
        #dots = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = self.attend(dots)  # 注意力 2,16,65,65  16个head，注意力map尺寸65*65，对应token（patch）[i,j]之间的注意力
        #print("attn: {}".format(attn.shape))
        # 每个token经过每个head的attention后的输出
        out = einsum('b h i j, b h j d -> b h i d', attn, v)  # atten@v 2,16,65,65 @ 2,16,65,64 -> 2,16,65,64
        #print("out dots(q,k)v: {}".format(out.shape))
        #out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')  # 合并所有head的输出(16*64) -> 1024 得到每个token当前的特征
        #print("out after permutation: {}".format(out.shape))
        #out = out.permute(0, 2, 1, 3).reshape(out.shape[0], out.shape[2], -1)
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
class BaseBlock(nn.Module):
    def __init__(self, inp, exp, out, se, stride, heads, dim):
        super(BaseBlock, self).__init__()
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
            #up = nn.ConvTranspose2d(o, o, f * 2, stride=f,
                                    #padding=f // 2, output_padding=0,
                                    #groups=o, bias=False)
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

def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

'Unoffical'
class MobileFormer52t(nn.Module):
    def __init__(self, heads, final_kernel, head_conv):
        super(MobileFormer52t, self).__init__()
        self.token = nn.Parameter(nn.Parameter(torch.randn(1, 3, 128)))
        # stem 3 224 224 -> 16 112 112
        self.stem = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(8),
            hswish(),
        )
        # bneck
        self.bneck = nn.Sequential(
            nn.Conv2d(8, 24, 3, stride=2, padding=1, groups=8),
            hswish(),
            nn.Conv2d(24, 12, kernel_size=1, stride=1),
            nn.BatchNorm2d(12)
        )
        '''
        cfg = {
            'name': 'mf52',
            'token': 3,  # num tokens
            'embed': 128,  # embed dim
            'stem': 8,
            'bneck': {'e': 24, 'o': 12, 's': 2},  # exp out stride
            'body': [
                # stage2
                {'inp': 12, 'exp': 36, 'out': 12, 'se': None, 'stride': 1, 'heads': 2, 'dim': 128},
                # stage3
                {'inp': 12, 'exp': 72, 'out': 24, 'se': None, 'stride': 2, 'heads': 2, 'dim': 128},
                {'inp': 24, 'exp': 72, 'out': 24, 'se': None, 'stride': 1, 'heads': 2, 'dim': 128},
                # stage4
                {'inp': 24, 'exp': 144, 'out': 48, 'se': None, 'stride': 2, 'heads': 2, 'dim': 128},
                {'inp': 48, 'exp': 192, 'out': 48, 'se': None, 'stride': 1, 'heads': 2, 'dim': 128},
                {'inp': 48, 'exp': 288, 'out': 64, 'se': None, 'stride': 1, 'heads': 2, 'dim': 128},
                # stage5
                {'inp': 64, 'exp': 384, 'out': 96, 'se': None, 'stride': 2, 'heads': 2, 'dim': 128},
                {'inp': 96, 'exp': 576, 'out': 96, 'se': None, 'stride': 1, 'heads': 2, 'dim': 128},
            ],
            'fc1': 1024,  # hid_layer
            'fc2': 1000  # num_clasess
            ,
        }
        '''
        # body
        #self.block = nn.ModuleList()
        #for kwargs in cfg['body']:
            #self.block.append(BaseBlock(**kwargs, dim=128))
        self.block0 = nn.Sequential(
            BaseBlock(inp=12, exp=36, out=12, se=None, stride=1, heads=2, dim=128),
        )
        self.block1 = nn.Sequential(
            BaseBlock(inp=12, exp=72, out=24, se=None, stride=2, heads=2, dim=128),
            BaseBlock(inp=24, exp=72, out=24, se=None, stride=1, heads=2, dim=128),
        )
        self.block2 = nn.Sequential(
            BaseBlock(inp=24, exp=144, out=48, se=None, stride=2, heads=2, dim=128),
            BaseBlock(inp=48, exp=192, out=48, se=None, stride=1, heads=2, dim=128),
            BaseBlock(inp=48, exp=288, out=64, se=None, stride=1, heads=2, dim=128),
        )
        #self.block3 = nn.Sequential(
            #BaseBlock(inp=64, exp=384, out=96, se=None, stride=2, heads=2, dim=128),
            #BaseBlock(inp=96, exp=576, out=96, se=None, stride=1, heads=2, dim=128),
        #)
        
        #inp = 96
        #exp = 576 ########## with 1x1 convolution and bn big start loss!!!!
        self.conv = nn.Conv2d(96, 576, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(576)
        self.init_params()
        
        #self.ida_up = IDAUp(12, [12, 24, 64, 576], [2 ** i for i in range(4)])
        self.ida_up = IDAUp(12, [12, 24, 64], [2 ** i for i in range(3)])
        
        self.heads = heads
        for head_id, head in enumerate(self.heads):
            classes = self.heads[head]
            if head_id == 0:
                fc = nn.Sequential(
                    nn.Conv2d(12, head_conv, kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_conv, classes, kernel_size=final_kernel, stride=1, padding=final_kernel // 2, bias=True),
                    )
                if 'hm' in head:
                    fc[-1].bias.data.fill_(-2.19)
                self.__setattr__(head, fc)
            else:
                fc = nn.Sequential(
                    nn.Conv2d(12, head_conv,
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
        b, _, _, _ = x.shape
        z = self.token.repeat(b, 1, 1)
        x = self.bneck(self.stem(x))
        
        x0, z0 = self.block0([x, z])
        x1, z1 = self.block1([x0, z0])
        x2, z2 = self.block2([x1, z1])
        #x3, z3 = self.block3([x2, z2])
        #x3 = self.bn(self.conv(x3))
        #out = [x0, x1, x2, x3]
        out = [x0, x1, x2]
        
        y = []
        for i in range(3):
            y.append(out[i].clone())
        self.ida_up(y, 0, len(y))
        
        
        z = {}
        for head in self.heads:
            z[head] = self.__getattr__(head)(y[-1])
        return [z]
