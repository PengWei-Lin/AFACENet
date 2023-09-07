from lib.opts import opts
from lib.models.model import create_model, load_model
from types import MethodType
import torch.onnx as onnx
import torch
from torch.onnx import OperatorExportTypes
from collections import OrderedDict

## onnx is not support dict return value
## for dla34
def pose_dla_forward(self, x):
    x = self.base(x)
    x = self.dla_up(x)
    y = []
    for i in range(self.last_level - self.first_level):
        y.append(x[i].clone())
    self.ida_up(y, 0, len(y))
    ret = []  ## change dict to list
    for head in self.heads:
        ret.append(self.__getattr__(head)(y[-1]))
    return ret
## for dla34v0
def dlav0_forward(self, x):
    x = self.base(x)
    x = self.dla_up(x[self.first_level:])
    # x = self.fc(x)
    # y = self.softmax(self.up(x))
    ret = []  ## change dict to list
    for head in self.heads:
        ret.append(self.__getattr__(head)(x))
    return ret
## for resdcn
def resnet_dcn_forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.deconv_layers(x)
    ret = []  ## change dict to list
    for head in self.heads:
        ret.append(self.__getattr__(head)(x))
    return ret

def mobilenet_forward(self, x):
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

def mobileformer_forward(self, x):
    b, _, _, _ = x.shape
    z = self.token.repeat(b, 1, 1)
    x = self.bneck(self.stem(x))
    x0, z0 = self.block0([x, z])
    x1, z1 = self.block1([x0, z0])
    x2, z2 = self.block2([x1, z1])
    x3, z3 = self.block3([x2, z2])
    x3 = self.bn(self.conv(x3))
    out = [x0, x1, x2, x3]
    y = []
    for i in range(4):
        y.append(out[i].clone())
    self.ida_up(y, 0, len(y))
    z = {}
    for head in self.heads:
        z[head] = self.__getattr__(head)(y[-1])
    return [z]

def mobilevit_forward(self, x):
    x = self.conv1(x)
    x = self.first_mv2(x)

    out0 = self.block0(x)
    out1 = self.block1(out0)
    out2 = self.block2(out1)
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

def mobilevitv2_forward(self, x):
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

def mobilevitv3d1d0_forward(self, x):
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

def mobilevitv3xxs_forward(self, x):
    x = self.conv1(x)
    x = self.first_mv2(x)

    out0 = self.block0(x)
    out1 = self.block1(out0)
    out2 = self.block2(out1)
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



forward = {'dla':pose_dla_forward, 'dlav0':dlav0_forward, 'resdcn':resnet_dcn_forward, 
           'mobilenetv3s':mobilenet_forward, 'mobilenetv3scsp':mobilenet_forward, 
           'mobilenetv3':mobilenet_forward, 'mobilenetv3csp':mobilenet_forward, 
           'mobileformer52':mobileformer_forward, 'mobileformer96':mobileformer_forward, 
           'mobileformer151':mobileformer_forward, 'mobileformer294':mobileformer_forward,
           'mobileformer508':mobileformer_forward, 'mobilevitxxs': mobilevit_forward,
           'mobilevitxs':mobilevit_forward, 'mobilevits':mobilevit_forward, 
           'mobilevitv2':mobilevitv2_forward, 'mobilevitv3d1d0':mobilevitv3d1d0_forward,
           'mobilevitv3xxs':mobilevitv3xxs_forward, 'mobilevitv3xxst':mobilevitv3xxs_forward, 
           'mobileforemer52t': mobileformer_forward, 'mobilevitv3xxstt':mobilevitv3xxs_forward}

opt = opts().init()  ## change lib/opts.py add_argument('task', default='ctdet'....) to add_argument('--task', default='ctdet'....)
#opt.arch = 'dla_34'
#opt.arch = 'mobileformer52'
#opt.arch = 'mobilevitxs'
#opt.arch = 'mobilevitv2'
#opt.arch = 'mobilevitv3d1d0'
#opt.arch = 'mobilevitv3xxs'
#opt.arch = 'mobilevitv3xxst'
opt.arch = 'mobilevitv3xxstt'
#opt.arch = 'mobilenetv3'
# opt.heads = OrderedDict([('hm', 80), ('reg', 2), ('wh', 2)])
# heads {'hm': 3, 'dep': 1, 'rot': 8, 'dim': 3, 'wh': 2, 'reg': 2}
opt.heads = OrderedDict([('hm', 3), ('dep', 1), ('rot', 8), ('dim', 3), ('wh', 2), ('reg', 2)])
#opt.head_conv = 256 if 'dla' in opt.arch else 64
if 'dla' in opt.arch:
    opt.head_conv = 256
#elif 'mobilevitv2' or 'mobilevitv3d1d0' in opt.arch:
    #opt.head_conv = 128
else:
    opt.head_conv = 64
print('opt', opt)
model = create_model(opt.arch, opt.heads, opt.head_conv)
#model.forward = MethodType(forward[opt.arch.split('_')[0]], model)
#load_model(model, '../models/ddd_3dop_ldla.pth')
load_model(model, '../models/ddd_3dop_mobilevitv3xxstt.pth')
#load_model(model, '../models/ddd_3dop_mobilenetv3csp.pth')
model.eval()
model.cuda()
# input = torch.zeros([1, 3, 512, 512]).cuda()
input = torch.zeros([1, 3, 384, 1280]).cuda()
#onnx.export(model, input, "../models/ddd_3dop_mobilenetv3csp.onnx", verbose=True,
onnx.export(model, input, "../models/ddd_3dop_mobilevitv3xxstt.onnx", verbose=True,
            operator_export_type=OperatorExportTypes.ONNX, opset_version=12)          