from torch import nn, Tensor
import argparse
from typing import Dict, Tuple, Optional, Union, Any, List
import math
import torch
from torch.nn import functional as F
from torch.nn import init
#from utils import logger


SUPPORTED_ACT_FNS = []
ACT_FN_REGISTRY = {}
SUPPORTED_NORM_FNS = []
NORM_LAYER_REGISTRY = {}
NORM_LAYER_CLS = []

    
class Att_Proj_Node_o(nn.Module):
    def __init__(self, chi, cho):
        super(Att_Proj_Node_o, self).__init__()
        self.conv = nn.Conv2d(chi, cho, kernel_size=3, stride=1, padding=1, dilation=1, groups=1) #bias=True)
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


def make_divisible(
    v: Union[float, int],
    divisor: Optional[int] = 8,
    min_value: Optional[Union[float, int]] = None,
) -> Union[float, int]:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return 



class BaseLayer(nn.Module):
    """
    Base class for neural network layers
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add layer specific arguments"""
        return parser

    def forward(self, *args, **kwargs) -> Any:
        pass

    def profile_module(self, *args, **kwargs) -> Tuple[Tensor, float, float]:
        raise NotImplementedError

    def __repr__(self):
        return "{}".format(self.__class__.__name__)


class Identity(BaseLayer):
    """
    This is a place-holder and returns the same tensor.
    """

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x

    def profile_module(self, x: Tensor) -> Tuple[Tensor, float, float]:
        return x, 0.0, 0.0



def build_normalization_layer(
    opts,
    num_features: int,
    norm_type: Optional[str] = None,
    num_groups: Optional[int] = None,
    *args,
    **kwargs
) -> torch.nn.Module:
    """
    Helper function to build the normalization layer.
    The function can be used in either of below mentioned ways:
    Scenario 1: Set the default normalization layers using command line arguments. This is useful when the same normalization
    layer is used for the entire network (e.g., ResNet).
    Scenario 2: Network uses different normalization layers. In that case, we can override the default normalization
    layer by specifying the name using `norm_type` argument
    """
    norm_type = (
        getattr(opts, "model.normalization.name", "batch_norm")
        if norm_type is None
        else norm_type
    )
    num_groups = (
        getattr(opts, "model.normalization.groups", 1)
        if num_groups is None
        else num_groups
    )
    momentum = getattr(opts, "model.normalization.momentum", 0.1)
    norm_layer = None
    norm_type = norm_type.lower() if norm_type is not None else None

    if norm_type in NORM_LAYER_REGISTRY:
        if torch.cuda.device_count() < 1 and norm_type.find("sync_batch") > -1:
            # for a CPU-device, Sync-batch norm does not work. So, change to batch norm
            norm_type = norm_type.replace("sync_", "")
        norm_layer = NORM_LAYER_REGISTRY[norm_type](
            normalized_shape=num_features,
            num_features=num_features,
            momentum=momentum,
            num_groups=num_groups,
        )
    elif norm_type == "identity":
        norm_layer = Identity()
    return 

def get_normalization_layer(
    opts,
    num_features: int,
    norm_type: Optional[str] = None,
    num_groups: Optional[int] = None,
    *args,
    **kwargs
) -> nn.Module:
    """
    Helper function to get normalization layers
    """
    return build_normalization_layer(opts, num_features, norm_type, num_groups)


class Dropout(nn.Dropout):
    """
    This layer, during training, randomly zeroes some of the elements of the input tensor with probability `p`
    using samples from a Bernoulli distribution.
    Args:
        p: probability of an element to be zeroed. Default: 0.5
        inplace: If set to ``True``, will do this operation in-place. Default: ``False``
    Shape:
        - Input: :math:`(N, *)` where :math:`N` is the batch size
        - Output: same as the input
    """

    def __init__(
        self, p: Optional[float] = 0.5, inplace: Optional[bool] = False, *args, **kwargs
    ) -> None:
        super().__init__(p=p, inplace=inplace)

    def profile_module(
        self, input: Tensor, *args, **kwargs
    ) -> Tuple[Tensor, float, float]:
        return input, 0.0, 0.0


class GlobalPool(BaseLayer):
    """
    This layers applies global pooling over a 4D or 5D input tensor
    Args:
        pool_type (Optional[str]): Pooling type. It can be mean, rms, or abs. Default: `mean`
        keep_dim (Optional[bool]): Do not squeeze the dimensions of a tensor. Default: `False`
    Shape:
        - Input: :math:`(N, C, H, W)` or :math:`(N, C, D, H, W)`
        - Output: :math:`(N, C, 1, 1)` or :math:`(N, C, 1, 1, 1)` if keep_dim else :math:`(N, C)`
    """

    pool_types = ["mean", "rms", "abs"]

    def __init__(
        self,
        pool_type: Optional[str] = "mean",
        keep_dim: Optional[bool] = False,
        *args,
        **kwargs
    ) -> None:
        super().__init__()
        '''
        if pool_type not in self.pool_types:
            logger.error(
                "Supported pool types are: {}. Got {}".format(
                    self.pool_types, pool_type
                )
            )
        '''
        self.pool_type = pool_type
        self.keep_dim = keep_dim

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        cls_name = "{} arguments".format(cls.__name__)
        group = parser.add_argument_group(title=cls_name, description=cls_name)
        group.add_argument(
            "--model.layer.global-pool",
            type=str,
            default="mean",
            help="Which global pooling?",
        )
        return parser

    def _global_pool(self, x: Tensor, dims: List):
        if self.pool_type == "rms":  # root mean square
            x = x**2
            x = torch.mean(x, dim=dims, keepdim=self.keep_dim)
            x = x**-0.5
        elif self.pool_type == "abs":  # absolute
            x = torch.mean(torch.abs(x), dim=dims, keepdim=self.keep_dim)
        else:
            # default is mean
            # same as AdaptiveAvgPool
            x = torch.mean(x, dim=dims, keepdim=self.keep_dim)
        return x

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 4:
            dims = [-2, -1]
        elif x.dim() == 5:
            dims = [-3, -2, -1]
        else:
            raise NotImplementedError("Currently 2D and 3D global pooling supported")
        return self._global_pool(x, dims=dims)

    def profile_module(self, input: Tensor) -> Tuple[Tensor, float, float]:
        input = self.forward(input)
        return input, 0.0, 0.0

    def __repr__(self):
        return "{}(type={})".format(self.__class__.__name__, self.pool_type)



class LinearLayer(BaseLayer):
    """
    Applies a linear transformation to the input data
    Args:
        in_features (int): number of features in the input tensor
        out_features (int): number of features in the output tensor
        bias  (Optional[bool]): use bias or not
        channel_first (Optional[bool]): Channels are first or last dimension. If first, then use Conv2d
    Shape:
        - Input: :math:`(N, *, C_{in})` if not channel_first else :math:`(N, C_{in}, *)` where :math:`*` means any number of dimensions.
        - Output: :math:`(N, *, C_{out})` if not channel_first else :math:`(N, C_{out}, *)`
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: Optional[bool] = True,
        channel_first: Optional[bool] = False,
        *args,
        **kwargs
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features)) if bias else None

        self.in_features = in_features
        self.out_features = out_features
        self.channel_first = channel_first

        self.reset_params()

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        parser.add_argument(
            "--model.layer.linear-init",
            type=str,
            default="xavier_uniform",
            help="Init type for linear layers",
        )
        parser.add_argument(
            "--model.layer.linear-init-std-dev",
            type=float,
            default=0.01,
            help="Std deviation for Linear layers",
        )
        return parser

    def reset_params(self):
        if self.weight is not None:
            torch.nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            torch.nn.init.constant_(self.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        if self.channel_first:
            #if not self.training:
                #logger.error("Channel-first mode is only supported during inference")
            #if x.dim() != 4:
                #logger.error("Input should be 4D, i.e., (B, C, H, W) format")
            # only run during conversion
            with torch.no_grad():
                return F.conv2d(
                    input=x,
                    weight=self.weight.clone()
                    .detach()
                    .reshape(self.out_features, self.in_features, 1, 1),
                    bias=self.bias,
                )
        else:
            x = F.linear(x, weight=self.weight, bias=self.bias)
        return x

    def __repr__(self):
        repr_str = (
            "{}(in_features={}, out_features={}, bias={}, channel_first={})".format(
                self.__class__.__name__,
                self.in_features,
                self.out_features,
                True if self.bias is not None else False,
                self.channel_first,
            )
        )
        return repr_str

    def profile_module(
        self, input: Tensor, *args, **kwargs
    ) -> Tuple[Tensor, float, float]:
        out_size = list(input.shape)
        out_size[-1] = self.out_features
        params = sum([p.numel() for p in self.parameters()])
        macs = params
        output = torch.zeros(size=out_size, dtype=input.dtype, device=input.device)
        return output, params, macs



class Conv2d(nn.Conv2d):
    """
    Applies a 2D convolution over an input
    Args:
        in_channels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H_{in}, W_{in})`
        out_channels (int): :math:`C_{out}` from an expected output of size :math:`(N, C_{out}, H_{out}, W_{out})`
        kernel_size (Union[int, Tuple[int, int]]): Kernel size for convolution.
        stride (Union[int, Tuple[int, int]]): Stride for convolution. Defaults to 1
        padding (Union[int, Tuple[int, int]]): Padding for convolution. Defaults to 0
        dilation (Union[int, Tuple[int, int]]): Dilation rate for convolution. Default: 1
        groups (Optional[int]): Number of groups in convolution. Default: 1
        bias (bool): Use bias. Default: ``False``
        padding_mode (Optional[str]): Padding mode. Default: ``zeros``
        use_norm (Optional[bool]): Use normalization layer after convolution. Default: ``True``
        use_act (Optional[bool]): Use activation layer after convolution (or convolution and normalization).
                                Default: ``True``
        act_name (Optional[str]): Use specific activation function. Overrides the one specified in command line args.
    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})`
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Optional[Union[int, Tuple[int, int]]] = 1,
        padding: Optional[Union[int, Tuple[int, int]]] = 0,
        dilation: Optional[Union[int, Tuple[int, int]]] = 1,
        groups: Optional[int] = 1,
        bias: Optional[bool] = False,
        padding_mode: Optional[str] = "zeros",
        *args,
        **kwargs
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )


class ConvLayer(BaseLayer):
    """
    Applies a 2D convolution over an input
    Args:
        opts: command line arguments
        in_channels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H_{in}, W_{in})`
        out_channels (int): :math:`C_{out}` from an expected output of size :math:`(N, C_{out}, H_{out}, W_{out})`
        kernel_size (Union[int, Tuple[int, int]]): Kernel size for convolution.
        stride (Union[int, Tuple[int, int]]): Stride for convolution. Default: 1
        dilation (Union[int, Tuple[int, int]]): Dilation rate for convolution. Default: 1
        padding (Union[int, Tuple[int, int]]): Padding for convolution. When not specified, 
                                               padding is automatically computed based on kernel size 
                                               and dilation rage. Default is ``None``
        groups (Optional[int]): Number of groups in convolution. Default: ``1``
        bias (Optional[bool]): Use bias. Default: ``False``
        padding_mode (Optional[str]): Padding mode. Default: ``zeros``
        use_norm (Optional[bool]): Use normalization layer after convolution. Default: ``True``
        use_act (Optional[bool]): Use activation layer after convolution (or convolution and normalization).
                                Default: ``True``
        act_name (Optional[str]): Use specific activation function. Overrides the one specified in command line args.
    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})`
    .. note::
        For depth-wise convolution, `groups=C_{in}=C_{out}`.
    """

    def __init__(
        self,
        opts,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Optional[Union[int, Tuple[int, int]]] = 1,
        dilation: Optional[Union[int, Tuple[int, int]]] = 1,
        padding: Optional[Union[int, Tuple[int, int]]] = None,
        groups: Optional[int] = 1,
        bias: Optional[bool] = False,
        padding_mode: Optional[str] = "zeros",
        use_norm: Optional[bool] = True,
        use_act: Optional[bool] = True,
        act_name: Optional[str] = None,
        *args,
        **kwargs
    ) -> None:
        super().__init__()

        if use_norm:
            norm_type = getattr(opts, "model.normalization.name", "batch_norm")
            if norm_type is not None and norm_type.find("batch") > -1:
                assert not bias, "Do not use bias when using normalization layers."
            elif norm_type is not None and norm_type.find("layer") > -1:
                bias = True
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        if isinstance(stride, int):
            stride = (stride, stride)

        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        assert isinstance(kernel_size, Tuple)
        assert isinstance(stride, Tuple)
        assert isinstance(dilation, Tuple)

        if padding is None:
            padding = (
                int((kernel_size[0] - 1) / 2) * dilation[0],
                int((kernel_size[1] - 1) / 2) * dilation[1],
            )

        #if in_channels % groups != 0:
            #logger.error(
                #"Input channels are not divisible by groups. {}%{} != 0 ".format(
                    #in_channels, groups
                #)
            #)
        #if out_channels % groups != 0:
            #logger.error(
                #"Output channels are not divisible by groups. {}%{} != 0 ".format(
                    #out_channels, groups
                #)
            #)

        block = nn.Sequential()

        conv_layer = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )

        block.add_module(name="conv", module=conv_layer)

        self.norm_name = None
        if use_norm:
            norm_layer = get_normalization_layer(opts=opts, num_features=out_channels)
            block.add_module(name="norm", module=norm_layer)
            self.norm_name = norm_layer.__class__.__name__

        self.act_name = None
        act_type = (
            getattr(opts, "model.activation.name", "prelu")
            if act_name is None
            else act_name
        )

        if act_type is not None and use_act:
            neg_slope = getattr(opts, "model.activation.neg_slope", 0.1)
            inplace = getattr(opts, "model.activation.inplace", False)
            act_layer = get_activation_fn(
                act_type=act_type,
                inplace=inplace,
                negative_slope=neg_slope,
                num_parameters=out_channels,
            )
            block.add_module(name="act", module=act_layer)
            self.act_name = act_layer.__class__.__name__

        self.block = block

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.kernel_size = conv_layer.kernel_size
        self.bias = bias
        self.dilation = dilation

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        cls_name = "{} arguments".format(cls.__name__)
        group = parser.add_argument_group(title=cls_name, description=cls_name)
        group.add_argument(
            "--model.layer.conv-init",
            type=str,
            default="kaiming_normal",
            help="Init type for conv layers",
        )
        parser.add_argument(
            "--model.layer.conv-init-std-dev",
            type=float,
            default=None,
            help="Std deviation for conv layers",
        )
        return parser

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)

    def __repr__(self):
        repr_str = self.block[0].__repr__()
        repr_str = repr_str[:-1]

        if self.norm_name is not None:
            repr_str += ", normalization={}".format(self.norm_name)

        if self.act_name is not None:
            repr_str += ", activation={}".format(self.act_name)
        repr_str += ")"
        return repr_str

    def profile_module(self, input: Tensor) -> (Tensor, float, float):
        #if input.dim() != 4:
            #logger.error(
                #"Conv2d requires 4-dimensional input (BxCxHxW). Provided input has shape: {}".format(
                    #input.size()
                #)
            #)

        b, in_c, in_h, in_w = input.size()
        assert in_c == self.in_channels, "{}!={}".format(in_c, self.in_channels)

        stride_h, stride_w = self.stride
        groups = self.groups

        out_h = in_h // stride_h
        out_w = in_w // stride_w

        k_h, k_w = self.kernel_size

        # compute MACS
        macs = (k_h * k_w) * (in_c * self.out_channels) * (out_h * out_w) * 1.0
        macs /= groups

        if self.bias:
            macs += self.out_channels * out_h * out_w

        # compute parameters
        params = sum([p.numel() for p in self.parameters()])

        output = torch.zeros(
            size=(b, self.out_channels, out_h, out_w),
            dtype=input.dtype,
            device=input.device,
        )
        # print(macs)
        return output, params, macs



def module_profile(module, x: Tensor, *args, **kwargs) -> Tuple[Tensor, float, float]:
    """
    Helper function to profile a module.
    .. note::
        Module profiling is for reference only and may contain errors as it solely relies on user implementation to
        compute theoretical FLOPs
    """

    if isinstance(module, nn.Sequential):
        n_macs = n_params = 0.0
        for l in module:
            try:
                x, l_p, l_macs = l.profile_module(x)
                n_macs += l_macs
                n_params += l_p
            except Exception as e:
                print(e, l)
                pass
    else:
        x, n_params, n_macs = module.profile_module(x)
    return x, n_params, 


def build_activation_layer(
    act_type: Optional[str] = "relu",
    num_parameters: Optional[int] = -1,
    inplace: Optional[bool] = True,
    negative_slope: Optional[float] = 0.1,
    *args,
    **kwargs
) -> torch.nn.Module:
    """
    Helper function to build the activation function
    """
    if act_type is None:
        act_type = "none"
    act_type = act_type.lower()
    act_layer = None
    if act_type in ACT_FN_REGISTRY:
        act_layer = ACT_FN_REGISTRY[act_type](
            num_parameters=num_parameters,
            inplace=inplace,
            negative_slope=negative_slope,
            *args,
            **kwargs
        )
    #else:
        #logger.error(
            #"Supported activation layers are: {}. Supplied argument is: {}".format(
                #SUPPORTED_ACT_FNS, act_type
            #)
        #)
    return act_layer


def get_activation_fn(
    act_type: Optional[str] = "relu",
    num_parameters: Optional[int] = -1,
    inplace: Optional[bool] = True,
    negative_slope: Optional[float] = 0.1,
    *args,
    **kwargs
) -> nn.Module:
    """
    Helper function to get activation (or non-linear) function
    """
    return build_activation_layer(
        act_type=act_type,
        num_parameters=num_parameters,
        negative_slope=negative_slope,
        inplace=inplace,
        *args,
        **kwargs
    )



class BaseModule(nn.Module):
    """Base class for all modules"""

    def __init__(self, *args, **kwargs):
        super(BaseModule, self).__init__()

    def forward(self, x: Any, *args, **kwargs) -> Any:
        raise NotImplementedError

    def profile_module(self, input: Any, *args, **kwargs) -> Tuple[Any, float, float]:
        raise NotImplementedError

    def __repr__(self):
        return "{}".format(self.__class__.__name__)



class SingleHeadAttention(BaseLayer):
    """
    This layer applies a single-head attention as described in `DeLighT <https://arxiv.org/abs/2008.00623>`_ paper
    Args:
        embed_dim (int): :math:`C_{in}` from an expected input of size :math:`(N, P, C_{in})`
        attn_dropout (Optional[float]): Attention dropout. Default: 0.0
        bias (Optional[bool]): Use bias or not. Default: ``True``
    Shape:
        - Input: :math:`(N, P, C_{in})` where :math:`N` is batch size, :math:`P` is number of patches,
        and :math:`C_{in}` is input embedding dim
        - Output: same shape as the input
    """

    def __init__(
        self,
        embed_dim: int,
        attn_dropout: Optional[float] = 0.0,
        bias: Optional[bool] = True,
        *args,
        **kwargs
    ) -> None:
        super().__init__()

        self.qkv_proj = LinearLayer(
            in_features=embed_dim, out_features=3 * embed_dim, bias=bias
        )

        self.attn_dropout = Dropout(p=attn_dropout)
        self.out_proj = LinearLayer(
            in_features=embed_dim, out_features=embed_dim, bias=bias
        )

        self.softmax = nn.Softmax(dim=-1)
        self.embed_dim = embed_dim
        self.scaling = self.embed_dim**-0.5

    def __repr__(self) -> str:
        return "{}(embed_dim={}, attn_dropout={})".format(
            self.__class__.__name__, self.embed_dim, self.attn_dropout.p
        )

    def forward(
        self,
        x: Tensor,
        x_kv: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        *args,
        **kwargs
    ) -> Tensor:
        # [N, P, C] --> [N, P, 3C]
        if x_kv is None:
            qkv = self.qkv_proj(x)
            # [N, P, 3C] --> [N, P, C] x 3
            query, key, value = torch.chunk(qkv, chunks=3, dim=-1)
        else:
            query = F.linear(
                x,
                weight=self.qkv_proj.weight[: self.embed_dim, ...],
                bias=self.qkv_proj.bias[: self.embed_dim],
            )

            # [N, P, C] --> [N, P, 2C]
            kv = F.linear(
                x_kv,
                weight=self.qkv_proj.weight[self.embed_dim :, ...],
                bias=self.qkv_proj.bias[self.embed_dim :],
            )
            key, value = torch.chunk(kv, chunks=2, dim=-1)

        query = query * self.scaling

        # [N, P, C] --> [N, C, P]
        key = key.transpose(-2, -1)

        # QK^T
        # [N, P, C] x [N, C, P] --> [N, P, P]
        attn = torch.matmul(query, key)

        if attn_mask is not None:
            # attn_mask shape should be the same as attn
            assert list(attn_mask.shape) == list(
                attn.shape
            ), "Shape of attention mask and attn should be the same. Got: {} and {}".format(
                attn_mask.shape, attn.shape
            )
            attn = attn + attn_mask

        if key_padding_mask is not None:
            # Do not attend to padding positions
            # key padding mask size is [N, P]
            batch_size, num_src_tokens, num_tgt_tokens = attn.shape
            assert key_padding_mask.dim() == 2 and list(key_padding_mask.shape) == [
                batch_size,
                num_tgt_tokens,
            ], "Key_padding_mask should be 2-dimension with shape [{}, {}]. Got: {}".format(
                batch_size, num_tgt_tokens, key_padding_mask.shape
            )
            attn = attn.masked_fill(
                key_padding_mask.unsqueeze(1).to(torch.bool),
                float("-inf"),
            )

        attn = self.softmax(attn)
        attn = self.attn_dropout(attn)

        # weighted sum
        # [N, P, P] x [N, P, C] --> [N, P, C]
        out = torch.matmul(attn, value)
        out = self.out_proj(out)

        return out

    def profile_module(self, input: Tensor) -> Tuple[Tensor, float, float]:
        b_sz, seq_len, in_channels = input.shape
        params = macs = 0.0

        qkv, p, m = module_profile(module=self.qkv_proj, x=input)
        params += p
        macs += m * seq_len * b_sz

        # number of operations in QK^T
        m_qk = (seq_len * in_channels * in_channels) * b_sz
        macs += m_qk

        # number of operations in computing weighted sum
        m_wt = (seq_len * in_channels * in_channels) * b_sz
        macs += m_wt

        out_p, p, m = module_profile(module=self.out_proj, x=input)
        params += p
        macs += m * seq_len * b_sz

        return input, params, 



class MultiHeadAttention(BaseLayer):
    """
    This layer applies a multi-head self- or cross-attention as described in
    `Attention is all you need <https://arxiv.org/abs/1706.03762>`_ paper
    Args:
        embed_dim (int): :math:`C_{in}` from an expected input of size :math:`(N, S, C_{in})`
        num_heads (int): Number of heads in multi-head attention
        attn_dropout (Optional[float]): Attention dropout. Default: 0.0
        bias (Optional[bool]): Use bias or not. Default: ``True``
    Shape:
        - Input:
           - Query tensor (x_q) :math:`(N, S, C_{in})` where :math:`N` is batch size, :math:`S` is number of source tokens,
        and :math:`C_{in}` is input embedding dim
           - Optional Key-Value tensor (x_kv) :math:`(N, T, C_{in})` where :math:`T` is number of target tokens
        - Output: same shape as the input
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attn_dropout: Optional[float] = 0.0,
        bias: Optional[bool] = True,
        output_dim: Optional[int] = None,
        coreml_compatible: Optional[bool] = False,
        *args,
        **kwargs
    ) -> None:
        if output_dim is None:
            output_dim = embed_dim
        super().__init__()
        #if embed_dim % num_heads != 0:
            #logger.error(
                #"Embedding dim must be divisible by number of heads in {}. Got: embed_dim={} and num_heads={}".format(
                    #self.__class__.__name__, embed_dim, num_heads
                #)
            #)

        self.qkv_proj = LinearLayer(
            in_features=embed_dim, out_features=3 * embed_dim, bias=bias
        )

        self.attn_dropout = Dropout(p=attn_dropout)
        self.out_proj = LinearLayer(
            in_features=embed_dim, out_features=output_dim, bias=bias
        )

        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5
        self.softmax = nn.Softmax(dim=-1)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.coreml_compatible = coreml_compatible
        self.use_separate_proj_weight = embed_dim != output_dim

    def __repr__(self):
        return "{}(head_dim={}, num_heads={}, attn_dropout={})".format(
            self.__class__.__name__, self.head_dim, self.num_heads, self.attn_dropout.p
        )

    def forward_tracing(
        self,
        x_q: Tensor,
        x_kv: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if x_kv is None:
            # [N, S, C] --> # [N, S, 3C] Here, T=S
            qkv = self.qkv_proj(x_q)
            # # [N, S, 3C] --> # [N, S, C] x 3
            query, key, value = torch.chunk(qkv, chunks=3, dim=-1)
        else:
            # [N, S, C]
            query = F.linear(
                x_q,
                weight=self.qkv_proj.weight[: self.embed_dim, ...],
                bias=self.qkv_proj.bias[: self.embed_dim]
                if self.qkv_proj.bias is not None
                else None,
            )

            # [N, T, C] --> [N, T, 2C]
            kv = F.linear(
                x_kv,
                weight=self.qkv_proj.weight[self.embed_dim :, ...],
                bias=self.qkv_proj.bias[self.embed_dim :]
                if self.qkv_proj.bias is not None
                else None,
            )
            key, value = torch.chunk(kv, chunks=2, dim=-1)

        query = query * self.scaling

        # [N, S, C] --> [N, S, c] x h, where C = c * h
        query = torch.chunk(query, chunks=self.num_heads, dim=-1)

        # [N, T, C] --> [N, T, c] x h, where C = c * h
        value = torch.chunk(value, chunks=self.num_heads, dim=-1)
        # [N, T, C] --> [N, T, c] x h, where C = c * h
        key = torch.chunk(key, chunks=self.num_heads, dim=-1)

        wt_out = []
        for h in range(self.num_heads):
            attn_h = torch.matmul(query[h], key[h].transpose(-1, -2))
            attn_h = self.softmax(attn_h)
            attn_h = self.attn_dropout(attn_h)
            out_h = torch.matmul(attn_h, value[h])
            wt_out.append(out_h)

        wt_out = torch.cat(wt_out, dim=-1)
        wt_out = self.out_proj(wt_out)
        return wt_out

    def forward_default(
        self,
        x_q: Tensor,
        x_kv: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        # [N, S, C]
        b_sz, S_len, in_channels = x_q.shape

        if x_kv is None:
            # self-attention
            # [N, S, C] --> [N, S, 3C] --> [N, S, 3, h, c] where C = hc
            qkv = self.qkv_proj(x_q).reshape(b_sz, S_len, 3, self.num_heads, -1)
            # [N, S, 3, h, c] --> [N, h, 3, S, C]
            qkv = qkv.transpose(1, 3).contiguous()

            # [N, h, 3, S, C] --> [N, h, S, C] x 3
            query, key, value = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        else:
            T_len = x_kv.shape[1]

            # cross-attention
            # [N, S, C]
            query = F.linear(
                x_q,
                weight=self.qkv_proj.weight[: self.embed_dim, ...],
                bias=self.qkv_proj.bias[: self.embed_dim]
                if self.qkv_proj.bias is not None
                else None,
            )
            # [N, S, C] --> [N, S, h, c] --> [N, h, S, c]
            query = (
                query.reshape(b_sz, S_len, self.num_heads, self.head_dim)
                .transpose(1, 2)
                .contiguous()
            )

            # [N, T, C] --> [N, T, 2C]
            kv = F.linear(
                x_kv,
                weight=self.qkv_proj.weight[self.embed_dim :, ...],
                bias=self.qkv_proj.bias[self.embed_dim :]
                if self.qkv_proj.bias is not None
                else None,
            )
            # [N, T, 2C] --> [N, T, 2, h, c]
            kv = kv.reshape(b_sz, T_len, 2, self.num_heads, self.head_dim)
            # [N, T, 2, h, c] --> [N, h, 2, T, c]
            kv = kv.transpose(1, 3).contiguous()
            key, value = kv[:, :, 0], kv[:, :, 1]

        query = query * self.scaling

        # [N h, T, c] --> [N, h, c, T]
        key = key.transpose(-1, -2)

        # QK^T
        # [N, h, S, c] x [N, h, c, T] --> [N, h, S, T]
        attn = torch.matmul(query, key)

        batch_size, num_heads, num_src_tokens, num_tgt_tokens = attn.shape
        if attn_mask is not None:
            # attn_mask shape should be the same as attn
            assert list(attn_mask.shape) == [
                batch_size,
                num_src_tokens,
                num_tgt_tokens,
            ], "Shape of attention mask should be [{}, {}, {}]. Got: {}".format(
                batch_size, num_src_tokens, num_tgt_tokens, attn_mask.shape
            )
            # [N, S, T] --> [N, 1, S, T]
            attn_mask = attn_mask.unsqueeze(1)
            attn = attn + attn_mask

        if key_padding_mask is not None:
            # Do not attend to padding positions
            # key padding mask size is [N, T]
            assert key_padding_mask.dim() == 2 and list(key_padding_mask.shape) == [
                batch_size,
                num_tgt_tokens,
            ], "Key_padding_mask should be 2-dimension with shape [{}, {}]. Got: {}".format(
                batch_size, num_tgt_tokens, key_padding_mask.shape
            )
            attn = attn.masked_fill(
                key_padding_mask.unsqueeze(1)
                .unsqueeze(2)
                .to(torch.bool),  # [N, T] --> [N, 1, 1, T]
                float("-inf"),
            )

        attn_dtype = attn.dtype
        attn_as_float = self.softmax(attn.float())
        attn = attn_as_float.to(attn_dtype)
        attn = self.attn_dropout(attn)

        # weighted sum
        # [N, h, S, T] x [N, h, T, c] --> [N, h, S, c]
        out = torch.matmul(attn, value)

        # [N, h, S, c] --> [N, S, h, c] --> [N, S, C]
        out = out.transpose(1, 2).reshape(b_sz, S_len, -1)
        out = self.out_proj(out)

        return out

    def forward_pytorch(
        self,
        x_q: Tensor,
        x_kv: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        out, _ = F.multi_head_attention_forward(
            query=x_q,
            key=x_kv if x_kv is not None else x_q,
            value=x_kv if x_kv is not None else x_q,
            embed_dim_to_check=self.embed_dim,
            num_heads=self.num_heads,
            in_proj_weight=torch.empty([0]),
            in_proj_bias=self.qkv_proj.bias,
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=self.attn_dropout.p,
            out_proj_weight=self.out_proj.weight,
            out_proj_bias=self.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            attn_mask=attn_mask,
            use_separate_proj_weight=True,
            q_proj_weight=self.qkv_proj.weight[: self.embed_dim, ...],
            k_proj_weight=self.qkv_proj.weight[
                self.embed_dim : 2 * self.embed_dim, ...
            ],
            v_proj_weight=self.qkv_proj.weight[2 * self.embed_dim :, ...],
        )
        return out

    def forward(
        self,
        x_q: Tensor,
        x_kv: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        *args,
        **kwargs
    ) -> Tensor:
        if self.coreml_compatible:
            # For CoreML, we follow batch-first format. Make sure the input is of the form
            # [Batch , Sequence, Hidden_dim]
            return self.forward_tracing(
                x_q=x_q,
                x_kv=x_kv,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
            )
        elif kwargs.get("use_pytorch_mha", False):
            # pytorch uses sequence-first format. Make sure that input is of the form [Sequence, Batch, Hidden dim]
            return self.forward_pytorch(
                x_q=x_q,
                x_kv=x_kv,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
            )
        else:
            # our default implementation format follows batch-first format. Make sure the input is of the form
            # [Batch , Sequence, Hidden_dim]
            return self.forward_default(
                x_q=x_q,
                x_kv=x_kv,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
            )

    def profile_module(self, input) -> Tuple[Tensor, float, float]:
        b_sz, seq_len, in_channels = input.shape
        params = macs = 0.0

        qkv, p, m = module_profile(module=self.qkv_proj, x=input)
        params += p
        macs += m * seq_len * b_sz

        # number of operations in QK^T
        m_qk = (seq_len * seq_len * in_channels) * b_sz
        macs += m_qk

        # number of operations in computing weighted sum
        m_wt = (seq_len * seq_len * in_channels) * b_sz
        macs += m_wt

        out_p, p, m = module_profile(module=self.out_proj, x=input)
        params += p
        macs += m * seq_len * b_sz

        return input, params, 



class TransformerEncoder(BaseModule):
    """
    This class defines the pre-norm `Transformer encoder <https://arxiv.org/abs/1706.03762>`_
    Args:
        opts: command line arguments
        embed_dim (int): :math:`C_{in}` from an expected input of size :math:`(N, P, C_{in})`
        ffn_latent_dim (int): Inner dimension of the FFN
        num_heads (Optional[int]) : Number of heads in multi-head attention. Default: 8
        attn_dropout (Optional[float]): Dropout rate for attention in multi-head attention. Default: 0.0
        dropout (Optional[float]): Dropout rate. Default: 0.0
        ffn_dropout (Optional[float]): Dropout between FFN layers. Default: 0.0
        transformer_norm_layer (Optional[str]): Normalization layer. Default: layer_norm
    Shape:
        - Input: :math:`(N, P, C_{in})` where :math:`N` is batch size, :math:`P` is number of patches,
        and :math:`C_{in}` is input embedding dim
        - Output: same shape as the input
    """

    def __init__(
        self,
        opts,
        embed_dim: int,
        ffn_latent_dim: int,
        num_heads: Optional[int] = 8,
        attn_dropout: Optional[float] = 0.0,
        dropout: Optional[float] = 0.0,
        ffn_dropout: Optional[float] = 0.0,
        transformer_norm_layer: Optional[str] = "layer_norm",
        *args,
        **kwargs
    ) -> None:

        super().__init__()

        attn_unit = SingleHeadAttention(
            embed_dim=embed_dim, attn_dropout=attn_dropout, bias=True
        )
        if num_heads > 1:
            attn_unit = MultiHeadAttention(
                embed_dim,
                num_heads,
                attn_dropout=attn_dropout,
                bias=True,
                coreml_compatible=getattr(
                    opts, "common.enable_coreml_compatible_module", False
                ),
            )

        self.pre_norm_mha = nn.Sequential(
            get_normalization_layer(
                opts=opts, norm_type=transformer_norm_layer, num_features=embed_dim
            ),
            attn_unit,
            Dropout(p=dropout),
        )

        act_name = self.build_act_layer(opts=opts)
        self.pre_norm_ffn = nn.Sequential(
            get_normalization_layer(
                opts=opts, norm_type=transformer_norm_layer, num_features=embed_dim
            ),
            LinearLayer(in_features=embed_dim, out_features=ffn_latent_dim, bias=True),
            act_name,
            Dropout(p=ffn_dropout),
            LinearLayer(in_features=ffn_latent_dim, out_features=embed_dim, bias=True),
            Dropout(p=dropout),
        )
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_latent_dim
        self.ffn_dropout = ffn_dropout
        self.std_dropout = dropout
        self.attn_fn_name = attn_unit.__class__.__name__
        self.act_fn_name = act_name.__class__.__name__
        self.norm_type = transformer_norm_layer

    @staticmethod
    def build_act_layer(opts) -> nn.Module:
        act_type = getattr(opts, "model.activation.name", "relu")
        neg_slope = getattr(opts, "model.activation.neg_slope", 0.1)
        inplace = getattr(opts, "model.activation.inplace", False)
        act_layer = get_activation_fn(
            act_type=act_type,
            inplace=inplace,
            negative_slope=neg_slope,
            num_parameters=1,
        )
        return act_layer

    def __repr__(self) -> str:
        return "{}(embed_dim={}, ffn_dim={}, dropout={}, ffn_dropout={}, attn_fn={}, act_fn={}, norm_fn={})".format(
            self.__class__.__name__,
            self.embed_dim,
            self.ffn_dim,
            self.std_dropout,
            self.ffn_dropout,
            self.attn_fn_name,
            self.act_fn_name,
            self.norm_type,
        )

    def forward(
        self,
        x: Tensor,
        x_prev: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        *args,
        **kwargs
    ) -> Tensor:

        # Multi-head attention
        res = x
        x = self.pre_norm_mha[0](x)  # norm
        x = self.pre_norm_mha[1](
            x_q=x,
            x_kv=x_prev,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            *args,
            **kwargs
        )  # mha
        x = self.pre_norm_mha[2](x)  # dropout
        x = x + res

        # Feed forward network
        x = x + self.pre_norm_ffn(x)
        return x

    def profile_module(
        self, input: Tensor, *args, **kwargs
    ) -> Tuple[Tensor, float, float]:
        b_sz, seq_len = input.shape[:2]

        out, p_mha, m_mha = module_profile(module=self.pre_norm_mha, x=input)

        out, p_ffn, m_ffn = module_profile(module=self.pre_norm_ffn, x=input)
        m_ffn = m_ffn * b_sz * seq_len

        macs = m_mha + m_ffn
        params = p_mha + p_ffn

        return input, params, 



class InvertedResidual(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: Union[int, float],
        dilation: int = 1,
        skip_connection: Optional[bool] = True,
    ):
        assert stride in [1, 2]
        hidden_dim = make_divisible(int(round(in_channels * expand_ratio)), 8)

        super().__init__()
        '''
        block = nn.Sequential()
        if expand_ratio != 1:
            block.add_module(
                name="exp_1x1",
                module=ConvLayer(
                    opts,
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    kernel_size=1,
                    use_act=True,
                    use_norm=True,
                ),
            )

        block.add_module(
            name="conv_3x3",
            module=ConvLayer(
                opts,
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                stride=stride,
                kernel_size=3,
                groups=hidden_dim,
                use_act=True,
                use_norm=True,
                dilation=dilation,
            ),
        )

        block.add_module(
            name="red_1x1",
            module=ConvLayer(
                opts,
                in_channels=hidden_dim,
                out_channels=out_channels,
                kernel_size=1,
                use_act=False,
                use_norm=True,
            ),
        )
        '''        
        block = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim, momentum=0.1),
            nn.PReLu(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim),
            nn.BatchNorm2d(hidden_dim, momentum=0.1),
            nn.PReLu(),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels, momentum=0.1)
        )

        self.block = block
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.exp = expand_ratio
        self.dilation = dilation
        self.stride = stride
        self.use_res_connect = (
            self.stride == 1 and in_channels == out_channels and skip_connection
        )

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.block(x)

    def profile_module(
        self, input: Tensor, *args, **kwargs
    ) -> Tuple[Tensor, float, float]:
        return module_profile(module=self.block, x=input)

    def __repr__(self) -> str:
        return "{}(in_channels={}, out_channels={}, stride={}, exp={}, dilation={}, skip_conn={})".format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels,
            self.stride,
            self.exp,
            self.dilation,
            self.use_res_connect,
        )


class MobileViTv3Block(nn.Module):
    """
        MobileViTv3 block
    """
    def __init__(self, in_channels: int, transformer_dim: int, ffn_dim: int,
                 n_transformer_blocks: Optional[int] = 2,
                 head_dim: Optional[int] = 32, attn_dropout: Optional[float] = 0.1,
                 dropout: Optional[int] = 0.1, ffn_dropout: Optional[int] = 0.1, patch_h: Optional[int] = 8,
                 patch_w: Optional[int] = 8, transformer_norm_layer: Optional[str] = "layer_norm",
                 conv_ksize: Optional[int] = 3,
                 dilation: Optional[int] = 1, var_ffn: Optional[bool] = False,
                 no_fusion: Optional[bool] = False,
                 ):

        # For MobileViTv3: Normal 3x3 convolution --> Depthwise 3x3 convolution
        '''
        conv_3x3_in = ConvLayer(
            opts=opts, in_channels=in_channels, out_channels=in_channels,
            kernel_size=conv_ksize, stride=1, use_norm=True, use_act=True, dilation=dilation,
            groups=in_channels
        )
        conv_1x1_in = ConvLayer(
            opts=opts, in_channels=in_channels, out_channels=transformer_dim,
            kernel_size=1, stride=1, use_norm=False, use_act=False
        )


        conv_1x1_out = ConvLayer(
            opts=opts, in_channels=transformer_dim, out_channels=in_channels,
            kernel_size=1, stride=1, use_norm=True, use_act=True
        )
        conv_3x3_out = None
        '''
        
        conv_3x3_in = nn.Sequential(
            nn.Conv2d(),
            nn.BatchNorm2d(),
            nn.PReLU(),
        )
        conv_1x1_in = nn.Sequential(
            nn.Conv2d()
        )
        conv_1x1_out = nn.Sequential(
            nn.Conv2d(),
            nn.BatchNorm2d(),
            nn.PReLU(),
        )

        # For MobileViTv3: input+global --> local+global
        if not no_fusion:
            #input_ch = tr_dim + in_ch
            conv_3x3_out = nn.Sequential(
                nn.Conv2d(in_channels=transformer_dim+in_channels, out_channels=in_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(),
                nn.PReLU(),
            )
        super(MobileViTv3Block, self).__init__()
        self.local_rep = nn.Sequential()
        self.local_rep.add_module(name="conv_3x3", module=conv_3x3_in)
        self.local_rep.add_module(name="conv_1x1", module=conv_1x1_in)

        assert transformer_dim % head_dim == 0
        num_heads = transformer_dim // head_dim

        ffn_dims = [ffn_dim] * n_transformer_blocks

        global_rep = [  #opts=opts
            TransformerEncoder(embed_dim=transformer_dim, ffn_latent_dim=ffn_dims[block_idx], num_heads=num_heads,
                               attn_dropout=attn_dropout, dropout=dropout, ffn_dropout=ffn_dropout,
                               transformer_norm_layer=transformer_norm_layer)
            for block_idx in range(n_transformer_blocks)
        ]
        global_rep.append(  #opts=opts
            get_normalization_layer(norm_type=transformer_norm_layer, num_features=transformer_dim)
        )
        self.global_rep = nn.Sequential(*global_rep)

        self.conv_proj = conv_1x1_out

        self.fusion = conv_3x3_out

        self.patch_h = patch_h
        self.patch_w = patch_w
        self.patch_area = self.patch_w * self.patch_h

        self.cnn_in_dim = in_channels
        self.cnn_out_dim = transformer_dim
        self.n_heads = num_heads
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.ffn_dropout = ffn_dropout
        self.dilation = dilation
        self.ffn_max_dim = ffn_dims[0]
        self.ffn_min_dim = ffn_dims[-1]
        self.var_ffn = var_ffn
        self.n_blocks = n_transformer_blocks
        self.conv_ksize = conv_ksize

    def __repr__(self):
        repr_str = "{}(".format(self.__class__.__name__)
        repr_str += "\n\tconv_in_dim={}, conv_out_dim={}, dilation={}, conv_ksize={}".format(self.cnn_in_dim, self.cnn_out_dim, self.dilation, self.conv_ksize)
        repr_str += "\n\tpatch_h={}, patch_w={}".format(self.patch_h, self.patch_w)
        repr_str += "\n\ttransformer_in_dim={}, transformer_n_heads={}, transformer_ffn_dim={}, dropout={}, " \
                    "ffn_dropout={}, attn_dropout={}, blocks={}".format(
            self.cnn_out_dim,
            self.n_heads,
            self.ffn_dim,
            self.dropout,
            self.ffn_dropout,
            self.attn_dropout,
            self.n_blocks
        )
        if self.var_ffn:
            repr_str += "\n\t var_ffn_min_mult={}, var_ffn_max_mult={}".format(
                self.ffn_min_dim, self.ffn_max_dim
            )

        repr_str += "\n)"
        return repr_str

    def unfolding(self, feature_map: Tensor) -> Tuple[Tensor, Dict]:
        patch_w, patch_h = self.patch_w, self.patch_h
        patch_area = int(patch_w * patch_h)
        batch_size, in_channels, orig_h, orig_w = feature_map.shape

        new_h = int(math.ceil(orig_h / self.patch_h) * self.patch_h)
        new_w = int(math.ceil(orig_w / self.patch_w) * self.patch_w)

        interpolate = False
        if new_w != orig_w or new_h != orig_h:
            # Note: Padding can be done, but then it needs to be handled in attention function.
            feature_map = F.interpolate(feature_map, size=(new_h, new_w), mode="bilinear", align_corners=False)
            interpolate = True

        # number of patches along width and height
        num_patch_w = new_w // patch_w # n_w
        num_patch_h = new_h // patch_h # n_h
        num_patches = num_patch_h * num_patch_w # N

        # [B, C, H, W] --> [B * C * n_h, p_h, n_w, p_w]
        reshaped_fm = feature_map.reshape(batch_size * in_channels * num_patch_h, patch_h, num_patch_w, patch_w)
        # [B * C * n_h, p_h, n_w, p_w] --> [B * C * n_h, n_w, p_h, p_w]
        transposed_fm = reshaped_fm.transpose(1, 2)
        # [B * C * n_h, n_w, p_h, p_w] --> [B, C, N, P] where P = p_h * p_w and N = n_h * n_w
        reshaped_fm = transposed_fm.reshape(batch_size, in_channels, num_patches, patch_area)
        # [B, C, N, P] --> [B, P, N, C]
        transposed_fm = reshaped_fm.transpose(1, 3)
        # [B, P, N, C] --> [BP, N, C]
        patches = transposed_fm.reshape(batch_size * patch_area, num_patches, -1)

        info_dict = {
            "orig_size": (orig_h, orig_w),
            "batch_size": batch_size,
            "interpolate": interpolate,
            "total_patches": num_patches,
            "num_patches_w": num_patch_w,
            "num_patches_h": num_patch_h
        }

        return patches, info_dict

    def folding(self, patches: Tensor, info_dict: Dict) -> Tensor:
        n_dim = patches.dim()
        assert n_dim == 3, "Tensor should be of shape BPxNxC. Got: {}".format(patches.shape)
        # [BP, N, C] --> [B, P, N, C]
        patches = patches.contiguous().view(info_dict["batch_size"], self.patch_area, info_dict["total_patches"], -1)

        batch_size, pixels, num_patches, channels = patches.size()
        num_patch_h = info_dict["num_patches_h"]
        num_patch_w = info_dict["num_patches_w"]

        # [B, P, N, C] --> [B, C, N, P]
        patches = patches.transpose(1, 3)

        # [B, C, N, P] --> [B*C*n_h, n_w, p_h, p_w]
        feature_map = patches.reshape(batch_size * channels * num_patch_h, num_patch_w, self.patch_h, self.patch_w)
        # [B*C*n_h, n_w, p_h, p_w] --> [B*C*n_h, p_h, n_w, p_w]
        feature_map = feature_map.transpose(1, 2)
        # [B*C*n_h, p_h, n_w, p_w] --> [B, C, H, W]
        feature_map = feature_map.reshape(batch_size, channels, num_patch_h * self.patch_h, num_patch_w * self.patch_w)
        if info_dict["interpolate"]:
            feature_map = F.interpolate(feature_map, size=info_dict["orig_size"], mode="bilinear", align_corners=False)
        return feature_map

    def forward(self, x: Tensor) -> Tensor:
        res = x

        # For MobileViTv3: Normal 3x3 convolution --> Depthwise 3x3 convolution
        fm_conv = self.local_rep(x)

        # convert feature map to patches
        patches, info_dict = self.unfolding(fm_conv)

        # learn global representations
        patches = self.global_rep(patches)

        # [B x Patch x Patches x C] --> [B x C x Patches x Patch]
        fm = self.folding(patches=patches, info_dict=info_dict)

        fm = self.conv_proj(fm)

        if self.fusion is not None:
            # For MobileViTv3: input+global --> local+global
            fm = self.fusion(
                torch.cat((fm_conv, fm), dim=1)
            )

        # For MobileViTv3: Skip connection
        fm = fm + res

        return fm

    def profile_module(self, input: Tensor) -> (Tensor, float, float):
        params = macs = 0.0

        res = input
        out_conv, p, m = module_profile(module=self.local_rep, x=input)
        params += p
        macs += m

        patches, info_dict = self.unfolding(feature_map=out_conv)

        patches, p, m = module_profile(module=self.global_rep, x=patches)
        params += p
        macs += m

        fm = self.folding(patches=patches, info_dict=info_dict)

        out, p, m = module_profile(module=self.conv_proj, x=fm)
        params += p
        macs += m

        if self.fusion is not None:
            out, p, m = module_profile(module=self.fusion, x=torch.cat((out, out_conv), dim=1))
            params += p
            macs += m

        return res, params, 


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)



class MobileViTv3(nn.Module):
    def __init__(self, heads, final_kernel, head_conv):
        super(MobileViTv3, self).__init__()
        
        self.conv_1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2),
            nn.BatchNorm2d(16, momentum=0.1),
            nn.PReLU(),
        )
        self.first_st = nn.Sequential(
            InvertedResidual(in_channels=16, out_channels=16, stride=1, expand_ratio=2),
        )
        self.block0 = nn.Sequential(
            InvertedResidual(in_channels=16, out_channels=24, stride=2, expand_ratio=2),
            InvertedResidual(in_channels=24, out_channels=24, stride=1, expand_ratio=2),
            InvertedResidual(in_channels=24, out_channels=24, stride=1, expand_ratio=2),
        )
        self.block1 = nn.Sequential(
            InvertedResidual(in_channels=24, out_channels=48, stride=2, expand_ratio=2),
            MobileViTv3Block(in_channels=48, transformer_dim=64, ffn_dim=128, n_transformer_blocks=2, path_h=2, path_w=2, expand_ratio=2),
        )
        self.block2 = nn.Sequential(
            InvertedResidual(in_channels=64, out_channels=64, stride=2, expand_ratio=2),
            MobileViTv3Block(in_channels=64, transformer_dim=80, ffn_dim=160, n_transformer_blocks=4, path_h=2, path_w=2, expand_ratio=2),
        )
        self.block3 = nn.Sequential(
            InvertedResidual(in_channels=80, out_channels=80, stride=2, expand_ratio=2),
            MobileViTv3Block(in_channels=80, transformer_dim=192, ffn_dim=192, n_transformer_blocks=3, path_h=2, path_w=2, expand_ratio=2),
        )
        self.conv_1x1_exp = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=768, kernel_size=1),
            nn.BatchNorm2d(768, momentum=0.1),
            nn.PReLU(),
        )
        # weight initialization
        #self.reset_parameters(opts=opts)
        
        self.ida_up = IDAUp(24, [24, 64, 80, 768], [2 ** i for i in range(4)])
        
        self.init_params()
        
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
                    fc[-2].bias.data.fill_(-2.19)
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
        x = self.first_st(x)
        out0 = self.block0(x)
        out1 = self.block1(out0)
        out2 = self.block2(out1)
        out3 = self.block3(out2)
        out3 = self.conv_1x1_exp(out3)
        out = [out0, out1, out2, out3]
        
        y = []
        for i in range(4):
            y.append(out[i].clone())
        self.ida_up(y, 0, len(y))
        
        z = {}
        
        for head in self.heads:
            z[head] = self.__getattr__(head)(y[-1])
        return [z]
        
        