from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torchvision.models as models
import torch
import torch.nn as nn
import os

from .networks.msra_resnet import get_pose_net
from .networks.dlav0 import get_pose_net as get_dlav0
from .networks.pose_dla_dcn import get_pose_net as get_dla_dcn
from .networks.resnet_dcn import get_pose_net as get_pose_net_dcn
from .networks.large_hourglass import get_large_hourglass_net
from .networks.mobilenetv3 import MobileNetV3
from .networks.mobilenetv3 import MobileNetV3S
from .networks.mobilenetv1 import MobileNetV1
from .networks.mobilenetv3 import MobileNetV3SCSP
from .networks.mobilenetv3 import MobileNetV3CSP
from .networks.mobileformer52 import MobileFormer52
from .networks.mobileformer96 import MobileFormer96
from .networks.mobileformer151 import MobileFormer151
#from .networks.mobileformer214 import MobileFormer214
from .networks.mobileformer294 import MobileFormer294
from .networks.mobileformer508 import MobileFormer508
from .networks.mobilevitxxs import MobileViTxxs
from .networks.mobilevitxs import MobileViTxs
from .networks.mobilevits import MobileViTs
from .networks.dat import DAT
from .networks.mobilevitv2 import MobileViTv2
from .networks.mobilevitv3xxs import MobileViTv3xxs
from .networks.mobilevitv3d1d0 import MobileViTv3d1d0
from .networks.mobilevitv3xxst import MobileViTv3xxst
from .networks.mobileformer52t import MobileFormer52t
from .networks.mobilevitv3d1d0t import MobileViTv3d1d0t
from .networks.mobilevitv3xxstt import MobileViTv3xxstt
from .networks.mobileformer52tt import MobileFormer52tt
from .networks.ldla import get_pose_net as get_ldla_dcn
from .networks.mobileformer52ttt import MobileFormer52ttt
from .networks.mvitdla_dcn import get_pose_net as mvitdla
from .networks.mfdla import get_pose_net as mfdla
from .networks.dlav2 import get_pose_net as dlav2
from .networks.dlap import get_pose_net as dlap
from .networks.dlapdoubleSE import get_pose_net as dlapdoubleSE
from .networks.dlapRWSEHead import get_pose_net as dlapRWSEHead
from .networks.dlapsingleRWSE import get_pose_net as dlapsingleRWSE
from .networks.dlapdoubleeSE import get_pose_net as dlapdoubleeSE
from .networks.dlapsingleECA import get_pose_net as dlapsingleECA
from .networks.dlapsingleSE import get_pose_net as dlapsingleSE
from .networks.dlapnon import get_pose_net as dlapnon
from .networks.dlapsingleandHeadRWSE import get_pose_net as dlapsingleandHeadRWSE
from .networks.dlapdoubleRWSE import get_pose_net as dlapdoubleRWSE
from .networks.dlapsingleeSE import get_pose_net as dlapsingleeSE
from .networks.dlapnoncarafe import get_pose_net as dlapnoncarafe
from .networks.dlapnontransposeconv import get_pose_net as dlapnontransposeconv
from .networks.dlapnoncarafefull import get_pose_net as dlapnoncarafefull
from .networks.dlapcarafehead import get_pose_net as dlapcarafehead
from .networks.dlapreason import get_pose_net as dlapreason
from .networks.dlapreasonbranch import get_pose_net as dlapreasonbranch
from .networks.dlapreasontwo import get_pose_net as dlapreasontwo
from .networks.dlapreasonbranchonly import get_pose_net as dlapreasonbranchonly
from .networks.dlapfinalkfpn import get_pose_net as dlapfinalkfpn
from .networks.dlapfinalkfpntwo import get_pose_net as dlapfinalkfpntwo
from .networks.dlapheadtest import get_pose_net as dlapheadtest
from .networks.dlapheadtesttwo import get_pose_net as dlapheadtesttwo
from .networks.dlapreasonnohead import get_pose_net as dlapreasonnohead
from .networks.dlapreasonfullcarafe import get_pose_net as dlapreasonfullcarafe
from .networks.dlapfcsff import get_pose_net as dlapfcsff
from .networks.dlapreasonat import get_pose_net as dlapreasonat
from .networks.dlapreasonats import get_pose_net as dlapreasonats
from .networks.dlapreasonatsbn import get_pose_net as dlapreasonatsbn
from .networks.dlaat import get_pose_net as dlaat
from .networks.dlaats import get_pose_net as dlaats
from .networks.dlaatbn import get_pose_net as dlaatbn
from .networks.dlaatsbn import get_pose_net as dlaatsbn
from .networks.dlapreasonbn import get_pose_net as dlapreasonbn
from .networks.dlapreasonnons import get_pose_net as dlapreasonnons
from .networks.dlapreasonnonsbn import get_pose_net as dlapreasonnonsbn
from .networks.dlaah import get_pose_net as dlaah
from .networks.dlaahnobn import get_pose_net as dlaahnobn
from .networks.dlaahs import get_pose_net as dlaahs
from .networks.dlaahnobns import get_pose_net as dlaahnobns
from .networks.dlaahp import get_pose_net as dlaahp
from .networks.dlaahnobnp import get_pose_net as dlaahnobnp
from .networks.dlaahpcsa import get_pose_net as dlaahpcsa
from .networks.dlaahnobnpcsa import get_pose_net as dlaahnobnpcsa
from .networks.dlaathsff import get_pose_net as dlaathsff
from .networks.dlahead import get_pose_net as dlahead
from .networks.dlaallhead import get_pose_net as dlaallhead
from .networks.dlaallheadnos import get_pose_net as dlaallheadnos
from .networks.dlafi import get_pose_net as dlafi
from .networks.dlaallheadnew import get_pose_net as dlaallheadnew
from .networks.dlaallheadnewnp import get_pose_net as dlaallheadnewnp
from .networks.dlaallheadlast import get_pose_net as dlaallheadlast
from .networks.dlaheads import get_pose_net as dlaheads
from .networks.dlaatPRelu import get_pose_net as dlaatPRelu
from .networks.dlaallheadPRelu import get_pose_net as dlaallheadPRelu
from .networks.dlaaware import get_pose_net as dlaaware
from .networks.dlasc import get_pose_net as dlasc
from .networks.dlaoffsetaware import get_pose_net as dlaoffsetaware
from .networks.dlaoffwhaware import get_pose_net as dlaoffwhaware
from .networks.dlawhaware import get_pose_net as dlawhaware
from .networks.dlaupt import get_pose_net as dlaupt
from .networks.dlaneckt import get_pose_net as dlaneckt
from .networks.dlasct import get_pose_net as dlasct
from .networks.dlaneckfpn import get_pose_net as dlaneckfpn
from .networks.dlaneckCM import get_pose_net as dlaneckCM
from .networks.dlaneckfpncarafe import get_pose_net as dlaneckfpncarafe
from .networks.dlanecknasfpn import get_pose_net as dlanecknasfpn
from .networks.dlaneckbfp import get_pose_net as dlaneckbfp
from .networks.dlaneckDE import get_pose_net as dlaneckDE
from .networks.dlacsa import get_pose_net as dlacsa
from .networks.dlase import get_pose_net as dlase
from .networks.dlaese import get_pose_net as dlaese
from .networks.dlaeca import get_pose_net as dlaeca
from .networks.dlacbam import get_pose_net as dlacbam
from .networks.dlada import get_pose_net as dlada
from .networks.dladc import get_pose_net as dladc
from .networks.dlady import get_pose_net as dlady
from .networks.dlask import get_pose_net as dlask
from .networks.dlafpn import get_pose_net as dlafpn
from .networks.dlaCM import get_pose_net as dlaCM
from .networks.dlafpncarafe import get_pose_net as dlafpncarafe
from .networks.dlaDE import get_pose_net as dlaDE
from .networks.dlaneckkfpn import get_pose_net as dlaneckkfpn
from .networks.dlascno import get_pose_net as dlascno
from .networks.dlascfc import get_pose_net as dlascfc
from .networks.dlascfi import get_pose_net as dlascfi
from .networks.dlafpnvt import get_pose_net as dlafpnvt
from .networks.dlafpncarafevt import get_pose_net as dlafpncarafevt


_model_factory = {
  'res': get_pose_net, # default Resnet with deconv
  'dlav0': get_dlav0, # default DLAup
  'dla': get_dla_dcn,
  'resdcn': get_pose_net_dcn,
  'hourglass': get_large_hourglass_net,
  'mobilenetv3': MobileNetV3,
  'mobilenetv3s': MobileNetV3S,
  'mobilenetv1': MobileNetV1,
  'mobilenetv3scsp': MobileNetV3SCSP,
  'mobilenetv3csp': MobileNetV3CSP,
  'mobileformer52': MobileFormer52,
  'mobileformer96': MobileFormer96,
  'mobileformer151': MobileFormer151,
  #'mobileformer214': MobileFormer214,
  'mobileformer294': MobileFormer294,
  'mobileformer508': MobileFormer508,
  'mobilevitxxs': MobileViTxxs,
  'mobilevitxs': MobileViTxs,
  'mobilevits': MobileViTs,
  'dat': DAT,
  'mobilevitv2': MobileViTv2,
  'mobilevitv3xxs': MobileViTv3xxs,
  'mobilevitv3d1d0': MobileViTv3d1d0,
  'mobilevitv3xxst': MobileViTv3xxst,
  'mobileformer52t': MobileFormer52t,
  'mobilevitv3d1d0t': MobileViTv3d1d0t,
  'mobilevitv3xxstt': MobileViTv3xxstt,
  'mobileformer52tt': MobileFormer52tt,
  'ldla': get_ldla_dcn,
  'mobileformer52ttt': MobileFormer52ttt,
  'mvitdla': mvitdla,
  'mfdla': mfdla,
  'dla2v': dlav2,
  'dlap': dlap,
  'dlapdoubleSE': dlapdoubleSE,
  'dlapRWSEHead': dlapRWSEHead,
  'dlapsingleRWSE': dlapsingleRWSE,
  'dlapdoubleeSE': dlapdoubleeSE,
  'dlapsingleECA': dlapsingleECA,
  'dlapsingleSE': dlapsingleSE,
  'dlapnon': dlapnon,
  'dlapsingleandHeadRWSE': dlapsingleandHeadRWSE,
  'dlapdoubleRWSE': dlapdoubleRWSE,
  'dlapsingleeSE': dlapsingleeSE,
  'dlapnoncarafe': dlapnoncarafe,
  'dlapnontransposeconv': dlapnontransposeconv,
  'dlapnoncarafefull': dlapnoncarafefull,
  'dlapcarafehead': dlapcarafehead,
  'dlapreason': dlapreason,
  'dlapreasonbranch': dlapreasonbranch,
  'dlapreasontwo': dlapreasontwo,
  'dlapreasonbranchonly': dlapreasonbranchonly,
  'dlapfinalkfpn': dlapfinalkfpn,
  'dlapfinalkfpntwo': dlapfinalkfpntwo,
  'dlapheadtest': dlapheadtest,
  'dlapheadtesttwo': dlapheadtesttwo,
  'dlapreasonnohead': dlapreasonnohead,
  'dlapreasonfullcarafe': dlapreasonfullcarafe,
  'dlapfcsff': dlapfcsff,
  'dlapreasonat': dlapreasonat,
  'dlapreasonats': dlapreasonats,
  'dlapreasonatsbn': dlapreasonatsbn,
  'dlaat': dlaat,
  'dlaats': dlaats,
  'dlaatbn': dlaatbn,
  'dlaatsbn': dlaatsbn,
  'dlapreasonbn': dlapreasonbn,
  'dlapreasonnons': dlapreasonnons,
  'dlapreasonnonsbn': dlapreasonnonsbn,
  'dlaah': dlaah,
  'dlaahnobn': dlaahnobn,
  'dlaahs': dlaahs,
  'dlaahnobns': dlaahnobns,
  'dlaahp': dlaahp,
  'dlaahnobnp': dlaahnobnp,
  'dlaahpcsa': dlaahpcsa,
  'dlaahnobnpcsa': dlaahnobnpcsa,
  'dlaathsff': dlaathsff,
  'dlahead': dlahead,
  'dlaallhead': dlaallhead,
  'dlaallheadnos': dlaallheadnos,
  'dlafi': dlafi,
  'dlaallheadnew': dlaallheadnew,
  'dlaallheadnewnp': dlaallheadnewnp,
  'dlaallheadlast': dlaallheadlast,
  'dlaheads': dlaheads,
  'dlaatPRelu': dlaatPRelu,
  'dlaallheadPRelu': dlaallheadPRelu,
  'dlaaware': dlaaware,
  'dlasc': dlasc,
  'dlaoffsetaware': dlaoffsetaware,
  'dlaoffwhaware': dlaoffwhaware,
  'dlawhaware': dlawhaware,
  'dlaupt': dlaupt,
  'dlaneckt':dlaneckt,
  'dlasct': dlasct,
  'dlaneckfpn': dlaneckfpn,
  'dlaneckCM': dlaneckCM,
  'dlaneckfpncarafe': dlaneckfpncarafe,
  'dlanecknasfpn': dlanecknasfpn,
  'dlaneckbfp': dlaneckbfp,
  'dlaneckDE': dlaneckDE,
  'dlacsa': dlacsa,
  'dlase': dlase,
  'dlaese': dlaese,
  'dlaeca': dlaeca,
  'dlacbam': dlacbam,
  'dlada': dlada,
  'dladc': dladc,
  'dlady': dlady,
  'dlask': dlask,
  'dlafpn': dlafpn,
  'dlaCM': dlaCM,
  'dlafpncarafe': dlafpncarafe,
  'dlaDE': dlaDE,
  'dlaneckkfpn': dlaneckkfpn,
  'dlascno': dlascno,
  'dlascfc': dlascfc,
  'dlascfi': dlascfi,
  'dlafpnvt': dlafpnvt,
  'dlafpncarafevt': dlafpncarafevt,
}

def create_model(arch, heads, head_conv):
  num_layers = int(arch[arch.find('_') + 1:]) if '_' in arch else 0
  arch = arch[:arch.find('_')] if '_' in arch else arch
  get_model = _model_factory[arch]
  #model = get_model(num_layers=num_layers, heads=heads, head_conv=head_conv)
  if num_layers == 0:
      model = get_model(heads, final_kernel=1, head_conv=head_conv)
  else:
      model = get_model(num_layers=num_layers, heads=heads, head_conv=head_conv)
  return model

def load_model(model, model_path, optimizer=None, resume=False, 
               lr=None, lr_step=None):
  start_epoch = 0
  checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
  print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
  state_dict_ = checkpoint['state_dict']
  state_dict = {}
  
  # convert data_parallal to model
  for k in state_dict_:
    if k.startswith('module') and not k.startswith('module_list'):
      state_dict[k[7:]] = state_dict_[k]
    else:
      state_dict[k] = state_dict_[k]
  model_state_dict = model.state_dict()

  # check loaded parameters and created model parameters
  msg = 'If you see this, your model does not fully load the ' + \
        'pre-trained weight. Please make sure ' + \
        'you have correctly specified --arch xxx ' + \
        'or set the correct --num_classes for your own dataset.'
  for k in state_dict:
    if k in model_state_dict:
      if state_dict[k].shape != model_state_dict[k].shape:
        print('Skip loading parameter {}, required shape{}, '\
              'loaded shape{}. {}'.format(
          k, model_state_dict[k].shape, state_dict[k].shape, msg))
        state_dict[k] = model_state_dict[k]
    else:
      print('Drop parameter {}.'.format(k) + msg)
  for k in model_state_dict:
    if not (k in state_dict):
      print('No param {}.'.format(k) + msg)
      state_dict[k] = model_state_dict[k]
  model.load_state_dict(state_dict, strict=False)

  # resume optimizer parameters
  if optimizer is not None and resume:
    if 'optimizer' in checkpoint:
      optimizer.load_state_dict(checkpoint['optimizer'])
      start_epoch = checkpoint['epoch']
      start_lr = lr
      for step in lr_step:
        if start_epoch >= step:
          start_lr *= 0.1
      for param_group in optimizer.param_groups:
        param_group['lr'] = start_lr
      print('Resumed optimizer with start lr', start_lr)
    else:
      print('No optimizer parameters in checkpoint.')
  if optimizer is not None:
    return model, optimizer, start_epoch
  else:
    return model

def save_model(path, epoch, model, optimizer=None):
  if isinstance(model, torch.nn.DataParallel):
    state_dict = model.module.state_dict()
  else:
    state_dict = model.state_dict()
  data = {'epoch': epoch,
          'state_dict': state_dict}
  if not (optimizer is None):
    data['optimizer'] = optimizer.state_dict()
  torch.save(data, path, _use_new_zipfile_serialization=False)

