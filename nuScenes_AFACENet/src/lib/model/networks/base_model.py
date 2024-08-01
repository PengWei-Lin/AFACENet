from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#import torch
from torch import nn

from mmcv.ops import CARAFEPack
import torch
import torch.nn.functional as F


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


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class BaseModel(nn.Module):
    def __init__(self, heads, head_convs, num_stacks, last_channel, opt=None):
        super(BaseModel, self).__init__()
        if opt is not None and opt.head_kernel != 3:
          print('Using head kernel:', opt.head_kernel)
          head_kernel = opt.head_kernel
        else:
          head_kernel = 3
          
        '''
        self.carafe_hm1_3 = CARAFEPack(channels = 10, scale_factor=4, up_kernel = 5, up_group = 1, encoder_kernel = 3, encoder_dilation = 1, compressed_channels = 64)
        self.carafe_hm2_3 = CARAFEPack(channels = 10, scale_factor=2, up_kernel = 5, up_group = 1, encoder_kernel = 3, encoder_dilation = 1, compressed_channels = 64)
        
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
        
        self.attention_head = CSAModuleV2(in_size=10)
        '''
        self.num_stacks = num_stacks
        self.heads = heads
        '''
        fpn_channels = [256, 128, 64]
        for fpn_idx, fpn_c in enumerate(fpn_channels):
            for head in sorted(self.heads):
                classes = self.heads[head]
                if 'hm' in head:
                    fc = nn.Sequential(
                        nn.Conv2d(fpn_c, 256, kernel_size=3, padding=1, bias=True),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(256, classes,
                                  kernel_size=1, stride=1,
                                  padding=0, bias=True),
                        )
                    fc[-1].bias.data.fill_(-2.19)
                else:
                    fc = nn.Sequential(
                        nn.Conv2d(fpn_c, 256, kernel_size=3, padding=1, bias=True),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(256, classes,
                                  kernel_size=1, stride=1,
                                  padding=0, bias=True))
                    fill_fc_weights(fc)
                self.__setattr__('fpn{}_{}'.format(fpn_idx, head), fc)
        '''
        
        for head in self.heads:
            classes = self.heads[head]
            head_conv = head_convs[head]
            if len(head_conv) > 0:
              out = nn.Conv2d(head_conv[-1], classes, 
                    kernel_size=1, stride=1, padding=0, bias=True)
              conv = nn.Conv2d(last_channel, head_conv[0],
                               kernel_size=head_kernel, 
                               padding=head_kernel // 2, bias=True)
              convs = [conv]
              for k in range(1, len(head_conv)):
                  convs.append(nn.Conv2d(head_conv[k - 1], head_conv[k], 
                               kernel_size=1, bias=True))
              if len(convs) == 1 and head == 'hm':
                fc = nn.Sequential(conv, nn.ReLU(inplace=True), out, CSAModuleV2(in_size=10))
                #print("Attention heatmap")
              elif len(convs) == 1:
                fc = nn.Sequential(conv, nn.ReLU(inplace=True), out)
                #print("Level 1?????")
              elif len(convs) == 2:
                fc = nn.Sequential(
                  convs[0], nn.ReLU(inplace=True), 
                  convs[1], nn.ReLU(inplace=True), out)
              elif len(convs) == 3:
                fc = nn.Sequential(
                    convs[0], nn.ReLU(inplace=True), 
                    convs[1], nn.ReLU(inplace=True), 
                    convs[2], nn.ReLU(inplace=True), out)
              elif len(convs) == 4:
                fc = nn.Sequential(
                    convs[0], nn.ReLU(inplace=True), 
                    convs[1], nn.ReLU(inplace=True), 
                    convs[2], nn.ReLU(inplace=True), 
                    convs[3], nn.ReLU(inplace=True), out)
              if 'hm' in head:
                #fc[-1].bias.data.fill_(opt.prior_bias)
                fc[-2].bias.data.fill_(opt.prior_bias)
              else:
                fill_fc_weights(fc)
            else:
              fc = nn.Conv2d(last_channel, classes, 
                  kernel_size=1, stride=1, padding=0, bias=True)
              if 'hm' in head:
                fc.bias.data.fill_(opt.prior_bias)
              else:
                fill_fc_weights(fc)
            self.__setattr__(head, fc)
        

    def img2feats(self, x):
      raise NotImplementedError
    
    def imgpre2feats(self, x, pre_img=None, pre_hm=None):
      raise NotImplementedError

    def forward(self, x, pre_img=None, pre_hm=None):
      if (pre_hm is not None) or (pre_img is not None):
        feats = self.imgpre2feats(x, pre_img, pre_hm)
      else:
        #print("Here!!!!!!!!!!!!!!!!!!!!!!!")
        #print("x:{}".format(x.shape))
        #_, _, input_h, input_w = x.size()
        #_, hm_w = input_h // 4, input_w // 4
        feats = self.img2feats(x)
        #concat_level1, concat_level2, concat_level3 = self.img2feats(x)
      out = []
      if self.opt.model_output_list:
        #print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        for s in range(self.num_stacks):
          z = []
          for head in sorted(self.heads):
              z.append(self.__getattr__(head)(feats[s]))
          out.append(z)
      else:
        #print("This place?????????????????????????????????")
        
        for s in range(self.num_stacks):
          z = {}
          for head in self.heads:
              z[head] = self.__getattr__(head)(feats[s])
          out.append(z)
        
        '''
        ret = {}
        for head in self.heads:
            temp_outs = []
            for fpn_idx, fdn_input in enumerate([concat_level1, concat_level2, concat_level3]):
                fpn_out = self.__getattr__(
                    'fpn{}_{}'.format(fpn_idx, head))(fdn_input)
                _, _, fpn_out_h, fpn_out_w = fpn_out.size()
                
                if hm_w // fpn_out_w == 4 and head == 'hm':
                    fpn_out = self.carafe_hm1_3(fpn_out)
                elif hm_w // fpn_out_w == 2 and head == 'hm':
                    fpn_out =self.carafe_hm2_3(fpn_out)
                # Add amodel center
                elif hm_w // fpn_out_w == 4 and head == 'amodel_offset':
                    fpn_out = self.carafe_act1_2(fpn_out)
                elif hm_w // fpn_out_w == 2 and head == 'amodel_offset':
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
                temp_outs.append(fpn_out)
                
            final_out = self.apply_kfpn(temp_outs)
            if head == 'hm':
                final_out = self.attention_head(final_out)
            ret[head] = final_out
        out.append(ret)
        '''
      return out
  
    '''
    def apply_kfpn(self, outs):
        outs = torch.cat([out.unsqueeze(-1) for out in outs], dim=-1)
        softmax_outs = F.softmax(outs, dim=-1)
        ret_outs = (outs * softmax_outs).sum(dim=-1)
        return ret_outs
    '''
