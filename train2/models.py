'''
NVIDIA from jtremblay@gmail.com
'''

# Networks
import numpy as np
import torch
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import torchvision.models as models
import torch.utils.data as data
import time 


# two classes taken from mobilenet 
# https://github.com/pytorch/vision/blob/master/torchvision/models/mobilenet.py
class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_planes),
            nn.ReLU6(inplace=True)
        )



class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, norm_layer=None):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class DopeMobileNet(nn.Module):
    def __init__(
            self,
            pretrained=False,
            numBeliefMap=9,
            numAffinity=16,
            stop_at_stage=6  # number of stages to process (if less than total number of stages)
        ):
        super(DopeMobileNet, self).__init__()

        self.mobile_feature = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True).features

        # upsample to 50x50 from 13x13
        self.upsample = nn.Sequential()
        self.upsample.add_module('0', nn.Upsample(scale_factor=2))

        # should this go before the upsample?
        # self.upsample.add_module('4', nn.Conv2d(1280, 640,
        #     kernel_size=3, stride=1, padding=1))
        self.upsample.add_module('44',InvertedResidual(1280, 640, stride=1, expand_ratio=6, norm_layer=nn.BatchNorm2d))
        # self.upsample.add_module('55',InvertedResidual(1280, 640, stride=1, expand_ratio=6, norm_layer=nn.BatchNorm2d))

        # self.upsample.add_module('5', nn.ReLU(inplace=True))

        # self.upsample.add_module('6', nn.Conv2d(640, 320,
        #     kernel_size=3, stride=1, padding=1))

        self.upsample.add_module('10', nn.Upsample(scale_factor=2))
        # self.upsample.add_module('14', nn.Conv2d(320, 160,
        #     kernel_size=3, stride=1, padding=1))
        # self.upsample.add_module('15', nn.ReLU(inplace=True))
        # self.upsample.add_module('16', nn.Conv2d(160, 64,
        #     kernel_size=3, stride=1, padding=0))
        self.upsample.add_module('55',InvertedResidual(640, 320, stride=1, expand_ratio=6, norm_layer=nn.BatchNorm2d))
        self.upsample.add_module('56',InvertedResidual(320, 64, stride=1, expand_ratio=6, norm_layer=nn.BatchNorm2d))

        # set 50,50
        self.upsample.add_module('4', nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0))

        # final output - change that for mobile block
        # self.heads_0 = nn.Sequential()

        def build_block(inputs, outputs, nb_layers = 2 ):
            layers = []
            layers.append(InvertedResidual(inputs, 64, stride=1, expand_ratio=6, norm_layer=nn.BatchNorm2d))
            for l in range(nb_layers-1):
                layers.append(InvertedResidual(64, 64, stride=1, expand_ratio=6, norm_layer=nn.BatchNorm2d))        
            layers.append(nn.Conv2d(64, outputs, kernel_size=3, stride=1, padding=1))
            # layers.append('4', nn.Conv2d(64, outputs, kernel_size=3, stride=1, padding=1))
            return nn.Sequential(*layers)

        self.head_0_beliefs = build_block(64,numBeliefMap)
        self.head_0_aff = build_block(64,(numBeliefMap-1)*2,3)

        self.head_1_beliefs = build_block(64+numBeliefMap+((numBeliefMap-1)*2),numBeliefMap,3)
        self.head_1_aff = build_block(64+numBeliefMap+(numBeliefMap-1)*2,(numBeliefMap-1)*2,2)

        self.head_2_beliefs = build_block(64+numBeliefMap+((numBeliefMap-1)*2),numBeliefMap,3)
        self.head_2_aff = build_block(64+numBeliefMap+(numBeliefMap-1)*2,(numBeliefMap-1)*2,1)



    def forward(self, x):
        '''Runs inference on the neural network'''
        # print(x.shape)
        out_features = self.mobile_feature(x)
        # print('out2_features',out_features.shape)
        output_up = self.upsample(out_features)
        # print('output_up',output_up.shape)

        # stages
        belief_0 = self.head_0_beliefs(output_up)
        aff_0 = self.head_0_aff(output_up)

        # print(belief_0.shape)

        out_0 = torch.cat([output_up, belief_0, aff_0], 1)

        # print(out_0.shape)
        # raise()
        belief_1 = self.head_1_beliefs(out_0)
        aff_1 = self.head_1_aff(out_0)

        out_1 = torch.cat([output_up, belief_1, aff_1], 1)

        belief_2 = self.head_2_beliefs(out_1)
        aff_2 = self.head_2_aff(out_1)

        return  [belief_0,belief_1,belief_2],\
                [aff_0,aff_1,aff_2]
        


class DopeNetwork(nn.Module):
    def __init__(
            self,
            pretrained=False,
            numBeliefMap=9,
            numAffinity=16,
            stop_at_stage=6  # number of stages to process (if less than total number of stages)
        ):
        super(DopeNetwork, self).__init__()

        self.stop_at_stage = stop_at_stage

        vgg_full = models.vgg19(pretrained=False).features
        self.vgg = nn.Sequential()
        for i_layer in range(24):
            self.vgg.add_module(str(i_layer), vgg_full[i_layer])

        # Add some layers
        i_layer = 23
        self.vgg.add_module(str(i_layer), nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1))
        self.vgg.add_module(str(i_layer+1), nn.ReLU(inplace=True))
        self.vgg.add_module(str(i_layer+2), nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1))
        self.vgg.add_module(str(i_layer+3), nn.ReLU(inplace=True))

        # print('---Belief------------------------------------------------')
        # _2 are the belief map stages
        self.m1_2 = DopeNetwork.create_stage(128, numBeliefMap, True)
        self.m2_2 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity,
                                             numBeliefMap, False)
        self.m3_2 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity,
                                             numBeliefMap, False)
        self.m4_2 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity,
                                             numBeliefMap, False)
        self.m5_2 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity,
                                             numBeliefMap, False)
        self.m6_2 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity,
                                             numBeliefMap, False)

        # print('---Affinity----------------------------------------------')
        # _1 are the affinity map stages
        self.m1_1 = DopeNetwork.create_stage(128, numAffinity, True)
        self.m2_1 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity,
                                             numAffinity, False)
        self.m3_1 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity,
                                             numAffinity, False)
        self.m4_1 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity,
                                             numAffinity, False)
        self.m5_1 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity,
                                             numAffinity, False)
        self.m6_1 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity,
                                             numAffinity, False)


    def forward(self, x):
        '''Runs inference on the neural network'''

        out1 = self.vgg(x)

        out1_2 = self.m1_2(out1)
        out1_1 = self.m1_1(out1)

        if self.stop_at_stage == 1:
            return [out1_2],\
                   [out1_1]

        out2 = torch.cat([out1_2, out1_1, out1], 1)
        out2_2 = self.m2_2(out2)
        out2_1 = self.m2_1(out2)

        if self.stop_at_stage == 2:
            return [out1_2, out2_2],\
                   [out1_1, out2_1]

        out3 = torch.cat([out2_2, out2_1, out1], 1)
        out3_2 = self.m3_2(out3)
        out3_1 = self.m3_1(out3)

        if self.stop_at_stage == 3:
            return [out1_2, out2_2, out3_2],\
                   [out1_1, out2_1, out3_1]

        out4 = torch.cat([out3_2, out3_1, out1], 1)
        out4_2 = self.m4_2(out4)
        out4_1 = self.m4_1(out4)

        if self.stop_at_stage == 4:
            return [out1_2, out2_2, out3_2, out4_2],\
                   [out1_1, out2_1, out3_1, out4_1]

        out5 = torch.cat([out4_2, out4_1, out1], 1)
        out5_2 = self.m5_2(out5)
        out5_1 = self.m5_1(out5)

        if self.stop_at_stage == 5:
            return [out1_2, out2_2, out3_2, out4_2, out5_2],\
                   [out1_1, out2_1, out3_1, out4_1, out5_1]

        out6 = torch.cat([out5_2, out5_1, out1], 1)
        out6_2 = self.m6_2(out6)
        out6_1 = self.m6_1(out6)

        return [out1_2, out2_2, out3_2, out4_2, out5_2, out6_2],\
               [out1_1, out2_1, out3_1, out4_1, out5_1, out6_1]
                        
    @staticmethod
    def create_stage(in_channels, out_channels, first=False):
        '''Create the neural network layers for a single stage.'''

        model = nn.Sequential()
        mid_channels = 128
        if first:
            padding = 1
            kernel = 3
            count = 6
            final_channels = 512
        else:
            padding = 3
            kernel = 7
            count = 10
            final_channels = mid_channels

        # First convolution
        model.add_module("0",
                         nn.Conv2d(
                             in_channels,
                             mid_channels,
                             kernel_size=kernel,
                             stride=1,
                             padding=padding)
                        )

        # Middle convolutions
        i = 1
        while i < count - 1:
            model.add_module(str(i), nn.ReLU(inplace=True))
            i += 1
            model.add_module(str(i),
                             nn.Conv2d(
                                 mid_channels,
                                 mid_channels,
                                 kernel_size=kernel,
                                 stride=1,
                                 padding=padding))
            i += 1

        # Penultimate convolution
        model.add_module(str(i), nn.ReLU(inplace=True))
        i += 1
        model.add_module(str(i), nn.Conv2d(mid_channels, final_channels, kernel_size=1, stride=1))
        i += 1

        # Last convolution
        model.add_module(str(i), nn.ReLU(inplace=True))
        i += 1
        model.add_module(str(i), nn.Conv2d(final_channels, out_channels, kernel_size=1, stride=1))
        i += 1

        return model



class BoundaryAwareNet(nn.Module):
    def __init__(
        self,
        pretrained_dope_path = None, # dope pretrained network    
        num_keypoints = 9,           # number of keypoints to refress to
        ):
        super(BoundaryAwareNet,self).__init__()

        self.dope = DopeNetwork()
        if not pretrained_dope_path is None:
            # print(pretrained_dope_path)
            # self.dope = torch.nn.DataParallel(self.dope)
            from collections import OrderedDict
            state_dict = torch.load(pretrained_dope_path)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v

            self.dope.load_state_dict(new_state_dict)
            print("DOPE pretrained loaded")
        #rest of vgg 
        vgg_full = models.vgg19(pretrained=True).features

        self.vgg = nn.Sequential()
        for i_layer in range(0,len(vgg_full)):
            self.vgg.add_module(str(i_layer), vgg_full[i_layer])

        # input resampling
        self.upsample = torch.nn.Upsample(scale_factor=8)

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_keypoints*2),
        )


    def forward(self,x):

        output_belief, output_affinity = self.dope(x)

        belief_summed = torch.sum(output_belief[-1],dim=1)

        # upsample
        belief_summed_up = self.upsample(belief_summed.unsqueeze(1)).detach()
        x_p = x * torch.cat([belief_summed_up,belief_summed_up,belief_summed_up],dim=1) + x * 0.5

        y_vgg = self.vgg(x_p)
        y = self.avgpool(y_vgg)
        y_class = self.classifier(y.flatten(1))


        return output_belief, output_affinity, y_class




class DreamHourglassMultiStage(nn.Module):
    def __init__(self, n_keypoints,
                       n_image_input_channels = 3,
                       internalize_spatial_softmax = True,
                       learned_beta = True,
                       initial_beta = 1.,
                       n_stages = 2,
                       joints_input = 0,
                       skip_connections = False,
                       deconv_decoder = False,
                       full_output = False):
        super(DreamHourglassMultiStage, self).__init__()

        self.n_keypoints = n_keypoints
        self.n_image_input_channels = n_image_input_channels
        self.internalize_spatial_softmax = internalize_spatial_softmax
        self.skip_connections = skip_connections
        self.deconv_decoder = deconv_decoder
        self.full_output = full_output 

        if self.internalize_spatial_softmax:
            # This warning is because the forward code just ignores the second head (spatial softmax)
            # Revisit later if we need multistage networks where each stage has multiple output heads that are needed
            print("WARNING: Keypoint softmax output head is currently unused. Prefer training new models of this type with internalize_spatial_softmax = False.")
            self.n_output_heads = 2
            self.learned_beta = learned_beta
            self.initial_beta = initial_beta
        else:
            self.n_output_heads = 1
            self.learned_beta = False
        self.joints_input = joints_input

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        assert isinstance(n_stages, int), \
            "Expected \"n_stages\" to be an integer, but it is {}.".format(type(n_stages))
        assert 0 < n_stages and n_stages <= 6, \
            "DreamHourglassMultiStage can only be constructed with 1 to 6 stages at this time."

        self.num_stages = n_stages

        # Stage 1
        self.stage1 = DreamHourglass(
            n_keypoints,
            n_image_input_channels,
            internalize_spatial_softmax,
            learned_beta,
            initial_beta,
            joints_input = joints_input,
            skip_connections = skip_connections,
            deconv_decoder = deconv_decoder,
            full_output = self.full_output,
        )

        # Stage 2
        if self.num_stages > 1:
            self.stage2 = DreamHourglass(
                n_keypoints,
                n_image_input_channels + n_keypoints + (n_keypoints-1)*2, # Includes the most previous stage
                internalize_spatial_softmax,
                learned_beta,
                initial_beta,
                joints_input = joints_input,
                skip_connections = skip_connections,
                deconv_decoder = deconv_decoder,
                full_output = self.full_output,
            )

        # Stage 3
        if self.num_stages > 2:
            self.stage3 = DreamHourglass(
                n_keypoints,
                n_image_input_channels + n_keypoints, # Includes the most previous stage
                internalize_spatial_softmax,
                learned_beta,
                initial_beta,
                joints_input = joints_input,
                skip_connections = skip_connections,
                deconv_decoder = deconv_decoder,
                full_output = self.full_output,
            )

        # Stage 4
        if self.num_stages > 3:
            self.stage4 = DreamHourglass(
                n_keypoints,
                n_image_input_channels + n_keypoints, # Includes the most previous stage
                internalize_spatial_softmax,
                learned_beta,
                initial_beta,
                joints_input = joints_input,
                skip_connections = skip_connections,
                deconv_decoder = deconv_decoder,
                full_output = self.full_output,
            )

        # Stage 5
        if self.num_stages > 4:
            self.stage5 = DreamHourglass(
                n_keypoints,
                n_image_input_channels + n_keypoints, # Includes the most previous stage
                internalize_spatial_softmax,
                learned_beta,
                initial_beta,
                joints_input = joints_input,
                skip_connections = skip_connections,
                deconv_decoder = deconv_decoder,
                full_output = self.full_output,
            )

        # Stage 6
        if self.num_stages > 5:
            self.stage6 = DreamHourglass(
                n_keypoints,
                n_image_input_channels + n_keypoints, # Includes the most previous stage
                internalize_spatial_softmax,
                learned_beta,
                initial_beta,
                joints_input = joints_input,
                skip_connections = skip_connections,
                deconv_decoder = deconv_decoder,
                full_output = self.full_output,
            )

    def forward(self, x, joints = None, verbose = False):

        y_output_stage1 = self.stage1(x, joints=joints)
        y_0_1 = y_output_stage1[0] # Just keeping belief maps for now
        y_1_1 = y_output_stage1[1]

        if self.num_stages == 1:
            return [y_0_1],[y_1_1]

        if self.num_stages > 1:
            # Upsample
            y_output_stage2 = self.stage2(torch.cat([x, y_0_1,y_1_1], dim=1), joints=joints)
            y2 = y_output_stage2[0] # Just keeping belief maps for now

            if self.num_stages == 2:
                return [y_0_1, y2],[y_1_1,y_output_stage2[1]]

        # if self.num_stages > 2:
        #     # Upsample
        #     if self.deconv_decoder or self.full_output:
        #         y2_upsampled = y2
        #     else:
        #         y2_upsampled = nn.functional.interpolate(y2, scale_factor=4) # TBD: change scale factor depending on image resolution
        #     y_output_stage3 = self.stage3(torch.cat([x, y2_upsampled], dim=1), joints=joints)
        #     y3 = y_output_stage3[0] # Just keeping belief maps for now

        #     if self.num_stages == 3:
        #         return [y_0_1, y2, y3]

        # if self.num_stages > 3:
        #     # Upsample
        #     if self.deconv_decoder or self.full_output:
        #         y3_upsampled = y3
        #     else:
        #         y3_upsampled = nn.functional.interpolate(y3, scale_factor=4) # TBD: change scale factor depending on image resolution
        #     y_output_stage4 = self.stage4(torch.cat([x, y3_upsampled], dim=1), joints=joints)
        #     y4 = y_output_stage4[0] # Just keeping belief maps for now

        #     if self.num_stages == 4:
        #         return [y_0_1, y2, y3, y4]

        # if self.num_stages > 4:
        #     # Upsample
        #     if self.deconv_decoder or self.full_output:
        #         y4_upsampled = y4
        #     else:
        #         y4_upsampled = nn.functional.interpolate(y4, scale_factor=4) # TBD: change scale factor depending on image resolution
        #     y_output_stage5 = self.stage5(torch.cat([x, y4_upsampled], dim=1), joints=joints)
        #     y5 = y_output_stage5[0] # Just keeping belief maps for now

        #     if self.num_stages == 5:
        #         return [y_0_1, y2, y3, y4, y5]

        # if self.num_stages > 5:
        #     # Upsample
        #     if self.deconv_decoder or self.full_output:
        #         y5_upsampled = y5
        #     else:
        #         y5_upsampled = nn.functional.interpolate(y5, scale_factor=4) # TBD: change scale factor depending on image resolution
        #     y_output_stage6 = self.stage6(torch.cat([x, y5_upsampled], dim=1), joints=joints)
        #     y6 = y_output_stage6[0] # Just keeping belief maps for now

        #     if self.num_stages == 6:
        #         return [y_0_1, y2, y3, y4, y5, y6]


# Based on DopeHourglassBlockSmall, not using skipped connections
class DreamHourglass(nn.Module):
    def __init__(self, n_keypoints,
                       n_image_input_channels = 3,
                       internalize_spatial_softmax = True,
                       learned_beta = True,
                       initial_beta = 1.,
                       joints_input = 0,
                       skip_connections = False,
                       deconv_decoder = False,
                       full_output = False):
        super(DreamHourglass, self).__init__()
        self.n_keypoints = n_keypoints
        self.n_image_input_channels = n_image_input_channels
        self.internalize_spatial_softmax = internalize_spatial_softmax
        self.skip_connections = skip_connections
        self.deconv_decoder = deconv_decoder
        self.full_output = full_output

        if self.internalize_spatial_softmax:
            self.n_output_heads = 2
            self.learned_beta = learned_beta
            self.initial_beta = initial_beta
        else:
            self.n_output_heads = 1
            self.learned_beta = False

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        vgg_t = models.vgg19(pretrained=True).features

        self.down_sample = nn.MaxPool2d(2)

        self.layer_0_1_down = nn.Sequential()
        self.layer_0_1_down.add_module('0', nn.Conv2d(self.n_image_input_channels, 64,
            kernel_size=3, stride=1, padding=1))
        for layer in range(1,4):
            self.layer_0_1_down.add_module(str(layer), vgg_t[layer])

        self.layer_0_2_down = nn.Sequential()
        for layer in range(5,9):
            self.layer_0_2_down.add_module(str(layer), vgg_t[layer])

        self.layer_0_3_down = nn.Sequential()
        for layer in range(10,18):
            self.layer_0_3_down.add_module(str(layer), vgg_t[layer])

        self.layer_0_4_down = nn.Sequential()
        for layer in range(19,27):
            self.layer_0_4_down.add_module(str(layer), vgg_t[layer])

        self.layer_0_5_down = nn.Sequential()
        for layer in range(28,36):
            self.layer_0_5_down.add_module(str(layer), vgg_t[layer])

        #Head 1 
        if self.deconv_decoder:
            # Decoder primarily uses ConvTranspose2d
            self.deconv_0_4 = nn.Sequential()
            deconv_input = 513 if joints_input > 0 else 512
            self.deconv_0_4.add_module('0', nn.ConvTranspose2d(deconv_input, 256,
                kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=1))
            self.deconv_0_4.add_module('1', nn.ReLU(inplace=True))
            self.deconv_0_4.add_module('2', nn.Conv2d(256, 256,
                kernel_size=3, stride=1, padding=1))
            self.deconv_0_4.add_module('3', nn.ReLU(inplace=True))

            self.deconv_0_3 = nn.Sequential()
            self.deconv_0_3.add_module('0', nn.ConvTranspose2d(256, 128,
                kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=1))
            self.deconv_0_3.add_module('1', nn.ReLU(inplace=True))
            self.deconv_0_3.add_module('2', nn.Conv2d(128, 128,
                kernel_size=3,stride=1,padding=1))
            self.deconv_0_3.add_module('3', nn.ReLU(inplace=True))

            self.deconv_0_2 = nn.Sequential()
            self.deconv_0_2.add_module('0', nn.ConvTranspose2d(128, 64,
                kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=1))
            self.deconv_0_2.add_module('1', nn.ReLU(inplace=True))
            self.deconv_0_2.add_module('2', nn.Conv2d(64, 64,
                kernel_size=3,stride=1,padding=1))
            self.deconv_0_2.add_module('3', nn.ReLU(inplace=True))

            self.deconv_0_1 = nn.Sequential()
            self.deconv_0_1.add_module('0', nn.ConvTranspose2d(64, 64,
                kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=1))
            self.deconv_0_1.add_module('1', nn.ReLU(inplace=True))

        else:
            # Decoder primarily uses Upsampling - for keypoints 
            self.upsample_0_4 = nn.Sequential()
            self.upsample_0_4.add_module('0', nn.Upsample(scale_factor=2))

            # should this go before the upsample?
            upsample_input = 513 if joints_input > 0 else 512
            self.upsample_0_4.add_module('4', nn.Conv2d(upsample_input, 256,
                kernel_size=3, stride=1, padding=1))
            self.upsample_0_4.add_module('5', nn.ReLU(inplace=True))
            self.upsample_0_4.add_module('6', nn.Conv2d(256, 256,
                kernel_size=3, stride=1, padding=1))

            self.upsample_0_3 = nn.Sequential()
            self.upsample_0_3.add_module('0', nn.Upsample(scale_factor=2))
            self.upsample_0_3.add_module('4', nn.Conv2d(256, 128,
                kernel_size=3, stride=1, padding=1))
            self.upsample_0_3.add_module('5', nn.ReLU(inplace=True))
            self.upsample_0_3.add_module('6', nn.Conv2d(128, 64,
                kernel_size=3, stride=1, padding=1))

            if self.full_output: 
                self.upsample_0_2 = nn.Sequential()
                self.upsample_0_2.add_module('0', nn.Upsample(scale_factor=2))
                self.upsample_0_2.add_module('2', nn.Conv2d(64, 64,
                    kernel_size=3,stride=1,padding=1))
                self.upsample_0_2.add_module('3', nn.ReLU(inplace=True))
                self.upsample_0_2.add_module('4', nn.Conv2d(64, 64,
                    kernel_size=3,stride=1,padding=1))
                self.upsample_0_2.add_module('5', nn.ReLU(inplace=True))


                self.upsample_0_1 = nn.Sequential()
                self.upsample_0_1.add_module('00', nn.Upsample(scale_factor=2))
                self.upsample_0_1.add_module('2', nn.Conv2d(64, 64,
                    kernel_size=3,stride=1,padding=1))
                self.upsample_0_1.add_module('3', nn.ReLU(inplace=True))
                self.upsample_0_1.add_module('4', nn.Conv2d(64, 64,
                    kernel_size=3,stride=1,padding=1))
                self.upsample_0_1.add_module('5', nn.ReLU(inplace=True))

            # Decoder primarily uses Upsampling - for affinities
            self.upsample_1_4 = nn.Sequential()
            self.upsample_1_4.add_module('0', nn.Upsample(scale_factor=2))

            # should this go before the upsample?
            upsample_input = 513 if joints_input > 0 else 512
            self.upsample_1_4.add_module('4', nn.Conv2d(upsample_input, 256,
                kernel_size=3, stride=1, padding=1))
            self.upsample_1_4.add_module('5', nn.ReLU(inplace=True))
            self.upsample_1_4.add_module('6', nn.Conv2d(256, 256,
                kernel_size=3, stride=1, padding=1))

            self.upsample_1_3 = nn.Sequential()
            self.upsample_1_3.add_module('0', nn.Upsample(scale_factor=2))
            self.upsample_1_3.add_module('4', nn.Conv2d(256, 128,
                kernel_size=3, stride=1, padding=1))
            self.upsample_1_3.add_module('5', nn.ReLU(inplace=True))
            self.upsample_1_3.add_module('6', nn.Conv2d(128, 64,
                kernel_size=3, stride=1, padding=1))

            if self.full_output: 
                self.upsample_1_2 = nn.Sequential()
                self.upsample_1_2.add_module('0', nn.Upsample(scale_factor=2))
                self.upsample_1_2.add_module('2', nn.Conv2d(64, 64,
                    kernel_size=3,stride=1,padding=1))
                self.upsample_1_2.add_module('3', nn.ReLU(inplace=True))
                self.upsample_1_2.add_module('4', nn.Conv2d(64, 64,
                    kernel_size=3,stride=1,padding=1))
                self.upsample_1_2.add_module('5', nn.ReLU(inplace=True))


                self.upsample_1_1 = nn.Sequential()
                self.upsample_1_1.add_module('00', nn.Upsample(scale_factor=2))
                self.upsample_1_1.add_module('2', nn.Conv2d(64, 64,
                    kernel_size=3,stride=1,padding=1))
                self.upsample_1_1.add_module('3', nn.ReLU(inplace=True))
                self.upsample_1_1.add_module('4', nn.Conv2d(64, 64,
                    kernel_size=3,stride=1,padding=1))
                self.upsample_1_1.add_module('5', nn.ReLU(inplace=True))


        # Output head - goes from [batch x 64 x height x width] -> [batch x n_keypoints x height x width]
        self.heads_0 = nn.Sequential()
        self.heads_0.add_module('0', nn.Conv2d(64, 64,
            kernel_size=3, stride=1, padding=1))
        self.heads_0.add_module('1', nn.ReLU(inplace=True))
        self.heads_0.add_module('2', nn.Conv2d(64, 32,
            kernel_size=3, stride=1, padding=1))
        self.heads_0.add_module('3', nn.ReLU(inplace=True))
        self.heads_0.add_module('4', nn.Conv2d(32, self.n_keypoints, 
            kernel_size=3, stride=1, padding=1))

        self.heads_1 = nn.Sequential()
        self.heads_1.add_module('0', nn.Conv2d(64, 64,
            kernel_size=3, stride=1, padding=1))
        self.heads_1.add_module('1', nn.ReLU(inplace=True))
        self.heads_1.add_module('2', nn.Conv2d(64, 32,
            kernel_size=3, stride=1, padding=1))
        self.heads_1.add_module('3', nn.ReLU(inplace=True))
        self.heads_1.add_module('4', nn.Conv2d(32, (self.n_keypoints-1)*2, 
            kernel_size=3, stride=1, padding=1))


    def forward(self, x, joints=None):

        # Encoder
        x_0_1   = self.layer_0_1_down(x)
        x_0_1_d = self.down_sample(x_0_1)
        x_0_2   = self.layer_0_2_down(x_0_1_d)
        x_0_2_d = self.down_sample(x_0_2)
        x_0_3   = self.layer_0_3_down(x_0_2_d)
        x_0_3_d = self.down_sample(x_0_3)
        x_0_4   = self.layer_0_4_down(x_0_3_d)
        x_0_4_d = self.down_sample(x_0_4)
        x_0_5   = self.layer_0_5_down(x_0_4_d)

        # Append joints to latent space if provided
        if joints is not None:
            joint_output = self.joint_head(joints)
            if self.skip_connections:
                decoder_input = torch.cat([x_0_5 + x_0_4_d, joint_output.reshape(joint_output.shape[0],1,25,25)], dim=1)
            else:
                decoder_input = torch.cat([x_0_5, joint_output.reshape(joint_output.shape[0],1,25,25)], dim=1)
        else:
            if self.skip_connections:
                decoder_input = x_0_5 + x_0_4_d
            else:
                decoder_input = x_0_5

        # Decoder
        if self.deconv_decoder:
            y_0_5 = self.deconv_0_4(decoder_input)

            if self.skip_connections:
                y_0_4 = self.deconv_0_3(y_0_5 + x_0_3_d)
            else:
                y_0_4 = self.deconv_0_3(y_0_5)

            if self.skip_connections:
                y_0_3 = self.deconv_0_2(y_0_4 + x_0_2_d)
            else:
                y_0_3 = self.deconv_0_2(y_0_4)

            if self.skip_connections:
                y_0_out = self.deconv_0_1(y_0_3 + x_0_1_d)
            else:
                y_0_out = self.deconv_0_1(y_0_3)

            if self.skip_connections:
                output_head_0 = self.heads_0(y_0_out + x_0_1)
            else:
                output_head_0 = self.heads_0(y_0_out)

        else:
            y_0_5 = self.upsample_0_4(decoder_input)

            if self.skip_connections:
                y_0_out = self.upsample_0_3(y_0_5 + x_0_3_d)
            else:
                y_0_out = self.upsample_0_3(y_0_5)

            if self.full_output:
                y_0_out = self.upsample_0_2(y_0_out)
                y_0_out = self.upsample_0_1(y_0_out)

            output_head_0 = self.heads_0(y_0_out)

            # SECOND HEAD
            y_1_5 = self.upsample_1_4(decoder_input)

            if self.skip_connections:
                y_1_out = self.upsample_1_3(y_1_5 + x_1_3_d)
            else:
                y_1_out = self.upsample_1_3(y_1_5)

            if self.full_output:
                y_1_out = self.upsample_1_2(y_1_out)
                y_1_out = self.upsample_1_1(y_1_out)

            output_head_1 = self.heads_1(y_1_out)

        # Output heads
        outputs = []
        outputs.append(output_head_0)

        # Return outputs
        return output_head_0,output_head_1


if __name__ == '__main__':
    import torch
    # n_keypoints = 7
    # n_joints = 10
    batch_size = 2

    # print('ResnetSimple')
    # net = ResnetSimple().cuda()
    # y = net(torch.zeros(batch_size, 3, 400, 400).cuda())
    # print(y[0][-1].shape)
    # print()
    # del net, y
    a = torch.sum(torch.zeros(2,9,50,50),dim=1)
    print(a.shape)
    # net = BoundaryAwareNet().cuda()
    net = DopeMobileNet().cuda()
    y = net(torch.zeros(batch_size, 3, 400, 400).cuda())

    # print(y.shape)