"""
NVIDIA from jtremblay@gmail.com
"""

# Networks
import torch
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torchvision.models as models


class DopeNetwork(nn.Module):
    def __init__(
        self,
        pretrained=False,
        numBeliefMap=9,
        numAffinity=16,
        stop_at_stage=6,  # number of stages to process (if less than total number of stages)
    ):
        super(DopeNetwork, self).__init__()

        self.stop_at_stage = stop_at_stage

        vgg_full = models.vgg19(pretrained=False).features
        self.vgg = nn.Sequential()
        for i_layer in range(24):
            self.vgg.add_module(str(i_layer), vgg_full[i_layer])

        # Add some layers
        i_layer = 23
        self.vgg.add_module(
            str(i_layer), nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        )
        self.vgg.add_module(str(i_layer + 1), nn.ReLU(inplace=True))
        self.vgg.add_module(
            str(i_layer + 2), nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        )
        self.vgg.add_module(str(i_layer + 3), nn.ReLU(inplace=True))

        # print('---Belief------------------------------------------------')
        # _2 are the belief map stages
        self.m1_2 = DopeNetwork.create_stage(128, numBeliefMap, True)
        self.m2_2 = DopeNetwork.create_stage(
            128 + numBeliefMap + numAffinity, numBeliefMap, False
        )
        self.m3_2 = DopeNetwork.create_stage(
            128 + numBeliefMap + numAffinity, numBeliefMap, False
        )
        self.m4_2 = DopeNetwork.create_stage(
            128 + numBeliefMap + numAffinity, numBeliefMap, False
        )
        self.m5_2 = DopeNetwork.create_stage(
            128 + numBeliefMap + numAffinity, numBeliefMap, False
        )
        self.m6_2 = DopeNetwork.create_stage(
            128 + numBeliefMap + numAffinity, numBeliefMap, False
        )

        # print('---Affinity----------------------------------------------')
        # _1 are the affinity map stages
        self.m1_1 = DopeNetwork.create_stage(128, numAffinity, True)
        self.m2_1 = DopeNetwork.create_stage(
            128 + numBeliefMap + numAffinity, numAffinity, False
        )
        self.m3_1 = DopeNetwork.create_stage(
            128 + numBeliefMap + numAffinity, numAffinity, False
        )
        self.m4_1 = DopeNetwork.create_stage(
            128 + numBeliefMap + numAffinity, numAffinity, False
        )
        self.m5_1 = DopeNetwork.create_stage(
            128 + numBeliefMap + numAffinity, numAffinity, False
        )
        self.m6_1 = DopeNetwork.create_stage(
            128 + numBeliefMap + numAffinity, numAffinity, False
        )

    def forward(self, x):
        """Runs inference on the neural network"""

        out1 = self.vgg(x)

        out1_2 = self.m1_2(out1)
        out1_1 = self.m1_1(out1)

        if self.stop_at_stage == 1:
            return [out1_2], [out1_1]

        out2 = torch.cat([out1_2, out1_1, out1], 1)
        out2_2 = self.m2_2(out2)
        out2_1 = self.m2_1(out2)

        if self.stop_at_stage == 2:
            return [out1_2, out2_2], [out1_1, out2_1]

        out3 = torch.cat([out2_2, out2_1, out1], 1)
        out3_2 = self.m3_2(out3)
        out3_1 = self.m3_1(out3)

        if self.stop_at_stage == 3:
            return [out1_2, out2_2, out3_2], [out1_1, out2_1, out3_1]

        out4 = torch.cat([out3_2, out3_1, out1], 1)
        out4_2 = self.m4_2(out4)
        out4_1 = self.m4_1(out4)

        if self.stop_at_stage == 4:
            return [out1_2, out2_2, out3_2, out4_2], [out1_1, out2_1, out3_1, out4_1]

        out5 = torch.cat([out4_2, out4_1, out1], 1)
        out5_2 = self.m5_2(out5)
        out5_1 = self.m5_1(out5)

        if self.stop_at_stage == 5:
            return [out1_2, out2_2, out3_2, out4_2, out5_2], [
                out1_1,
                out2_1,
                out3_1,
                out4_1,
                out5_1,
            ]

        out6 = torch.cat([out5_2, out5_1, out1], 1)
        out6_2 = self.m6_2(out6)
        out6_1 = self.m6_1(out6)

        return [out1_2, out2_2, out3_2, out4_2, out5_2, out6_2], [
            out1_1,
            out2_1,
            out3_1,
            out4_1,
            out5_1,
            out6_1,
        ]

    @staticmethod
    def create_stage(in_channels, out_channels, first=False):
        """Create the neural network layers for a single stage."""

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
        model.add_module(
            "0",
            nn.Conv2d(
                in_channels, mid_channels, kernel_size=kernel, stride=1, padding=padding
            ),
        )

        # Middle convolutions
        i = 1
        while i < count - 1:
            model.add_module(str(i), nn.ReLU(inplace=True))
            i += 1
            model.add_module(
                str(i),
                nn.Conv2d(
                    mid_channels,
                    mid_channels,
                    kernel_size=kernel,
                    stride=1,
                    padding=padding,
                ),
            )
            i += 1

        # Penultimate convolution
        model.add_module(str(i), nn.ReLU(inplace=True))
        i += 1
        model.add_module(
            str(i), nn.Conv2d(mid_channels, final_channels, kernel_size=1, stride=1)
        )
        i += 1

        # Last convolution
        model.add_module(str(i), nn.ReLU(inplace=True))
        i += 1
        model.add_module(
            str(i), nn.Conv2d(final_channels, out_channels, kernel_size=1, stride=1)
        )
        i += 1

        return model
