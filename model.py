import torch
from torch import nn, Tensor
import torch.nn.functional as F

from typing import Callable, Optional, Tuple, List

from utils import BasicConv2d, InceptionA, InceptionB, InceptionC, make_layers


class MyInception(nn.Module):
    """
    Modified version of Inception V3.
    """

    def __init__(
        self,
        inception_blocks: Optional[List[Callable[..., nn.Module]]] = None,
    ) -> None:
        super(MyInception, self).__init__()

        if inception_blocks is None:
            inception_blocks = [
                BasicConv2d,
                InceptionA,
                InceptionB,
                InceptionC,
            ]

        conv_block = inception_blocks[0]
        inception_a = inception_blocks[1]
        inception_b = inception_blocks[2]
        inception_c = inception_blocks[3]

        self.Conv2d_1a_3x3 = conv_block(3, 32, kernel_size=3, stride=2, padding=1)
        self.Conv2d_2a_3x3 = conv_block(32, 32, kernel_size=3, padding=1)
        self.Conv2d_2b_3x3 = conv_block(32, 64, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.Conv2d_3b_1x1 = conv_block(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = conv_block(80, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.Mixed_5b = inception_a(192, pool_features=32)
        self.Mixed_5c = inception_a(256, pool_features=64)
        self.Mixed_5d = inception_a(288, pool_features=64)

        self.Mixed_6a = inception_b(288)
        self.Mixed_6b = inception_c(768, channels_7x7=128)

    def _forward(self, x: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        # N x 3 x 384 x 288
        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 192 x 144
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 192 x 144
        x = self.Conv2d_2b_3x3(x)
        # N x 64 x 192 x 144
        x = self.maxpool1(x)
        # N x 64 x 96 x 72

        # N x 64 x 96 x 72
        x = self.Conv2d_3b_1x1(x)
        # N x 80 x 96 x 72
        x = self.Conv2d_4a_3x3(x)
        # N x 192 x 96 x 72
        x = self.maxpool2(x)
        # N x 192 x 48 x 36
        feature1 = torch.clone(x)
        # N x 192 x 48 x 36

        # N x 192 x 48 x 36
        x = self.Mixed_5b(x)
        # N x 256 x 48 x 36
        x = self.Mixed_5c(x)
        # N x 288 x 48 x 36
        x = self.Mixed_5d(x)
        # N x 288 x 48 x 36
        feature2 = torch.clone(x)
        # N x 288 x 48 x 36

        # N x 288 x 48 x 36
        x = self.Mixed_6a(x)
        # N x 768 x 24 x 18
        feature3 = self.Mixed_6b(x)

        feature3 = F.interpolate(feature3, size=feature2.size()[-2:], mode="bilinear", align_corners=True)

        return feature1, feature2, feature3

    def forward(self, img):
        return self._forward(img)


class ContextualModule(nn.Module):
    """
    Modified contextual module from Context-Aware Crowd Counting
    (https://github.com/weizheliu/Context-Aware-Crowd-Counting).
    """
    def __init__(self, in_features, out_features=512, sizes=(1, 2, 3, 6), eps=1e-6):
        super(ContextualModule, self).__init__()
        self.scales = nn.ModuleList([self._make_scale(in_features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(in_features * 2, out_features, kernel_size=1)
        self.relu = nn.ReLU()
        self.weight_net = nn.Conv2d(in_features, in_features, kernel_size=1)
        self.eps = eps

    def __make_weight(self, features, scale_features):
        # Calculate contrast features.
        contrast_features = features - scale_features

        # Calculate weights: contrast_features -> 1x1 conv -> sigmoid.
        weights = torch.sigmoid(self.weight_net(contrast_features))

        return weights

    def _make_scale(self, features, size):
        # Add an average-pooling layer.
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))

        # Add a convolutional layer.
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)

        return nn.Sequential(prior, conv)

    def forward(self, features):
        # Remember the input has 4 dimensions.
        h, w = features.shape[2], features.shape[3]

        # Calculate scale-aware features.
        multi_scales = [
            F.interpolate(
                input=stage(features),
                size=(h, w),
                mode='bilinear',
                align_corners=True
            ) for stage in self.scales
        ]

        # Calculate weights.
        weights = [self.__make_weight(features, scale_features) for scale_features in multi_scales]

        # Weight-average multi-scale features.
        weighted_features = sum(multi_scales[i] * weights[i] for i in range(len(self.scales)))
        # Add `eps` to the denominator in case of division by 0.
        weighted_features = weighted_features / (sum(weights) + self.eps)

        # Concatenation.
        overall_features = torch.cat([weighted_features, features], dim=1)

        bottle = self.bottleneck(overall_features)

        return self.relu(bottle)


class ICC(nn.Module):
    """
    MyInception -> ContextualModule -> Decoder
    """
    def __init__(self):
        super(ICC, self).__init__()
        self.encoder = MyInception()

        self.context = ContextualModule(in_features=192, out_features=192)

        decoder_cfg = [
            ("Conv2d", {"in_channels": 1248, "out_channels": 256, "kernel_size": 1}),
            ("ReLU", {"inplace": True}),
            ("Conv2d", {"in_channels": 256, "out_channels": 256, "kernel_size": 3, "padding": 2, "dilation": 2}),
            ("ReLU", {"inplace": True}),
            ("Conv2d", {"in_channels": 256, "out_channels": 128, "kernel_size": 1}),
            ("ReLU", {"inplace": True}),
            ("Conv2d", {"in_channels": 128, "out_channels": 128, "kernel_size": 3, "padding": 2, "dilation": 2}),
            ("ReLU", {"inplace": True}),
            ("Conv2d", {"in_channels": 128, "out_channels": 64, "kernel_size": 1}),
            ("ReLU", {"inplace": True}),
            ("Conv2d", {"in_channels": 64, "out_channels": 64, "kernel_size": 3, "padding": 2, "dilation": 2}),
            ("ReLU", {"inplace": True}),
            ("Conv2d", {"in_channels": 64, "out_channels": 1, "kernel_size": 1}),
            ("ReLU", {"inplace": True}),
        ]

        self.decoder = make_layers(decoder_cfg)

        self._initialize_weights()

    def _initialize_weights(self):

        for layer in self.modules():

            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, std=0.01)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

            elif isinstance(layer, nn.ConvTranspose2d):
                nn.init.normal_(layer.weight, std=0.01)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)

    def forward(self, img):
        feature1, feature2, feature3 = self.encoder(img)

        feature1 = self.context(feature1)

        contextual_features = torch.cat([
            feature1,
            feature2,
            feature3
        ], dim=1)

        density_map = self.decoder(contextual_features)

        return density_map
