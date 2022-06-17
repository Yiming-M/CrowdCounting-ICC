import torch
from torch import nn, Tensor
import torch.nn.functional as F
import warnings
from typing import Callable, Any, Optional, Tuple, List


class Upsample(nn.Module):
    def __init__(self,
                 size=None,
                 scale_factor=None,
                 mode="nearest",
                 align_corners=None,
                 recompute_scale_factor=None):
        super(Upsample, self).__init__()
        self.interpolate = F.interpolate
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        self.recompute_scale_factor = recompute_scale_factor

    def forward(self, input):
        return self.interpolate(input=input,
                                size=self.size,
                                scale_factor=self.scale_factor,
                                mode=self.mode,
                                align_corners=self.align_corners,
                                recompute_scale_factor=self.recompute_scale_factor)


def make_layers(config):
    """
    Building a sequential model according to `config`.

    Param:
    - `config`: a list of tuples `(layer_type, conf_dict)`.
        - `layer_type` specifies the type of the layer, such as `"Conv2d"` & `"MaxPool2d"`.
        - `layer_config` is a dictionary which contains parameters of the layer. For example
          `{"in_channels": 256, "out_channels": 512, "kernel_size": 3}` is a valid dict.
    """
    # Initialization.
    layers = []

    for layer_num, (layer_type, layer_config) in enumerate(config):
        # Check `layer_type`.
        try:
            assert layer_type in ["Conv2d", "MaxPool2d", "BatchNorm2d", "ReLU", "Upsample", "ConvTranspose2d"]
        except AssertionError:
            warnings.warn(f"Wrong type for layer {layer_num}. Layer not added.")
            continue

        # Check `layer_config`.
        if layer_type == "Conv2d":
            try:
                layer = nn.Conv2d(**layer_config)
            except TypeError:
                warnings.warn(f"Wrong configuration for layer {layer_num}. Layer not added.")
                continue
        elif layer_type == "MaxPool2d":
            try:
                layer = nn.MaxPool2d(**layer_config)
            except TypeError:
                warnings.warn(f"Wrong configuration for layer {layer_num}. Layer not added.")
                continue
        elif layer_type == "BatchNorm2d":
            try:
                layer = nn.BatchNorm2d(**layer_config)
            except TypeError:
                warnings.warn(f"Wrong configuration for layer {layer_num}. Layer not added.")
                continue
        elif layer_type == "ReLU":
            try:
                layer = nn.ReLU(**layer_config)
            except TypeError:
                warnings.warn(f"Wrong configuration for layer {layer_num}. Layer not added.")
                continue
        elif layer_type == "Upsample":
            try:
                layer = Upsample(**layer_config)
            except TypeError:
                warnings.warn(f"Wrong configuration for layer {layer_num}. Layer not added.")
                continue
        else:
            try:
                layer = nn.ConvTranspose2d(**layer_config)
            except TypeError:
                warnings.warn(f"Wrong configuration for layer {layer_num}. Layer not added.")
                continue

        # Add the layer.
        layers.append(layer)

    return nn.Sequential(*layers)


class BasicConv2d(nn.Module):
    """
    Implement Conv2d -> BatchNorm2d -> ReLU.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=False,
                 eps=1e-3,
                 **kwargs: Any):
        """
        Params:
        - in_channels: for nn.Conv2d.
        - out_channels: for nn.Conv2d, which is equal to num_features in nn.BatchNorm2d.
        - bias: for nn.Conv2d.
        - eps: for nn.BatchNorm2d.
        - kwargs: other parameters for nn.Conv2d.
        """
        super(BasicConv2d, self).__init__()

        conv2d = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "bias": bias,
        }
        conv2d.update(kwargs)

        batchnorm2d = {
            "num_features": out_channels,
            "eps": eps,
        }

        self.basicconv2d = make_layers([
            ("Conv2d", conv2d),
            ("BatchNorm2d", batchnorm2d),
            ("ReLU", {"inplace": True})
        ])

    def forward(self, X):
        return self.basicconv2d(X)


class InceptionA(nn.Module):
    """
    Code from PyTorch
    (https://pytorch.org/vision/stable/_modules/torchvision/models/inception.html#inception_v3).

    Structure of Inception-A:
          ┌- conv1-`out_channels1` --------------------------------------------------------> ┐
          |                                                                                  |
          |- conv1-`interm_channels1` -> conv5-`out_channels2` ----------------------------> |
    Base -|                                                                                  |- Concat
          |- conv1-`interm_chennels2` -> conv3-`out_channels3` -> conv3-`out_channels3` ---> |
          |                                                                                  |
          └- avgpool-3 -> conv1-`pool_features` -------------------------------------------> ┘
    """

    def __init__(
        self,
        in_channels: int,
        pool_features: int,
        interm_channels1: int = 48,
        interm_channels2: int = 64,
        out_channels1: int = 64,
        out_channels2: int = 64,
        out_channels3: int = 96,
        conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:

        super(InceptionA, self).__init__()

        if conv_block is None:
            conv_block = BasicConv2d

        self.branch1x1 = conv_block(in_channels, out_channels1, kernel_size=1)

        self.branch5x5_1 = conv_block(in_channels, interm_channels1, kernel_size=1)
        self.branch5x5_2 = conv_block(interm_channels1, out_channels2, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = conv_block(in_channels, interm_channels2, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(interm_channels2, out_channels3, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = conv_block(out_channels3, out_channels3, kernel_size=3, padding=1)

        self.branch_pool = conv_block(in_channels, pool_features, kernel_size=1)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]

        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)

        return torch.cat(outputs, 1)


class InceptionB(nn.Module):
    """
    Code from PyTorch
    (https://pytorch.org/vision/stable/_modules/torchvision/models/inception.html#inception_v3).

    Structure of Inception-B:
          ┌- conv3-`out_channels1` --------------------------------------------------------> ┐
          |                                                                                  |
    Base -|- conv1-`interm_channels1` -> conv3-`out_channels2` -> conv3-`out_channels2` ---> |- Concat
          |                                                                                  |
          └- maxpool-3 --------------------------------------------------------------------> ┘
    """

    def __init__(
        self,
        in_channels: int,
        interm_channels1: int = 64,
        out_channels1: int = 384,
        out_channels2: int = 96,
        conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:

        super(InceptionB, self).__init__()

        if conv_block is None:
            conv_block = BasicConv2d

        self.branch3x3 = conv_block(in_channels, out_channels1, kernel_size=3, stride=2, padding=1)

        self.branch3x3dbl_1 = conv_block(in_channels, interm_channels1, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(interm_channels1, out_channels2, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = conv_block(out_channels2, out_channels2, kernel_size=3, stride=2, padding=1)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)

        outputs = [branch3x3, branch3x3dbl, branch_pool]

        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)

        return torch.cat(outputs, 1)


class InceptionC(nn.Module):
    """
    Code from PyTorch
    (https://pytorch.org/vision/stable/_modules/torchvision/models/inception.html#inception_v3).

    Structure of Inception-C:
          ┌- conv1-`out_channels1` -----------------------------------------------------------------------------------------------------------------------> ┐
          |                                                                                                                                                 |
          |- conv1-`channels_7x7` -> conv(1, 7)-`channels_7x7` -> conv(7, 1)-`out_channels2` -------------------------------------------------------------> |
    Base -|                                                                                                                                                 |- Concat
          |- conv1-`channels_7x7` -> conv(7, 1)-`channels_7x7` -> conv(1, 7)-`channels_7x7` -> conv(7, 1)-`channels_7x7` -> conv(1, 7)-`out_channels3` ---> |
          |                                                                                                                                                 |
          └- avgpool-3 -> conv1-`out_channels4` ----------------------------------------------------------------------------------------------------------> ┘
    """

    def __init__(
        self,
        in_channels: int,
        channels_7x7: int,
        out_channels1: int = 192,
        out_channels2: int = 192,
        out_channels3: int = 192,
        out_channels4: int = 192,
        conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:

        super(InceptionC, self).__init__()

        if conv_block is None:
            conv_block = BasicConv2d

        self.branch1x1 = conv_block(in_channels, out_channels1, kernel_size=1)

        c7 = channels_7x7
        self.branch7x7_1 = conv_block(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = conv_block(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = conv_block(c7, out_channels2, kernel_size=(7, 1), padding=(3, 0))

        self.branch7x7dbl_1 = conv_block(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = conv_block(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = conv_block(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = conv_block(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = conv_block(c7, out_channels3, kernel_size=(1, 7), padding=(0, 3))

        self.branch_pool = conv_block(in_channels, out_channels4, kernel_size=1)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]

        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)

        return torch.cat(outputs, 1)


class InceptionD(nn.Module):
    """
    Code from PyTorch
    (https://pytorch.org/vision/stable/_modules/torchvision/models/inception.html#inception_v3).

    Structure of Inception-D:
          ┌- conv1-`interm_channels1` -> conv3-`out_channels1`  ------------------------------------------------------------------> ┐
          |                                                                                                                         |
    Base -|- conv1-`interm_channels2` -> conv(1, 7)-`interm_channels2` -> conv(7, 1)-`interm_channels2` -> conv3-`out_channels2` -> |- Concat
          |                                                                                                                         |
          └- maxpool-3 -----------------------------------------------------------------------------------------------------------> ┘
    """

    def __init__(
        self,
        in_channels: int,
        interm_channels1: int = 192,
        interm_channels2: int = 192,
        out_channels1: int = 320,
        out_channels2: int = 192,
        conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:

        super(InceptionD, self).__init__()

        if conv_block is None:
            conv_block = BasicConv2d

        self.branch3x3_1 = conv_block(in_channels, interm_channels1, kernel_size=1)
        self.branch3x3_2 = conv_block(interm_channels1, out_channels1, kernel_size=3, stride=2, padding=1)

        self.branch7x7x3_1 = conv_block(in_channels, interm_channels2, kernel_size=1)
        self.branch7x7x3_2 = conv_block(interm_channels2, interm_channels2, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7x3_3 = conv_block(interm_channels2, interm_channels2, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7x3_4 = conv_block(interm_channels2, out_channels2, kernel_size=3, stride=2, padding=1)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)

        outputs = [branch3x3, branch7x7x3, branch_pool]

        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionE(nn.Module):
    """
    Code from PyTorch
    (https://pytorch.org/vision/stable/_modules/torchvision/models/inception.html#inception_v3).

    Structure of Inception-D:
          ┌- conv1-`out_channels1`  -------------------------------------------------------------------------> ┐
          |                                                                                                    |
          |                            ┌- conv(1, 3)-`out_channels2` -> ┐                                      |
          |- conv1-`interm_channels1` -|                                |- Concat ---------------------------> |
          |                            └- conv(3, 1)-`out_channels2` -> ┘                                      |
    Base -|                                                                                                    |- Concat
          |                                                     ┌- conv(1, 3)-`out_channels3` -> ┐             |
          |- conv1-`interm_channels2` -> conv3-`out_channels3` -|                                |- Concat --> |
          |                                                     └- conv(3, 1)-`out_channels3` -> ┘             |
          |                                                                                                    |
          └- avgpool-3 -> conv1-`out_channels4`--------------------------------------------------------------> ┘
    """

    def __init__(
        self,
        in_channels: int,
        interm_channels1: int = 384,
        interm_channels2: int = 448,
        out_channels1: int = 320,
        out_channels2: int = 384,
        out_channels3: int = 384,
        out_channels4: int = 192,
        conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:

        super(InceptionE, self).__init__()

        if conv_block is None:
            conv_block = BasicConv2d

        self.branch1x1 = conv_block(in_channels, out_channels1, kernel_size=1)

        self.branch3x3_1 = conv_block(in_channels, interm_channels1, kernel_size=1)
        self.branch3x3_2a = conv_block(interm_channels1, out_channels2, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = conv_block(interm_channels1, out_channels2, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3dbl_1 = conv_block(in_channels, interm_channels2, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(interm_channels2, out_channels3, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = conv_block(out_channels3, out_channels3, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = conv_block(out_channels3, out_channels3, kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = conv_block(in_channels, out_channels4, kernel_size=1)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]

        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)

        return torch.cat(outputs, 1)
