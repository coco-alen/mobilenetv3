import torch
from params import args
import torch.nn as nn
import nni

def _make_divisible(v, divisor, min_value=None):
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
    return new_v

def _get_layer_by_name(name):
    if name == "Conv":
        return ConvBNReLU
    elif name == "DwConv":
        return InvertedResidual
    elif name == "None":
        return nn.Identity
    else:
        raise ValueError("Invalid layer name: {}".format(name))

def _get_nni_shape():
    params = nni.get_next_parameter()
    inverted_residual_setting = []
    layer_num = 1
    while f"expand_ratio{layer_num}" in params.keys():
        inverted_residual_setting.append([
            params[f"expand_ratio{layer_num}"],
            params[f"channel{layer_num}"],
            params[f"num{layer_num}"],
            params[f"stride{layer_num}"],
            params[f"type{layer_num}"]
        ])
        layer_num += 1
    return inverted_residual_setting

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, **kwargs):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes, momentum=args.momentum),
            # Replace with ReLU
            nn.ReLU(inplace=False)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup, momentum=args.momentum),
        ])
        self.conv = nn.Sequential(*layers)

        # # Replace torch.add with floatfunctional
        # self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        if self.use_res_connect:
            # return self.skip_add.add(x, self.conv(x))
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0, inverted_residual_setting=None, round_nearest=8):
        """
        MobileNet V2 main class
        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
        """
        super().__init__()
        input_channel = 32
        # input_channel = 16
        last_channel = 1280

        if inverted_residual_setting is None:
            
            if args.nni_NAS:
                inverted_residual_setting = _get_nni_shape()
            else:
                inverted_residual_setting = [
                    # t, c, n, s
                    [1, 16, 1, 1, "DwCov"],
                    [6, 24, 2, 2, "DwCov"],
                    [6, 32, 3, 2, "DwCov"],
                    [6, 64, 4, 2, "DwCov"],
                    [6, 96, 3, 1, "DwCov"],
                    [6, 160, 3, 2, "DwCov"],
                    [6, 320, 1, 1, "DwCov"],
                ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 5:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 5-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)

        features = [ConvBNReLU(3, input_channel, stride=2)]
        # building inverted residual blocks
        for t, c, n, s, name in inverted_residual_setting:

            output_channel = _make_divisible(c * width_mult, round_nearest)
            block = _get_layer_by_name(name)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                if name != "None":
                    input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # self.quant = quant.QuantStub()
        # self.dequant = quant.DeQuantStub()

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(args.drop_path),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.ones_(m.weight)
                torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
            # x = self.quant(x)
        x = self.features(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        #     x = self.dequant(x)
        return x

    # # Fuse Conv+BN and Conv+BN+Relu modules prior to quantization
    # # This operation does not change the numerics
    # def fuse_model(self):
    #     for m in self.modules():
    #         if type(m) == ConvBNReLU:
    #             torch.quantization.fuse_modules(m, ['0', '1', '2'], inplace=True)
    #         if type(m) == InvertedResidual:
    #             for idx in range(len(m.conv)):
    #                 if type(m.conv[idx]) == nn.Conv2d:
    #                     torch.quantization.fuse_modules(m.conv, [str(idx), str(idx + 1)], inplace=True)