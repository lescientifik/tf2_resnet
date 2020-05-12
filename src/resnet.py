import tensorflow as tf
from tensorflow.keras import layers, Sequential, Model


def conv3x3(out_planes, strides=1, dilation=1):
    """3x3 convolution with padding"""
    return layers.Conv2D(filters=out_planes, kernel_size=3, strides=strides,
                         padding="same", use_bias=False, dilation_rate=dilation, kernel_initializer='he_normal')


def conv1x1(out_planes, strides=1, dilation=1):
    """1x1 convolution"""
    return layers.Conv2D(filters=out_planes, kernel_size=3, strides=strides,
                         padding="same", use_bias=False, dilation_rate=dilation, kernel_initializer='he_normal')


def basic_block(planes, strides, downsample=None, base_width=64, dilation=1, norm=None):
    def basic_block_in(x):
        if norm is None:
            norm_layer = layers.BatchNormalization
        if base_width != 64:
            raise ValueError('BasicBlock only supports base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        identity = x
        residual = layers.ReLU()(norm_layer()(conv3x3(planes, strides)(x)))
        out = norm_layer()(conv3x3(planes)(residual))
        if downsample is not None:
            identity = downsample(x)

        out += identity
        out = layers.ReLU()(out)

    return basic_block_in


class BasicBlock(layers.Layer):
    expansion = 1

    def __init__(self, planes, strides=1, downsample=None, base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__(name="BasicBlock")
        if norm_layer is None:
            norm_layer = layers.BatchNormalization
        if base_width != 64:
            raise ValueError('BasicBlock only supports base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(planes, strides)
        self.bn1 = norm_layer()
        self.relu = tf.nn.relu
        self.conv2 = conv3x3(planes)
        self.bn2 = norm_layer()
        self.downsample = downsample
        self.stride = strides

    def call(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out




class Bottleneck(layers.Layer):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, planes, strides=1, downsample=None, base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__(name="Bottleneck")
        if norm_layer is None:
            norm_layer = layers.BatchNormalization
        width = int(planes * (base_width / 64.))
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(width)
        self.bn1 = norm_layer()
        self.conv2 = conv3x3(width, strides, dilation)
        self.bn2 = norm_layer()
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer()
        self.relu = tf.nn.relu
        self.downsample = downsample
        self.stride = strides

    def call(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(Model):

    def __init__(self, block, _layers, num_classes=1000, zero_init_residual=False, width=64,
                 replace_stride_with_dilation=None, norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = layers.BatchNormalization
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.base_width = width
        self.pad = layers.ZeroPadding2D(3)
        self.conv1 = layers.Conv2D(self.inplanes, kernel_size=3, strides=2,
                                   padding="valid", use_bias=False, kernel_initializer='he_normal')
        self.bn1 = norm_layer()
        self.relu = tf.nn.relu
        self.maxpool = layers.MaxPooling2D(pool_size=3, strides=2, padding="same")
        self.layer1 = self._make_layer(block, 64, _layers[0])
        self.layer2 = self._make_layer(block, 128, _layers[1], dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, _layers[2], dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, _layers[3], dilate=replace_stride_with_dilation[2])
        self.avgpool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(num_classes)

        # Already done before
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

        # TODO port this part
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        # if zero_init_residual:
        #     for m in self.modules():
        #         if isinstance(m, Bottleneck):
        #             nn.init.constant_(m.bn3.weight, 0)
        #         elif isinstance(m, BasicBlock):
        #             nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, strides=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= strides
            strides = 1
        if strides != 1 or self.inplanes != planes * block.expansion:
            downsample = Sequential(layers=[
                conv1x1(planes * block.expansion, strides),
                norm_layer()], name="downsample"
            )

        _layers = [block(planes, strides, downsample,
                         self.base_width, previous_dilation, norm_layer)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            _layers.append(block(planes,
                                 base_width=self.base_width, dilation=self.dilation,
                                 norm_layer=norm_layer))

        return Sequential(layers=_layers, name=str(blocks) + "_layer")

    def call(self, x):
        x = self.pad(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.fc(x)

        return x


def _resnet(arch, block, layers_size, pretrained, progress, **kwargs):
    model = ResNet(block, layers_size, **kwargs)
    # if pretrained:
    #     state_dict = load_state_dict_from_url(model_urls[arch],
    #                                           progress=progress)
    #     model.load_state_dict(state_dict)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


"""

from resnet import resnet18

model = resnet18()
model.build((None,64,64,3))

"""
