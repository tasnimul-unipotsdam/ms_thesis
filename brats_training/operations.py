import torch
import torch.nn as nn
import torch.nn.functional as F


def weights_init_normal(m):
    if isinstance(m, nn.Conv3d):
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.Linear):
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm3d):
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def weights_init_uniform(m):
    if isinstance(m, nn.Conv3d):
        torch.nn.init.uniform_(m.weight.data)
    elif isinstance(m, nn.Linear):
        torch.nn.init.uniform_(m.weight.data)
    elif isinstance(m, nn.BatchNorm3d):
        torch.nn.init.uniform_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)


def weights_init_xavier_normal(m):
    if isinstance(m, nn.Conv3d):
        torch.nn.init.xavier_normal_(m.weight.data, gain=1)
    elif isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight.data, gain=1)
    elif isinstance(m, nn.BatchNorm3d):
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def weights_init_xavier_uniform(m):
    if isinstance(m, nn.Conv3d):
        torch.nn.init.xavier_uniform_(m.weight.data, gain=1)
    elif isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data, gain=1)
    elif isinstance(m, nn.BatchNorm3d):
        torch.nn.init.uniform_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)


def weights_init_kaiming_normal(m):
    if isinstance(m, nn.Conv3d):
        torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif isinstance(m, nn.BatchNorm3d):
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def weights_init_kaiming_uniform(m):
    if isinstance(m, nn.Conv3d):
        torch.nn.init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in')
    elif isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in')
    elif isinstance(m, nn.BatchNorm3d):
        torch.nn.init.uniform_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    if isinstance(m, nn.Conv3d):
        torch.nn.init.orthogonal_(m.weight.data, gain=1)
    elif isinstance(m, nn.Linear):
        torch.nn.init.orthogonal_(m.weight.data, gain=1)
    elif isinstance(m, nn.BatchNorm3d):
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    print("init weights")
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'uniform':
        net.apply(weights_init_uniform)
    elif init_type == 'xavier_normal':
        net.apply(weights_init_xavier_normal)
    elif init_type == 'xavier_uniform':
        net.apply(weights_init_xavier_uniform)
    elif init_type == 'kaming_normal':
        net.apply(weights_init_kaiming_normal)
    elif init_type == 'kaming_uniform':
        net.apply(weights_init_kaiming_uniform)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def conv_layer_3d(in_size, out_size, kernel_size=(3, 3, 1), stride=(1, 1, 1), padding=(1, 1, 0),
                  dropout_rate=None):
    conv_layer = []
    conv_layer.append(nn.Conv3d(in_size, out_size, kernel_size, stride, padding))
    conv_layer.append(nn.BatchNorm3d(out_size))
    conv_layer.append(nn.ReLU())
    if dropout_rate is not None:
        conv_layer.append(nn.Dropout3d(p=dropout_rate))
    return nn.Sequential(*conv_layer)


class UnetConv3(nn.Module):
    """
    3D Convolution block
    """
    def __init__(self, in_size, out_size, kernel_size=(3, 3, 1), padding=(1, 1, 0),
                 stride=(1, 1, 1), dropout_rate=None):
        """
        :param in_size:
        :param out_size:
        :param kernel_size:
        :param padding:
        :param stride:
        :param dropout_rate:
        """
        super(UnetConv3, self).__init__()
        self.conv1 = conv_layer_3d(in_size, out_size, kernel_size, stride, padding, dropout_rate)
        self.conv2 = conv_layer_3d(out_size, out_size, kernel_size, 1, padding, dropout_rate)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class UnetUp3(nn.Module):
    """
    UpSample 3D convolution
    """
    def __init__(self, in_size, out_size, use_deconv=True, dropout_rate=None):
        """
        :param in_size:
        :param out_size:
        :param use_deconv:
        :param dropout_rate:
        """
        super(UnetUp3, self).__init__()
        if use_deconv:
            self.conv = conv_layer_3d(in_size, out_size, dropout_rate=dropout_rate)
            self.up = nn.ConvTranspose3d(in_size, out_size, kernel_size=(4, 4, 1), stride=(2, 2, 1),
                                         padding=(1, 1, 0))
        else:
            self.conv = conv_layer_3d(in_size + out_size, out_size, dropout_rate=dropout_rate)
            self.up = nn.Upsample(scale_factor=(2, 2, 1), mode='trilinear')

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2, 0]
        outputs1 = F.pad(inputs1, padding)
        return self.conv(torch.cat([outputs1, outputs2], 1))
