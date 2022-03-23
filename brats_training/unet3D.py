import torch
import torch.nn as nn
from torch.nn import functional as F

from torchsummary import summary

# setting device
device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'device:{device}')
torch.cuda.set_device(device)


def _get_dropout_(drop, drop_mode, i, repetitions):
    if drop_mode == "all":
        return drop

    if drop_mode == "first" and i == 0:
        return drop

    if drop_mode == "last" and i == repetitions - 1:
        return drop

    if drop_mode == "no":
        return drop


def _get_dropout_mode(drop_center, curr_depth, depth, is_down):
    if drop_center is None:
        return "all"

    if curr_depth == depth:
        return "no"

    if curr_depth + drop_center >= depth:
        return "last" if is_down else "first"
    return "no"


class Conv(nn.Module):
    def __init__(self, in_channel, out_channel, drop=None, bn=True, padding=1, kernel=(3, 3, 3),
                 activation=True):
        super(Conv, self).__init__()

        self.conv = nn.Sequential()
        self.conv.add_module("conv",
                             nn.Conv3d(in_channel, out_channel, kernel_size=kernel,
                                       padding=padding))

        if drop is not None:
            self.conv.add_module("dropout", nn.Dropout3d(p=drop))

        if bn:
            self.conv.add_module("batch_normalization", nn.BatchNorm3d(out_channel))

        if activation:
            self.conv.add_module("relu", nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class DoubleConv(nn.Module):
    def __init__(self, in_channel, out_channel, mid_channel=None, drop=None, drop_mode="all",
                 bn=True, repetitions=2):
        super(DoubleConv, self).__init__()

        if not mid_channel:
            mid_channel = out_channel

        convs = []

        in_ch_tmp = in_channel

        for i in range(repetitions):
            do = _get_dropout_(drop, drop_mode, i, repetitions)

            convs.append(Conv(in_ch_tmp, mid_channel, drop=do, bn=bn))

            in_ch_tmp = mid_channel
            mid_channel = out_channel

        self.block = nn.Sequential(*convs)

    def forward(self, x):
        x = self.block(x)
        return x


class Down(nn.Module):
    def __init__(self, in_channel, out_channel, drop=None, drop_center="all",
                 curr_depth=0, depth=4, bn=True):
        super().__init__()

        do_mode = _get_dropout_mode(drop_center, curr_depth, depth, True)
        self.maxpool_conv = nn.Sequential(nn.MaxPool3d(2),
                                          DoubleConv(in_channel, out_channel, drop=drop,
                                                     drop_mode=do_mode,
                                                     bn=bn))

    def forward(self, x):
        x = self.maxpool_conv(x)
        return x


class Up(nn.Module):
    def __init__(self, in_channel, out_channel, drop=None, drop_center="all", curr_depth=0, depth=4,
                 bn=True, bilinear=True):
        super(Up, self).__init__()

        do_mode = _get_dropout_mode(drop_center, curr_depth, depth, is_down=False)

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
            self.conv = DoubleConv(in_channel, out_channel, in_channel // 2, drop, do_mode, bn)

        else:
            self.up = nn.ConvTranspose3d(in_channel, in_channel // 2, kernel_size=(3, 3, 3),
                                         stride=(3, 3, 3))
            self.conv = DoubleConv(in_channel, out_channel, drop=drop, drop_mode=do_mode, bn=bn)

    def forward(self, x1, x2):
        # print("x1:", x1.shape)
        # print("x2:", x2.shape)
        x1 = self.up(x1)
        # input is CHWD
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        diffZ = x2.size()[4] - x1.size()[4]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2, diffZ // 2,
                        diffZ - diffZ // 2])

        cat = torch.cat([x2, x1], dim=1)

        x = self.conv(cat)
        return x


class OutConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = nn.Conv3d(in_channel, out_channel, kernel_size=(1, 1, 1))

    def forward(self, x):
        x = self.conv(x)
        return x


class OutConvActivation(nn.Module):
    def __init__(self, in_channel, out_channel, activation=True):
        super(OutConvActivation, self).__init__()
        self.conv = nn.Sequential()
        self.conv.add_module("last_conv", nn.Conv3d(in_channel, out_channel,
                                                    kernel_size=(1, 1, 1)))

        if activation:
            self.conv.add_module("sigmoid", nn.Sigmoid())

    def forward(self, x):
        x = self.conv(x)
        return x


class Unet3D(nn.Module):
    print("getting model")
    DEFAULT_DEPTH = 4
    DEFAULT_DROPOUT = 0.05  # 0.1
    DEFAULT_FILTERS = 16  # 16

    def __init__(self, n_channels, n_classes, n_filters=DEFAULT_FILTERS, depth=DEFAULT_DEPTH,
                 drop=DEFAULT_DROPOUT, drop_center=None, bn=True, bilinear=True):
        super(Unet3D, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_filters = n_filters
        self.drop = drop
        self.drop_center = drop_center
        self.bn = bn
        self.bilinear = bilinear

        curr_depth = 0
        do_mode = _get_dropout_mode(drop_center, curr_depth, depth, is_down=True)
        self.inc = DoubleConv(in_channel=n_channels, out_channel=n_filters, mid_channel=None,
                              drop=drop, drop_mode=do_mode, bn=bn)

        curr_depth = 1
        self.down1 = Down(in_channel=n_filters, out_channel=n_filters * 2, drop=drop,
                          drop_center=drop_center, curr_depth=curr_depth, depth=depth, bn=bn)

        curr_depth = 2
        self.down2 = Down(in_channel=n_filters * 2, out_channel=n_filters * 4, drop=drop,
                          drop_center=drop_center, curr_depth=curr_depth, depth=depth, bn=bn)

        curr_depth = 3
        self.down3 = Down(in_channel=n_filters * 4, out_channel=n_filters * 8, drop=drop,
                          drop_center=drop_center, curr_depth=curr_depth, depth=depth, bn=bn)

        factor = 2 if self.bilinear else 1

        self.down4 = Down(in_channel=n_filters * 8, out_channel=n_filters * 16 // factor, drop=drop,
                          drop_center=drop_center, curr_depth=depth, depth=depth, bn=bn)

        curr_depth = 3
        self.up1 = Up(in_channel=n_filters * 16, out_channel=n_filters * 8 // factor, drop=drop,
                      drop_center=drop_center, curr_depth=curr_depth, depth=depth, bn=bn,
                      bilinear=bilinear)

        curr_depth = 2
        self.up2 = Up(in_channel=n_filters * 8, out_channel=n_filters * 4 // factor, drop=drop,
                      drop_center=drop_center, curr_depth=curr_depth, depth=depth, bn=bn,
                      bilinear=bilinear)

        curr_depth = 1
        self.up3 = Up(in_channel=n_filters * 4, out_channel=n_filters * 2 // factor, drop=drop,
                      drop_center=drop_center, curr_depth=curr_depth, depth=depth, bn=bn,
                      bilinear=bilinear)

        curr_depth = 0
        self.up4 = Up(in_channel=n_filters * 2, out_channel=n_filters, drop=drop,
                      drop_center=drop_center, curr_depth=curr_depth, depth=depth, bn=bn,
                      bilinear=bilinear)

        self.out = OutConv(n_filters, n_classes)

    def forward(self, x):
        x1 = self.inc(x)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        output = self.out(x)
        return output


if __name__ == '__main__':
    print("model")
    # model = Unet3D(n_channels=1, n_classes=1)
    # print(model)
    # model.to(device)
    # summary(model, (1, 128, 128, 128))
