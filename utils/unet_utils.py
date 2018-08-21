import torch
import torch.nn as nn


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, num_layers=2):
        super(UNetConvBlock, self).__init__()

        self.convs = nn.ModuleList()
        if is_batchnorm:
            conv = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, padding=1),
                                 nn.BatchNorm2d(out_size), nn.ReLU)
            self.convs.append(conv)
            for i in range(1, num_layers):
                conv = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, padding=1),
                                     nn.BatchNorm2d(out_size), nn.ReLU())
                self.convs.append(conv)
        else:
            conv = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, padding=1),
                                 nn.ReLU())
            self.convs.append(conv)

    def forward(self, inputs):
        outputs = inputs
        for conv in self.convs:
            outputs = conv(outputs)
        return outputs


class UNetDown(nn.Module):

    def __init__(self, in_size, out_size, is_batchnorm):
        super(UNetDown, self).__init__()
        self.conv = UNetConvBlock(in_size, out_size, is_batchnorm, num_layers=2)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, inputs):
        residual = self.conv(inputs)
        outputs = self.pool(residual)
        return residual, outputs


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv=False, residual_size=2, is_batchnorm=False):
        super(UNetUp, self).__init__()
        if residual_size is None:
            residual_size = out_size
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, in_size, kernel_size=2, stride=2)
            self.conv = UNetConvBlock(in_size + residual_size, out_size, is_batchnorm=is_batchnorm, num_layers=2)
        else:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear")
            self.conv = UNetConvBlock(in_size + residual_size, out_size, is_batchnorm=is_batchnorm, num_layers=3)
        print('UnetUp convBlock::{}->{}'.format(in_size + residual_size, out_size))

    def forward(self, residual, previous):
        upsampled = self.up(previous)
        print('previous ({}) -> upsampled ({})'.format(previous.size()[1], upsampled.size()[1]))
        print('residual.size(), upsampled.size()', residual.size(), upsampled.size())
        result = self.conv(torch.cat([residual, upsampled], 1))
        print('Result size:', result.size())
        return result


class UNet(nn.Module):

    def __init__(self, feature_scale=1, n_classes=1, is_deconv=True, in_channels=3,
                 is_batchnorm=True, filters=None):
        super(UNet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batechnorm = is_batchnorm
        self.feature_scale = feature_scale

        if filters is None:
            filters = [32, 64, 64, 128, 128, 256]
        print("UNet filter sizes:", filters)

        filters = [x / self.feature_scale for x in filters]

        self.down1 = UNetDown(self.in_channels, filters[0], self.is_batechnorm)
        self.down2 = UNetDown(filters[0], filters[1], self.is_batechnorm)
        self.down3 = UNetDown(filters[1], filters[2], self.is_batechnorm)
        self.down4 = UNetDown(filters[2], filters[3], self.is_batechnorm)
        self.down5 = UNetDown(filters[3], filters[4], self.is_batechnorm)

        self.center = UNetConvBlock(filters[4], filters[5], self.is_batechnorm)

        self.up5 = UNetUp(filters[5], filters[4], self.is_deconv, is_batchnorm=self.is_batechnorm)
        self.up4 = UNetUp(filters[4], filters[3], self.is_deconv, is_batchnorm=self.is_batechnorm)
        self.up3 = UNetUp(filters[3], filters[2], self.is_deconv, is_batchnorm=self.is_batechnorm)
        self.up2 = UNetUp(filters[2], filters[1], self.is_deconv, is_batchnorm=self.is_batechnorm)
        self.up1 = UNetUp(filters[1], filters[0], self.is_deconv, is_batchnorm=self.is_batechnorm)

        self.final = nn.Conv2d(filters[0], n_classes, kernel_size=1)

    def forward(self, inputs):
        res1, out = self.down1(inputs)
        res2, out = self.down2(out)
        res3, out = self.down3(out)
        res4, out = self.down4(out)
        res5, out = self.down5(out)
        out = self.center(out)
        out = self.up5(res5, out)
        out = self.up4(res4, out)
        out = self.up3(res3, out)
        out = self.up2(res2, out)
        out = self.up1(res1, out)
        return self.final(out)






















































