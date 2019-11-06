import torch
import torch.nn as nn
from model.conv_bn_lrelu import ConvBNLRelu
class Encoder(nn.Module):
    def __init__(self, sec_len=8192):
        super(Encoder, self).__init__()
        self.H = 128
        self.W = 128
        self.conv_channels = 64
        self.num_blocks = 4

        layers = [ConvBNLRelu(9, self.conv_channels)]
        for _ in range(self.num_blocks-1):
            layer = ConvBNLRelu(self.conv_channels, self.conv_channels)
            layers.append(layer)

        self.conv_layers = nn.Sequential(*layers)
        self.final_layer = nn.Sequential( nn.Conv2d(self.conv_channels, 3, kernel_size=1),
                                        nn.Sigmoid(),)

        self.PreMessage = Process(sec_len)
        self.PreKey = Process()
    def forward(self, image, message, key):
        feature_message = self.PreMessage(message)
        feature_key = self.PreKey(key)
        concat = torch.cat([image, feature_message, feature_key], dim=1)
        encoded_image = self.conv_layers(concat)
        im_w = self.final_layer(encoded_image)
        return im_w

class Process(nn.Module):
    def __init__(self, sec_len = 1024):
        super(Process, self).__init__()
        self.sec_len = sec_len
        self.pre_32x32 = nn.Sequential(
            nn.ConvTranspose2d(1, 2, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=True),
            nn.BatchNorm2d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.ConvTranspose2d(2, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=True),
            nn.BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.pre_128x64 = nn.Sequential(
            nn.ConvTranspose2d(1, 2, kernel_size=(3, 4), stride=(1, 2), padding=(1, 1), bias=True),
            nn.BatchNorm2d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.ConvTranspose2d(2, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True),
            nn.BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
    def forward(self, input):
        len = input.size()[1]
        if self.sec_len == 32*32:
            input = input.view(-1, 1, 32, 32)
            out = self.pre_32x32(input)
            return out
        elif self.sec_len == 128*64:
            input = input.view(-1, 1, 128, 64)
            out = self.pre_128x64(input)
            return out

        