import torch
import torch.nn as nn
from model.conv_bn_lrelu import ConvBNLRelu
class pkgenerator(nn.Module):
    def __init__(self, output_function=nn.Sigmoid):
        super(pkgenerator, self).__init__()
        self.layer = ConvBNLRelu(1,1)
        self.out = nn.Sequential(
            nn.Conv2d(1,1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Sigmoid()
        )
    def forward(self, input):
        input = input.view(-1, 1, 32, 32)
        out = self.layer(input)
        out = self.out(out)
        out = out.view(-1, 1024)
        return out