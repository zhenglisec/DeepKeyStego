import torch.nn as nn

class ConvBNLRelu(nn.Module):
    def __init__(self, channels_in, channels_out, stride=1):

        super(ConvBNLRelu, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 3, stride, padding=1),
            nn.BatchNorm2d(channels_out),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)
