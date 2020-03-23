import torch.nn as nn
from model.conv_bn_lrelu import ConvBNLRelu

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        num_blocks = 3
        layers = [ConvBNLRelu(1, 64)]
        for _ in range(num_blocks-1):
            layers.append(ConvBNLRelu(64, 64))
        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.before_linear = nn.Sequential(*layers)
        self.linear = nn.Linear(64, 1)

    def forward(self, image):
        X = self.before_linear(image)
        X.squeeze_(3).squeeze_(2)
        X = self.linear(X)
        return X