import torch
import torch.nn as nn

class PreKey(nn.Module):
    def __init__(self):
        super(PreKey,self).__init__()
        self.key_pre = nn.Sequential(
            nn.ConvTranspose2d(1, 1, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=True),
            nn.BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.ConvTranspose2d(1, 1, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=True),
            nn.BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
    def forward(self, input):
        input = input.view(-1, 1, 32, 32)
        out = self.key_pre(input)
        out = self.key_pre(out)
        return out

class syDecoder(nn.Module):
    def __init__(self, nc=1, nhf=64, sec_len = 128*64, output_function=nn.Sigmoid):
        super(syDecoder, self).__init__()
        self.sec_len = sec_len
        self.skey_pre = PreKey()
        self.main = nn.Sequential(
            nn.Conv2d(nc+1, nhf, 3, 1, 1),
            nn.BatchNorm2d(nhf),
            nn.LeakyReLU(True),
            nn.Conv2d(nhf, nhf, 3, 1, 1),
            nn.BatchNorm2d(nhf),
            nn.LeakyReLU(True)
        )
        self.revel_512x256 = nn.Sequential(
            nn.Conv2d(nhf, nhf, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True),
            nn.BatchNorm2d(nhf),
            nn.ReLU(True),
            nn.Conv2d(nhf, 1, kernel_size=(3, 4), stride=(1, 2), padding=(1, 1), bias=True),
            output_function()
        )
    def forward(self, input, skey):
        skey = skey.view(-1, 1, 32, 32)
        skey_feature = self.skey_pre(skey)
        input_key = torch.cat([input, skey_feature], dim=1)
        ste_feature = self.main(input_key)
        
        out = self.revel_512x256(ste_feature)
        out = out.view(-1, 512 * 256)
        return out
class asyDecoder(nn.Module):
    def __init__(self, nc=1, nhf=64, sec_len=128*64, output_function=nn.Sigmoid):
        super(asyDecoder, self).__init__()
        self.sec_len = sec_len
        self.pkey_pre = PreKey()
        self.skey_pre = PreKey()
        self.main = nn.Sequential(
            nn.Conv2d(nc+2, nhf, 3, 1, 1),
            nn.BatchNorm2d(nhf),
            nn.LeakyReLU(True),
            nn.Conv2d(nhf, nhf, 3, 1, 1),
            nn.BatchNorm2d(nhf),
            nn.LeakyReLU(True)
        )
        self.revel_512x256 = nn.Sequential(
            nn.Conv2d(nhf, nhf, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True),
            nn.BatchNorm2d(nhf),
            nn.ReLU(True),
            nn.Conv2d(nhf, 1, kernel_size=(3, 4), stride=(1, 2), padding=(1, 1), bias=True),
            output_function()
        )
    def forward(self, input, pkey, skey):
        skey = skey.view(-1, 1, 32, 32)
        pkey_feature = self.pkey_pre(pkey)
        skey_feature = self.skey_pre(skey)
        out = torch.cat([input, pkey_feature, skey_feature], dim=1)
        ste_feature = self.main(out)
        out = self.revel_512x256(ste_feature)
        out = out.view(-1, 512 * 256)
        return out


