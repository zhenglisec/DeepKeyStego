from __future__ import print_function

import argparse
import os
import random
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from model import SSIM
from model.encoder import Encoder
from model.decoder import syDecoder, asyDecoder
from model.discriminator import Discriminator
from model.pkgenerator import pkgenerator

GPU = '0,1'
os.environ['CUDA_VISIBLE_DEVICES'] = GPU
parser = argparse.ArgumentParser(
    description='Pytorch Implement with ImageNet')
parser.add_argument('--type', default='asymmeric', help='symmeric or asymmeric')
parser.add_argument('--dataroot', default='xxx/xxx/xxx')
parser.add_argument('--train', type=bool, default=True)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--batchsize', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--beta', '--list', nargs='+',
                    default=[0.5, 0.5, 0.03, 0.1])
parser.add_argument('--seed', default=22, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--secret_len', type=int, default=8192)
parser.add_argument('--key_len', type=int, default=1024)
args = parser.parse_args()

if torch.cuda.is_available():
    cudnn.benchmark = True
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have cho5sen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')


print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),])

trainset = torchvision.datasets.ImageFolder(
    root=args.dataroot+'train', transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batchsize, shuffle=True, num_workers=2, drop_last=True)
testset = torchvision.datasets.ImageFolder(
    root=args.dataroot+'test', transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=args.batchsize, shuffle=False, num_workers=2, drop_last=True)

# Adversarial ground truths
valid = torch.cuda.FloatTensor(args.batchsize, 1).fill_(1.0)
fake = torch.cuda.FloatTensor(args.batchsize, 1).fill_(0.0)

best_real_acc, best_wm_acc, best_wm_input_acc = 0, 0, 0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
train_loss, test_loss = [[], []], [[], []]
train_acc, test_acc = [[], []], [[], []]

# Model
print('==> Building model..')
if args.type == 'symmeric':
    Decoder = syDecoder(sec_len = args.secret_len, output_function=nn.Sigmoid)
elif args.type == 'asymmeric':
    Decoder = asyDecoder(sec_len = args.secret_len, output_function=nn.Sigmoid)
    Pkgenerator = pkgenerator()
Encoder = Encoder(sec_len = args.secret_len)
Discriminator = Discriminator()

Encoder = nn.DataParallel(Encoder.cuda())
Decoder = nn.DataParallel(Decoder.cuda())
Discriminator = nn.DataParallel(Discriminator.cuda())
if args.type == 'asymmeric':
    Pkgenerator = nn.DataParallel(Pkgenerator.cuda())
# loss function
criterionE_mse = nn.MSELoss().cuda()
criterionE_ssim = SSIM().cuda()
criterionD = nn.L1Loss().cuda()
optimizerE = optim.Adam(Encoder.parameters(), lr=args.lr, betas=(0.5, 0.999))
optimizerD = optim.Adam(Decoder.parameters(), lr=args.lr, betas=(0.5, 0.999))
criterionDis = nn.BCEWithLogitsLoss().cuda()
optimizerDis = optim.Adam(Discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

if args.type == 'asymmeric':
    optimizerPkGen = optim.Adam(Pkgenerator.parameters(), lr=args.lr, betas=(0.5, 0.999))

print(Encoder)
print(Decoder)
print(Discriminator)
def train(epoch):
    print('\nEpoch: %d' % epoch)
    Encoder.train()
    Decoder.train()
    Discriminator.train()
    if args.type == 'asymmeric':
        Pkgenerator.train()

    for batch_idx, (input, _) in enumerate(trainloader):
        input = input.cuda()
        messages = torch.from_numpy(np.random.randint(2, size=(args.batchsize, args.secret_len))).float().cuda()
        skey = torch.from_numpy(np.random.randint(2, size=(args.batchsize, args.key_len))).float().cuda() # secrect key
        if args.type == 'asymmeric':
            pkey = Pkgenerator(skey)
        #############optimize Discriminator##############
        optimizerDis.zero_grad()
        if args.type == 'symmeric':
            stego = Encoder(input, messages, skey)
        elif args.type == 'asymmeric':
            stego = Encoder(input, messages, pkey)
        stego_dis_output = Discriminator(stego.detach())
        real_dis_output = Discriminator(input)
        loss_D_stego = criterionDis(stego_dis_output, fake)
        loss_D_real = criterionDis(real_dis_output, valid)
        loss_D = loss_D_stego + loss_D_real
        loss_D.backward()
        optimizerDis.step()
        ################optimize Encoder Decoder or Pkgenerator#############
        optimizerE.zero_grad()
        optimizerD.zero_grad()
        if args.type == 'symmeric':
            decoded_messages = Decoder(stego, skey)
        elif args.type == 'asymmeric':
            optimizerPkGen.zero_grad()
            decoded_messages = Decoder(stego, pkey, skey)
        stego_dis_output = Discriminator(stego)
        loss_mse = criterionE_mse(input, stego)
        loss_ssim = criterionE_ssim(input, stego)
        loss_adv = criterionDis(stego_dis_output, valid)
        loss_message = criterionD(decoded_messages, messages)
        loss_H = args.beta[0] * loss_mse + args.beta[1] * \
            (1 - loss_ssim) + args.beta[2] * loss_adv + args.beta[3] * loss_message
        loss_H.backward()
        optimizerE.step()
        optimizerD.step()
        if args.type == 'asymmeric':
            optimizerPkGen.step()
        decoded_rounded = torch.round(decoded_messages.detach())

        bitwise_avg_correct = torch.sum(torch.eq(messages, decoded_rounded))/args.batchsize

        print('[%d/%d][%d/%d]  Loss D: %.4f () Loss_H: %.4f (mse: %.4f ssim: %.4f adv: %.4f) bitcorrect: %.4f' % (
            epoch, args.num_epochs, batch_idx, len(trainloader),
            loss_D.item(), loss_H.item(), loss_mse.item(
            ), loss_ssim.item(), loss_adv.item(), bitwise_avg_correct))

def test(epoch):
    Encoder.eval()
    Decoder.eval()
    if args.type == 'asymmeric':
        Pkgenerator.eval()

    with torch.no_grad():
        for batch_idx, (input, _) in enumerate(testloader):
            input = input.cuda()
            messages = torch.from_numpy(np.random.randint(2, size=(args.batchsize, args.secret_len))).float().cuda()
            skey = torch.from_numpy(np.random.randint(2, size=(args.batchsize, args.key_len))).float().cuda()
            if args.type == 'symmeric':
                stego = Encoder(input, messages, skey)
                decoded_messages = Decoder(stego, skey)
                save_img = 'results/symmeric.png'
            if args.type == 'asymmeric':
                pkey = Pkgenerator(skey)
                stego = Encoder(input, messages, pkey)
                decoded_messages = Decoder(stego, pkey, skey)
                save_img = 'results/asymmeric.png'

            decoded_rounded = torch.round(decoded_messages.detach())#.cpu().numpy().round().clip(0, 1)
            bitwise_avg_correct = torch.sum(torch.eq(messages, decoded_rounded))/args.batchsize
            
            concat_img = torch.cat([input[0:10], stego[0:10]], dim=0)
            torchvision.utils.save_image(concat_img, save_img, nrow=10, padding=0)
            print('BitCorrect: %.4f' % (bitwise_avg_correct))

for epoch in range(args.num_epochs):
    train(epoch)
    test(epoch)