from __future__ import print_function

import sys
sys.path.append('./pytorch-cifar')
sys.path.append('./pytorch-cifar/models')

from models import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from utils import progress_bar
import wass_smooth_utils

parser = argparse.ArgumentParser(description='Wasserstein Certificate Evaluation')
parser.add_argument('--stdev', default=0.02, type=float, help='std. dev. of smoothing')
parser.add_argument('--model',  required=True, help='checkpoint to certify')
parser.add_argument('--alpha', default=0.05, type=float, help='Certify to 1-alpha probability')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--predsamples', default=1000, type=int, help='samples for prediction')
parser.add_argument('--boundsamples', default=10000, type=int, help='samples for bound')

args = parser.parse_args()
checkpoint_dir = 'checkpoints'
radii_dir = 'radii'
if not os.path.exists('./radii'):
    os.makedirs('./radii')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data
print('==> Preparing data..')

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

mu = torch.Tensor((0.4914, 0.4822, 0.4465)).unsqueeze(-1).unsqueeze(-1).to(device)
std = torch.Tensor((0.2023, 0.1994, 0.2010)).unsqueeze(-1).unsqueeze(-1).to(device)
unnormalize = lambda x: x*std + mu
normalize = lambda x: (x-mu)/std


testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

# Model
print('==> Building model..')

net = ResNet18()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True


assert os.path.isdir(checkpoint_dir), 'Error: no checkpoint directory found!'
resume_file = '{}/{}'.format(checkpoint_dir, args.model)
#print(resume_file)
assert os.path.isfile(resume_file)
checkpoint = torch.load(resume_file)
net.load_state_dict(checkpoint['net'])

net.eval()
all_batches = []
for batch_idx, (inputs, targets) in enumerate(testloader):
    inputs, targets = inputs.to(device), targets.to(device)
    with torch.no_grad():
        #breakpoint()
        batch_radii = wass_smooth_utils.wass_smooth_certify(inputs, targets, net, args.alpha, args.stdev, args.predsamples, args.boundsamples,channel_norm=normalize )
        all_batches.append(batch_radii)
        progress_bar(batch_idx, len(testloader))
out = torch.cat(all_batches)
sortd,indices = torch.sort(out)
torch.save(sortd, radii_dir +'/'+args.model+'_alpha_'+str(args.alpha)+'_boundsamples_'+str(args.boundsamples)+'_predsamples_'+str(args.predsamples) + '.pth')