from __future__ import print_function

import sys

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

parser = argparse.ArgumentParser(description='MNIST Evaluation')
parser.add_argument('--stdev', default=0.02, type=float, help='std. dev. of smoothing')
parser.add_argument('--model',  required=True, help='checkpoint to predict')
parser.add_argument('--alpha', default=0.05, type=float, help='Predict to 1-alpha probability')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--samples', default=10000, type=int, help='number of samples')

args = parser.parse_args()
checkpoint_dir = 'checkpoints'
acc_dir = 'accuracies'
if not os.path.exists('./accuracies'):
    os.makedirs('./accuracies')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data
print('==> Preparing data..')

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=False, num_workers=2)

# Model
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

net = nn.Sequential(
        nn.Conv2d(1, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(32*7*7,100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True


assert os.path.isdir(checkpoint_dir), 'Error: no checkpoint directory found!'
resume_file = '{}/{}'.format(checkpoint_dir, args.model)
assert os.path.isfile(resume_file)
checkpoint = torch.load(resume_file)
net.load_state_dict(checkpoint['net'])

net.eval()
tot = 0
correct = 0
abstain = 0
for batch_idx, (inputs, targets) in enumerate(testloader):
    inputs, targets = inputs.to(device), targets.to(device)
    with torch.no_grad():
        #breakpoint()
        predicted = wass_smooth_utils.wass_predict(inputs, net, args.stdev, args.samples, args.alpha)
        correct += (predicted == targets.cpu()).sum()
        abstain += (predicted == -1).sum()
        tot += predicted.shape[0]
        progress_bar(batch_idx, len(testloader), 'Acc: %.3f%% (%d/%d)' % (100.*correct/tot, correct, tot))
out = {
    'total': tot,
    'correct': correct,
    'abstain': abstain
}
print('Wasserstein Mnist Accuracies')
print('stdev:' +str(args.stdev))
print('total:'+str(tot))
print('correct:'+str(correct))
print('abstain:'+str(abstain))
torch.save(out, acc_dir +'/'+args.model+'_alpha_'+str(args.alpha)+'_samples_'+str(args.samples)+'.pth')
