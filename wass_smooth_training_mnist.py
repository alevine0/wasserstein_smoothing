from __future__ import print_function

import sys
sys.path.append('./pytorch-cifar')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import wass_smooth_utils
from utils import progress_bar


parser = argparse.ArgumentParser(description='PyTorch MNIST Training')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--stdev', default=0.02, type=float, help='std. dev. of smoothing')
parser.add_argument('--testsamples', default=100, type=int, help='number of smoothing samples for test.')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--seed', default=0, type=int, help='random seed')

args = parser.parse_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

checkpoint_dir = 'checkpoints'
if not os.path.exists('./checkpoints'):
    os.makedirs('./checkpoints')
checkpoint_file = f'./{checkpoint_dir}/mnist_smooth_base_lr_{args.lr}_stddev_{args.stdev}_epoch_{{}}.pth'

print("==> Checkpoint directory", checkpoint_dir)
print("==> Saving to", checkpoint_file)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# Model
print('==> Building model..')

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

criterion = nn.NLLLoss()

def test_nominal(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = torch.log(wass_smooth_utils.wass_smooth_forward_train(inputs, net, args.stdev))
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total


if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(checkpoint_dir), 'Error: no checkpoint directory found!'
    resume_file = '{}/{}'.format(checkpoint_dir, args.resume)
    assert os.path.isfile(resume_file)
    checkpoint = torch.load(resume_file)
    net.load_state_dict(checkpoint['net'])
    start_epoch = checkpoint['epoch']+1
    test_nominal(start_epoch)
    checkpoint_file = './{}/mnist_smooth_base_lr_{}_stddev_{}_epoch_{}_resume_{}.pth'.format(checkpoint_dir, args.lr, args.stdev, '{}', args.resume)

optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    nominal_correct = 0
    total_epsilon = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs_nominal = torch.log(wass_smooth_utils.wass_smooth_forward_train(inputs, net, args.stdev))
        _, predicted_nominal = outputs_nominal.max(1)
        nominal_correct += predicted_nominal.eq(targets).sum().item()

        loss = criterion(outputs_nominal, targets)
        loss.backward()
        optimizer.step()
        total += targets.size(0)

        train_loss += loss.item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*nominal_correct/total, nominal_correct, total))

def test(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    nominal_correct = 0
    total = 0
    total_epsilon = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.no_grad():

            outputs_nominal = torch.log(wass_smooth_utils.wass_smooth_forward(inputs, net, args.testsamples, args.stdev))
            loss = criterion(outputs_nominal, targets)
            _, predicted_nominal = outputs_nominal.max(1)
            nominal_correct += predicted_nominal.eq(targets).sum().item()

            test_loss += loss.item()
            total += targets.size(0)

            progress_bar(batch_idx, len(testloader), 'Test Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*nominal_correct/total, nominal_correct, total))

    # Save checkpoint.
    if (epoch ==199):
        acc = 100.*correct/total
        avg_epsilon = total_epsilon/total
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'eps': avg_epsilon,
            'epoch': epoch,
        }
        if not os.path.isdir(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        torch.save(state, checkpoint_file.format(epoch))


for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test_nominal(epoch)
    if (epoch == 199):
        test(epoch)
