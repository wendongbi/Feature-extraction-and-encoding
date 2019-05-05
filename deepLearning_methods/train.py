from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from util.dataloader import Datasets
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
import os
import argparse

# from models import *
from ori_resnet import resnet101
from utils import progress_bar
import numpy as np

train_id = 7
resume_id = 7

log_train_path = './logFiles/' + str(train_id) + '/logs/train'
log_test_path = './logFiles/' + str(train_id) + '/logs/test'

if not os.path.exists(log_train_path):
	os.makedirs(log_train_path)
if not os.path.exists(log_test_path):
	os.makedirs(log_test_path)
writer_train = SummaryWriter(log_train_path)
writer_test = SummaryWriter(log_test_path)

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--lr', default=5e-3, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--optim', default='Adam', type=str)
parser.add_argument('--epoch', default=200, type=int, metavar='N')
parser.add_argument('--weight-decay','--wd',default=1e-4,type=float, metavar='W')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
parser.add_argument('--logspace', default=2, type=int)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch



# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    # transforms.ColorJitter(1, 1, 1, 0.5),
    transforms.ToTensor(),
])

# transform_test = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])

totalset = Datasets('./data/my_data/total.csv','./data/my_data/Total_data', transform=transform_train)


trainset = Datasets('./data/my_data/train.csv','./data/my_data/train_data', transform=transform_train)
trainloader = torch.utils.data.DataLoader(totalset, batch_size=64, shuffle=True, num_workers=12)


testset = Datasets('./data/test2.csv','./data/my_data2/test_data', transform=transform_train)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=12)

proposal = Datasets('./proposal.csv','./proposals', transform=transform_train)


# Model
print('==> Building model..')

# net = torchvision.models.resnet101()
net = resnet101(pretrained=True)


net = net.to(device)


if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint/' + str(resume_id) + '/'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('checkpoint/' + str(resume_id) + '/ckpt.t7')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    print('best accu:', best_acc)

criterion = nn.CrossEntropyLoss()
if args.optim == 'SGD':
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
elif args.optim == 'Adam':
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
logspace_lr = torch.logspace(np.log10(args.lr), np.log10(args.lr) - args.logspace, args.epoch)
# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    if args.logspace != 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = logspace_lr[epoch]
    else:
        adjust_learning_rate(optimizer, epoch)
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs, _ = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    writer_train.add_scalar('loss', train_loss / (batch_idx + 1), global_step=epoch)
    writer_train.add_scalar('acc', 100.*correct/total, global_step=epoch)
        #print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _ = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        print('###############################')
        print('Test Accu:', 100.*correct/total)
        print('###############################')
        writer_test.add_scalar('loss', test_loss/(batch_idx+1), global_step=epoch)
        writer_test.add_scalar('acc', 100.*correct/total, global_step=epoch)
            #print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        save_path = 'checkpoint/' + str(train_id) + '/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(state, save_path + 'ckpt.t7')
        best_acc = acc

def saveFeature():
    featureloader = torch.utils.data.DataLoader(proposal, batch_size=128, shuffle=False, num_workers=12)
    # print('==> Resuming from checkpoint..')
    # assert os.path.isdir('checkpoint/' + str(train_id) + '/'), 'Error: no checkpoint directory found!'
    # checkpoint = torch.load('checkpoint/' + str(resume_id) + '/ckpt.t7')
    # net.load_state_dict(checkpoint['net'])

    # best_acc = checkpoint['acc']
    # start_epoch = checkpoint['epoch']
    net.eval()
    flag = True
    count = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(featureloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, feature = net(inputs)
            if flag:
                featureList = feature.cpu().numpy()
                label = targets.cpu().numpy().reshape(-1, 1)
                print(np.shape(featureList))
                flag = False
            else:
                featureList = np.vstack((featureList, feature.cpu().numpy()))
                label = np.vstack((label, targets.cpu().numpy().reshape(-1,1)))
                print(np.shape(featureList), np.shape(label))
            
            np.save('./proposal_feature2', featureList)
            np.save('./proposal_label2', label.reshape(-1))
            # if(batch_idx == 10):
            #     break
            

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 100))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == "__main__":
    # test(start_epoch)
    # for epoch in range(start_epoch, start_epoch+80):
    #     train(epoch)
    #     test(epoch)
    saveFeature()
    print(len(totalset))
    feature = np.load('proposal_feature2.npy')
    print(np.shape(feature))
    label = np.load('./proposal_label2.npy')
    print(np.shape(label))

