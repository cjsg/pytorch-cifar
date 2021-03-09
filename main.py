'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import time
from math import log10, ceil

from models import *
from utils import *
# from utils import progress_bar


class Progress(object):
    def __init__(self, *names):
        self.monitor = dict()
        for name in names:
            self.monitor[name] = []

    def update(self, name, val):
        if name not in self.monitor:
            self.monitor[name] = [val]
        else:
            self.monitor[name].append(val)

    def __getitem__(self, key):                                                                                                                                                                         
        return self.monitor[key]

    def display_all(self):
        entries = ['Progress:']
        for name in self.monitor:
            entries.append(f'\n  {name}:\t{torch.tensor(self.monitor[name])}')
        print(''.join(entries))

    def display(self, epoch):
        entries = [f'{epoch:03d}']
        for name in self.monitor:
            entries.append(f'   {name}: {self.monitor[name][-1]:.4f}')
        print(''.join(entries))


def load_data(bs=128, num_workers=2, img_size=32):
    # img_size = size (height) of input image (can be 4, 8, 16 or 32)

    transform_train = transforms.Compose([
        transforms.ColorJitter(brightness=.2, contrast=.2, saturation=.2, hue=.2),
        transforms.RandomResizedCrop(32, scale=(.8, 1.)),
        transforms.RandomAffine(degrees=30, translate=(.2, .2), scale=(.8, 1.2), shear=(.1, .1, .1, .1)),
        torchvision.transforms.RandomGrayscale(p=0.1),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=datadir, train=True, download=False, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=bs, shuffle=True, num_workers=num_workers)

    testset = torchvision.datasets.CIFAR10(
        root=datadir, train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=bs, shuffle=False, num_workers=num_workers)

    # classes = ('plane', 'car', 'bird', 'cat', 'deer',
    #            'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader


def train(epoch):
    # print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    start_time = time.time()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()  # if scheduler = OneCycleLR

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if args.verbose:
            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    progress.update('tr_time', time.time()-start_time)
    progress.update('tr_loss', train_loss/(batch_idx+1))
    progress.update('tr_acc', 100.*correct/total)


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    start_time = time.time()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
  
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if args.verbose:
                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

        te_acc = 100.*correct/total
        best_acc = max(te_acc, best_acc)
        progress.update('te_time', time.time()-start_time)
        progress.update('te_loss', test_loss/(batch_idx+1))
        progress.update('te_acc', te_acc)


def save_if_needed(epoch, ckptdir):
    if not os.path.isdir(ckptdir):
        os.makedirs(ckptdir)

    acc = progress['te_acc'][-1]
    # Save every 10 epochs or if best or last

    if ((epoch+1) % 10 == 0) or (epoch+1 == args.epochs) or (acc == best_acc):
        state = {
            'args': args,
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'progress': progress,
            'best_acc': best_acc,
        }

    if ((epoch+1) % 10 == 0) or (epoch+1 == args.epochs):
        torch.save(state, os.path.join(f'{ckptdir}', f'{epoch:03d}.pth'))
        
    if acc == best_acc:
        torch.save(state, os.path.join(f'{ckptdir}', 'best.pth'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--name', '-nm', type=str, help='Mandatory: experiment name')
    parser.add_argument('--epochs', '-ep', default=500, type=int,
                        help='number of epochs')
    parser.add_argument('--lr', '-lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--adameps', default=1e-8, type=float,
                        help="adam's eps parameter")
    parser.add_argument('--amsgrad', default=0, type=int,
                        help='if >0, use amsgrad in adam')
    parser.add_argument('--ckpt_path', default=None, type=str,
                        help='path to checkpoint')
    parser.add_argument('--img_size', '-d', default=32, type=int,
                        help='desired image size '
                        '(height or width in [4, 8, 16, 32)')
    parser.add_argument('--batch_size', '-bs', default=128, type=int,
                        help='batch size per (cuda) device.')
    parser.add_argument('--dropout_rate', '-dr', default=0., type=float)
    parser.add_argument('--num_workers', '-nw', default=2, type=int,
                        help='number of workers in data loader per (cuda) device')
    parser.add_argument('--verbose', '-v', default=0, type=int)
    args = parser.parse_args()

    datadir = os.path.expanduser('~/datasets/cifar10')
    n_dig = ceil(-log10(args.lr))
    fmt = '.' + str(n_dig) + 'f' if n_dig > 0 else str(-n_dig+1) + '.0f'
    string = 'lr={lr:' + fmt + '}_adameps={adameps:6.4f}'
    ckptdir = os.path.join('checkpoint', args.name, string.format(
        lr=args.lr, adameps=args.adameps))
    # ckptdir = os.path.join('checkpoint', args.name, f'lr={args.lr:f}')
    args.verbose = (args.verbose != 0)
    args.amsgrad = (args.amsgrad != 0)
    if args.verbose:
        initialize_progress_bar_settings()

    if torch.cuda.is_available():
        device = 'cuda'
        device_count = torch.cuda.device_count()
    else:
        device = 'cpu'
        device_count = 1

    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    args.batch_size = device_count * args.batch_size
    args.num_workers = device_count * args.num_workers

    # Data
    print('==> Preparing data..')
    trainloader, testloader = load_data(
        args.batch_size, args.num_workers, img_size=args.img_size)

    # Model
    print('==> Building model..')
    # net = VGG('VGG19')
    # net = ResNet18()
    # net = PreActResNet18()
    # net = GoogLeNet()
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    # net = MobileNet()
    # net = MobileNetV2()
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()
    # net = ShuffleNetV2(1)
    # net = EfficientNetB0()
    # net = RegNetX_200MF()
    # net = SimpleDLA()
    # net = SmallResNet18(args.img_size)
    # net = ViT_L2_H4_P4(args.dropout_rate)
    net = ViT_L8_H4_P4(args.dropout_rate)
    init_params(net)
    net = net.to(device)
    # print(net)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    progress = Progress()
    if args.ckpt_path is not None:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(args.ckpt_path)   # './checkpoint/ckpt.pth')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']+1
        progress = checkpoint['progress']
        epochs = args.epochs
        args = checkpoint['args']
        args.epochs = epochs

    # Training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=0.1,
                           eps=args.adameps, amsgrad=args.amsgrad)
    pct_start = 1e4 / (len(trainloader)*args.epochs*args.batch_size)  # 10K warm-up
    # print(f'pct_start {pct_start:.2e}')
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs,
        steps_per_epoch=len(trainloader), pct_start=pct_start,
        anneal_strategy='linear')
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, T_max=args.epochs)
    # optimizer = optim.SGD(net.parameters(), lr=args.lr,
    #                       momentum=0.9, weight_decay=5e-4)

    print("==> Starting to train")
    for epoch in range(start_epoch, args.epochs):
        train(epoch)
        test(epoch)
        # scheduler.step()
        save_if_needed(epoch, ckptdir)
        progress.display(epoch)
