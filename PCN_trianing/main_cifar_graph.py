"""
Train CIFAR10 with PyTorch. This code was used for graphing the experiments we conducted.
"""
from __future__ import print_function
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import argparse
import pandas as pd
from prednet import *
from utils import progress_bar
from torch.autograd import Variable

def main_cifar(model='PredNetBpD', circles=5, gpunum=1, Tied=False, weightDecay=1e-3, nesterov=False):
    use_cuda = torch.cuda.is_available()
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    batchsize = 128
    root = './'
    rep = 1
    lr = 0.01

    # Lists to store metrics for CSV logging
    epochs_list = []
    train_loss_list = []
    train_acc_list = []
    test_loss_list = []
    test_acc_list = []

    models = {'PredNetBpD':PredNetBpD}
    modelname = model+'_'+str(circles)+'CLS_'+str(nesterov)+'Nes_'+str(weightDecay)+'WD_'+str(Tied)+'TIED_'+str(rep)+'REP'

    # clearn folder
    checkpointpath = root+'checkpoint/'
    logpath = root+'log/'
    if not os.path.isdir(checkpointpath):
        os.mkdir(checkpointpath)
    if not os.path.isdir(logpath):
        os.mkdir(logpath)
    while(os.path.isfile(logpath+'training_stats_'+modelname+'.txt')):
        rep += 1
        modelname = model+'_'+str(circles)+'CLS_'+str(nesterov)+'Nes_'+str(weightDecay)+'WD_'+str(Tied)+'TIED_'+str(rep)+'REP'

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
    trainset = torchvision.datasets.CIFAR100(root='../data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR100(root='../data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=False, num_workers=2)

    # Model
    print('==> Building model..')
    net = models[model](num_classes=100,cls=circles,Tied=Tied)


    # Define objective function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), momentum=0.9, lr=lr, weight_decay=weightDecay, nesterov=nesterov)

    # Parallel computing
    if use_cuda:
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=range(gpunum))
        cudnn.benchmark = True

    # item() is a recent addition, so this helps with backward compatibility.
    def to_python_float(t):
        if hasattr(t, 'item'):
            return t.item()
        else:
            return t[0]

    # Training
    def train(epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0

        training_setting = 'batchsize=%d | epoch=%d | lr=%.1e ' % (batchsize, epoch, optimizer.param_groups[0]['lr'])
        statfile.write('\nTraining Setting: '+training_setting+'\n')

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += to_python_float(loss.data)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).float().cpu().sum()

            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

        train_epoch_loss = train_loss/(batch_idx+1)
        train_epoch_acc = 100.*correct/total

        statstr = 'Training: Epoch=%d | Loss: %.3f |  Acc: %.3f%% (%d/%d) | best acc: %.3f' \
                  % (epoch, train_epoch_loss, train_epoch_acc, correct, total, best_acc)
        statfile.write(statstr+'\n')

        # Append metrics for CSV logging
        epochs_list.append(epoch)
        train_loss_list.append(train_epoch_loss)
        train_acc_list.append(train_epoch_acc)

        return train_epoch_loss, train_epoch_acc

    # Testing
    def test(epoch):
        nonlocal best_acc
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                if use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                inputs, targets = Variable(inputs), Variable(targets)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += to_python_float(loss.data)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += predicted.eq(targets.data).float().cpu().sum()

                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

        test_epoch_loss = test_loss/(batch_idx+1)
        test_epoch_acc = 100.*correct/total

        statstr = 'Testing: Epoch=%d | Loss: %.3f |  Acc: %.3f%% (%d/%d) | best_acc: %.3f' \
                  % (epoch, test_epoch_loss, test_epoch_acc, correct, total, best_acc)
        statfile.write(statstr+'\n')

        # Save checkpoint.
        acc = test_epoch_acc
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        torch.save(state, checkpointpath + modelname + '_last_ckpt.t7')
        if acc >= best_acc:
            print('Saving..')
            torch.save(state, checkpointpath + modelname + '_best_ckpt.t7')
            best_acc = acc

        # Append metrics for CSV logging
        test_loss_list.append(test_epoch_loss)
        test_acc_list.append(test_epoch_acc)

        return test_epoch_loss, test_epoch_acc

    # Set adaptive learning rates
    def decrease_learning_rate():
        """Decay the previous learning rate by 10"""
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 10


    for epoch in range(start_epoch, start_epoch+300):
        statfile = open(logpath+'training_stats_'+modelname+'.txt', 'a+')
        if epoch==150 or epoch==225 or epoch == 262:
            decrease_learning_rate()
        train(epoch)
        test(epoch)
        statfile.close()

    # Create DataFrame from metrics
    metrics_df = pd.DataFrame({
        'Epoch': epochs_list,
        'Train Loss': train_loss_list,
        'Train Accuracy': train_acc_list,
        'Test Loss': test_loss_list,
        'Test Accuracy': test_acc_list
    })

    # Save metrics to CSV
    csv_filename = logpath + 'SM_training_metrics_' + modelname + '.csv'
    metrics_df.to_csv(csv_filename, index=False)

if __name__ == '__main__':
    cls_ct = [5,10,15]
    for cls_ct_i in cls_ct:
        print("running model with circles=" + str(cls_ct_i))
        main_cifar(circles=cls_ct_i)
    #main_cifar(circles=10)