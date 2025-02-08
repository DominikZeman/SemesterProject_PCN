"""
Train Human action Recognition (HAR) with PyTorch. This code was used for graphing the experiments we conducted.
"""
from __future__ import print_function
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import argparse
from prednet import *
from utils import progress_bar
from torch.autograd import Variable

class HumanActionDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.labels_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.class_to_idx = {
            'calling': 0, 'clapping': 1, 'cycling': 2, 'dancing': 3,
            'drinking': 4, 'eating': 5, 'fighting': 6, 'hugging': 7,
            'laughing': 8, 'listening_to_music': 9, 'running': 10,
            'sitting': 11, 'sleeping': 12, 'texting': 13, 'using_laptop': 14
        }

    def __len__(self):
        return len(self.labels_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.labels_frame.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        label = self.class_to_idx[self.labels_frame.iloc[idx, 1]]

        if self.transform:
            image = self.transform(image)

        return image, label

def main_har(model='PredNetBpD', circles=5, gpunum=2, Tied=False, weightDecay=1e-3, nesterov=False):
    use_cuda = torch.cuda.is_available()
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    batchsize = 16
    root = './'
    root = './'
    rep = 1
    lr = 0.001

    # Lists to store metrics for CSV logging
    epochs_list = []
    train_loss_list = []
    train_acc_list = []
    test_loss_list = []
    test_acc_list = []

    models = {'PredNetBpD':PredNetBpD}
    modelname = 'HAR'+model+'_'+str(circles)+'CLS_'+str(nesterov)+'Nes_'+str(weightDecay)+'WD_'+str(Tied)+'TIED_'+str(rep)+'REP'

    # clearn folder
    checkpointpath = root+'checkpoint/'
    logpath = root+'log/'
    if not os.path.isdir(checkpointpath):
        os.mkdir(checkpointpath)
    if not os.path.isdir(logpath):
        os.mkdir(logpath)
    while(os.path.isfile(logpath+'training_stats_'+modelname+'.txt')):
        rep += 1
        modelname = 'HAR'+model+'_'+str(circles)+'CLS_'+str(nesterov)+'Nes_'+str(weightDecay)+'WD_'+str(Tied)+'TIED_'+str(rep)+'REP'

    # Data
    print('==> Preparing data..')
    # Modify transforms to handle variable image sizes
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to a standard size
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Replace dataset loading section
    trainset = HumanActionDataset(
        csv_file='har/processed/train_labels.csv',
        root_dir='har/processed/train',
        transform=transform_train
    )
    trainloader = DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=4, pin_memory=True)

    testset = HumanActionDataset(
        csv_file='har/processed/val_labels.csv',
        root_dir='har/processed/val',
        transform=transform_test
    )
    testloader = DataLoader(testset, batch_size=batchsize, shuffle=False, num_workers=4, pin_memory=True)

    # Model
    print('==> Building model..')
    net = models[model](num_classes=15,cls=circles,Tied=Tied)

    """
    # Load checkpoint if exists
    checkpoint_file = checkpointpath + 'HARPredNetBpD_15CLS_FalseNes_0.001WD_FalseTIED_10REP_last_ckpt.t7'
    if os.path.isfile(checkpoint_file):
        checkpoint = torch.load(checkpoint_file)
        net.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['acc']
    """
    # Load checkpoint if exists
    checkpoint_file = checkpointpath + 'HARPredNetBpD_15CLS_FalseNes_0.001WD_FalseTIED_10REP_last_ckpt.t7'
    if os.path.isfile(checkpoint_file):
        checkpoint = torch.load(checkpoint_file, map_location='cuda' if use_cuda else 'cpu')

        # Remove 'module.' prefix from state dict keys
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint['net'].items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v

        net.load_state_dict(new_state_dict)
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['acc']

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
                  % (epoch, train_loss/(batch_idx+1), 100.*correct/total, correct, total, best_acc)
        statfile.write(statstr+'\n')

        # Append metrics for CSV logging
        epochs_list.append(epoch)
        train_loss_list.append(train_epoch_loss)
        train_acc_list.append(train_epoch_acc)


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
                  % (epoch, test_loss/(batch_idx+1), 100.*correct/total, correct, total, best_acc)
        statfile.write(statstr+'\n')

        # Save checkpoint.
        acc = 100.*correct/total
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


    for epoch in range(start_epoch, start_epoch+150):
        statfile = open(logpath+'training_stats_'+modelname+'.txt', 'a+')
        if epoch==75 or epoch==110 or epoch == 135:
            decrease_learning_rate()
        train(epoch)
        test(epoch)

    # Create DataFrame from metrics
    metrics_df = pd.DataFrame({
        'Epoch': epochs_list,
        'Train Loss': train_loss_list,
        'Train Accuracy': train_acc_list,
        'Test Loss': test_loss_list,
        'Test Accuracy': test_acc_list
    })

    # Save metrics to CSV
    csv_filename = logpath + 'training_metrics_' + modelname + '.csv'
    metrics_df.to_csv(csv_filename, index=False)

if __name__ == '__main__':
    #cls_ct = [15]
    #for cls_ct_i in cls_ct:
    #    print("running model with circles=" + str(cls_ct_i))
    main_har(circles=15)
