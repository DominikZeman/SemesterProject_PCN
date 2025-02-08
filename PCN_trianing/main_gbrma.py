"""
Train GBRMA with PyTorch. This code ended up not beiong used due to the dataset being incompatible with the network architecture.
"""

import argparse
import os
import shutil
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from prednet import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EMGDataset(Dataset):
    """Custom Dataset for EMG gesture data"""
    def __init__(self, data, labels):
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def load_and_preprocess_data(data_path):
    """Load and preprocess EMG data from CSV files"""
    all_data = []
    all_labels = []

    # Load data for each gesture (0-3)
    for gesture in range(4):
        file_path = os.path.join(data_path, f"{gesture}.csv")
        data = pd.read_csv(file_path, header=None)

        # Assuming the last column is the label
        gesture_data = data.iloc[:, :-1].values
        gesture_labels = data.iloc[:, -1].values

        # Flatten the data to 64-element vectors and normalize
        flattened_data = gesture_data.reshape(-1, 64)
        normalized_data = (flattened_data - flattened_data.min()) / (flattened_data.max() - flattened_data.min())

        all_data.append(normalized_data)
        all_labels.extend(gesture_labels)

    all_data = np.vstack(all_data)
    all_labels = np.array(all_labels)

    return all_data, all_labels

def create_data_loaders(data, labels, batch_size, num_workers):
    """Create train, validation, and test data loaders with 70/15/15 split"""
    # First split: 70% train, 30% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        data, labels, test_size=0.3, random_state=42, stratify=labels
    )

    # Second split: Split temp into validation and test (50/50)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # Create datasets
    train_dataset = EMGDataset(X_train, y_train)
    val_dataset = EMGDataset(X_val, y_val)
    test_dataset = EMGDataset(X_test, y_test)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader

def train_emg():
    parser = argparse.ArgumentParser(description='PyTorch EMG Gesture Training')
    parser.add_argument('--data', default='gbrma', type=str, help='path to dataset')
    parser.add_argument('--workers', default=4, type=int, help='number of data loading workers')
    parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number')
    parser.add_argument('--batch-size', default=32, type=int, help='mini-batch size')
    parser.add_argument('--circles', default=3, type=int, help='PCN circles')
    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--resume', default='', type=str, help='path to checkpoint')
    args = parser.parse_args()

    # Create model
    model = EMGPredNetBpE(num_classes=4, cls=args.circles)
    model = torch.nn.DataParallel(model).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                 weight_decay=args.weight_decay)

    # Optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")

    # Load and prepare data
    data, labels = load_and_preprocess_data(args.data)
    train_loader, val_loader, test_loader = create_data_loaders(
        data, labels, args.batch_size, args.workers
    )

    # Training loop
    best_acc = 0
    for epoch in range(args.start_epoch, args.epochs):
        # Train for one epoch
        train_one_epoch(train_loader, model, criterion, optimizer, epoch, args)

        # Evaluate on validation set
        acc = validate(val_loader, model, criterion, epoch, args)

        # Remember best accuracy and save checkpoint
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best)

    # Final evaluation on test set
    print("Evaluating on test set...")
    test_acc = validate(test_loader, model, criterion, args.epochs, args)
    print(f"Final test accuracy: {test_acc:.2f}%")

def train_one_epoch(train_loader, model, criterion, optimizer, epoch, args):
    """Training loop for one epoch"""
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    model.train()
    end = time.time()

    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        input = input.to(device)
        target = target.to(device)

        # Compute output and loss
        output = model(input)
        loss = criterion(output, target)

        # Measure accuracy and record loss
        prec = accuracy(output.data, target)
        losses.update(loss.item(), input.size(0))
        acc.update(prec.item(), input.size(0))

        # Compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                  f'Acc {acc.val:.3f} ({acc.avg:.3f})')

def validate(val_loader, model, criterion, epoch, args):
    """Validation/Testing loop"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    model.eval()
    end = time.time()

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input = input.to(device)
            target = target.to(device)

            # Compute output and loss
            output = model(input)
            loss = criterion(output, target)

            # Measure accuracy and record loss
            prec = accuracy(output.data, target)
            losses.update(loss.item(), input.size(0))
            acc.update(prec.item(), input.size(0))

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print(f'Test: [{i}/{len(val_loader)}]\t'
                      f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                      f'Acc {acc.val:.3f} ({acc.avg:.3f})')

    print(f' * Accuracy {acc.avg:.3f}')
    return acc.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target):
    """Computes the accuracy"""
    batch_size = target.size(0)
    _, pred = output.max(1)
    correct = pred.eq(target).float().sum(0)
    return correct.mul_(100.0 / batch_size)

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Save checkpoint"""
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    filepath = os.path.join('checkpoints', filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join('checkpoints', 'model_best.pth.tar'))

if __name__ == '__main__':
    train_emg()