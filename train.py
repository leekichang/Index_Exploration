import time

import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader

from utils.data_gen import VaeDataset
from models.SegNet import SegNet
from utils.utils import *


def train(epoch, train_loader, model, optimizer):
    model.train()

    batch_time = ExpoAverageMeter()  # forward prop. + back prop. time
    losses = ExpoAverageMeter()  # loss (per word decoded)

    start = time.time()

    for i_batch, (x, y) in enumerate(train_loader):
        # Set device options
        x = x.to(device)
        y = y.to(device)

        # Zero gradients
        optimizer.zero_grad()

        y_hat = model(x)

        loss = torch.sqrt((y_hat - y).pow(2).mean())
        loss.backward()

        optimizer.step()

        # Keep track of metrics
        losses.update(loss.item())
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i_batch % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batcåh Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.7f} ({loss.avg:.7f})\t'.format(epoch, i_batch, len(train_loader),
                                                                  batch_time=batch_time,
                                                                  loss=losses))


def valid(val_loader, model):
    model.eval()  # eval mode (no dropout or batchnorm)

    # Loss function

    batch_time = ExpoAverageMeter()  # forward prop. + back prop. time
    losses = ExpoAverageMeter()  # loss (per word decoded)

    start = time.time()

    with torch.no_grad():
        # Batches
        for i_batch, (x, y) in enumerate(val_loader):
            # Set device options
            x = x.to(device)
            y = y.to(device)

            y_hat = model(x)
            loss = torch.sqrt((y_hat - y).pow(2).mean())

            # Keep track of metrics
            losses.update(loss.item())
            batch_time.update(time.time() - start)

            start = time.time()

            # Print status
            if i_batch % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.7f} ({loss.avg:.7f})\t'.format(i_batch, len(val_loader),
                                                                      batch_time=batch_time,
                                                                      loss=losses))

    return losses.avg


def main():
    train_loader = DataLoader(dataset=VaeDataset('train'), batch_size=batch_size, shuffle=True,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(dataset=VaeDataset('valid'), batch_size=batch_size, shuffle=False,
                            pin_memory=True, drop_last=True)
    # Create SegNet model
    label_nbr = 3
    model = SegNet(label_nbr)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    # Use appropriate device
    model = model.to(device)

    # define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_loss = 100000
    epochs_since_improvement = 0

    # Epochs
    for epoch in range(start_epoch, epochs):
        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == 20:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(optimizer, 0.8)

        # One epoch's training
        train(epoch, train_loader, model, optimizer)

        # One epoch's validation
        val_loss = valid(val_loader, model)
        print('\n * LOSS - {loss:.8f}\n'.format(loss=val_loss))

        # Check if there was an improvement
        is_best = val_loss < best_loss
        best_loss = min(best_loss, val_loss)

        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(epoch, model, optimizer, val_loss, is_best)


if __name__ == '__main__':
    main()
