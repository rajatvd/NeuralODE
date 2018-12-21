"""
Train on batch function for training a ConvNet on MNIST
"""
import torch
from torch import nn
from torch import optim

import pytorch_utils.sacred_trainer as st


def train_on_batch(model, batch, optimizer):
    """One train step on batch of MNIST data. Uses CrossEntropyLoss.

    Parameters
    ----------
    model : nn.Module
        Model for MNIST classification.
    batch : tuple
        Tuple of images and labels
    optimizer : torch Optimizer
        Description of parameter `optimizer`.

    Returns
    -------
    tuple: loss, accuracy
        Both are numpy
    """


    criterion = nn.CrossEntropyLoss()

    images, labels = batch
    outputs = model(images)
    loss = criterion(outputs, labels)

    # backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.cpu().detach().numpy(), st.accuracy(outputs.cpu(), labels.cpu())


def validate(model, val_loader):
    """Find loss and accuracy on the given dataloader using the model.

    Parameters
    ----------
    model : nn.Module
        Model for MNIST classification.
    val_loader : DataLoader
        Data over which to validate.

    Returns
    -------
    tuple: val_loss, accuracy
        Both are numpy
    """

    with torch.no_grad():
        model = model.eval()

        val_loss = 0
        accuracy = 0
        total = 0
        for images, labels in val_loader:

            criterion = nn.CrossEntropyLoss()
            batch_size = images.shape[0]

            # forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss*batch_size
            accuracy += st.accuracy(outputs, labels)*batch_size
            total += batch_size

        val_loss /= total
        accuracy /= total

        model = model.train()
        return val_loss.cpu().numpy(), accuracy

def scheduler_generator(optimizer, milestones, gamma):
    """A generator which performs lr scheduling on the given optimizer using
    a MultiStepLR scheduler with given milestones and gamma."""
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones,
                                               gamma)
    while True:
        scheduler.step()
        yield (optimizer.param_groups[0]['lr'],) # yield to return the lr

def create_scheduler_callback(optimizer, milestones, gamma):
    """Returns a function which can be used as callback for lr scheduling
    on the given optimizer using a MultiStepLR scheduler with given
    milestones and gamma."""

    g = scheduler_generator(optimizer, milestones, gamma)
    def scheduler_callback(model, val_loader, batch_metrics_dict):
        """LR scheduler callback using the next function of a
        scheduler_generator"""

        return next(g)

    return scheduler_callback

def create_val_scheduler_callback(optimizer, milestones, gamma):
    """Returns a function which can be used as callback for lr scheduling
    on the given optimizer using a MultiStepLR scheduler with given
    milestones and gamma.

    It also computes loss on the validation data loader.
    """

    g = scheduler_generator(optimizer, milestones, gamma)
    def scheduler_callback(model, val_loader, batch_metrics_dict):
        """LR scheduler callback using the next function of a
        scheduler_generator"""

        val_loss, val_accuracy = validate(model, val_loader)
        lr = next(g)

        return val_loss, val_accuracy, lr[0]

    return scheduler_callback
