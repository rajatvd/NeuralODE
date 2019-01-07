"""
Train on batch and other functions for training a ODEnet on MNIST
"""
import torch
from torch import nn
from torch import optim
from tqdm import tqdm
import logging

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
    if isinstance(model, nn.DataParallel):
        ode_model = model.module

    criterion = nn.CrossEntropyLoss()

    images, labels = batch

    ode_model.odefunc.nfe = 0
    outputs = model(images)
    nfe_forward = ode_model.odefunc.nfe
    loss = criterion(outputs, labels)

    # backward and optimize
    ode_model.odefunc.nfe = 0
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    nfe_backward = ode_model.odefunc.nfe

    loss = loss.cpu().detach().numpy()
    acc = st.accuracy(outputs.cpu(), labels.cpu())
    return loss, acc, nfe_forward, nfe_backward



def validate(model, val_loader, _log=logging.getLogger('validate')):
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


    model = model.eval()
    val_loss = 0
    accuracy = 0
    total = 0
    _log.info(f"Running validate with {len(val_loader)} steps")
    for images, labels in tqdm(val_loader):
        with torch.no_grad():
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
