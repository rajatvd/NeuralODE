"""
Run this script to train a ConvNet on MNIST.
"""
import torch
from torch import optim

from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
from visdom_observer.visdom_observer import VisdomObserver
import pytorch_utils.sacred_trainer as st
from pytorch_utils.updaters import averager

from model_ingredient import model_ingredient, make_model
from data_ingredient import data_ingredient, make_dataloaders

from training_functions import train_on_batch, create_val_scheduler_callback

torch.backends.cudnn.benchmark = True

SETTINGS.CAPTURE_MODE = 'no'

ex = Experiment('odenet_mnist',
                ingredients=[model_ingredient, data_ingredient])
SAVE_DIR = 'ODEMnistClassification'
ex.observers.append(FileStorageObserver.create(SAVE_DIR))
ex.observers.append(VisdomObserver())


# ----------------OPTIMIZER-----------------

@ex.config
def optimizer_config():
    """Config for optimzier
    Currently available opts (types of optimizers):
        adam
        adamax
        rmsprop
    """
    lr = 0.001 # learning rate
    opt = 'adam' # type of optimzier
    weight_decay = 0 # l2 regularization weight_decay (lambda)


@ex.capture
def make_optimizer(model, lr, opt, weight_decay):
    """Make an optimizer of the given type (opt), for the given model's
    parameters with the given learning rate (lr)"""
    optimizers = {
        'adam':optim.Adam,
        'adamax':optim.Adamax,
        'rmsprop':optim.RMSprop,
    }

    optimizer = optimizers[opt](model.parameters(), lr=lr,
                                weight_decay=weight_decay)

    return optimizer


# -----------CALLBACK FOR LR SCHEDULING-------------

@ex.config
def scheduler_config():
    """Config for lr scheduler"""
    milestones = [50, 100]
    gamma = 0.5 # factor to reduce lr by at each milestone

@ex.capture
def make_scheduler_callback(optimizer, milestones, gamma):
    """Create a MultiStepLR scheduler callback for the optimizer
    using the config"""
    return create_val_scheduler_callback(optimizer, milestones, gamma)


@ex.config
def train_config():
    epochs = 100
    save_every = 1
    start_epoch = 1

@ex.automain
def main(_run):

    dset, train, val, test = make_dataloaders()
    model = make_model()
    optimizer = make_optimizer(model)
    callback = make_scheduler_callback(optimizer)

    st.loop(
        **{**_run.config,
           **dict(_run=_run,
                  model=model,
                  optimizer=optimizer,
                  save_dir=SAVE_DIR,
                  trainOnBatch=train_on_batch,
                  train_loader=train,
                  val_loader=val,
                  callback=callback,
                  callback_metric_names=['val_loss', 'val_acc', 'learning_rate'],
                  batch_metric_names=['loss', 'acc', 'nfef', 'nfeb'],
                  updaters=[averager]*4)})