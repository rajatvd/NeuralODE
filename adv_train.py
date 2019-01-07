"""
Run this script to train a ConvNet on MNIST.
"""
from functools import partial

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

from adversarial import fgsm, pgd, AdversarialLoader

torch.backends.cudnn.benchmark = True
SETTINGS.CAPTURE_MODE = 'no'

ex = Experiment('adv-odenet_mnist_randtime',
                ingredients=[model_ingredient, data_ingredient])
SAVE_DIR = 'runs/AdvODEnetRandTimeMnist'
ex.observers.append(FileStorageObserver.create(SAVE_DIR))
ex.observers.append(VisdomObserver())


ATTACKS = {
    'fgsm':fgsm,
    'pgd':pgd
}

# ------- COMBINING TRAIN AND ADVERSARIAL LOADERS -------
class CombineDataloaders:
    """Combine mutliple dataloaders or iterators which yield images and labels
    into one iterator.

    Parameters
    ----------
    *loaders : type
        List of dataloaders to combine.

    """
    def __init__(self, *loaders):
        self.loaders = loaders

    def __iter__(self):
        iters = [iter(loader) for loader in self.loaders]
        while True:
            items = [next(it) for it in iters]
            images = torch.cat([i[0] for i in items])
            labels = torch.cat([i[1] for i in items])
            yield images, labels

    def __len__(self):
        return min([len(loader) for loader in self.loaders])


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

@ex.config
def attack_config():
    attack = 'pgd'
    epsilon = 0.3
    pgd_step_size = 0.01
    pgd_num_steps = 40
    pgd_random_start = True

@ex.automain
def main(_run,
         attack,
         epsilon,
         pgd_step_size,
         pgd_num_steps,
         pgd_random_start,):

    dset, train, val, test = make_dataloaders()
    model = make_model()
    optimizer = make_optimizer(model)
    callback = make_scheduler_callback(optimizer)

    if attack == 'pgd':
        attack_fn = partial(ATTACKS[attack],
                            epsilon=epsilon,
                            step_size=pgd_step_size,
                            num_steps=pgd_num_steps,
                            random_start=pgd_random_start)
    else:
        attack_fn = partial(ATTACKS[attack], epsilon=epsilon)

    adv_train = AdversarialLoader(model, train, attack_fn)
    final_train = CombineDataloaders(train, adv_train)

    st.loop(
        **{**_run.config,
           **dict(_run=_run,
                  model=model,
                  optimizer=optimizer,
                  save_dir=SAVE_DIR,
                  trainOnBatch=train_on_batch,
                  train_loader=final_train,
                  val_loader=val,
                  callback=callback,
                  callback_metric_names=['val_loss', 'val_acc', 'learning_rate'],
                  batch_metric_names=['loss', 'acc', 'nfef', 'nfeb'],
                  updaters=[averager]*4)})