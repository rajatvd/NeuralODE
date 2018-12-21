"""Ingredient for making mnist data loaders."""

import torch
from mnist_dataset import MyMNIST
from sacred import Ingredient

data_ingredient = Ingredient('dataset')

@data_ingredient.config
def data_config():
    """Config for data source and loading"""
    batch_size = 32
    device = 'cpu'
    val_split = 0.05
    num_workers = 0 # number of subprocesses apart from main for data loading

@data_ingredient.capture
def make_dataloaders(batch_size,
                     num_workers,
                     val_split,
                     device,
                     _log):
    """Make the required DataLoaders and datasets.

    Parameters
    ----------
    batch_size : int
        batch_size for DataLoader, default is 32.
    num_workers : int
        num_workers for DataLoader, default is 0.
    val_split : float
        ratio of dataset used for validation, default is 0.01.
    device : str
        device to load the DataLoader
    _log : logger
        logger instance

    Returns
    -------
    tuple: dataset, train_loader, val_loader, test_loader
        Returns the dataset, and the train, validation, and test DataLoaders.

    """

    dset = MyMNIST("data", download=True, device=device)
    test_dset = MyMNIST("data", download=True, train=False, device=device)

    _log.info(f"Loaded dataset on {device}")

    total = len(dset)
    train_num = int(total*(1-val_split))
    val_num = total-train_num

    _log.info(f"Split dataset into {train_num} train samples and {val_num} \
    validation samples")

    train, val = torch.utils.data.dataset.random_split(dset,
                                                       [train_num, val_num])

    train_loader = torch.utils.data.DataLoader(
        train,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,)

    val_loader = torch.utils.data.DataLoader(
        val,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,)

    test_loader = torch.utils.data.DataLoader(
        test_dset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,)

        # next(iter(train_loader))

    return dset, train_loader, val_loader, test_loader
