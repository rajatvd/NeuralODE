"""Test a ConvNet on the MNIST dataset.
"""
# %%
import os
import json

import torch
from sacred import Experiment

from training_functions import validate

from model_ingredient import make_model
from data_ingredient import make_dataloaders

# %%
def remove_key(d, key):
    """Remove the key from the given dictionary and all its sub dictionaries.
    Mutates the dictionary.

    Parameters
    ----------
    d : dict
        Input dictionary.
    key : type
        Key to recursively remove.

    Returns
    -------
    None
    """
    for k in list(d.keys()):
        if isinstance(d[k], dict):
            remove_key(d[k], key)
        elif k == key:
            del d[k]

def read_config(run_dir):
    """Read the config json from the given run directory"""
    with open(os.path.join(run_dir, 'config.json')) as file:
        config = json.loads(file.read())
        remove_key(config, key='__doc__')

    return config

def get_model_path(run_dir, epoch):
    """Get the path to the saved model state_dict with the given epoch number
    If epoch is 'latest', the latest model state dict path will be returned.
    """
    if epoch == 'latest':
        return os.path.join(run_dir, 'latest.statedict.pkl')

    filenames = os.listdir(run_dir)

    for filename in filenames:
        if 'statedict' not in filename:
            continue
        if filename.startswith('epoch'):
            number = int(filename[len('epoch'):].split('_')[0])
            if epoch == number:
                return os.path.join(run_dir, filename)

    raise ValueError(f"No statedict found with epoch number '{epoch}'")

# %%
ex = Experiment('test_mnist')

@ex.config
def input_config():
    """Parameters for sampling using the given model"""
    run_dir = 'MnistClassification/8'
    epoch = 'latest'
    device = 'cpu'

@ex.automain
def main(run_dir,
         epoch,
         device,
         _log):

    config = read_config(run_dir)
    _log.info(f"Read config from {run_dir}")

    model = make_model(**{**config['model'], 'device':device}, _log=_log)
    path = get_model_path(run_dir, epoch)
    model.load_state_dict(torch.load(path))
    model = model.eval()
    _log.info(f"Loaded state dict from {path}")

    dset, tl, vl, test_loader = make_dataloaders(**{**config['dataset'],
                                                    'device':device},
                                                 _log=_log)
    _log.info("Testing model...")
    test_loss, test_acc = validate(model, test_loader)

    _log.info(f"Test loss = {test_loss:.6f}, Test accuracy = {test_acc:.4f}")

