"""Test a ODENet on the MNIST dataset.
"""
# %%
import os
import json

import torch
from sacred import Experiment
from pytorch_utils.sacred_trainer import read_config, get_model_path
from training_functions import validate

from model_ingredient import make_model
from data_ingredient import make_dataloaders

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

